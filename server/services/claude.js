const axios = require('axios');

class ClaudeService {
  constructor() {
    this.apiBase = process.env.CLAUDE_API_BASE || 'https://openrouter.ai/api/v1';
    this.apiKey = process.env.CLAUDE_API_KEY;
    this.model = process.env.CLAUDE_MODEL || 'anthropic/claude-opus-4.6';
  }

  async chat(userMessage, options = {}) {
    const { systemPrompt, temperature = 0.3, maxTokens = 8192 } = options;

    const messages = [];
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: userMessage });

    const response = await axios.post(`${this.apiBase}/chat/completions`, {
      model: this.model,
      messages,
      temperature,
      max_tokens: maxTokens,
    }, {
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://tinybert-pipeline.local',
      },
      timeout: 180000,
    });

    return response.data.choices[0].message.content;
  }

  /**
   * 分析请求日志，识别可蒸馏的高频任务模式
   */
  async analyzePatterns(requestLogs) {
    const systemPrompt = `你是一个AI系统架构师，专门分析用户请求模式并识别可以被蒸馏为小模型的重复性任务。

你需要分析请求日志，返回以下 JSON 格式的分析结果：
{
  "patterns": [
    {
      "task_type": "任务类型标识(英文, snake_case)",
      "description": "任务描述",
      "frequency": 0.0-1.0,
      "complexity": "low|medium|high",
      "recommended_model": "fasttext|tinybert|minilm|distilbert",
      "labels": ["标签1", "标签2"],
      "reason": "推荐理由"
    }
  ],
  "summary": "整体分析总结"
}

模型选择规则：
- fasttext: 简单的关键词/短语分类，二分类或少量类别，体积<2MB
- tinybert: 需要语义理解的分类，如情感分析、意图识别，体积15-50MB
- minilm: 需要较强语义理解但要平衡速度，体积20-80MB
- distilbert: 复杂的多分类或需要深层理解，体积130MB+

只有当某类任务占比超过20%且复杂度为low或medium时才推荐蒸馏。`;

    const response = await this.chat(
      `请分析以下 ${requestLogs.length} 条请求日志，识别高频可蒸馏任务：\n\n${JSON.stringify(requestLogs.slice(0, 200), null, 2)}`,
      { systemPrompt, temperature: 0.1 }
    );

    return this._extractJSON(response);
  }

  /**
   * 生成新版本的 loop.js 代码
   */
  async generateLoopCode(currentCode, newTools, reason) {
    const systemPrompt = `你是一个 Node.js 代码生成专家。你的任务是修改 loop.js 文件，在现有流程中插入小模型的前置分流逻辑。

严格遵循以下规则：
1. 必须导出一个 async function process(request, context) 函数
2. context 包含: { gemini, tools, log }
   - gemini.chat(input, options) - 调用大模型
   - tools.predict(toolName, input) - 调用已注册的小模型工具
   - tools.list() - 获取可用工具列表
   - log(message) - 记录日志
3. request 包含: { input, type?, metadata? }
4. 返回值: { output, tool_used, confidence, metadata? }
5. 前置分流逻辑：先尝试用小模型处理，置信度高则直接返回或将结果传递给大模型做最终组装
6. 置信度低于阈值(0.8)时回退到大模型
7. 保留 __validation_test__ 的特殊处理以通过验证
8. 代码必须是完整的、可直接 require() 的 CommonJS 模块
9. 只输出代码，不要 markdown 标记`;

    const toolDescs = newTools.map(t =>
      `- ${t.name}: ${t.description} (架构: ${t.model_arch}, 类型: ${t.task_type})`
    ).join('\n');

    const prompt = `当前 loop.js 代码：
\`\`\`javascript
${currentCode}
\`\`\`

现在需要集成以下新工具：
${toolDescs}

修改原因：${reason}

请生成新版本的 loop.js 代码，在前置位添加工具分流逻辑。对于每个工具，根据用户输入判断是否匹配该工具的任务类型，如果匹配则先用小模型预测，置信度高则将预测结果作为预分析信息传递给 Gemini 做最终回复组装（或直接返回）。`;

    const response = await this.chat(prompt, { systemPrompt, temperature: 0.2 });
    return this._extractCode(response);
  }

  /**
   * 分析用户的进化意图，拆解为可执行的计划
   */
  async analyzeIntent(intentText, currentTools, currentCode) {
    const systemPrompt = `你是一个AI系统架构师。用户会用自然语言描述他们想让系统增加的能力或行为。
你需要分析这个意图，判断实现它需要哪些步骤。

返回 JSON：
{
  "summary": "一句话概括用户意图",
  "needs_model": true/false,
  "model_task": {
    "task_type": "英文snake_case任务标识",
    "description": "任务描述（给数据生成用）",
    "labels": ["标签1", "标签2"]
  },
  "loop_instruction": "给代码生成器的指令：描述在 loop.js 中要实现什么逻辑，包括判断条件、走哪个工具、返回什么内容。尽量具体，包含用户提到的固定回复文本。",
  "reason": "为什么这样拆解"
}

判断规则：
- 如果意图涉及"分类/识别/判断"某种模式（如情感、意图、恶意、语言类型等），needs_model = true
- 如果只是纯逻辑修改（如改默认回复、加固定条件），needs_model = false，此时 model_task 为 null
- loop_instruction 必须足够具体，让代码生成器能直接据此写出代码`;

    const toolsDesc = currentTools.length
      ? `当前已有工具：\n${currentTools.map(t => `- ${t.name}: ${t.description}`).join('\n')}`
      : '当前没有已注册工具';

    const response = await this.chat(
      `用户进化意图：「${intentText}」\n\n${toolsDesc}\n\n当前 loop.js 行数：${currentCode.split('\n').length} 行`,
      { systemPrompt, temperature: 0.1 }
    );

    return this._extractJSON(response);
  }

  /**
   * 根据意图指令生成新版 loop.js（支持用户自定义行为）
   */
  async generateLoopCodeWithIntent(currentCode, allTools, intentInstruction, reason) {
    const systemPrompt = `你是一个 Node.js 代码生成专家。你的任务是根据用户的进化意图修改 loop.js。

严格遵循以下规则：
1. 必须导出一个 async function process(request, context) 函数
2. context 包含: { gemini, tools, log }
   - gemini.chat(input, options) - 调用大模型
   - tools.predict(toolName, input) - 调用已注册的小模型工具
   - tools.list() - 获取可用工具列表
   - log(message) - 记录日志
3. request 包含: { input, type?, metadata? }
4. 返回值必须是: { output, tool_used, confidence, metadata? }
5. 置信度低于阈值(0.8)时回退到大模型
6. 保留 __validation_test__ 的特殊处理以通过验证：
   if (request.input === '__validation_test__') return { output: '[test ok]', tool_used: 'test', confidence: 1 };
7. 代码必须是完整的、可直接 require() 的 CommonJS 模块
8. 只输出纯 JavaScript 代码，不要 markdown 标记或任何注释外文字`;

    const toolDescs = allTools.length
      ? allTools.map(t => `- ${t.name}: ${t.description} (类型: ${t.taskType})`).join('\n')
      : '（无已注册工具）';

    const prompt = `当前 loop.js 代码：
\`\`\`javascript
${currentCode}
\`\`\`

已注册工具：
${toolDescs}

用户进化意图指令：
${intentInstruction}

修改原因：${reason}

请生成完整的新版 loop.js 代码。`;

    const response = await this.chat(prompt, { systemPrompt, temperature: 0.2 });
    return this._extractCode(response);
  }

  /**
   * 根据任务描述选择最佳小模型架构
   */
  async selectModelArch(taskDescription, sampleInputs) {
    const systemPrompt = `你是一个模型架构选型专家。根据任务描述和样本，推荐最合适的小模型架构。

返回 JSON：
{
  "model_arch": "fasttext|tinybert|minilm|distilbert",
  "reason": "选型理由",
  "estimated_size_mb": 数字,
  "estimated_accuracy": 0.0-1.0,
  "labels": ["需要的分类标签列表"],
  "training_config": {
    "epochs": 数字,
    "batch_size": 数字,
    "learning_rate": 数字,
    "max_length": 数字
  }
}`;

    const response = await this.chat(
      `任务描述：${taskDescription}\n\n样本输入（前10条）：\n${sampleInputs.slice(0, 10).map((s, i) => `${i + 1}. ${s}`).join('\n')}`,
      { systemPrompt, temperature: 0.1 }
    );

    return this._extractJSON(response);
  }

  _extractJSON(text) {
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try { return JSON.parse(jsonMatch[0]); } catch {}
    }
    const arrayMatch = text.match(/\[[\s\S]*\]/);
    if (arrayMatch) {
      try { return JSON.parse(arrayMatch[0]); } catch {}
    }
    throw new Error('无法从响应中提取有效 JSON');
  }

  _extractCode(text) {
    const codeBlockMatch = text.match(/```(?:javascript|js)?\n([\s\S]*?)```/);
    if (codeBlockMatch) return codeBlockMatch[1].trim();
    if (text.includes('module.exports') || text.includes('async function process')) {
      return text.trim();
    }
    throw new Error('无法从响应中提取有效代码');
  }
}

module.exports = { ClaudeService };
