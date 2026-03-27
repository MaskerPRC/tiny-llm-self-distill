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

    const body = {
      model: this.model,
      messages,
      temperature,
    };
    if (/^(gpt-|o[1-9])/.test(this.model)) {
      body.max_completion_tokens = maxTokens;
    } else {
      body.max_tokens = maxTokens;
    }

    try {
      const response = await axios.post(`${this.apiBase}/chat/completions`, body, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': 'https://tinybert-pipeline.local',
        },
        timeout: 180000,
      });
      return response.data.choices[0].message.content;
    } catch (err) {
      const detail = err.response?.data?.error?.message || err.response?.data || err.message;
      console.error(`[Claude] API ${err.response?.status}: ${typeof detail === 'object' ? JSON.stringify(detail) : detail}`);
      throw err;
    }
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
   - tools.predict(toolName, input) - 调用小模型工具
   - tools.list() - 获取可用工具列表
   - log(message) - 记录日志
3. request 包含: { input, type?, metadata? }
4. 返回值: { output, tool_used, confidence, metadata? }
5. 前置分流逻辑：先尝试用小模型处理，置信度高则直接返回或传给大模型做最终组装
6. 置信度低于阈值(0.8)时回退到大模型
7. 保留 __validation_test__ 的特殊处理以通过验证
8. 代码必须是完整的、可直接 require() 的 CommonJS 模块
9. 只输出代码，不要 markdown 标记

不同 task_mode 的工具调用方式和返回值：
- classify: tools.predict(name, "字符串") → { label, confidence, probabilities }
- ner: tools.predict(name, "字符串") → { entities: [{label, start, end, text}], entity_count }
- similarity: tools.predict(name, {text_a, text_b}) → { label, confidence } 或 { score }
- regression: tools.predict(name, "字符串") → { score, confidence }

每个工具的 labels 是训练时的精确标签名，代码中必须使用这些标签名判断。`;

    const toolDescs = newTools.map(t => {
      const mode = t.task_mode || t.taskMode || 'classify';
      const labelInfo = t.labels?.length ? ` | labels: [${t.labels.map(l => `"${l}"`).join(', ')}]` : '';
      return `- ${t.name}: ${t.description} (mode: ${mode}, arch: ${t.model_arch}, type: ${t.task_type}${labelInfo})`;
    }).join('\n');

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
   * 根据意图指令生成新版 loop.js（支持用户自定义行为）
   */
  async generateLoopCodeWithIntent(currentCode, allTools, intentInstruction, reason) {
    const systemPrompt = `你是一个 Node.js 代码生成专家。你的任务是根据用户的进化意图修改 loop.js。

严格遵循以下规则：
1. 必须导出一个 async function process(request, context) 函数
2. context 包含: { gemini, tools, log }
   - gemini.chat(input, options) - 调用大模型
   - tools.predict(toolName, input) - 调用小模型工具
   - tools.list() - 获取可用工具列表
   - log(message) - 记录日志
3. request 包含: { input, type?, metadata? }
4. 返回值必须是: { output, tool_used, confidence, metadata? }
5. 置信度低于阈值(0.8)时回退到大模型
6. 保留 __validation_test__ 的特殊处理以通过验证：
   if (request.input === '__validation_test__') return { output: '[test ok]', tool_used: 'test', confidence: 1 };
7. 代码必须是完整的、可直接 require() 的 CommonJS 模块
8. 只输出纯 JavaScript 代码，不要 markdown 标记或任何注释外文字

不同 task_mode 的工具调用方式和返回值：

### classify（分类）
调用: tools.predict(toolName, "纯文本字符串")
返回: { label: "精确标签名", confidence: 0.0-1.0, probabilities: { "标签1": 0.x, ... } }
用法: 根据 label 值判断分类结果，必须使用工具 labels 字段中的精确标签名

### ner（实体识别）
调用: tools.predict(toolName, "纯文本字符串")
返回: { entities: [{ label: "实体类型", start: 0, end: 3, text: "提取的文本" }], entity_count: 数字 }
用法: 遍历 entities 数组获取提取的实体

### similarity（句子对关系）
调用: tools.predict(toolName, { text_a: "文本A", text_b: "文本B" })
返回: { label: "关系标签", confidence: 0.0-1.0 } 或 { score: 0.0-1.0 }
用法: 注意输入是对象不是字符串

### regression（文本打分）
调用: tools.predict(toolName, "纯文本字符串")
返回: { score: 0.0-1.0, confidence: 1.0 }
用法: 根据 score 值判断程度`;

    const toolDescs = allTools.length
      ? allTools.map(t => {
          const mode = t.taskMode || 'classify';
          const labelInfo = t.labels?.length ? ` | labels: [${t.labels.map(l => `"${l}"`).join(', ')}]` : '';
          return `- ${t.name}: ${t.description} (mode: ${mode}, type: ${t.taskType}${labelInfo})`;
        }).join('\n')
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
