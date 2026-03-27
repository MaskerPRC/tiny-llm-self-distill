const axios = require('axios');

/**
 * IntentService - 使用 GPT-5.4 分析用户的进化意图
 *
 * 职责：将用户的自然语言意图拆解为可执行的计划，
 * 判断是否需要训练小模型，生成给代码生成器的具体指令。
 */
class IntentService {
  constructor() {
    this.apiBase = process.env.INTENT_API_BASE || process.env.SELECTOR_API_BASE || 'https://openrouter.ai/api/v1';
    this.apiKey = process.env.INTENT_API_KEY || process.env.SELECTOR_API_KEY;
    this.model = process.env.INTENT_MODEL || process.env.SELECTOR_MODEL || 'openai/gpt-5.4';
  }

  async chat(userMessage, options = {}) {
    const { systemPrompt, temperature = 0.1, maxTokens = 4096 } = options;

    const messages = [];
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: userMessage });

    const body = {
      model: this.model,
      messages,
      temperature,
      response_format: { type: 'json_object' },
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
        timeout: 120000,
      });
      return response.data.choices[0].message.content;
    } catch (err) {
      const detail = err.response?.data?.error?.message || err.response?.data || err.message;
      console.error(`[Intent] API ${err.response?.status}: ${typeof detail === 'object' ? JSON.stringify(detail) : detail}`);
      throw err;
    }
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
  "needs_model": true或false,
  "model_task": {
    "task_type": "英文snake_case任务标识",
    "task_mode": "classify|ner|similarity|regression",
    "description": "任务描述（给数据生成用）",
    "labels": ["标签列表：classify用分类标签，ner用实体类型（如PER/LOC/ORG），similarity用关系标签"]
  },
  "loop_instruction": "给代码生成器的详细指令：描述在 loop.js 的 process 函数中要实现什么逻辑，包括判断条件、走哪个工具、返回什么内容。必须包含用户提到的固定回复文本原文。",
  "reason": "为什么这样拆解"
}

task_mode 判断规则：
- classify: 判断文本属于哪个类别（情感、意图、恶意等）
- ner: 从文本中提取人名/地名/组织/产品等实体
- similarity: 判断两段文本的关系（相似/矛盾/蕴含/问答匹配）
- regression: 给文本打一个连续分数（质量/严重度/相关性 0~1）

needs_model 判断规则：
- 如果意图涉及AI来"分类/识别/判断/提取/打分"某种模式，needs_model = true
- 如果只是纯逻辑修改（改默认回复、加固定条件、关键词过滤），needs_model = false，model_task 设为 null
- loop_instruction 极其重要，必须足够具体
- 如果 needs_model 为 true，loop_instruction 中要说明 tool 名称（格式: {task_type}_{model_arch}）和该 tool 的 task_mode`;

    const toolsDesc = currentTools.length
      ? `当前已有工具：\n${currentTools.map(t => `- ${t.name}: ${t.description}`).join('\n')}`
      : '当前没有已注册工具';

    const response = await this.chat(
      `用户进化意图：「${intentText}」\n\n${toolsDesc}\n\n当前 loop.js 行数：${currentCode.split('\n').length} 行`,
      { systemPrompt }
    );

    return this._extractJSON(response);
  }

  _extractJSON(text) {
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try { return JSON.parse(jsonMatch[0]); } catch {}
    }
    throw new Error('无法从GPT-5.4响应中提取有效JSON');
  }
}

module.exports = { IntentService };
