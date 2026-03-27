const axios = require('axios');

class SelectorService {
  constructor() {
    this.apiBase = process.env.SELECTOR_API_BASE || 'https://openrouter.ai/api/v1';
    this.apiKey = process.env.SELECTOR_API_KEY;
    this.model = process.env.SELECTOR_MODEL || 'openai/gpt-5.4';
    this.candidates = (process.env.MODEL_CANDIDATES || 'fasttext,tinybert,minilm,distilbert')
      .split(',').map(s => s.trim());
  }

  async chat(userMessage, options = {}) {
    const { systemPrompt, temperature = 0.1, maxTokens = 4096 } = options;

    const messages = [];
    if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
    messages.push({ role: 'user', content: userMessage });

    const body = { model: this.model, messages, temperature, response_format: { type: 'json_object' } };
    if (/^(gpt-|o[1-9])/.test(this.model)) {
      body.max_completion_tokens = maxTokens;
    } else {
      body.max_tokens = maxTokens;
    }

    try {
      const response = await axios.post(`${this.apiBase}/chat/completions`, body, {
        headers: { 'Authorization': `Bearer ${this.apiKey}`, 'Content-Type': 'application/json', 'HTTP-Referer': 'https://tinybert-pipeline.local' },
        timeout: 120000,
      });
      return response.data.choices[0].message.content;
    } catch (err) {
      const detail = err.response?.data?.error?.message || err.response?.data || err.message;
      console.error(`[Selector] API ${err.response?.status}: ${typeof detail === 'object' ? JSON.stringify(detail) : detail}`);
      throw err;
    }
  }

  async selectModelArch(taskDescription, sampleInputs, previousAttempt) {
    const archSpecs = {
      fasttext: { name: 'FastText', size: '<2MB', task_modes: ['classify'], strengths: '关键词级分类，极快(<1ms)', fit_for: '简单意图路由、脏话检测' },
      tinybert: { name: 'RBT3 (3层)', size: '15-50MB', task_modes: ['classify', 'ner', 'similarity', 'regression'], strengths: '语义理解，蒸馏成熟', fit_for: '情感分析、意图识别、实体提取、语义匹配' },
      minilm: { name: 'RBT4 (4层)', size: '20-80MB', task_modes: ['classify', 'ner', 'similarity', 'regression'], strengths: '更强的语义理解', fit_for: '复杂多分类、句子对任务、NER' },
      distilbert: { name: 'RBT6 (6层)', size: '100MB+', task_modes: ['classify', 'ner', 'similarity', 'regression'], strengths: '接近BERT全性能', fit_for: '复杂长文本、高精度NER、细粒度回归' },
    };

    const availableArchs = this.candidates.filter(c => archSpecs[c])
      .map(c => `- ${c} (${archSpecs[c].name}): ${archSpecs[c].size}, 支持 ${archSpecs[c].task_modes.join('/')} | ${archSpecs[c].strengths} | 适用: ${archSpecs[c].fit_for}`)
      .join('\n');

    let retryContext = '';
    if (previousAttempt) {
      retryContext = `\n\n⚠️ 上次用 "${previousAttempt.arch}" + "${previousAttempt.task_mode || 'classify'}" 训练，准确率 ${previousAttempt.accuracy} 不达标，请选更强的架构或调整任务模式。`;
    }

    const systemPrompt = `你是AI模型选型专家。根据任务描述，同时决定：
1. task_mode（微调方式）：
   - classify: 文本分类（输入文本→输出标签）最常用
   - ner: 命名实体识别（输入文本→输出实体列表，包含 start/end/label）
   - similarity: 句子对关系（输入两段文本→输出关系标签或相似度分数）
   - regression: 文本打分（输入文本→输出 0~1 连续分数）

2. model_arch（模型架构）——能用小的就不用大的：
${availableArchs}

返回JSON：
{
  "task_mode": "classify|ner|similarity|regression",
  "model_arch": "架构ID",
  "reason": "为什么选这个 task_mode 和架构",
  "labels": ["标签列表（classify用分类标签，ner用实体类型，similarity用关系标签）"],
  "data_format_hint": "给数据生成器的说明：每条数据应该长什么样",
  "training_config": { "epochs": 数字, "batch_size": 数字, "learning_rate": 数字, "max_length": 数字 }
}

判断规则：
- 需要从文本中抽取人名/地名/产品名等 → ner
- 需要判断两段文本的关系（相似/矛盾/蕴含）→ similarity
- 需要给文本打一个连续分数（质量/严重程度/相关性）→ regression
- 其他大部分场景 → classify（最通用）
- fasttext 只能做 classify，不支持其他任务模式`;

    const samplesText = sampleInputs?.length > 0
      ? `\n\n样本（前10条）：\n${sampleInputs.slice(0, 10).map((s, i) => `${i + 1}. ${s}`).join('\n')}`
      : '';

    const response = await this.chat(`任务描述：${taskDescription}${samplesText}${retryContext}`, { systemPrompt });
    return this._extractJSON(response);
  }

  _extractJSON(text) {
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) { try { return JSON.parse(jsonMatch[0]); } catch {} }
    throw new Error('无法从GPT-5.4响应中提取有效JSON');
  }
}

module.exports = { SelectorService };
