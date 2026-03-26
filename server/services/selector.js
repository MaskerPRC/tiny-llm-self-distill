const axios = require('axios');

/**
 * SelectorService - 使用 GPT-5.4 做蒸馏模型架构选型
 * 
 * 职责：根据任务描述、样本数据、候选架构列表，
 * 选出最小且能胜任的小模型架构，并给出训练配置建议。
 */
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
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: userMessage });

    const response = await axios.post(`${this.apiBase}/chat/completions`, {
      model: this.model,
      messages,
      temperature,
      max_tokens: maxTokens,
      response_format: { type: 'json_object' },
    }, {
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://tinybert-pipeline.local',
      },
      timeout: 120000,
    });

    return response.data.choices[0].message.content;
  }

  /**
   * 根据任务描述和样本，选择最合适的小模型架构
   */
  async selectModelArch(taskDescription, sampleInputs, previousAttempt) {
    const archSpecs = {
      fasttext: {
        name: 'FastText',
        size: '<2MB',
        params: '<1M',
        strengths: '关键词/短语级分类，速度极快（<1ms），二分类或少量类别',
        weaknesses: '不理解语序，无法处理反讽、双重否定等语义逻辑',
        fit_for: '垃圾邮件过滤、脏话检测、简单意图路由',
      },
      tinybert: {
        name: 'TinyBERT (4层)',
        size: '15-50MB',
        params: '14.5M',
        strengths: '理解上下文语义，能处理反讽和委婉表达，蒸馏技术成熟',
        weaknesses: '比FastText慢100倍，需要GPU加速效果更好',
        fit_for: '情感分析、意图识别、文本蕴含、语义相似度',
      },
      minilm: {
        name: 'MiniLM-L6',
        size: '20-80MB',
        params: '22M',
        strengths: '学习注意力关系图，语义理解力比TinyBERT更均衡',
        weaknesses: '体积稍大',
        fit_for: '需要较强语义理解的多分类、句子对任务',
      },
      distilbert: {
        name: 'DistilBERT (6层)',
        size: '130MB+',
        params: '66M',
        strengths: '保留BERT 97%性能，能处理复杂长难句',
        weaknesses: '体积大，推理慢，边际收益递减',
        fit_for: '复杂多分类、长文本理解、需要极高准确率的场景',
      },
    };

    const availableArchs = this.candidates
      .filter(c => archSpecs[c])
      .map(c => `### ${archSpecs[c].name} (${c})\n- 体积: ${archSpecs[c].size} | 参数: ${archSpecs[c].params}\n- 擅长: ${archSpecs[c].strengths}\n- 弱点: ${archSpecs[c].weaknesses}\n- 适用: ${archSpecs[c].fit_for}`)
      .join('\n\n');

    let retryContext = '';
    if (previousAttempt) {
      retryContext = `\n\n⚠️ 上一次尝试使用 "${previousAttempt.arch}" 训练，准确率只有 ${previousAttempt.accuracy}，不达标。请选一个更强的架构。`;
    }

    const systemPrompt = `你是一个AI模型选型专家。你的任务是根据NLP任务描述，从候选架构中选出"最小且能胜任"的模型。

核心原则：能用小的就不用大的。只有当任务确实需要语义理解时才上Transformer。

可用候选架构：
${availableArchs}

请返回JSON：
{
  "model_arch": "架构ID",
  "reason": "一句话说明为什么选这个而不是更小/更大的",
  "labels": ["任务需要的分类标签列表"],
  "training_config": {
    "epochs": 数字,
    "batch_size": 数字,
    "learning_rate": 数字,
    "max_length": 数字
  }
}`;

    const samplesText = sampleInputs && sampleInputs.length > 0
      ? `\n\n样本输入（前10条）：\n${sampleInputs.slice(0, 10).map((s, i) => `${i + 1}. ${s}`).join('\n')}`
      : '';

    const response = await this.chat(
      `任务描述：${taskDescription}${samplesText}${retryContext}`,
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

module.exports = { SelectorService };
