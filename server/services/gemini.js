const axios = require('axios');

class GeminiService {
  constructor() {
    this.apiBase = process.env.GEMINI_API_BASE || 'https://openrouter.ai/api/v1';
    this.apiKey = process.env.GEMINI_API_KEY;
    this.model = process.env.GEMINI_MODEL || 'google/gemini-3.1-pro-preview';
  }

  async chat(userMessage, options = {}) {
    const { systemPrompt, temperature = 0.7, maxTokens = 4096, jsonMode = false } = options;

    const messages = [];
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: userMessage });

    const body = {
      model: this.model,
      messages,
      temperature,
      max_tokens: maxTokens,
    };

    if (jsonMode) {
      body.response_format = { type: 'json_object' };
    }

    const response = await axios.post(`${this.apiBase}/chat/completions`, body, {
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
   * 批量标注：给定输入文本数组和任务描述，使用 Gemini 进行标注
   */
  async labelBatch(texts, taskDescription, labels) {
    const systemPrompt = `你是一个数据标注专家。请对以下文本进行分类标注。

任务描述：${taskDescription}
可用标签：${labels.join(', ')}

请对每条文本返回一个标签。严格按照JSON数组格式返回，每个元素包含 text 和 label 字段。`;

    const batchSize = 20;
    const results = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const userMessage = `请标注以下 ${batch.length} 条文本:\n${batch.map((t, idx) => `${idx + 1}. ${t}`).join('\n')}`;

      const response = await this.chat(userMessage, {
        systemPrompt,
        jsonMode: true,
        temperature: 0.1,
      });

      try {
        const parsed = JSON.parse(response);
        const items = parsed.results || parsed.data || parsed;
        if (Array.isArray(items)) {
          results.push(...items);
        }
      } catch {
        console.error(`[Gemini] 批次 ${i / batchSize + 1} 标注解析失败，尝试逐条处理`);
        for (const text of batch) {
          try {
            const single = await this.chat(
              `请对以下文本进行分类标注，只返回标签名：\n"${text}"`,
              { systemPrompt, temperature: 0.1 }
            );
            results.push({ text, label: single.trim() });
          } catch (err) {
            results.push({ text, label: labels[0], error: err.message });
          }
        }
      }
    }

    return results;
  }

  /**
   * 生成训练数据：给定任务描述和标签，让 Gemini 合成训练样本
   */
  async generateTrainingData(taskDescription, labels, count) {
    const perLabel = Math.ceil(count / labels.length);
    const allData = [];

    for (const label of labels) {
      const prompt = `你是一个训练数据生成专家。请为以下NLP分类任务生成训练数据。

任务描述：${taskDescription}
目标标签：${label}
生成数量：${perLabel} 条

要求：
1. 生成多样化的、真实的文本样本
2. 包含各种表达方式、语气、长度
3. 包含一些边界案例和容易混淆的样本
4. 返回 JSON 数组，每个元素有 text 和 label 字段

只返回 JSON，不要其他文字。`;

      const response = await this.chat(prompt, {
        jsonMode: true,
        temperature: 0.9,
        maxTokens: 8192,
      });

      try {
        const parsed = JSON.parse(response);
        const items = Array.isArray(parsed) ? parsed : (parsed.data || parsed.results || []);
        allData.push(...items.map(item => ({
          text: item.text,
          label: label,
        })));
      } catch {
        console.error(`[Gemini] 生成标签 "${label}" 的数据解析失败`);
      }
    }

    return allData;
  }
}

module.exports = { GeminiService };
