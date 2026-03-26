const axios = require('axios');

class GeminiService {
  constructor() {
    this.apiBase = process.env.GEMINI_API_BASE || 'https://openrouter.ai/api/v1';
    this.apiKey = process.env.GEMINI_API_KEY;
    this.model = process.env.GEMINI_MODEL || 'google/gemini-3.1-pro-preview';

    this.datagenApiBase = process.env.DATAGEN_API_BASE || this.apiBase;
    this.datagenApiKey = process.env.DATAGEN_API_KEY || this.apiKey;
    this.datagenModel = process.env.DATAGEN_MODEL || this.model;
  }

  async chat(userMessage, options = {}) {
    const { systemPrompt, temperature = 0.7, maxTokens = 4096, jsonMode = false, useDatagen = false } = options;

    const apiBase = useDatagen ? this.datagenApiBase : this.apiBase;
    const apiKey = useDatagen ? this.datagenApiKey : this.apiKey;
    const model = useDatagen ? this.datagenModel : this.model;

    const messages = [];
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: userMessage });

    const body = {
      model,
      messages,
      temperature,
      max_tokens: maxTokens,
    };

    if (jsonMode) {
      body.response_format = { type: 'json_object' };
    }

    const response = await axios.post(`${apiBase}/chat/completions`, body, {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
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
        useDatagen: true,
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
              { systemPrompt, temperature: 0.1, useDatagen: true }
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
   * 生成训练数据：所有标签并发生成，每个标签内部也并发分批
   */
  async generateTrainingData(taskDescription, labels, count, onProgress) {
    const BATCH_SIZE = 50;
    const CONCURRENCY = parseInt(process.env.DATAGEN_CONCURRENCY) || 10;
    const perLabel = Math.ceil(count / labels.length);
    const seenTexts = new Set();

    const labelResults = await Promise.all(
      labels.map(label => this._generateForLabel(
        taskDescription, label, perLabel, BATCH_SIZE, CONCURRENCY, seenTexts, onProgress,
        () => labels.reduce((sum, l) => sum, 0)
      ))
    );

    const allData = labelResults.flat();
    console.log(`[Datagen] 数据生成完毕: 共 ${allData.length}/${count} 条，去重文本 ${seenTexts.size} 条`);
    return allData;
  }

  async _generateForLabel(taskDescription, label, target, batchSize, concurrency, seenTexts, onProgress, _) {
    const labelData = [];
    let consecutiveEmpty = 0;
    const maxEmptyRounds = 3;

    while (labelData.length < target && consecutiveEmpty < maxEmptyRounds) {
      const remaining = target - labelData.length;
      const batchCount = Math.min(batchSize, remaining);
      const parallelCount = Math.min(
        concurrency,
        Math.ceil(remaining / batchSize)
      );

      const batchPromises = [];
      for (let i = 0; i < parallelCount; i++) {
        const batchIndex = Math.floor(labelData.length / batchSize) + i + 1;
        batchPromises.push(this._generateOneBatch(taskDescription, label, batchCount, batchIndex, labelData.length));
      }

      const batchResults = await Promise.allSettled(batchPromises);

      let roundAdded = 0;
      for (const result of batchResults) {
        if (result.status !== 'fulfilled' || !result.value) continue;

        for (const item of result.value) {
          if (!item.text || typeof item.text !== 'string') continue;
          const text = item.text.trim();
          if (text.length === 0) continue;

          if (seenTexts.has(text)) continue;
          seenTexts.add(text);

          labelData.push({ text, label });
          roundAdded++;
          if (labelData.length >= target) break;
        }
        if (labelData.length >= target) break;
      }

      if (roundAdded === 0) {
        consecutiveEmpty++;
      } else {
        consecutiveEmpty = 0;
      }

      if (onProgress) {
        onProgress({ label, generated: labelData.length, target, totalGenerated: labelData.length, totalTarget: target });
      }
    }

    if (consecutiveEmpty >= maxEmptyRounds) {
      console.warn(`[Datagen] 标签 "${label}" 连续 ${maxEmptyRounds} 轮无新数据，停止（${labelData.length}/${target}）`);
    }
    console.log(`[Datagen] 标签 "${label}" 完成: ${labelData.length}/${target} 条`);
    return labelData;
  }

  async _generateOneBatch(taskDescription, label, batchCount, batchIndex, existingCount) {
    const prompt = `你是一个训练数据生成专家。请为以下NLP分类任务生成 ${batchCount} 条训练数据。

任务描述：${taskDescription}
目标标签：${label}
批次编号：${batchIndex}（用于多样性种子，请尽量让不同批次的数据风格、场景差异化）

要求：
1. 生成多样化的、真实的文本样本
2. 包含各种表达方式、语气、长度、场景
3. 包含一些边界案例和容易混淆的样本
4. 每条文本必须不同，不要重复
5. 返回 JSON 数组，每个元素格式: {"text": "文本内容", "label": "${label}"}
6. 严格生成 ${batchCount} 条

只返回 JSON 数组，不要其他文字。`;

    try {
      const response = await this.chat(prompt, {
        jsonMode: true,
        temperature: 0.95,
        maxTokens: 8192,
        useDatagen: true,
      });

      const parsed = JSON.parse(response);
      return Array.isArray(parsed) ? parsed : (parsed.data || parsed.results || []);
    } catch (err) {
      console.error(`[Datagen] 标签 "${label}" 批次 ${batchIndex} 失败: ${err.message}`);
      return [];
    }
  }
}

module.exports = { GeminiService };
