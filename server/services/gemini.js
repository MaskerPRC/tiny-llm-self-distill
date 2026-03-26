const axios = require('axios');
const fs = require('fs');
const path = require('path');

const DIVERSITY_SCENES = [
  '日常对话', '社交媒体评论', '客服工单', '产品评价', '新闻评论',
  '论坛帖子', '即时消息', '邮件内容', '搜索查询', '语音转文字',
  '博客留言', '视频弹幕', '问答社区', '投诉反馈', '朋友圈动态',
  '工作汇报', '面试对话', '教学场景', '医疗咨询', '法律咨询',
  '餐饮点评', '旅游分享', '游戏聊天', '亲子对话', '情侣对话',
  '商务谈判', '学术讨论', '技术问答', '求助帖', '吐槽抱怨',
];

const DIVERSITY_STYLES = [
  '口语化/随意', '正式/书面', '讽刺/反语', '夸张/情绪化', '冷静/理性',
  '简短/一句话', '长段落/详细', '带错别字/不规范', '夹杂英文', '方言/网络用语',
  '礼貌/委婉', '直接/粗暴', '幽默/调侃', '含蓄/暗示', '严肃/权威',
];

class GeminiService {
  constructor() {
    this.apiBase = process.env.GEMINI_API_BASE || 'https://openrouter.ai/api/v1';
    this.apiKey = process.env.GEMINI_API_KEY;
    this.model = process.env.GEMINI_MODEL || 'google/gemini-3.1-pro-preview';

    this.datagenApiBase = process.env.DATAGEN_API_BASE || this.apiBase;
    this.datagenApiKey = process.env.DATAGEN_API_KEY || this.apiKey;
    this.datagenModel = process.env.DATAGEN_MODEL || this.model;
  }

  async _callWithRetry(apiBase, apiKey, body, maxRetries = 3) {
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const response = await axios.post(`${apiBase}/chat/completions`, body, {
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://tinybert-pipeline.local',
          },
          timeout: 120000,
        });
        return response.data.choices[0].message.content;
      } catch (err) {
        const status = err.response?.status;
        if ((status === 429 || status === 403) && attempt < maxRetries) {
          const wait = Math.min(2000 * Math.pow(2, attempt), 30000);
          console.warn(`[API] ${status} 限频，${wait / 1000}s 后重试 (${attempt + 1}/${maxRetries})`);
          await new Promise(r => setTimeout(r, wait));
          continue;
        }
        throw err;
      }
    }
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

    return this._callWithRetry(apiBase, apiKey, body);
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
   * 生成训练数据：每条数据独立生成，两步流水线（推理 → 结构化），高并发
   * 每个并发请求分配不同的场景+风格种子，避免重复
   *
   * @param {string} savePath - 实时保存路径（JSONL 格式），每生成一条立即追加写入
   */
  async generateTrainingData(taskDescription, labels, count, onProgress, savePath) {
    const CONCURRENCY = parseInt(process.env.DATAGEN_CONCURRENCY) || 20;
    const perLabel = Math.ceil(count / labels.length);
    const seenTexts = new Set();
    const totalProgress = { generated: 0, target: count };

    if (savePath) {
      fs.mkdirSync(path.dirname(savePath), { recursive: true });
      if (fs.existsSync(savePath)) {
        const existing = fs.readFileSync(savePath, 'utf-8').trim().split('\n').filter(Boolean);
        for (const line of existing) {
          try {
            const item = JSON.parse(line);
            if (item.text) seenTexts.add(item.text.trim());
          } catch {}
        }
        if (seenTexts.size > 0) {
          console.log(`[Datagen] 从 ${savePath} 恢复了 ${seenTexts.size} 条已有数据，继续生成`);
        }
      }
    }

    const ctx = { seenTexts, totalProgress, savePath };

    const labelResults = await Promise.all(
      labels.map(label => this._generateForLabel(
        taskDescription, label, labels, perLabel, CONCURRENCY, ctx, onProgress
      ))
    );

    const allData = labelResults.flat();
    console.log(`[Datagen] 数据生成完毕: 共 ${allData.length}/${count} 条，去重文本 ${seenTexts.size} 条`);
    if (savePath) {
      console.log(`[Datagen] 数据已实时保存到: ${savePath}`);
    }
    return allData;
  }

  async _generateForLabel(taskDescription, label, allLabels, target, concurrency, ctx, onProgress) {
    const { seenTexts, totalProgress, savePath } = ctx;

    const existingForLabel = [];
    if (savePath && fs.existsSync(savePath)) {
      const lines = fs.readFileSync(savePath, 'utf-8').trim().split('\n').filter(Boolean);
      for (const line of lines) {
        try {
          const item = JSON.parse(line);
          if (item.label === label && item.text) {
            existingForLabel.push({ text: item.text.trim(), label });
          }
        } catch {}
      }
    }

    const labelData = [...existingForLabel];
    if (labelData.length >= target) {
      console.log(`[Datagen] 标签 "${label}" 已有 ${labelData.length}/${target} 条（从文件恢复），跳过`);
      return labelData;
    }
    if (labelData.length > 0) {
      console.log(`[Datagen] 标签 "${label}" 从文件恢复 ${labelData.length} 条，继续生成`);
    }

    let consecutiveLowYield = 0;
    const maxLowYieldRounds = 5;
    let globalIndex = labelData.length;

    while (labelData.length < target && consecutiveLowYield < maxLowYieldRounds) {
      const remaining = target - labelData.length;
      const parallelCount = Math.min(concurrency, remaining);

      const promises = [];
      for (let i = 0; i < parallelCount; i++) {
        const scene = DIVERSITY_SCENES[globalIndex % DIVERSITY_SCENES.length];
        const style = DIVERSITY_STYLES[globalIndex % DIVERSITY_STYLES.length];
        globalIndex++;
        promises.push(this._generateOneItem(taskDescription, label, allLabels, globalIndex, scene, style));
      }

      const results = await Promise.allSettled(promises);

      let roundAdded = 0;
      for (const result of results) {
        if (result.status !== 'fulfilled' || !result.value) continue;
        const text = result.value.text;
        if (!text || typeof text !== 'string') continue;
        const trimmed = text.trim();
        if (trimmed.length === 0 || seenTexts.has(trimmed)) continue;

        seenTexts.add(trimmed);
        const item = { text: trimmed, label };
        labelData.push(item);
        roundAdded++;
        totalProgress.generated++;

        if (savePath) {
          fs.appendFileSync(savePath, JSON.stringify(item) + '\n');
        }

        if (labelData.length >= target) break;
      }

      const yieldRate = roundAdded / parallelCount;
      if (yieldRate < 0.1) {
        consecutiveLowYield++;
      } else {
        consecutiveLowYield = 0;
      }

      if (onProgress) {
        onProgress({
          label,
          generated: labelData.length,
          target,
          totalGenerated: totalProgress.generated,
          totalTarget: totalProgress.target,
        });
      }
    }

    if (consecutiveLowYield >= maxLowYieldRounds) {
      console.warn(`[Datagen] 标签 "${label}" 连续 ${maxLowYieldRounds} 轮低产出，停止（${labelData.length}/${target}）`);
    }
    console.log(`[Datagen] 标签 "${label}" 完成: ${labelData.length}/${target} 条`);
    return labelData;
  }

  /**
   * 单条数据两步生成：
   * Step 1 - 自由推理：基于指定场景和风格深度思考后生成一条真实文本
   * Step 2 - 结构化提取：从推理结果中提取干净的 JSON
   */
  async _generateOneItem(taskDescription, label, allLabels, index, scene, style) {
    const reasoningPrompt = `你是一个高质量NLP训练数据的生成专家。请为以下分类任务生成一条训练样本。

任务描述：${taskDescription}
所有标签：${allLabels.join(', ')}
本条目标标签：${label}

★ 强制约束（必须遵守）：
- 场景：${scene}
- 风格：${style}
- 序号：${index}

请按以下步骤思考：
1. 在「${scene}」场景下，一个真实的人会怎么表达属于「${label}」类别的内容？
2. 用「${style}」的风格来写，注意语气、词汇和句式要匹配这个风格。
3. 确保文本独特，不要写"太典型"的例子，尽量贴近真实场景。
4. 写出最终的文本样本。

请详细推理，最后一行写出生成的文本样本（用【】包裹）。`;

    const reasoning = await this.chat(reasoningPrompt, {
      temperature: 0.95,
      maxTokens: 1024,
      useDatagen: true,
    });

    const extractPrompt = `从以下推理过程中，提取最终生成的训练文本样本。

推理过程：
${reasoning}

要求：
- 提取推理中最终确定的文本样本（通常在【】中或在最后）
- 如果推理中有多个候选，取最终选定的那个
- 返回 JSON：{"text": "提取的文本", "label": "${label}"}
- text 中不要包含【】符号
- 只返回 JSON`;

    const structured = await this.chat(extractPrompt, {
      jsonMode: true,
      temperature: 0.1,
      maxTokens: 512,
      useDatagen: true,
    });

    try {
      const parsed = JSON.parse(structured);
      return { text: parsed.text, label };
    } catch {
      const bracketMatch = reasoning.match(/【(.+?)】/);
      if (bracketMatch) return { text: bracketMatch[1], label };
      return null;
    }
  }
}

module.exports = { GeminiService };
