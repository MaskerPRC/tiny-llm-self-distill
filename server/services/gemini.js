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
    };

    const useNewTokenParam = /^(gpt-|o[1-9]|claude-)/.test(model);
    if (useNewTokenParam) {
      body.max_completion_tokens = maxTokens;
    } else {
      body.max_tokens = maxTokens;
    }

    if (jsonMode) {
      body.response_format = { type: 'json_object' };
    }

    return this._callWithRetry(apiBase, apiKey, body);
  }

  /**
   * 批量标注：给定输入文本数组和任务描述，使用 Gemini 进行标注
   */
  async labelBatch(texts, taskDescription, labels) {
    const labelsStr = labels.join(', ');
    const systemPrompt = `你是数据标注专家。任务：${taskDescription}\n可用标签：${labelsStr}\n对每条文本返回标签。只输出纯JSON数组，不要用markdown代码块包裹，不要添加任何解释文字。格式：[{"text":"原文","label":"标签"},...]`;

    const batchSize = 20;
    const results = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchNum = Math.floor(i / batchSize) + 1;
      const userMessage = `标注以下 ${batch.length} 条文本，只返回JSON数组:\n${batch.map((t, idx) => `${idx + 1}. ${t}`).join('\n')}`;

      let batchOk = false;
      try {
        const response = await this.chat(userMessage, {
          systemPrompt,
          temperature: 0.1,
          useDatagen: true,
        });

        const items = this._extractJsonArray(response);
        if (items && items.length > 0) {
          results.push(...items);
          batchOk = true;
        }
      } catch (err) {
        console.error(`[Gemini] 批次 ${batchNum} API调用失败: ${err.message}`);
      }

      if (!batchOk) {
        console.warn(`[Gemini] 批次 ${batchNum} JSON解析失败，逐条标注`);
        for (const text of batch) {
          try {
            const single = await this.chat(
              `对以下文本分类，只返回一个标签名（${labelsStr}），不要返回任何其他内容：\n"${text}"`,
              { temperature: 0.1, useDatagen: true }
            );
            const label = this._matchLabel(single.trim(), labels);
            results.push({ text, label });
          } catch (err) {
            results.push({ text, label: labels[0] });
          }
        }
      }
    }

    return results;
  }

  _extractJsonArray(raw) {
    if (!raw || typeof raw !== 'string') return null;

    const fenced = raw.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
    const str = fenced ? fenced[1].trim() : raw.trim();

    const arrMatch = str.match(/\[[\s\S]*\]/);
    if (!arrMatch) return null;

    try {
      const arr = JSON.parse(arrMatch[0]);
      if (Array.isArray(arr)) return arr;
    } catch {}

    try {
      const obj = JSON.parse(str);
      const arr = obj.results || obj.data || obj.items || obj;
      if (Array.isArray(arr)) return arr;
    } catch {}

    return null;
  }

  _matchLabel(text, labels) {
    const lower = text.toLowerCase().replace(/["`'.\s]/g, '');
    for (const l of labels) {
      if (lower.includes(l.toLowerCase())) return l;
    }
    return labels[0];
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
        const cleaned = this._cleanText(text);
        if (!cleaned || seenTexts.has(cleaned)) continue;

        seenTexts.add(cleaned);
        const item = { text: cleaned, label };
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
   * 单条数据生成：LLM 推理生成 + 正则提取（不再用第二次 LLM 调用）
   * API 调用：每条数据只调一次
   */
  async _generateOneItem(taskDescription, label, allLabels, index, scene, style) {
    const prompt = `你是一个高质量NLP训练数据生成专家。

任务：生成一条用于文本分类模型训练的样本。
分类目标：${allLabels.map(l => `"${l}"`).join(' / ')}
任务描述：${taskDescription}
本条标签：${label}
场景：${scene}
风格：${style}

请先用2-3句话简短思考这个人物和场景，然后直接给出最终文本。

输出格式（必须严格遵守）：
THINKING: （2-3句简短思考）
RESULT: （最终的文本样本，至少10个字，不包含任何编号、标签名、场景名）`;

    const response = await this.chat(prompt, {
      temperature: 0.95,
      maxTokens: 800,
      useDatagen: true,
    });

    const extracted = this._extractResult(response);
    if (extracted) return { text: extracted, label };
    return null;
  }

  _extractResult(response) {
    if (!response || typeof response !== 'string') return null;

    const resultMatch = response.match(/RESULT:\s*([\s\S]+?)$/i);
    if (resultMatch) {
      const cleaned = this._cleanText(resultMatch[1]);
      if (cleaned) return cleaned;
    }

    const angleMatch = response.match(/<<<([\s\S]+?)>>>/);
    if (angleMatch) {
      const cleaned = this._cleanText(angleMatch[1]);
      if (cleaned) return cleaned;
    }

    const bracketMatch = response.match(/【([\s\S]+?)】/);
    if (bracketMatch) {
      const cleaned = this._cleanText(bracketMatch[1]);
      if (cleaned) return cleaned;
    }

    const lines = response.split('\n').filter(l => l.trim().length > 0);
    if (lines.length > 0) {
      const lastLine = this._cleanText(lines[lines.length - 1]);
      if (lastLine && lastLine.length >= 10) return lastLine;
    }

    return null;
  }

  _cleanText(text) {
    if (!text || typeof text !== 'string') return null;

    let cleaned = text.trim();
    cleaned = cleaned.replace(/^<<<|>>>$/g, '');
    cleaned = cleaned.replace(/^【|】$/g, '');
    cleaned = cleaned.replace(/^"([\s\S]+)"$/, '$1');
    cleaned = cleaned.replace(/^\d+[\.\、\:\：\s\|]+/g, '');
    cleaned = cleaned.replace(/^序号[\s\：:]*\d*[\s\.\、\:\：\|]*/gi, '');
    cleaned = cleaned.replace(/^RESULT:\s*/i, '');
    cleaned = cleaned.replace(/^THINKING:[\s\S]*?RESULT:\s*/i, '');
    cleaned = cleaned.replace(/^(malicious_or_abusive|malicious_or_hostile|benign|allow|block|hostile|malicious)[\s\|\\t:]*/gi, '');
    cleaned = cleaned.replace(/场景[\：:][^\n]*/g, '');
    cleaned = cleaned.replace(/风格[\：:][^\n]*/g, '');
    cleaned = cleaned.replace(/标签[\：:][^\n]*/g, '');
    cleaned = cleaned.replace(/文本[\：:]\s*/g, '');
    cleaned = cleaned.replace(/>>>[^<]*$/g, '');
    cleaned = cleaned.replace(/<<<[^>]*$/g, '');
    cleaned = cleaned.trim();

    if (cleaned.length < 8) return null;
    if (/^[\d\s序号：:\.、\|]+$/.test(cleaned)) return null;
    if (/^(包裹|符合|元信息|最终|输出|确认|草稿)/i.test(cleaned)) return null;

    return cleaned;
  }

  // ========== 多任务模式数据生成入口 ==========

  async generateTrainingDataByMode(taskMode, taskDescription, labels, count, onProgress, savePath) {
    switch (taskMode) {
      case 'ner':
        return this._generateNERData(taskDescription, labels, count, onProgress, savePath);
      case 'similarity':
        return this._generateSimilarityData(taskDescription, labels, count, onProgress, savePath);
      case 'regression':
        return this._generateRegressionData(taskDescription, count, onProgress, savePath);
      default:
        return this.generateTrainingData(taskDescription, labels, count, onProgress, savePath);
    }
  }

  // ========== NER 数据生成 ==========

  async _generateNERData(taskDescription, entityLabels, count, onProgress, savePath) {
    const CONCURRENCY = parseInt(process.env.DATAGEN_CONCURRENCY) || 20;
    const allData = [];

    if (savePath) {
      fs.mkdirSync(path.dirname(savePath), { recursive: true });
      if (fs.existsSync(savePath)) {
        const lines = fs.readFileSync(savePath, 'utf-8').trim().split('\n').filter(Boolean);
        for (const line of lines) {
          try { allData.push(JSON.parse(line)); } catch {}
        }
        if (allData.length > 0) console.log(`[Datagen-NER] Resumed ${allData.length} items`);
      }
    }

    let consecutiveLow = 0;
    let idx = allData.length;

    while (allData.length < count && consecutiveLow < 5) {
      const batch = Math.min(CONCURRENCY, count - allData.length);
      const promises = [];
      for (let i = 0; i < batch; i++) {
        const scene = DIVERSITY_SCENES[idx % DIVERSITY_SCENES.length];
        const style = DIVERSITY_STYLES[idx % DIVERSITY_STYLES.length];
        idx++;
        promises.push(this._generateOneNERItem(taskDescription, entityLabels, scene, style));
      }

      const results = await Promise.allSettled(promises);
      let added = 0;
      for (const r of results) {
        if (r.status !== 'fulfilled' || !r.value) continue;
        allData.push(r.value);
        added++;
        if (savePath) fs.appendFileSync(savePath, JSON.stringify(r.value) + '\n');
        if (allData.length >= count) break;
      }

      consecutiveLow = (added / batch < 0.1) ? consecutiveLow + 1 : 0;
      if (onProgress) onProgress({ label: 'NER', generated: allData.length, target: count, totalGenerated: allData.length, totalTarget: count });
    }

    console.log(`[Datagen-NER] Done: ${allData.length}/${count}`);
    return allData;
  }

  async _generateOneNERItem(taskDescription, entityLabels, scene, style) {
    const prompt = `你是NLP训练数据生成专家。生成一条NER（命名实体识别）训练样本。

任务描述：${taskDescription}
需要标注的实体类型：${entityLabels.join(', ')}
场景：${scene}
风格：${style}

请生成一段自然文本（至少20字），并标注其中的实体。
输出纯JSON，不要markdown包裹：
{"text": "文本内容", "entities": [{"start": 起始字符位置, "end": 结束字符位置, "label": "实体类型"}]}

注意：start/end 是字符级偏移量（从0开始），end 是不包含的位置。确保偏移量精确匹配文本。`;

    const response = await this.chat(prompt, { temperature: 0.9, maxTokens: 600, useDatagen: true });

    try {
      const json = this._extractJsonObject(response);
      if (json?.text && Array.isArray(json.entities)) {
        const valid = json.entities.filter(e =>
          typeof e.start === 'number' && typeof e.end === 'number' && e.label &&
          e.start >= 0 && e.end <= json.text.length && e.start < e.end
        );
        if (json.text.length >= 10) return { text: json.text, entities: valid };
      }
    } catch {}
    return null;
  }

  // ========== 句子对数据生成 ==========

  async _generateSimilarityData(taskDescription, labels, count, onProgress, savePath) {
    const CONCURRENCY = parseInt(process.env.DATAGEN_CONCURRENCY) || 20;
    const allData = [];
    const perLabel = Math.ceil(count / labels.length);

    if (savePath) {
      fs.mkdirSync(path.dirname(savePath), { recursive: true });
      if (fs.existsSync(savePath)) {
        const lines = fs.readFileSync(savePath, 'utf-8').trim().split('\n').filter(Boolean);
        for (const line of lines) { try { allData.push(JSON.parse(line)); } catch {} }
        if (allData.length > 0) console.log(`[Datagen-Sim] Resumed ${allData.length} items`);
      }
    }

    for (const label of labels) {
      const existing = allData.filter(d => d.label === label).length;
      let needed = perLabel - existing;
      let idx = existing;
      let consecutiveLow = 0;

      while (needed > 0 && consecutiveLow < 5) {
        const batch = Math.min(CONCURRENCY, needed);
        const promises = [];
        for (let i = 0; i < batch; i++) {
          const scene = DIVERSITY_SCENES[idx % DIVERSITY_SCENES.length];
          idx++;
          promises.push(this._generateOneSimilarityItem(taskDescription, label, labels, scene));
        }

        const results = await Promise.allSettled(promises);
        let added = 0;
        for (const r of results) {
          if (r.status !== 'fulfilled' || !r.value) continue;
          allData.push(r.value);
          added++;
          needed--;
          if (savePath) fs.appendFileSync(savePath, JSON.stringify(r.value) + '\n');
          if (needed <= 0) break;
        }

        consecutiveLow = (added / batch < 0.1) ? consecutiveLow + 1 : 0;
        if (onProgress) onProgress({ label, generated: allData.filter(d => d.label === label).length, target: perLabel, totalGenerated: allData.length, totalTarget: count });
      }
    }

    console.log(`[Datagen-Sim] Done: ${allData.length}/${count}`);
    return allData;
  }

  async _generateOneSimilarityItem(taskDescription, label, allLabels, scene) {
    const prompt = `你是NLP训练数据生成专家。生成一条句子对关系判断的训练样本。

任务描述：${taskDescription}
关系标签选项：${allLabels.join(', ')}
本条标签：${label}
场景：${scene}

生成两段文本，它们的关系是「${label}」。
输出纯JSON，不要markdown包裹：
{"text_a": "第一段文本", "text_b": "第二段文本", "label": "${label}"}

每段文本至少10个字，内容要自然真实。`;

    const response = await this.chat(prompt, { temperature: 0.9, maxTokens: 600, useDatagen: true });
    try {
      const json = this._extractJsonObject(response);
      if (json?.text_a?.length >= 5 && json?.text_b?.length >= 5 && json?.label) return json;
    } catch {}
    return null;
  }

  // ========== 回归/打分数据生成 ==========

  async _generateRegressionData(taskDescription, count, onProgress, savePath) {
    const CONCURRENCY = parseInt(process.env.DATAGEN_CONCURRENCY) || 20;
    const allData = [];

    if (savePath) {
      fs.mkdirSync(path.dirname(savePath), { recursive: true });
      if (fs.existsSync(savePath)) {
        const lines = fs.readFileSync(savePath, 'utf-8').trim().split('\n').filter(Boolean);
        for (const line of lines) { try { allData.push(JSON.parse(line)); } catch {} }
        if (allData.length > 0) console.log(`[Datagen-Reg] Resumed ${allData.length} items`);
      }
    }

    const scoreRanges = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'];
    let idx = allData.length;
    let consecutiveLow = 0;

    while (allData.length < count && consecutiveLow < 5) {
      const batch = Math.min(CONCURRENCY, count - allData.length);
      const promises = [];
      for (let i = 0; i < batch; i++) {
        const scene = DIVERSITY_SCENES[idx % DIVERSITY_SCENES.length];
        const range = scoreRanges[idx % scoreRanges.length];
        idx++;
        promises.push(this._generateOneRegressionItem(taskDescription, range, scene));
      }

      const results = await Promise.allSettled(promises);
      let added = 0;
      for (const r of results) {
        if (r.status !== 'fulfilled' || !r.value) continue;
        allData.push(r.value);
        added++;
        if (savePath) fs.appendFileSync(savePath, JSON.stringify(r.value) + '\n');
        if (allData.length >= count) break;
      }

      consecutiveLow = (added / batch < 0.1) ? consecutiveLow + 1 : 0;
      if (onProgress) onProgress({ label: 'score', generated: allData.length, target: count, totalGenerated: allData.length, totalTarget: count });
    }

    console.log(`[Datagen-Reg] Done: ${allData.length}/${count}`);
    return allData;
  }

  async _generateOneRegressionItem(taskDescription, scoreRange, scene) {
    const prompt = `你是NLP训练数据生成专家。生成一条文本打分/回归训练样本。

任务描述：${taskDescription}
目标分数区间：${scoreRange}（0.0最低，1.0最高）
场景：${scene}

生成一段文本，并给出精确的浮点分数。
输出纯JSON，不要markdown包裹：
{"text": "文本内容", "score": 0.xx}

文本至少10个字，分数必须在 ${scoreRange} 范围内。`;

    const response = await this.chat(prompt, { temperature: 0.9, maxTokens: 400, useDatagen: true });
    try {
      const json = this._extractJsonObject(response);
      if (json?.text?.length >= 8 && typeof json?.score === 'number' && json.score >= 0 && json.score <= 1) return json;
    } catch {}
    return null;
  }

  _extractJsonObject(raw) {
    if (!raw || typeof raw !== 'string') return null;
    const fenced = raw.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
    const str = fenced ? fenced[1].trim() : raw.trim();
    const objMatch = str.match(/\{[\s\S]*\}/);
    if (objMatch) { try { return JSON.parse(objMatch[0]); } catch {} }
    return null;
  }
}

module.exports = { GeminiService };
