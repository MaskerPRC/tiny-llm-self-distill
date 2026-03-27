const path = require('path');
const fs = require('fs');

class Predictor {
  constructor(toolConfig) {
    this.name = toolConfig.name;
    this.modelArch = toolConfig.modelArch;
    this.taskMode = toolConfig.config?.task_mode || 'classify';
    this.modelPath = toolConfig.modelPath;
    this.onnxPath = toolConfig.onnxPath;
    this.config = toolConfig.config || {};
    this.session = null;
    this.tokenizer = null;
    this.labels = this.config.labels || [];
  }

  async load() {
    if (this.modelArch === 'fasttext') {
      await this._loadFastText();
    } else {
      await this._loadOnnx();
    }
  }

  async _loadOnnx() {
    const ort = require('onnxruntime-node');
    const onnxFile = this.onnxPath || this.modelPath;

    if (!onnxFile || !fs.existsSync(onnxFile)) {
      throw new Error(`ONNX model not found: ${onnxFile}`);
    }

    this.session = await ort.InferenceSession.create(onnxFile);

    const modelDir = path.dirname(onnxFile);
    const vocabPath = path.join(modelDir, 'vocab.json');
    if (fs.existsSync(vocabPath)) {
      this.tokenizer = JSON.parse(fs.readFileSync(vocabPath, 'utf-8'));
    }

    const labelsPath = path.join(modelDir, 'labels.json');
    if (fs.existsSync(labelsPath)) {
      this.labels = JSON.parse(fs.readFileSync(labelsPath, 'utf-8'));
    }

    const taskConfigPath = path.join(modelDir, 'task_config.json');
    if (fs.existsSync(taskConfigPath)) {
      this.taskConfig = JSON.parse(fs.readFileSync(taskConfigPath, 'utf-8'));
    }

    if (this.taskMode === 'ner') {
      const tag2idPath = path.join(modelDir, 'tag2id.json');
      if (fs.existsSync(tag2idPath)) {
        this.tag2id = JSON.parse(fs.readFileSync(tag2idPath, 'utf-8'));
        this.id2tag = Object.fromEntries(Object.entries(this.tag2id).map(([k, v]) => [v, k]));
      }
      const entityLabelsPath = path.join(modelDir, 'entity_labels.json');
      if (fs.existsSync(entityLabelsPath)) {
        this.entityLabels = JSON.parse(fs.readFileSync(entityLabelsPath, 'utf-8'));
      }
    }

    console.log(`[Predictor] Loaded ${this.name} (${this.taskMode}, ${this.labels.length} labels)`);
  }

  async _loadFastText() {
    const modelDir = path.dirname(this.modelPath);
    const labelsPath = path.join(modelDir, 'labels.json');
    const weightsPath = path.join(modelDir, 'weights.json');

    if (fs.existsSync(labelsPath)) this.labels = JSON.parse(fs.readFileSync(labelsPath, 'utf-8'));
    if (fs.existsSync(weightsPath)) this.fastTextWeights = JSON.parse(fs.readFileSync(weightsPath, 'utf-8'));
    console.log(`[Predictor] Loaded FastText: ${this.name}`);
  }

  async predict(input) {
    const text = typeof input === 'object' && input !== null
      ? (input.text || input.input || JSON.stringify(input))
      : String(input);

    if (this.modelArch === 'fasttext') {
      return this._predictFastText(text);
    }

    switch (this.taskMode) {
      case 'ner': return this._predictNER(text);
      case 'similarity': return this._predictSimilarity(input);
      case 'regression': return this._predictRegression(text);
      default: return this._predictClassify(text);
    }
  }

  // ========== 分类 ==========
  async _predictClassify(text) {
    const ort = require('onnxruntime-node');
    const maxLen = this.config.max_length || 128;
    const { inputIds, attentionMask } = this._tokenize(text, maxLen);

    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, maxLen]),
      attention_mask: new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, maxLen]),
    };
    if (this.session.inputNames.includes('token_type_ids')) {
      feeds.token_type_ids = new ort.Tensor('int64', new BigInt64Array(maxLen), [1, maxLen]);
    }

    const results = await this.session.run(feeds);
    const logits = Array.from(results[this.session.outputNames[0]].data);
    const probs = this._softmax(logits);
    const maxIdx = probs.indexOf(Math.max(...probs));

    return {
      label: this.labels[maxIdx] || `class_${maxIdx}`,
      confidence: probs[maxIdx],
      probabilities: Object.fromEntries(this.labels.map((l, i) => [l, probs[i] || 0])),
    };
  }

  // ========== NER ==========
  async _predictNER(text) {
    const ort = require('onnxruntime-node');
    const maxLen = this.config.max_length || 128;
    const chars = [...text];
    const { inputIds, attentionMask, charOffsets } = this._tokenizeWithOffsets(text, maxLen);

    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, maxLen]),
      attention_mask: new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, maxLen]),
    };
    if (this.session.inputNames.includes('token_type_ids')) {
      feeds.token_type_ids = new ort.Tensor('int64', new BigInt64Array(maxLen), [1, maxLen]);
    }

    const results = await this.session.run(feeds);
    const output = results[this.session.outputNames[0]];
    const numTags = this.labels.length || (this.id2tag ? Object.keys(this.id2tag).length : 3);

    const entities = [];
    let currentEntity = null;

    for (let i = 0; i < maxLen; i++) {
      if (attentionMask[i] === 0) break;
      const offset = charOffsets[i];
      if (offset === undefined || offset < 0) continue;

      const tagLogits = [];
      for (let j = 0; j < numTags; j++) {
        tagLogits.push(Number(output.data[i * numTags + j]));
      }
      const tagIdx = tagLogits.indexOf(Math.max(...tagLogits));
      const tag = this.id2tag ? this.id2tag[tagIdx] : `TAG_${tagIdx}`;

      if (tag.startsWith('B-')) {
        if (currentEntity) entities.push(currentEntity);
        currentEntity = { label: tag.slice(2), start: offset, end: offset + 1, text: chars[offset] || '' };
      } else if (tag.startsWith('I-') && currentEntity && tag.slice(2) === currentEntity.label) {
        currentEntity.end = offset + 1;
        currentEntity.text += chars[offset] || '';
      } else {
        if (currentEntity) { entities.push(currentEntity); currentEntity = null; }
      }
    }
    if (currentEntity) entities.push(currentEntity);

    return { entities, entity_count: entities.length };
  }

  // ========== 句子对相似度 ==========
  async _predictSimilarity(input) {
    const ort = require('onnxruntime-node');
    const maxLen = this.config.max_length || 128;
    const textA = typeof input === 'object' ? (input.text_a || input.a || '') : '';
    const textB = typeof input === 'object' ? (input.text_b || input.b || '') : '';

    const { inputIds, attentionMask, tokenTypeIds } = this._tokenizePair(textA, textB, maxLen);

    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, maxLen]),
      attention_mask: new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, maxLen]),
    };
    if (this.session.inputNames.includes('token_type_ids')) {
      feeds.token_type_ids = new ort.Tensor('int64', BigInt64Array.from(tokenTypeIds.map(BigInt)), [1, maxLen]);
    }

    const results = await this.session.run(feeds);
    const logits = Array.from(results[this.session.outputNames[0]].data);

    const isRegression = this.taskConfig?.is_regression || logits.length === 1;
    if (isRegression) {
      return { score: logits[0], confidence: 1.0 };
    }

    const probs = this._softmax(logits);
    const maxIdx = probs.indexOf(Math.max(...probs));
    return {
      label: this.labels[maxIdx] || `class_${maxIdx}`,
      confidence: probs[maxIdx],
      probabilities: Object.fromEntries(this.labels.map((l, i) => [l, probs[i] || 0])),
    };
  }

  // ========== 回归/打分 ==========
  async _predictRegression(text) {
    const ort = require('onnxruntime-node');
    const maxLen = this.config.max_length || 128;
    const { inputIds, attentionMask } = this._tokenize(text, maxLen);

    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, maxLen]),
      attention_mask: new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, maxLen]),
    };
    if (this.session.inputNames.includes('token_type_ids')) {
      feeds.token_type_ids = new ort.Tensor('int64', new BigInt64Array(maxLen), [1, maxLen]);
    }

    const results = await this.session.run(feeds);
    const score = Number(results[this.session.outputNames[0]].data[0]);
    return { score, confidence: 1.0 };
  }

  // ========== FastText ==========
  _predictFastText(text) {
    if (!this.fastTextWeights) {
      return { label: this.labels[0], confidence: 0.5, probabilities: {} };
    }
    const words = text.toLowerCase().split(/\s+/);
    const scores = {};
    for (const label of this.labels) scores[label] = 0;

    for (const word of words) {
      if (this.fastTextWeights[word]) {
        for (const [label, weight] of Object.entries(this.fastTextWeights[word])) {
          if (scores[label] !== undefined) scores[label] += weight;
        }
      }
    }

    const probs = this._softmax(Object.values(scores));
    const entries = Object.keys(scores);
    const maxIdx = probs.indexOf(Math.max(...probs));
    return {
      label: entries[maxIdx],
      confidence: probs[maxIdx],
      probabilities: Object.fromEntries(entries.map((k, i) => [k, probs[i]])),
    };
  }

  // ========== Tokenizers ==========

  _tokenize(text, maxLen) {
    const inputIds = new Array(maxLen).fill(0);
    const attentionMask = new Array(maxLen).fill(0);

    const clsId = this.tokenizer?.['[CLS]'] || 101;
    const sepId = this.tokenizer?.['[SEP]'] || 102;
    const unkId = this.tokenizer?.['[UNK]'] || 100;

    inputIds[0] = clsId;
    attentionMask[0] = 1;

    const chars = [...text].slice(0, maxLen - 2);
    for (let i = 0; i < chars.length; i++) {
      inputIds[i + 1] = (this.tokenizer && this.tokenizer[chars[i]] !== undefined)
        ? this.tokenizer[chars[i]] : unkId;
      attentionMask[i + 1] = 1;
    }
    inputIds[chars.length + 1] = sepId;
    attentionMask[chars.length + 1] = 1;

    return { inputIds, attentionMask };
  }

  _tokenizeWithOffsets(text, maxLen) {
    const inputIds = new Array(maxLen).fill(0);
    const attentionMask = new Array(maxLen).fill(0);
    const charOffsets = new Array(maxLen).fill(-1);

    const clsId = this.tokenizer?.['[CLS]'] || 101;
    const sepId = this.tokenizer?.['[SEP]'] || 102;
    const unkId = this.tokenizer?.['[UNK]'] || 100;

    inputIds[0] = clsId;
    attentionMask[0] = 1;

    const chars = [...text].slice(0, maxLen - 2);
    for (let i = 0; i < chars.length; i++) {
      inputIds[i + 1] = (this.tokenizer && this.tokenizer[chars[i]] !== undefined)
        ? this.tokenizer[chars[i]] : unkId;
      attentionMask[i + 1] = 1;
      charOffsets[i + 1] = i;
    }
    inputIds[chars.length + 1] = sepId;
    attentionMask[chars.length + 1] = 1;

    return { inputIds, attentionMask, charOffsets };
  }

  _tokenizePair(textA, textB, maxLen) {
    const inputIds = new Array(maxLen).fill(0);
    const attentionMask = new Array(maxLen).fill(0);
    const tokenTypeIds = new Array(maxLen).fill(0);

    const clsId = this.tokenizer?.['[CLS]'] || 101;
    const sepId = this.tokenizer?.['[SEP]'] || 102;
    const unkId = this.tokenizer?.['[UNK]'] || 100;

    const halfLen = Math.floor((maxLen - 3) / 2);
    const charsA = [...textA].slice(0, halfLen);
    const charsB = [...textB].slice(0, halfLen);

    let pos = 0;
    inputIds[pos] = clsId; attentionMask[pos] = 1; pos++;

    for (const c of charsA) {
      inputIds[pos] = (this.tokenizer && this.tokenizer[c] !== undefined) ? this.tokenizer[c] : unkId;
      attentionMask[pos] = 1; pos++;
    }
    inputIds[pos] = sepId; attentionMask[pos] = 1; pos++;

    const segBStart = pos;
    for (const c of charsB) {
      inputIds[pos] = (this.tokenizer && this.tokenizer[c] !== undefined) ? this.tokenizer[c] : unkId;
      attentionMask[pos] = 1; tokenTypeIds[pos] = 1; pos++;
    }
    inputIds[pos] = sepId; attentionMask[pos] = 1; tokenTypeIds[pos] = 1;

    return { inputIds, attentionMask, tokenTypeIds };
  }

  _softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }
}

module.exports = { Predictor };
