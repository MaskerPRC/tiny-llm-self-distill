const path = require('path');
const fs = require('fs');

class Predictor {
  constructor(toolConfig) {
    this.name = toolConfig.name;
    this.modelArch = toolConfig.modelArch;
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
      throw new Error(`ONNX 模型文件不存在: ${onnxFile}`);
    }

    this.session = await ort.InferenceSession.create(onnxFile);

    const vocabPath = path.join(path.dirname(onnxFile), 'vocab.json');
    const tokenizerConfigPath = path.join(path.dirname(onnxFile), 'tokenizer_config.json');

    if (fs.existsSync(vocabPath)) {
      this.tokenizer = JSON.parse(fs.readFileSync(vocabPath, 'utf-8'));
    }

    if (fs.existsSync(tokenizerConfigPath)) {
      this.tokenizerConfig = JSON.parse(fs.readFileSync(tokenizerConfigPath, 'utf-8'));
    }

    const labelsPath = path.join(path.dirname(onnxFile), 'labels.json');
    if (fs.existsSync(labelsPath)) {
      this.labels = JSON.parse(fs.readFileSync(labelsPath, 'utf-8'));
    }

    console.log(`[Predictor] ONNX 模型加载完成: ${this.name} (${this.labels.length} 个标签)`);
  }

  async _loadFastText() {
    const modelDir = path.dirname(this.modelPath);
    const labelsPath = path.join(modelDir, 'labels.json');
    const weightsPath = path.join(modelDir, 'weights.json');

    if (fs.existsSync(labelsPath)) {
      this.labels = JSON.parse(fs.readFileSync(labelsPath, 'utf-8'));
    }

    if (fs.existsSync(weightsPath)) {
      this.fastTextWeights = JSON.parse(fs.readFileSync(weightsPath, 'utf-8'));
    }

    console.log(`[Predictor] FastText 模型加载完成: ${this.name}`);
  }

  async predict(input) {
    if (this.modelArch === 'fasttext') {
      return this._predictFastText(input);
    }
    return this._predictOnnx(input);
  }

  async _predictOnnx(input) {
    const ort = require('onnxruntime-node');
    const maxLen = this.config.max_length || 128;
    const { inputIds, attentionMask } = this._tokenize(input, maxLen);

    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, maxLen]),
      attention_mask: new ort.Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, maxLen]),
    };

    if (this.session.inputNames.includes('token_type_ids')) {
      feeds.token_type_ids = new ort.Tensor('int64', new BigInt64Array(maxLen), [1, maxLen]);
    }

    const results = await this.session.run(feeds);
    const outputName = this.session.outputNames[0];
    const logits = Array.from(results[outputName].data);

    const probs = this._softmax(logits);
    const maxIdx = probs.indexOf(Math.max(...probs));

    return {
      label: this.labels[maxIdx] || `class_${maxIdx}`,
      confidence: probs[maxIdx],
      probabilities: Object.fromEntries(
        this.labels.map((label, i) => [label, probs[i] || 0])
      ),
    };
  }

  _predictFastText(input) {
    if (!this.fastTextWeights) {
      return { label: this.labels[0], confidence: 0.5, probabilities: {} };
    }

    const words = input.toLowerCase().split(/\s+/);
    const scores = {};
    for (const label of this.labels) {
      scores[label] = 0;
    }

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

  _tokenize(text, maxLen) {
    const inputIds = new Array(maxLen).fill(0);
    const attentionMask = new Array(maxLen).fill(0);

    if (this.tokenizer) {
      inputIds[0] = this.tokenizer['[CLS]'] || 101;
      attentionMask[0] = 1;

      const chars = text.toLowerCase().split('');
      for (let i = 0; i < Math.min(chars.length, maxLen - 2); i++) {
        const token = this.tokenizer[chars[i]] || this.tokenizer['[UNK]'] || 100;
        inputIds[i + 1] = token;
        attentionMask[i + 1] = 1;
      }

      const seqLen = Math.min(chars.length, maxLen - 2) + 1;
      inputIds[seqLen] = this.tokenizer['[SEP]'] || 102;
      attentionMask[seqLen] = 1;
    } else {
      inputIds[0] = 101;
      attentionMask[0] = 1;
      const charCodes = [...text].slice(0, maxLen - 2);
      for (let i = 0; i < charCodes.length; i++) {
        inputIds[i + 1] = charCodes[i].charCodeAt(0) % 30000;
        attentionMask[i + 1] = 1;
      }
      inputIds[charCodes.length + 1] = 102;
      attentionMask[charCodes.length + 1] = 1;
    }

    return { inputIds, attentionMask };
  }

  _softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }
}

module.exports = { Predictor };
