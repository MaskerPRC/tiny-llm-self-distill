const express = require('express');
const router = express.Router();

router.get('/', (req, res) => {
  res.json({
    gemini_model: process.env.GEMINI_MODEL || 'google/gemini-3.1-pro-preview',
    claude_model: process.env.CLAUDE_MODEL || 'anthropic/claude-opus-4.6',
    default_model_arch: process.env.DEFAULT_MODEL_ARCH || 'tinybert',
    evolve_min_requests: parseInt(process.env.EVOLVE_MIN_REQUESTS) || 50,
    evolve_pattern_threshold: parseFloat(process.env.EVOLVE_PATTERN_THRESHOLD) || 0.3,
    predict_confidence_threshold: parseFloat(process.env.PREDICT_CONFIDENCE_THRESHOLD) || 0.8,
    train_data_count: parseInt(process.env.TRAIN_DATA_COUNT) || 5000,
    train_epochs: parseInt(process.env.TRAIN_EPOCHS) || 5,
    train_val_split: parseFloat(process.env.TRAIN_VAL_SPLIT) || 0.2,
    supported_architectures: [
      { id: 'fasttext', name: 'FastText', size: '<2MB', complexity: 'low' },
      { id: 'tinybert', name: 'TinyBERT (4层)', size: '15-50MB', complexity: 'medium' },
      { id: 'minilm', name: 'MiniLM-L6', size: '20-80MB', complexity: 'medium' },
      { id: 'distilbert', name: 'DistilBERT', size: '130MB+', complexity: 'high' },
    ],
  });
});

module.exports = router;
