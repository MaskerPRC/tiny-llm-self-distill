const path = require('path');
require('dotenv').config();
const { initDB, getDB } = require('../server/db');
initDB();
const db = getDB();

const taskId = '5b47abe2-0000-0000-0000-000000000000';
const savePath = path.resolve(__dirname, '..', 'data', 'training-data', 'malicious_intent_detection_5b47abe2.jsonl');

const state = {
  plan: {
    summary: '\u5728\u5904\u7406\u6700\u524d\u9762\u52a0\u610f\u56fe\u8bc6\u522b\uff0c\u6076\u610f\u5219\u8fd4\u56de\u8bf7\u597d\u597d\u8bf4\u8bdd',
    needs_model: true,
    model_task: {
      task_type: 'malicious_intent_detection',
      description: '\u5224\u65ad\u7528\u6237\u8f93\u5165\u662f\u5426\u4e3a\u6076\u610f/\u8fb1\u9a82\u5185\u5bb9',
      labels: ['benign', 'malicious_or_abusive'],
    },
    loop_instruction: '\u5728 loop.js \u6700\u524d\u9762\u52a0\u610f\u56fe\u8bc6\u522b\u5206\u6d41\uff0c\u5982\u679c\u662f\u6076\u610f\u5219\u76f4\u63a5\u8fd4\u56de\u8bf7\u597d\u597d\u8bf4\u8bdd',
  },
  summary: '\u5728\u5904\u7406\u6700\u524d\u9762\u52a0\u610f\u56fe\u8bc6\u522b\uff0c\u6076\u610f\u5219\u8fd4\u56de\u8bf7\u597d\u597d\u8bf4\u8bdd',
  needsModel: true,
  loopInstruction: '\u5728 loop.js \u6700\u524d\u9762\u52a0\u610f\u56fe\u8bc6\u522b\u5206\u6d41\uff0c\u5982\u679c\u662f\u6076\u610f\u5219\u76f4\u63a5\u8fd4\u56de\u8bf7\u597d\u597d\u8bf4\u8bdd',
  modelTask: {
    task_type: 'malicious_intent_detection',
    description: '\u5224\u65ad\u7528\u6237\u8f93\u5165\u662f\u5426\u4e3a\u6076\u610f/\u8fb1\u9a82\u5185\u5bb9',
    labels: ['benign', 'malicious_or_abusive'],
  },
  archConfig: {
    model_arch: 'tinybert',
    labels: ['benign', 'malicious_or_abusive'],
    training_config: { epochs: 5, batch_size: 16, learning_rate: 2e-5, max_length: 128 },
  },
  finalLabels: ['benign', 'malicious_or_abusive'],
  trainConfig: { epochs: 5, batch_size: 16, learning_rate: 2e-5, max_length: 128 },
  distillId: '5b47abe2',
  savePath,
  trainingDataCount: 5000,
};

db.prepare('DELETE FROM evolution_tasks WHERE id = ?').run(taskId);

db.prepare(
  'INSERT INTO evolution_tasks (id, type, intent_text, status, current_step, state) VALUES (?, ?, ?, ?, ?, ?)'
).run(
  taskId,
  'intent',
  '\u5728\u5904\u7406\u6700\u524d\u9762\u52a0\u4e00\u4e2a\u610f\u56fe\u8bc6\u522b\uff0c\u5982\u679c\u662f\u6076\u610f\uff0c\u5219\u76f4\u63a5\u8fd4\u56de\u6587\u5b57\uff1a\u8bf7\u597d\u597d\u8bf4\u8bdd\u3002',
  'failed',
  'intent_distill_verify',
  JSON.stringify(state),
);

const row = db.prepare('SELECT * FROM evolution_tasks WHERE id = ?').get(taskId);
console.log('Task inserted OK');
console.log('  id:', row.id);
console.log('  status:', row.status);
console.log('  current_step:', row.current_step);
console.log('  intent:', row.intent_text);
console.log('  data path:', JSON.parse(row.state).savePath);
console.log('  data count:', JSON.parse(row.state).trainingDataCount);
