require('dotenv').config();
const { initDB, getDB } = require('../server/db');
initDB();
const db = getDB();

const taskId = '5b47abe2-0000-0000-0000-000000000000';
const task = db.prepare('SELECT * FROM evolution_tasks WHERE id = ?').get(taskId);

if (!task) {
  console.log('Task not found');
  process.exit(1);
}

db.prepare("DELETE FROM tools WHERE name = 'malicious_intent_detection_tinybert'").run();
console.log('Deleted old tool registration');

db.prepare(`UPDATE evolution_tasks SET status = 'failed', current_step = 'intent_distill_train', error = NULL, updated_at = datetime('now') WHERE id = ?`).run(taskId);

const row = db.prepare('SELECT id, status, current_step FROM evolution_tasks WHERE id = ?').get(taskId);
console.log('Task reset:', row.id, row.status, row.current_step);
console.log('Ready to resume from training step with Chinese model (hfl/rbt3)');
