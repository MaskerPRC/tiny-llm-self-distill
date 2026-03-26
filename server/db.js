const Database = require('better-sqlite3');
const path = require('path');

let db;

function initDB() {
  const dbPath = path.join(__dirname, '..', 'tinybert-pipeline.db');
  db = new Database(dbPath);
  db.pragma('journal_mode = WAL');

  db.exec(`
    CREATE TABLE IF NOT EXISTS request_logs (
      id TEXT PRIMARY KEY,
      user_input TEXT NOT NULL,
      task_type TEXT,
      tool_used TEXT,
      llm_response TEXT,
      confidence REAL,
      latency_ms INTEGER,
      created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS tools (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL UNIQUE,
      description TEXT,
      task_type TEXT NOT NULL,
      model_arch TEXT NOT NULL,
      model_path TEXT,
      onnx_path TEXT,
      accuracy REAL,
      status TEXT DEFAULT 'pending',
      config TEXT,
      created_at TEXT DEFAULT (datetime('now')),
      updated_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS training_jobs (
      id TEXT PRIMARY KEY,
      tool_id TEXT,
      task_type TEXT NOT NULL,
      model_arch TEXT NOT NULL,
      status TEXT DEFAULT 'pending',
      data_count INTEGER,
      epochs INTEGER,
      metrics TEXT,
      log TEXT,
      error TEXT,
      created_at TEXT DEFAULT (datetime('now')),
      updated_at TEXT DEFAULT (datetime('now')),
      FOREIGN KEY (tool_id) REFERENCES tools(id)
    );

    CREATE TABLE IF NOT EXISTS training_data (
      id TEXT PRIMARY KEY,
      job_id TEXT NOT NULL,
      input_text TEXT NOT NULL,
      label TEXT NOT NULL,
      source TEXT DEFAULT 'gemini',
      created_at TEXT DEFAULT (datetime('now')),
      FOREIGN KEY (job_id) REFERENCES training_jobs(id)
    );

    CREATE TABLE IF NOT EXISTS loop_versions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      version TEXT NOT NULL,
      code TEXT NOT NULL,
      reason TEXT,
      active INTEGER DEFAULT 0,
      created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS evolution_logs (
      id TEXT PRIMARY KEY,
      analysis TEXT,
      decision TEXT,
      action TEXT,
      result TEXT,
      created_at TEXT DEFAULT (datetime('now'))
    );
  `);

  return db;
}

function getDB() {
  if (!db) throw new Error('Database not initialized. Call initDB() first.');
  return db;
}

module.exports = { initDB, getDB };
