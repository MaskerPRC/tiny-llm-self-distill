const express = require('express');
const { v4: uuidv4 } = require('uuid');
const { getDB } = require('../db');
const { broadcast } = require('../ws');
const { GeminiService } = require('../services/gemini');

const router = express.Router();

router.post('/', async (req, res) => {
  const { input, type, metadata } = req.body;

  if (!input) {
    return res.status(400).json({ error: '缺少 input 字段' });
  }

  const startTime = Date.now();
  const requestId = uuidv4();

  try {
    const loopManager = req.app.locals.loopManager;
    const toolRegistry = req.app.locals.toolRegistry;
    const gemini = new GeminiService();

    const context = {
      gemini,
      tools: {
        predict: (toolName, inp) => toolRegistry.predict(toolName, inp),
        list: () => toolRegistry.list(),
      },
      log: (message) => {
        console.log(`[Loop] ${message}`);
        broadcast({ type: 'loop_log', requestId, message });
      },
    };

    const request = { input, type, metadata };
    const result = await loopManager.executeLoop(request, context);

    const latencyMs = Date.now() - startTime;

    const db = getDB();
    db.prepare(`
      INSERT INTO request_logs (id, user_input, task_type, tool_used, llm_response, confidence, latency_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).run(
      requestId,
      input,
      result.metadata?.task_type || type || null,
      result.tool_used || null,
      typeof result.output === 'string' ? result.output : JSON.stringify(result.output),
      result.confidence || null,
      latencyMs,
    );

    res.json({
      id: requestId,
      output: result.output,
      tool_used: result.tool_used,
      confidence: result.confidence,
      latency_ms: latencyMs,
      loop_version: loopManager.getCurrentVersion(),
      metadata: result.metadata,
    });
  } catch (err) {
    const latencyMs = Date.now() - startTime;
    console.error(`[Chat] 处理失败:`, err.message);

    try {
      const db = getDB();
      db.prepare(`
        INSERT INTO request_logs (id, user_input, task_type, tool_used, llm_response, confidence, latency_ms)
        VALUES (?, ?, ?, 'error', ?, 0, ?)
      `).run(requestId, input, type || null, err.message, latencyMs);
    } catch {}

    res.status(500).json({ error: err.message, id: requestId });
  }
});

router.get('/history', (req, res) => {
  const db = getDB();
  const limit = parseInt(req.query.limit) || 50;
  const offset = parseInt(req.query.offset) || 0;
  const logs = db.prepare(
    'SELECT * FROM request_logs ORDER BY created_at DESC LIMIT ? OFFSET ?'
  ).all(limit, offset);
  const total = db.prepare('SELECT COUNT(*) as count FROM request_logs').get().count;
  res.json({ logs, total, limit, offset });
});

router.get('/stats', (req, res) => {
  const db = getDB();

  const totalRequests = db.prepare('SELECT COUNT(*) as count FROM request_logs').get().count;
  const toolUsage = db.prepare(`
    SELECT tool_used, COUNT(*) as count, AVG(latency_ms) as avg_latency, AVG(confidence) as avg_confidence
    FROM request_logs
    WHERE tool_used IS NOT NULL
    GROUP BY tool_used
    ORDER BY count DESC
  `).all();
  const avgLatency = db.prepare('SELECT AVG(latency_ms) as avg FROM request_logs').get().avg;

  const recentHour = db.prepare(`
    SELECT COUNT(*) as count FROM request_logs
    WHERE created_at > datetime('now', '-1 hour')
  `).get().count;

  res.json({
    total_requests: totalRequests,
    recent_hour: recentHour,
    avg_latency_ms: Math.round(avgLatency || 0),
    tool_usage: toolUsage,
  });
});

module.exports = router;
