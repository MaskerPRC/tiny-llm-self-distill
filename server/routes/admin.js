const express = require('express');
const { getDB } = require('../db');

const router = express.Router();

router.get('/tools', (req, res) => {
  const db = getDB();
  const tools = db.prepare('SELECT * FROM tools ORDER BY created_at DESC').all();
  res.json({ tools });
});

router.get('/tools/:name', (req, res) => {
  const toolRegistry = req.app.locals.toolRegistry;
  const tool = toolRegistry.get(req.params.name);
  if (!tool) return res.status(404).json({ error: '工具不存在' });
  res.json({ tool });
});

router.delete('/tools/:name', async (req, res) => {
  try {
    const toolRegistry = req.app.locals.toolRegistry;
    await toolRegistry.deactivate(req.params.name);
    res.json({ message: `工具 ${req.params.name} 已停用` });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/tools/:name/test', async (req, res) => {
  const { input } = req.body;
  if (!input) return res.status(400).json({ error: '缺少 input' });

  try {
    const toolRegistry = req.app.locals.toolRegistry;
    const result = await toolRegistry.predict(req.params.name, input);
    res.json({ result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/loop', (req, res) => {
  const loopManager = req.app.locals.loopManager;
  res.json({
    version: loopManager.getCurrentVersion(),
    code: loopManager.getCurrentCode(),
    history: loopManager.getVersionHistory(),
  });
});

router.get('/loop/history', (req, res) => {
  const loopManager = req.app.locals.loopManager;
  res.json({ versions: loopManager.getVersionHistory() });
});

router.post('/loop/rollback', (req, res) => {
  const { versionId } = req.body;
  if (!versionId) return res.status(400).json({ error: '缺少 versionId' });

  try {
    const loopManager = req.app.locals.loopManager;
    const result = loopManager.rollback(versionId);
    res.json({ message: '回滚成功', ...result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/evolve', async (req, res) => {
  try {
    const { Evolver } = require('../evolver');
    const loopManager = req.app.locals.loopManager;
    const toolRegistry = req.app.locals.toolRegistry;
    const evolver = new Evolver(loopManager, toolRegistry);

    res.json({ message: '进化流程已启动，请通过 WebSocket 查看进度' });

    evolver.evolve().catch(err => {
      console.error('[Evolve] 进化失败:', err.message);
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/evolve/distill', async (req, res) => {
  const { taskType, description, labels, modelArch, dataCount } = req.body;

  if (!taskType || !description || !labels) {
    return res.status(400).json({ error: '缺少必要字段: taskType, description, labels' });
  }

  try {
    const { Evolver } = require('../evolver');
    const loopManager = req.app.locals.loopManager;
    const toolRegistry = req.app.locals.toolRegistry;
    const evolver = new Evolver(loopManager, toolRegistry);

    res.json({ message: `蒸馏任务 "${taskType}" 已启动` });

    evolver.distillTask({
      taskType,
      description,
      labels: Array.isArray(labels) ? labels : labels.split(',').map(s => s.trim()),
      modelArch: modelArch || process.env.DEFAULT_MODEL_ARCH || 'tinybert',
      dataCount: dataCount || parseInt(process.env.TRAIN_DATA_COUNT) || 5000,
    }).catch(err => {
      console.error(`[Distill] 蒸馏 "${taskType}" 失败:`, err.message);
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/training', (req, res) => {
  const db = getDB();
  const jobs = db.prepare('SELECT * FROM training_jobs ORDER BY created_at DESC').all();
  res.json({ jobs: jobs.map(j => ({ ...j, metrics: j.metrics ? JSON.parse(j.metrics) : null })) });
});

router.get('/training/:id', (req, res) => {
  const db = getDB();
  const job = db.prepare('SELECT * FROM training_jobs WHERE id = ?').get(req.params.id);
  if (!job) return res.status(404).json({ error: '训练任务不存在' });
  res.json({ job: { ...job, metrics: job.metrics ? JSON.parse(job.metrics) : null } });
});

router.get('/evolution', (req, res) => {
  const db = getDB();
  const logs = db.prepare('SELECT * FROM evolution_logs ORDER BY created_at DESC LIMIT 20').all();
  res.json({ logs: logs.map(l => ({
    ...l,
    analysis: l.analysis ? JSON.parse(l.analysis) : null,
    decision: l.decision ? JSON.parse(l.decision) : null,
  })) });
});

module.exports = router;
