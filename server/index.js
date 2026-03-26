require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { initDB } = require('./db');
const { setupWebSocket } = require('./ws');
const chatRoutes = require('./routes/chat');
const adminRoutes = require('./routes/admin');
const configRoutes = require('./routes/config');
const { LoopManager } = require('./loop-manager');
const { ToolRegistry } = require('./services/tool-registry');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '5mb' }));

const dataDir = path.join(__dirname, '..', 'data');
['models', 'datasets', 'logs', 'loop-versions', 'temp'].forEach(d => {
  fs.mkdirSync(path.join(dataDir, d), { recursive: true });
});

app.use('/data', express.static(dataDir));

initDB();

const loopManager = new LoopManager();
const toolRegistry = new ToolRegistry();

app.locals.loopManager = loopManager;
app.locals.toolRegistry = toolRegistry;

app.use('/api/chat', chatRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/config', configRoutes);

app.get('/', (req, res) => {
  res.json({
    name: 'TinyBERT Pipeline - Self-Evolving Agent Service',
    version: '1.0.0',
    status: 'running',
    loop_version: loopManager.getCurrentVersion(),
    tools_count: toolRegistry.getToolCount(),
  });
});

const server = app.listen(PORT, () => {
  console.log(`[TinyBERT Pipeline] 服务启动: http://localhost:${PORT}`);
  console.log(`[Loop] 当前版本: ${loopManager.getCurrentVersion()}`);
  console.log(`[Tools] 已注册工具: ${toolRegistry.getToolCount()}`);
});

setupWebSocket(server);
