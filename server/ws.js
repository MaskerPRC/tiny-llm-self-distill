const { WebSocketServer } = require('ws');

let wss;

function setupWebSocket(server) {
  wss = new WebSocketServer({ server, path: '/ws' });

  wss.on('connection', (ws) => {
    console.log('[WS] 客户端连接');
    ws.on('close', () => console.log('[WS] 客户端断开'));
  });
}

function broadcast(data) {
  if (!wss) return;
  const msg = JSON.stringify(data);
  wss.clients.forEach((client) => {
    if (client.readyState === 1) {
      client.send(msg);
    }
  });
}

module.exports = { setupWebSocket, broadcast };
