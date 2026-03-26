const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || res.statusText)
  }
  return res.json()
}

export default {
  getStatus: () => request('/status'),
  getConfig: () => request('/config'),

  chat: (input, type) => request('/chat', { method: 'POST', body: { input, type } }),
  getChatHistory: (limit = 50, offset = 0) => request(`/chat/history?limit=${limit}&offset=${offset}`),
  getChatStats: () => request('/chat/stats'),

  getTools: () => request('/admin/tools'),
  testTool: (name, input) => request(`/admin/tools/${name}/test`, { method: 'POST', body: { input } }),
  deleteTool: (name) => request(`/admin/tools/${name}`, { method: 'DELETE' }),

  getLoop: () => request('/admin/loop'),
  getLoopHistory: () => request('/admin/loop/history'),
  rollbackLoop: (versionId) => request('/admin/loop/rollback', { method: 'POST', body: { versionId } }),

  triggerEvolve: () => request('/admin/evolve', { method: 'POST' }),
  evolveIntent: (intent) => request('/admin/evolve/intent', { method: 'POST', body: { intent } }),
  distill: (params) => request('/admin/evolve/distill', { method: 'POST', body: params }),
  getEvolveTasks: () => request('/admin/evolve/tasks'),
  resumeEvolveTask: (id) => request(`/admin/evolve/resume/${id}`, { method: 'POST' }),

  getTraining: () => request('/admin/training'),
  getTrainingJob: (id) => request(`/admin/training/${id}`),
  getEvolutionLogs: () => request('/admin/evolution'),
}
