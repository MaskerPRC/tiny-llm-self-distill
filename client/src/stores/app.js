import { defineStore } from 'pinia'
import { ref, onMounted, onUnmounted } from 'vue'

export const useAppStore = defineStore('app', () => {
  const status = ref(null)
  const config = ref(null)
  const wsLogs = ref([])
  const wsConnected = ref(false)
  let ws = null

  function connectWS() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(`${protocol}//${location.host}/ws`)

    ws.onopen = () => { wsConnected.value = true }
    ws.onclose = () => {
      wsConnected.value = false
      setTimeout(connectWS, 3000)
    }
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        wsLogs.value.unshift({ ...data, _time: new Date().toLocaleTimeString('zh-CN') })
        if (wsLogs.value.length > 200) wsLogs.value.length = 200
      } catch {}
    }
  }

  async function fetchStatus() {
    try {
      const res = await fetch('/api/status')
      status.value = await res.json()
    } catch {}
  }

  async function fetchConfig() {
    try {
      const res = await fetch('/api/config')
      config.value = await res.json()
    } catch {}
  }

  return { status, config, wsLogs, wsConnected, connectWS, fetchStatus, fetchConfig }
})
