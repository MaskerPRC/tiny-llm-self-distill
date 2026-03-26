<template>
  <div class="layout">
    <aside class="sidebar">
      <div class="sidebar-brand">
        <h1>TinyBERT Pipeline</h1>
        <div class="version">v{{ store.status?.loop_version || '...' }}</div>
      </div>
      <nav>
        <router-link to="/" :class="{ active: $route.path === '/' }">
          <span>📊</span> 概览
        </router-link>
        <router-link to="/chat" :class="{ active: $route.path === '/chat' }">
          <span>💬</span> 对话
        </router-link>
        <router-link to="/tools" :class="{ active: $route.path === '/tools' }">
          <span>🔧</span> 工具
        </router-link>
        <router-link to="/evolution" :class="{ active: $route.path === '/evolution' }">
          <span>🧬</span> 进化
        </router-link>
        <router-link to="/loop" :class="{ active: $route.path === '/loop' }">
          <span>🔄</span> 流程
        </router-link>
      </nav>
      <div class="sidebar-footer">
        <span class="ws-dot" :class="store.wsConnected ? 'on' : 'off'"></span>
        {{ store.wsConnected ? 'WebSocket 已连接' : '连接中...' }}
      </div>
    </aside>
    <main class="main">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useAppStore } from './stores/app'
import './style.css'

const store = useAppStore()

onMounted(() => {
  store.fetchStatus()
  store.fetchConfig()
  store.connectWS()
  setInterval(() => store.fetchStatus(), 15000)
})
</script>
