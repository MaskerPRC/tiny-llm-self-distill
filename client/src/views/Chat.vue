<template>
  <div>
    <h2 class="page-title">对话测试</h2>

    <div class="card">
      <div class="chat-messages" ref="msgBox">
        <div v-for="msg in messages" :key="msg.id" class="chat-msg" :class="msg.role">
          {{ msg.content }}
          <div class="chat-meta" v-if="msg.meta">
            {{ msg.meta }}
          </div>
        </div>
        <div v-if="loading" class="chat-msg assistant" style="opacity:.6">思考中...</div>
      </div>

      <div style="display:flex;gap:8px;margin-top:12px">
        <input
          v-model="input"
          placeholder="输入消息..."
          @keydown.enter="send"
          :disabled="loading"
        />
        <button class="btn btn-primary" @click="send" :disabled="loading || !input.trim()">
          发送
        </button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">请求历史</div>
      <table v-if="history.length">
        <thead>
          <tr>
            <th>输入</th>
            <th>工具</th>
            <th>置信度</th>
            <th>延迟</th>
            <th>时间</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="h in history" :key="h.id">
            <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{{ h.user_input }}</td>
            <td>
              <span class="tag" :class="h.tool_used?.includes('gemini') ? 'tag-blue' : 'tag-green'">
                {{ h.tool_used || '–' }}
              </span>
            </td>
            <td>{{ h.confidence?.toFixed(2) ?? '–' }}</td>
            <td>{{ h.latency_ms }}ms</td>
            <td style="color:var(--text-dim);font-size:12px">{{ h.created_at }}</td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty">暂无历史</div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import api from '../api'

const input = ref('')
const loading = ref(false)
const messages = ref([])
const history = ref([])
const msgBox = ref(null)

async function send() {
  if (!input.value.trim() || loading.value) return
  const text = input.value.trim()
  input.value = ''
  messages.value.push({ id: Date.now(), role: 'user', content: text })
  loading.value = true
  scrollToBottom()

  try {
    const res = await api.chat(text)
    messages.value.push({
      id: res.id,
      role: 'assistant',
      content: res.output,
      meta: `${res.tool_used} | ${res.confidence?.toFixed(2)} | ${res.latency_ms}ms | loop v${res.loop_version}`,
    })
    loadHistory()
  } catch (err) {
    messages.value.push({ id: Date.now(), role: 'assistant', content: `错误: ${err.message}` })
  } finally {
    loading.value = false
    scrollToBottom()
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (msgBox.value) msgBox.value.scrollTop = msgBox.value.scrollHeight
  })
}

async function loadHistory() {
  try {
    const res = await api.getChatHistory(20)
    history.value = res.logs
  } catch {}
}

onMounted(loadHistory)
</script>
