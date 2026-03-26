<template>
  <div>
    <h2 class="page-title">系统概览</h2>

    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-value">{{ stats.total_requests ?? '–' }}</div>
        <div class="stat-label">总请求数</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ stats.recent_hour ?? '–' }}</div>
        <div class="stat-label">近1小时请求</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ stats.avg_latency_ms ?? '–' }}<small>ms</small></div>
        <div class="stat-label">平均延迟</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ store.status?.tools_count ?? '–' }}</div>
        <div class="stat-label">已注册工具</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">v{{ store.status?.loop_version ?? '–' }}</div>
        <div class="stat-label">Loop 版本</div>
      </div>
    </div>

    <div class="card" style="margin-top: 20px">
      <div class="card-title">工具使用分布</div>
      <table v-if="stats.tool_usage?.length">
        <thead>
          <tr>
            <th>工具</th>
            <th>调用次数</th>
            <th>平均延迟</th>
            <th>平均置信度</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="t in stats.tool_usage" :key="t.tool_used">
            <td>
              <span class="tag" :class="t.tool_used.includes('gemini') ? 'tag-blue' : 'tag-green'">
                {{ t.tool_used }}
              </span>
            </td>
            <td>{{ t.count }}</td>
            <td>{{ Math.round(t.avg_latency) }}ms</td>
            <td>{{ t.avg_confidence?.toFixed(2) ?? '–' }}</td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty">暂无请求数据</div>
    </div>

    <div class="card">
      <div class="card-title">模型配置</div>
      <table v-if="store.config">
        <tbody>
          <tr>
            <td style="color:var(--text-dim)">用户请求模型</td>
            <td><span class="tag tag-blue">{{ store.config.gemini_model }}</span></td>
          </tr>
          <tr>
            <td style="color:var(--text-dim)">代码迭代模型</td>
            <td><span class="tag tag-purple">{{ store.config.claude_model }}</span></td>
          </tr>
          <tr>
            <td style="color:var(--text-dim)">架构选型模型</td>
            <td><span class="tag tag-green">{{ store.config.selector_model }}</span></td>
          </tr>
          <tr>
            <td style="color:var(--text-dim)">候选架构</td>
            <td>
              <span class="tag tag-purple" v-for="c in store.config.model_candidates" :key="c" style="margin-right:6px">
                {{ c }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-title">实时日志</div>
      <div class="log-panel">
        <div class="log-line" v-for="(log, i) in store.wsLogs.slice(0, 30)" :key="i">
          <span style="color:var(--text-dim)">{{ log._time }}</span>
          <span class="tag tag-blue" style="margin:0 6px">{{ log.type }}</span>
          {{ log.message || '' }}
        </div>
        <div v-if="!store.wsLogs.length" class="empty" style="padding:20px">等待日志...</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useAppStore } from '../stores/app'
import api from '../api'

const store = useAppStore()
const stats = ref({})

onMounted(async () => {
  try { stats.value = await api.getChatStats() } catch {}
})
</script>
