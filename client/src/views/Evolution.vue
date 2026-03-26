<template>
  <div>
    <h2 class="page-title">自进化引擎</h2>

    <div class="stat-grid" style="margin-bottom:20px">
      <div class="stat-card">
        <div class="stat-value" style="font-size:20px">
          <button class="btn btn-primary" @click="triggerEvolve" :disabled="evolving">
            {{ evolving ? '进化中...' : '触发自动进化' }}
          </button>
        </div>
        <div class="stat-label">分析请求日志 → 识别模式 → 蒸馏 → 更新流程</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">手动蒸馏任务</div>
      <div class="form-row">
        <div class="form-group">
          <label>任务类型 (英文)</label>
          <input v-model="distill.taskType" placeholder="如: sentiment_analysis" />
        </div>
        <div class="form-group" style="flex:2">
          <label>任务描述</label>
          <input v-model="distill.description" placeholder="如: 判断用户评论是消极还是积极情绪" />
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>标签 (逗号分隔)</label>
          <input v-model="distill.labels" placeholder="如: 消极,积极" />
        </div>
        <div class="form-group" style="flex:0 0 120px">
          <label>数据量</label>
          <input v-model.number="distill.dataCount" type="number" />
        </div>
        <button class="btn btn-primary" style="height:38px" @click="startDistill" :disabled="distilling || !distill.taskType || !distill.description || !distill.labels">
          {{ distilling ? '蒸馏中...' : '开始蒸馏' }}
        </button>
      </div>
      <p style="font-size:12px;color:var(--text-dim);margin-top:4px">
        模型架构由 GPT-5.4 自动选择，准确率不达标时自动升级到更大架构重试
      </p>
    </div>

    <div class="card">
      <div class="card-title">训练任务</div>
      <table v-if="jobs.length">
        <thead>
          <tr>
            <th>ID</th>
            <th>任务类型</th>
            <th>架构</th>
            <th>数据量</th>
            <th>状态</th>
            <th>准确率</th>
            <th>时间</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="j in jobs" :key="j.id">
            <td style="font-family:monospace;font-size:12px">{{ j.id?.substring(0, 8) }}</td>
            <td>{{ j.task_type }}</td>
            <td><span class="tag tag-purple">{{ j.model_arch }}</span></td>
            <td>{{ j.data_count }}</td>
            <td>
              <span class="tag" :class="statusClass(j.status)">{{ j.status }}</span>
            </td>
            <td style="color:var(--green)">{{ j.metrics?.accuracy ? (j.metrics.accuracy * 100).toFixed(1) + '%' : '–' }}</td>
            <td style="color:var(--text-dim);font-size:12px">{{ j.created_at }}</td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty">暂无训练任务</div>
    </div>

    <div class="card">
      <div class="card-title">进化日志</div>
      <div v-if="evoLogs.length" style="display:flex;flex-direction:column;gap:8px">
        <div v-for="l in evoLogs" :key="l.id" style="border-bottom:1px solid var(--border);padding-bottom:8px">
          <div style="display:flex;justify-content:space-between">
            <span class="tag" :class="l.result === 'success' ? 'tag-green' : l.result === 'skipped' ? 'tag-yellow' : 'tag-red'">
              {{ l.action }}
            </span>
            <span style="font-size:12px;color:var(--text-dim)">{{ l.created_at }}</span>
          </div>
          <div v-if="l.analysis?.summary" style="font-size:13px;margin-top:4px;color:var(--text-dim)">
            {{ l.analysis.summary }}
          </div>
        </div>
      </div>
      <div v-else class="empty">暂无进化记录</div>
    </div>

    <div class="card">
      <div class="card-title">实时进度</div>
      <div class="log-panel">
        <div class="log-line" v-for="(log, i) in evolverLogs" :key="i">{{ log._time }} {{ log.message }}</div>
        <div v-if="!evolverLogs.length" style="padding:12px;color:var(--text-dim)">等待进化事件...</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useAppStore } from '../stores/app'
import api from '../api'

const store = useAppStore()
const evolving = ref(false)
const distilling = ref(false)
const jobs = ref([])
const evoLogs = ref([])

const distill = ref({
  taskType: '',
  description: '',
  labels: '',
  dataCount: 5000,
})

const evolverLogs = computed(() =>
  store.wsLogs.filter(l => l.type?.startsWith('evolver_') || l.type?.startsWith('training_') || l.type?.startsWith('distill'))
)

function statusClass(s) {
  if (s === 'completed') return 'tag-green'
  if (s === 'training' || s === 'preparing') return 'tag-yellow'
  if (s === 'failed' || s === 'aborted') return 'tag-red'
  return 'tag-blue'
}

async function triggerEvolve() {
  evolving.value = true
  try {
    await api.triggerEvolve()
  } catch (err) {
    alert(err.message)
  }
  setTimeout(() => { evolving.value = false }, 3000)
}

async function startDistill() {
  distilling.value = true
  try {
    await api.distill({
      taskType: distill.value.taskType,
      description: distill.value.description,
      labels: distill.value.labels,
      dataCount: distill.value.dataCount,
    })
  } catch (err) {
    alert(err.message)
  }
  setTimeout(() => { distilling.value = false }, 3000)
}

async function load() {
  try {
    const [t, e] = await Promise.all([api.getTraining(), api.getEvolutionLogs()])
    jobs.value = t.jobs
    evoLogs.value = e.logs
  } catch {}
}

onMounted(load)
</script>
