<template>
  <div>
    <h2 class="page-title">已注册工具</h2>

    <div v-if="tools.length" style="display:flex;flex-direction:column;gap:12px">
      <div class="card" v-for="tool in tools" :key="tool.id">
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
          <div>
            <h3 style="font-size:16px;margin-bottom:4px">{{ tool.name }}</h3>
            <p style="color:var(--text-dim);font-size:13px">{{ tool.description }}</p>
          </div>
          <div style="display:flex;gap:6px">
            <span class="tag tag-purple">{{ tool.model_arch }}</span>
            <span class="tag" :class="tool.status === 'active' ? 'tag-green' : 'tag-red'">
              {{ tool.status }}
            </span>
          </div>
        </div>

        <div style="display:flex;gap:20px;margin-top:12px;font-size:13px;color:var(--text-dim)">
          <span>任务类型: <b style="color:var(--text)">{{ tool.task_type }}</b></span>
          <span>准确率: <b style="color:var(--green)">{{ tool.accuracy ? (tool.accuracy * 100).toFixed(1) + '%' : '–' }}</b></span>
          <span>创建: {{ tool.created_at }}</span>
        </div>

        <div style="margin-top:12px;display:flex;gap:8px;align-items:flex-end">
          <div class="form-group" style="flex:3">
            <label>测试输入</label>
            <input v-model="testInputs[tool.name]" placeholder="输入文本测试..." @keydown.enter="testTool(tool.name)" />
          </div>
          <button class="btn btn-primary" @click="testTool(tool.name)" :disabled="!testInputs[tool.name]">
            测试
          </button>
          <button class="btn btn-danger" @click="deleteTool(tool.name)">停用</button>
        </div>

        <div v-if="testResults[tool.name]" class="code-block" style="margin-top:8px;max-height:150px">{{ JSON.stringify(testResults[tool.name], null, 2) }}</div>
      </div>
    </div>

    <div v-else class="card empty">
      <p>暂无已注册工具</p>
      <p style="margin-top:8px;font-size:13px">系统会在积累足够请求后自动蒸馏工具，或到「进化」页面手动触发蒸馏</p>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import api from '../api'

const tools = ref([])
const testInputs = reactive({})
const testResults = reactive({})

async function load() {
  try {
    const res = await api.getTools()
    tools.value = res.tools
  } catch {}
}

async function testTool(name) {
  if (!testInputs[name]) return
  try {
    const res = await api.testTool(name, testInputs[name])
    testResults[name] = res.result
  } catch (err) {
    testResults[name] = { error: err.message }
  }
}

async function deleteTool(name) {
  if (!confirm(`确定停用工具 "${name}"？`)) return
  try {
    await api.deleteTool(name)
    load()
  } catch {}
}

onMounted(load)
</script>
