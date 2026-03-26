<template>
  <div>
    <h2 class="page-title">元流程管理</h2>

    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <div>
          <div class="card-title" style="margin-bottom:0">当前 loop.js</div>
          <span class="tag tag-green" style="margin-top:4px">v{{ loopData.version }}</span>
        </div>
        <button class="btn btn-ghost" @click="load">刷新</button>
      </div>
      <div class="code-block">{{ loopData.code || '加载中...' }}</div>
    </div>

    <div class="card">
      <div class="card-title">版本历史</div>
      <p style="font-size:12px;color:var(--text-dim);margin-bottom:12px">
        每个版本由 Claude Opus 4.6 在进化时自动生成，包含小模型前置分流逻辑。点击「回滚」可恢复到任意历史版本。
      </p>
      <table v-if="loopData.history?.length">
        <thead>
          <tr>
            <th>ID</th>
            <th>版本</th>
            <th>原因</th>
            <th>状态</th>
            <th>创建时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="v in loopData.history" :key="v.id">
            <td style="font-family:monospace">{{ v.id }}</td>
            <td><b>v{{ v.version }}</b></td>
            <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{{ v.reason }}</td>
            <td>
              <span class="tag" :class="v.active ? 'tag-green' : 'tag-blue'">
                {{ v.active ? '当前' : '历史' }}
              </span>
            </td>
            <td style="color:var(--text-dim);font-size:12px">{{ v.created_at }}</td>
            <td>
              <button
                v-if="!v.active"
                class="btn btn-ghost"
                style="padding:4px 10px;font-size:12px"
                @click="rollback(v.id, v.version)"
              >
                回滚
              </button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty">只有初始版本</div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import api from '../api'

const loopData = ref({ version: '', code: '', history: [] })

async function load() {
  try {
    loopData.value = await api.getLoop()
  } catch {}
}

async function rollback(id, version) {
  if (!confirm(`确定回滚到 v${version}？`)) return
  try {
    await api.rollbackLoop(id)
    load()
  } catch (err) {
    alert(err.message)
  }
}

onMounted(load)
</script>
