const fs = require('fs');
const path = require('path');
const { getDB } = require('../db');
const { broadcast } = require('../ws');

class ToolRegistry {
  constructor() {
    this.tools = new Map();
    this.predictors = new Map();
    this._loadFromDB();
  }

  _loadFromDB() {
    try {
      const db = getDB();
      const tools = db.prepare("SELECT * FROM tools WHERE status = 'active'").all();
      for (const tool of tools) {
        this._registerInMemory(tool);
      }
      console.log(`[ToolRegistry] 从数据库加载了 ${tools.length} 个工具`);
    } catch (err) {
      console.error('[ToolRegistry] 加载工具失败:', err.message);
    }
  }

  _registerInMemory(toolRow) {
    const config = toolRow.config ? JSON.parse(toolRow.config) : {};
    this.tools.set(toolRow.name, {
      id: toolRow.id,
      name: toolRow.name,
      description: toolRow.description,
      taskType: toolRow.task_type,
      modelArch: toolRow.model_arch,
      modelPath: toolRow.model_path,
      onnxPath: toolRow.onnx_path,
      accuracy: toolRow.accuracy,
      config,
    });
  }

  async register(toolInfo) {
    const db = getDB();
    const { v4: uuidv4 } = require('uuid');
    const id = uuidv4();

    db.prepare(`
      INSERT INTO tools (id, name, description, task_type, model_arch, model_path, onnx_path, accuracy, status, config)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
    `).run(
      id,
      toolInfo.name,
      toolInfo.description,
      toolInfo.taskType,
      toolInfo.modelArch,
      toolInfo.modelPath || null,
      toolInfo.onnxPath || null,
      toolInfo.accuracy || null,
      JSON.stringify(toolInfo.config || {}),
    );

    const tool = { id, ...toolInfo, status: 'active' };
    this._registerInMemory({ ...tool, task_type: tool.taskType, model_arch: tool.modelArch, model_path: tool.modelPath, onnx_path: tool.onnxPath });

    broadcast({ type: 'tool_registered', tool: { id, name: toolInfo.name, taskType: toolInfo.taskType } });
    console.log(`[ToolRegistry] 注册工具: ${toolInfo.name} (${toolInfo.modelArch})`);
    return tool;
  }

  async predict(toolName, input) {
    const tool = this.tools.get(toolName);
    if (!tool) throw new Error(`工具 "${toolName}" 未注册`);

    let predictor = this.predictors.get(toolName);
    if (!predictor) {
      predictor = await this._loadPredictor(tool);
      this.predictors.set(toolName, predictor);
    }

    return predictor(input);
  }

  async _loadPredictor(tool) {
    const { Predictor } = require('./predictor');
    const predictor = new Predictor(tool);
    await predictor.load();
    return (input) => predictor.predict(input);
  }

  list() {
    return Array.from(this.tools.values()).map(t => ({
      name: t.name,
      description: t.description,
      taskType: t.taskType,
      modelArch: t.modelArch,
      accuracy: t.accuracy,
    }));
  }

  get(name) {
    return this.tools.get(name);
  }

  getToolCount() {
    return this.tools.size;
  }

  getByTaskType(taskType) {
    return Array.from(this.tools.values()).filter(t => t.taskType === taskType);
  }

  async deactivate(toolName) {
    const db = getDB();
    db.prepare("UPDATE tools SET status = 'inactive', updated_at = datetime('now') WHERE name = ?").run(toolName);
    this.tools.delete(toolName);
    this.predictors.delete(toolName);
    broadcast({ type: 'tool_deactivated', name: toolName });
  }
}

module.exports = { ToolRegistry };
