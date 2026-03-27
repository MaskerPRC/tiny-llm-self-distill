const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');
const { getDB } = require('./db');
const { broadcast } = require('./ws');
const { GeminiService } = require('./services/gemini');
const { ClaudeService } = require('./services/claude');
const { SelectorService } = require('./services/selector');
const { IntentService } = require('./services/intent');
const { Trainer } = require('./services/trainer');

const STEPS = {
  INTENT_ANALYZE: 'intent_analyze',
  INTENT_DISTILL_ARCH: 'intent_distill_arch',
  INTENT_DISTILL_DATAGEN: 'intent_distill_datagen',
  INTENT_DISTILL_VERIFY: 'intent_distill_verify',
  INTENT_DISTILL_TRAIN: 'intent_distill_train',
  INTENT_DISTILL_REGISTER: 'intent_distill_register',
  INTENT_CODEGEN: 'intent_codegen',
  INTENT_SWAP: 'intent_swap',
  COMPLETED: 'completed',
};

class Evolver {
  constructor(loopManager, toolRegistry) {
    this.loopManager = loopManager;
    this.toolRegistry = toolRegistry;
    this.gemini = new GeminiService();
    this.claude = new ClaudeService();
    this.selector = new SelectorService();
    this.intent = new IntentService();
    this.trainer = new Trainer();
    this.isEvolving = false;
  }

  // ========== 状态持久化 ==========

  _saveTask(taskId, updates) {
    const db = getDB();
    const existing = db.prepare('SELECT * FROM evolution_tasks WHERE id = ?').get(taskId);
    if (!existing) {
      db.prepare(`
        INSERT INTO evolution_tasks (id, type, intent_text, status, current_step, state)
        VALUES (?, ?, ?, ?, ?, ?)
      `).run(
        taskId,
        updates.type || 'intent',
        updates.intent_text || null,
        updates.status || 'running',
        updates.current_step || null,
        updates.state ? JSON.stringify(updates.state) : '{}',
      );
    } else {
      const sets = [];
      const vals = [];
      if (updates.status) { sets.push('status = ?'); vals.push(updates.status); }
      if (updates.current_step) { sets.push('current_step = ?'); vals.push(updates.current_step); }
      if (updates.state) {
        const merged = { ...(JSON.parse(existing.state || '{}')), ...updates.state };
        sets.push('state = ?');
        vals.push(JSON.stringify(merged));
      }
      if (updates.error !== undefined) { sets.push('error = ?'); vals.push(updates.error); }
      sets.push("updated_at = datetime('now')");
      vals.push(taskId);
      db.prepare(`UPDATE evolution_tasks SET ${sets.join(', ')} WHERE id = ?`).run(...vals);
    }
  }

  _getTask(taskId) {
    const db = getDB();
    const row = db.prepare('SELECT * FROM evolution_tasks WHERE id = ?').get(taskId);
    if (row && row.state) row.state = JSON.parse(row.state);
    return row;
  }

  _getResumableTasks() {
    const db = getDB();
    const rows = db.prepare(
      "SELECT * FROM evolution_tasks WHERE status IN ('running', 'failed') ORDER BY updated_at DESC"
    ).all();
    return rows.map(r => ({ ...r, state: r.state ? JSON.parse(r.state) : {} }));
  }

  // ========== 意图进化（带状态持久化） ==========

  async evolveWithIntent(intentText, resumeTaskId) {
    if (this.isEvolving) throw new Error('进化流程已在运行中');
    this.isEvolving = true;

    let taskId, task, startStep;

    if (resumeTaskId) {
      task = this._getTask(resumeTaskId);
      if (!task) throw new Error(`任务 ${resumeTaskId} 不存在`);
      taskId = resumeTaskId;
      intentText = task.intent_text;
      startStep = task.current_step;
      this._log(taskId, `🔄 恢复进化任务，从 ${startStep} 继续...`);
      this._saveTask(taskId, { status: 'running', error: null });
    } else {
      taskId = uuidv4();
      startStep = STEPS.INTENT_ANALYZE;
      this._saveTask(taskId, {
        type: 'intent',
        intent_text: intentText,
        status: 'running',
        current_step: STEPS.INTENT_ANALYZE,
        state: {},
      });
    }

    this._broadcast('intent_start', { id: taskId, intent: intentText });
    this._log(taskId, `🎯 进化意图: "${intentText}"`);

    try {
      const state = task?.state || {};

      // Step 1: 分析意图
      if (this._shouldRun(startStep, STEPS.INTENT_ANALYZE)) {
        this._log(taskId, '🧠 Step 1: GPT-5.4 分析进化意图...');
        let plan;
        try {
          plan = await this.intent.analyzeIntent(
            intentText,
            this.toolRegistry.list(),
            this.loopManager.getCurrentCode()
          );
        } catch (err) {
          this._log(taskId, `   GPT-5.4 分析失败: ${err.message}，使用原始意图`);
          plan = {};
        }

        state.plan = plan;
        state.summary = plan.summary || plan.description || intentText;
        state.needsModel = plan.needs_model === true;
        state.loopInstruction = plan.loop_instruction || plan.instruction || plan.code_instruction || intentText;

        this._log(taskId, `   分析结果: ${state.summary}`);
        this._log(taskId, `   需要训练模型: ${state.needsModel ? '是' : '否'}`);

        const nextStep = state.needsModel && plan.model_task
          ? STEPS.INTENT_DISTILL_ARCH
          : STEPS.INTENT_CODEGEN;

        if (state.needsModel && plan.model_task) {
          state.modelTask = plan.model_task;
        }

        this._saveTask(taskId, { current_step: nextStep, state });
        startStep = nextStep;
      }

      // Step 2: 蒸馏（如果需要）
      if (state.needsModel && state.modelTask) {
        const mt = state.modelTask;

        // Step 2A: 选择架构
        if (this._shouldRun(startStep, STEPS.INTENT_DISTILL_ARCH)) {
          this._log(taskId, `\n🏭 Step 2A: GPT-5.4 选择任务模式+模型架构 (${mt.task_type})...`);
          let archConfig;
          try {
            archConfig = await this.selector.selectModelArch(mt.description, []);
            this._log(taskId, `   推荐: ${archConfig.task_mode || 'classify'} + ${archConfig.model_arch} | ${archConfig.reason}`);
          } catch (err) {
            this._log(taskId, `   选型失败，使用 classify + tinybert 兜底`);
            archConfig = {
              task_mode: 'classify',
              model_arch: 'tinybert',
              labels: mt.labels,
              training_config: { epochs: 5, batch_size: 16, learning_rate: 2e-5, max_length: 128 },
            };
          }

          state.taskMode = archConfig.task_mode || mt.task_mode || 'classify';
          state.archConfig = archConfig;
          state.finalLabels = archConfig.labels || mt.labels;
          state.trainConfig = archConfig.training_config || {};
          state.distillId = uuidv4().substring(0, 8);

          const dataDir = path.join(__dirname, '..', 'data', 'training-data');
          fs.mkdirSync(dataDir, { recursive: true });
          state.savePath = path.join(dataDir, `${mt.task_type}_${state.distillId}.jsonl`);

          this._saveTask(taskId, { current_step: STEPS.INTENT_DISTILL_DATAGEN, state });
          startStep = STEPS.INTENT_DISTILL_DATAGEN;
        }

        // Step 2B: 生成训练数据
        if (this._shouldRun(startStep, STEPS.INTENT_DISTILL_DATAGEN)) {
          const dataCount = parseInt(process.env.TRAIN_DATA_COUNT) || 5000;
          const taskMode = state.taskMode || 'classify';
          this._log(taskId, `\n📝 Step 2B: 生成 ${dataCount} 条 ${taskMode} 训练数据（${path.basename(state.savePath)}）...`);

          const trainingData = await this.gemini.generateTrainingDataByMode(
            taskMode, mt.description, state.finalLabels, dataCount,
            (progress) => {
              this._log(taskId, `   [${progress.label}] ${progress.generated}/${progress.target} | 总计 ${progress.totalGenerated}/${progress.totalTarget}`);
            },
            state.savePath,
          );

          state.trainingDataCount = trainingData.length;
          this._log(taskId, `   生成完毕: ${trainingData.length} 条`);

          if (trainingData.length < 100) {
            this._log(taskId, `   ⚠️ 数据不足 (${trainingData.length} < 100)，跳过蒸馏`);
            state.needsModel = false;
            this._saveTask(taskId, { current_step: STEPS.INTENT_CODEGEN, state });
            startStep = STEPS.INTENT_CODEGEN;
          } else {
            this._saveTask(taskId, { current_step: STEPS.INTENT_DISTILL_VERIFY, state });
            startStep = STEPS.INTENT_DISTILL_VERIFY;
          }
        }

        // Step 2C: 验证标注质量
        if (state.needsModel && this._shouldRun(startStep, STEPS.INTENT_DISTILL_VERIFY)) {
          const taskMode = state.taskMode || 'classify';
          this._log(taskId, `\n🏷️ Step 2C: 验证数据质量 (${taskMode})...`);
          const trainingData = this._loadTrainingData(state.savePath);

          if (taskMode === 'classify') {
            const sampleSize = Math.min(50, Math.floor(trainingData.length * 0.1));
            const sample = trainingData.slice(0, sampleSize);
            try {
              const verified = await this.gemini.labelBatch(
                sample.map(d => d.text), mt.description, state.finalLabels
              );
              let matchCount = 0;
              for (let i = 0; i < Math.min(verified.length, sample.length); i++) {
                if (verified[i]?.label === sample[i].label) matchCount++;
              }
              this._log(taskId, `   标注一致性: ${(matchCount / Math.max(verified.length, 1) * 100).toFixed(1)}%`);
            } catch (err) {
              this._log(taskId, `   标注验证失败 (${err.message})，跳过`);
            }
          } else if (taskMode === 'ner') {
            const withEntities = trainingData.filter(d => d.entities?.length > 0).length;
            this._log(taskId, `   NER: ${withEntities}/${trainingData.length} 条含实体标注`);
          } else if (taskMode === 'similarity') {
            const valid = trainingData.filter(d => d.text_a && d.text_b).length;
            this._log(taskId, `   Similarity: ${valid}/${trainingData.length} 条有效句子对`);
          } else if (taskMode === 'regression') {
            const scores = trainingData.filter(d => typeof d.score === 'number').map(d => d.score);
            if (scores.length > 0) {
              const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
              this._log(taskId, `   Regression: ${scores.length} 条, 均分 ${avg.toFixed(3)}, 范围 [${Math.min(...scores).toFixed(2)}, ${Math.max(...scores).toFixed(2)}]`);
            }
          }

          this._saveTask(taskId, { current_step: STEPS.INTENT_DISTILL_TRAIN, state });
          startStep = STEPS.INTENT_DISTILL_TRAIN;
        }

        // Step 2D: 训练模型
        if (state.needsModel && this._shouldRun(startStep, STEPS.INTENT_DISTILL_TRAIN)) {
          const trainingData = this._loadTrainingData(state.savePath);
          const currentArch = state.archConfig.model_arch;
          const tc = state.trainConfig;

          const taskMode = state.taskMode || 'classify';
          this._log(taskId, `\n🏃 Step 2D: 训练 ${currentArch} (${taskMode})...`);

          const trainResult = await this.trainer.train({
            taskType: mt.task_type,
            taskMode,
            modelArch: currentArch,
            trainingData,
            labels: state.finalLabels,
            epochs: tc.epochs || parseInt(process.env.TRAIN_EPOCHS) || 5,
            batchSize: tc.batch_size || 16,
            learningRate: tc.learning_rate || (currentArch === 'fasttext' ? 0.1 : 2e-5),
            maxLength: tc.max_length || 128,
            valSplit: parseFloat(process.env.TRAIN_VAL_SPLIT) || 0.2,
          });

          state.trainResult = {
            accuracy: trainResult.metrics.accuracy,
            modelPath: trainResult.modelPath,
            onnxPath: trainResult.onnxPath,
            metrics: trainResult.metrics,
          };
          this._log(taskId, `   训练完成! Accuracy: ${trainResult.metrics.accuracy?.toFixed(4)}`);

          this._saveTask(taskId, { current_step: STEPS.INTENT_DISTILL_REGISTER, state });
          startStep = STEPS.INTENT_DISTILL_REGISTER;
        }

        // Step 2E: 注册工具
        if (state.needsModel && this._shouldRun(startStep, STEPS.INTENT_DISTILL_REGISTER)) {
          const toolName = `${mt.task_type}_${state.archConfig.model_arch}`;
          this._log(taskId, `\n🔧 Step 2E: 注册工具 ${toolName}...`);

          const taskMode = state.taskMode || 'classify';
          await this.toolRegistry.register({
            name: toolName,
            description: `${mt.description} (${taskMode}/${state.archConfig.model_arch}, acc=${state.trainResult.accuracy?.toFixed(4)})`,
            taskType: mt.task_type,
            modelArch: state.archConfig.model_arch,
            modelPath: state.trainResult.modelPath,
            onnxPath: state.trainResult.onnxPath,
            accuracy: state.trainResult.accuracy,
            config: {
              task_mode: taskMode,
              labels: state.finalLabels,
              max_length: (state.trainConfig || {}).max_length || 128,
              threshold: parseFloat(process.env.PREDICT_CONFIDENCE_THRESHOLD) || 0.8,
            },
          });

          state.newToolName = toolName;
          this._log(taskId, `   工具已注册: ${toolName}`);

          this._saveTask(taskId, { current_step: STEPS.INTENT_CODEGEN, state });
          startStep = STEPS.INTENT_CODEGEN;
        }
      } else if (this._shouldRun(startStep, STEPS.INTENT_CODEGEN)) {
        this._log(taskId, '\n📝 Step 2: 无需训练模型，跳过蒸馏');
      }

      // Step 3: Claude 生成新 loop.js
      if (this._shouldRun(startStep, STEPS.INTENT_CODEGEN)) {
        this._log(taskId, '\n🔧 Step 3: Claude 生成新流程代码...');
        const allTools = this.toolRegistry.list();
        const fullInstruction = `用户原始意图：「${intentText}」\n\n具体实现指令：${state.loopInstruction}`;

        const newCode = await this.claude.generateLoopCodeWithIntent(
          this.loopManager.getCurrentCode(),
          allTools,
          fullInstruction,
          `用户意图: ${state.summary}`
        );

        state.newCode = newCode;
        this._saveTask(taskId, { current_step: STEPS.INTENT_SWAP, state });
        startStep = STEPS.INTENT_SWAP;
      }

      // Step 4: 双缓冲替换
      if (this._shouldRun(startStep, STEPS.INTENT_SWAP)) {
        this._log(taskId, '\n🔄 Step 4: 双缓冲替换...');
        const result = await this.loopManager.updateLoop(state.newCode, `意图进化: ${state.summary}`);
        this._log(taskId, `✅ Loop 已更新到 v${result.version}`);
      }

      // 完成
      this._saveTask(taskId, { status: 'completed', current_step: STEPS.COMPLETED });
      this._saveEvolutionLog(taskId, { intent: intentText, summary: state.summary, plan: state.plan }, null, 'intent_evolve', 'success');
      this._broadcast('intent_complete', { id: taskId, summary: state.summary, version: this.loopManager.getCurrentVersion() });
      this._log(taskId, '\n✅ 意图进化完成!');

      return { status: 'success', taskId, plan: state.plan, newToolName: state.newToolName, version: this.loopManager.getCurrentVersion() };

    } catch (err) {
      this._log(taskId, `❌ 进化失败: ${err.message}`);
      this._saveTask(taskId, { status: 'failed', error: err.message });
      throw err;
    } finally {
      this.isEvolving = false;
    }
  }

  // ========== 自动进化 ==========

  async evolve() {
    if (this.isEvolving) throw new Error('进化流程已在运行中');
    this.isEvolving = true;
    const evolveId = uuidv4();

    try {
      this._broadcast('evolve_start', { id: evolveId });
      this._log(evolveId, '🧬 开始自进化流程');

      this._log(evolveId, '📊 Step 1: 分析请求模式...');
      const analysis = await this._analyzePatterns();

      if (!analysis || !analysis.patterns || analysis.patterns.length === 0) {
        this._log(evolveId, '📊 未发现可蒸馏的任务模式，进化结束');
        this._saveEvolutionLog(evolveId, analysis, null, 'no_patterns', 'skipped');
        return { status: 'skipped', reason: '未发现可蒸馏的任务模式' };
      }

      this._log(evolveId, `📊 发现 ${analysis.patterns.length} 个候选任务模式`);

      const results = [];
      for (const pattern of analysis.patterns) {
        try {
          this._log(evolveId, `\n🔬 处理任务模式: ${pattern.task_type} (${pattern.description})`);
          const result = await this.distillTask({
            taskType: pattern.task_type,
            description: pattern.description,
            labels: pattern.labels,
            dataCount: parseInt(process.env.TRAIN_DATA_COUNT) || 5000,
          });
          results.push({ pattern, result, success: true });
        } catch (err) {
          this._log(evolveId, `⚠️ 蒸馏 "${pattern.task_type}" 失败: ${err.message}`);
          results.push({ pattern, error: err.message, success: false });
        }
      }

      const successCount = results.filter(r => r.success).length;
      if (successCount === 0) {
        this._log(evolveId, '❌ 所有蒸馏任务均失败');
        this._saveEvolutionLog(evolveId, analysis, results, 'distill_all_failed', 'failed');
        return { status: 'failed', reason: '所有蒸馏任务均失败' };
      }

      this._log(evolveId, `\n🧠 Step 3: 使用 Claude Opus 4.6 生成新流程代码...`);
      const newTools = results.filter(r => r.success).map(r => ({
        name: r.result.toolName,
        description: r.pattern.description,
        model_arch: r.pattern.recommended_model,
        task_type: r.pattern.task_type,
      }));
      await this._updateLoopCode(newTools, evolveId);

      this._log(evolveId, `\n✅ 进化完成! 成功蒸馏 ${successCount}/${analysis.patterns.length} 个任务`);
      this._saveEvolutionLog(evolveId, analysis, results, 'evolve_complete', 'success');
      this._broadcast('evolve_complete', { id: evolveId, successCount, totalPatterns: analysis.patterns.length, newVersion: this.loopManager.getCurrentVersion() });

      return { status: 'success', successCount, results };
    } catch (err) {
      this._log(evolveId, `❌ 进化失败: ${err.message}`);
      this._saveEvolutionLog(evolveId, null, null, 'evolve_error', err.message);
      throw err;
    } finally {
      this.isEvolving = false;
    }
  }

  // ========== 单任务蒸馏（由自动进化调用） ==========

  async distillTask(config) {
    const { taskType, description, labels, dataCount } = config;
    const distillId = uuidv4();
    this._broadcast('distill_start', { id: distillId, taskType });

    this._log(distillId, '🧠 A. GPT-5.4 选择模型架构...');
    let archConfig;
    try {
      archConfig = await this.selector.selectModelArch(description, []);
      this._log(distillId, `   GPT-5.4 推荐: ${archConfig.model_arch} | 理由: ${archConfig.reason}`);
    } catch (err) {
      this._log(distillId, `   GPT-5.4 选型失败 (${err.message})，使用 tinybert 兜底`);
      archConfig = { model_arch: 'tinybert', labels, training_config: { epochs: 5, batch_size: 16, learning_rate: 2e-5, max_length: 128 } };
    }

    const finalLabels = archConfig.labels || labels;
    const trainConfig = archConfig.training_config || {};

    const dataDir = path.join(__dirname, '..', 'data', 'training-data');
    fs.mkdirSync(dataDir, { recursive: true });
    const savePath = path.join(dataDir, `${taskType}_${distillId.substring(0, 8)}.jsonl`);

    this._log(distillId, `📝 B. Flash 2.5 生成 ${dataCount} 条训练数据...`);
    const trainingData = await this.gemini.generateTrainingData(
      description, finalLabels, dataCount,
      (progress) => { this._log(distillId, `   [${progress.label}] ${progress.generated}/${progress.target} | 总计 ${progress.totalGenerated}/${progress.totalTarget}`); },
      savePath,
    );
    this._log(distillId, `   生成完毕: ${trainingData.length} 条`);

    if (trainingData.length < 100) throw new Error(`训练数据不足 (${trainingData.length} < 100)`);

    this._log(distillId, '🏷️ C. 验证标注质量...');
    const sampleSize = Math.min(50, Math.floor(trainingData.length * 0.1));
    const sample = trainingData.slice(0, sampleSize);
    try {
      const verified = await this.gemini.labelBatch(sample.map(d => d.text), description, finalLabels);
      let matchCount = 0;
      for (let i = 0; i < Math.min(verified.length, sample.length); i++) {
        if (verified[i]?.label === sample[i].label) matchCount++;
      }
      this._log(distillId, `   标注一致性: ${(matchCount / Math.max(verified.length, 1) * 100).toFixed(1)}%`);
    } catch (err) {
      this._log(distillId, `   标注验证失败 (${err.message})，跳过`);
    }

    const minAccuracy = 0.85;
    let currentArch = archConfig.model_arch;
    let currentTrainConfig = trainConfig;
    let previousAttempt = null;
    const maxRetries = this.selector.candidates.length;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      this._log(distillId, `🏃 D${attempt > 0 ? ` (第${attempt + 1}次)` : ''}. 训练 ${currentArch}...`);
      const trainResult = await this.trainer.train({
        taskType, modelArch: currentArch, trainingData, labels: finalLabels,
        epochs: currentTrainConfig.epochs || parseInt(process.env.TRAIN_EPOCHS) || 5,
        batchSize: currentTrainConfig.batch_size || 16,
        learningRate: currentTrainConfig.learning_rate || (currentArch === 'fasttext' ? 0.1 : 2e-5),
        maxLength: currentTrainConfig.max_length || 128,
        valSplit: parseFloat(process.env.TRAIN_VAL_SPLIT) || 0.2,
      });

      const accuracy = trainResult.metrics.accuracy;
      this._log(distillId, `   训练完成! Accuracy: ${accuracy?.toFixed(4)}`);

      if (accuracy >= minAccuracy) {
        const toolName = `${taskType}_${currentArch}`;
        this._log(distillId, `🔧 E. 注册工具: ${toolName}`);
        await this.toolRegistry.register({
          name: toolName,
          description: `自动蒸馏的${description}工具 (${currentArch}, acc=${accuracy.toFixed(4)})`,
          taskType, modelArch: currentArch,
          modelPath: trainResult.modelPath, onnxPath: trainResult.onnxPath, accuracy,
          config: { labels: finalLabels, max_length: currentTrainConfig.max_length || 128, threshold: parseFloat(process.env.PREDICT_CONFIDENCE_THRESHOLD) || 0.8 },
        });
        this._log(distillId, `✅ 蒸馏完成: ${toolName}`);
        this._broadcast('distill_complete', { id: distillId, toolName, accuracy, modelSize: trainResult.metrics.model_size_mb });
        return { toolName, metrics: trainResult.metrics, modelPath: trainResult.modelPath, onnxPath: trainResult.onnxPath };
      }

      this._log(distillId, `⚠️ ${currentArch} 准确率 ${accuracy.toFixed(4)} 不达标`);
      previousAttempt = { arch: currentArch, accuracy: accuracy.toFixed(4) };
      try {
        const retryConfig = await this.selector.selectModelArch(description, [], previousAttempt);
        if (retryConfig.model_arch === currentArch) break;
        currentArch = retryConfig.model_arch;
        currentTrainConfig = retryConfig.training_config || {};
        this._log(distillId, `   升级到: ${currentArch}`);
      } catch { break; }
    }

    throw new Error(`所有候选架构均未达到 ${minAccuracy} 准确率`);
  }

  // ========== 辅助方法 ==========

  _shouldRun(startStep, targetStep) {
    const order = Object.values(STEPS);
    return order.indexOf(startStep) <= order.indexOf(targetStep);
  }

  _loadTrainingData(savePath) {
    if (!fs.existsSync(savePath)) return [];
    const lines = fs.readFileSync(savePath, 'utf-8').trim().split('\n').filter(Boolean);
    const data = [];
    for (const line of lines) {
      try { data.push(JSON.parse(line)); } catch {}
    }
    return data;
  }

  async _analyzePatterns() {
    const db = getDB();
    const minRequests = parseInt(process.env.EVOLVE_MIN_REQUESTS) || 50;
    const totalCount = db.prepare('SELECT COUNT(*) as count FROM request_logs').get().count;
    if (totalCount < minRequests) return { patterns: [], summary: `请求数 ${totalCount} 不足 ${minRequests}` };

    const existingTools = this.toolRegistry.list().map(t => t.taskType);
    const logs = db.prepare(`SELECT user_input, task_type, tool_used, confidence FROM request_logs WHERE tool_used = 'gemini_direct' OR tool_used = 'gemini_fallback' ORDER BY created_at DESC LIMIT 500`).all();
    if (logs.length < 20) return { patterns: [], summary: '纯 LLM 请求不足 20 条' };

    const analysis = await this.claude.analyzePatterns(logs);
    if (analysis.patterns) {
      const threshold = parseFloat(process.env.EVOLVE_PATTERN_THRESHOLD) || 0.3;
      analysis.patterns = analysis.patterns.filter(p => !existingTools.includes(p.task_type) && p.frequency >= threshold);
    }
    return analysis;
  }

  async _updateLoopCode(newTools, evolveId) {
    const newCode = await this.claude.generateLoopCode(
      this.loopManager.getCurrentCode(),
      this.toolRegistry.list(),
      `自动进化: 新增 ${newTools.map(t => t.name).join(', ')} 工具`
    );
    const reason = `自进化 [${evolveId.substring(0, 8)}]: 集成 ${newTools.map(t => t.name).join(', ')}`;
    const result = await this.loopManager.updateLoop(newCode, reason);
    this._log(evolveId, `🔄 Loop 已更新到 v${result.version}`);
    return result;
  }

  _broadcast(type, data) { broadcast({ type: `evolver_${type}`, ...data }); }

  _log(id, message) {
    const timestamp = new Date().toLocaleTimeString('zh-CN');
    console.log(`[Evolver] ${message}`);
    broadcast({ type: 'evolver_log', id, message: `[${timestamp}] ${message}` });
  }

  _saveEvolutionLog(evolveId, analysis, decision, action, result) {
    try {
      getDB().prepare('INSERT INTO evolution_logs (id, analysis, decision, action, result) VALUES (?, ?, ?, ?, ?)').run(
        evolveId,
        analysis ? JSON.stringify(analysis) : null,
        decision ? JSON.stringify(decision) : null,
        action, result,
      );
    } catch (err) { console.error('[Evolver] 保存日志失败:', err.message); }
  }
}

module.exports = { Evolver };
