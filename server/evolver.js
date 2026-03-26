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

/**
 * Evolver - 自进化引擎
 * 
 * 职责：
 * 1. 分析请求日志，识别高频可蒸馏任务
 * 2. 选择最合适的小模型架构
 * 3. 使用 Gemini 生成/标注训练数据
 * 4. 启动训练，产出小模型
 * 5. 注册为 Agent 工具
 * 6. 使用 Claude Opus 4.6 生成新版 loop.js 代码
 * 7. 双缓冲替换元流程
 */
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

  /**
   * 完整进化流程：分析 → 决策 → 蒸馏 → 集成
   */
  async evolve() {
    if (this.isEvolving) {
      throw new Error('进化流程已在运行中');
    }

    this.isEvolving = true;
    const evolveId = uuidv4();

    try {
      this._broadcast('evolve_start', { id: evolveId });
      this._log(evolveId, '🧬 开始自进化流程');

      // Step 1: 分析请求日志
      this._log(evolveId, '📊 Step 1: 分析请求模式...');
      const analysis = await this._analyzePatterns();

      if (!analysis || !analysis.patterns || analysis.patterns.length === 0) {
        this._log(evolveId, '📊 未发现可蒸馏的任务模式，进化结束');
        this._saveEvolutionLog(evolveId, analysis, null, 'no_patterns', 'skipped');
        return { status: 'skipped', reason: '未发现可蒸馏的任务模式' };
      }

      this._log(evolveId, `📊 发现 ${analysis.patterns.length} 个候选任务模式`);

      // Step 2: 对每个模式执行蒸馏
      const results = [];
      for (const pattern of analysis.patterns) {
        try {
          this._log(evolveId, `\n🔬 处理任务模式: ${pattern.task_type} (${pattern.description})`);
          const result = await this.distillTask({
            taskType: pattern.task_type,
            description: pattern.description,
            labels: pattern.labels,
            modelArch: pattern.recommended_model,
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

      // Step 3: 使用 Claude 生成新 loop.js
      this._log(evolveId, `\n🧠 Step 3: 使用 Claude Opus 4.6 生成新流程代码...`);
      const newTools = results
        .filter(r => r.success)
        .map(r => ({
          name: r.result.toolName,
          description: r.pattern.description,
          model_arch: r.pattern.recommended_model,
          task_type: r.pattern.task_type,
        }));

      await this._updateLoopCode(newTools, evolveId);

      this._log(evolveId, `\n✅ 进化完成! 成功蒸馏 ${successCount}/${analysis.patterns.length} 个任务`);
      this._saveEvolutionLog(evolveId, analysis, results, 'evolve_complete', 'success');

      this._broadcast('evolve_complete', {
        id: evolveId,
        successCount,
        totalPatterns: analysis.patterns.length,
        newVersion: this.loopManager.getCurrentVersion(),
      });

      return { status: 'success', successCount, results };
    } catch (err) {
      this._log(evolveId, `❌ 进化失败: ${err.message}`);
      this._saveEvolutionLog(evolveId, null, null, 'evolve_error', err.message);
      throw err;
    } finally {
      this.isEvolving = false;
    }
  }

  /**
   * 单个任务的蒸馏流程
   * GPT-5.4 选型 → Gemini 生数据 → 训练 → 不达标则 GPT-5.4 换更大架构重试
   */
  async distillTask(config) {
    const { taskType, description, labels, dataCount } = config;
    const distillId = uuidv4();

    this._broadcast('distill_start', { id: distillId, taskType });

    // Step A: GPT-5.4 选择模型架构
    this._log(distillId, '🧠 A. GPT-5.4 选择模型架构...');
    let archConfig;
    try {
      archConfig = await this.selector.selectModelArch(description, []);
      this._log(distillId, `   GPT-5.4 推荐: ${archConfig.model_arch} | 理由: ${archConfig.reason}`);
    } catch (err) {
      this._log(distillId, `   GPT-5.4 选型失败 (${err.message})，使用 tinybert 兜底`);
      archConfig = {
        model_arch: 'tinybert',
        labels,
        training_config: { epochs: 5, batch_size: 16, learning_rate: 2e-5, max_length: 128 },
      };
    }

    const finalLabels = archConfig.labels || labels;
    const trainConfig = archConfig.training_config || {};

    // Step B: Flash 2.5 分批生成训练数据（实时落盘，支持断点续生成）
    const dataDir = path.join(__dirname, '..', 'data', 'training-data');
    fs.mkdirSync(dataDir, { recursive: true });
    const savePath = path.join(dataDir, `${taskType}_${distillId.substring(0, 8)}.jsonl`);

    this._log(distillId, `📝 B. Flash 2.5 生成 ${dataCount} 条训练数据（实时保存: ${path.basename(savePath)}）...`);
    const trainingData = await this.gemini.generateTrainingData(
      description, finalLabels, dataCount,
      (progress) => {
        this._log(distillId, `   [${progress.label}] ${progress.generated}/${progress.target} 条 | 总计 ${progress.totalGenerated}/${progress.totalTarget}`);
      },
      savePath
    );
    this._log(distillId, `   生成完毕: 共 ${trainingData.length} 条（去重后），已保存到 ${path.basename(savePath)}`);

    if (trainingData.length < 100) {
      throw new Error(`训练数据不足 (${trainingData.length} < 100)`);
    }

    // Step C: 验证标注质量
    this._log(distillId, '🏷️ C. 验证标注质量...');
    const sampleSize = Math.min(50, Math.floor(trainingData.length * 0.1));
    const sample = trainingData.slice(0, sampleSize);
    const verified = await this.gemini.labelBatch(
      sample.map(d => d.text), description, finalLabels
    );
    let matchCount = 0;
    for (let i = 0; i < Math.min(verified.length, sample.length); i++) {
      if (verified[i]?.label === sample[i].label) matchCount++;
    }
    this._log(distillId, `   标注一致性: ${(matchCount / Math.max(verified.length, 1) * 100).toFixed(1)}%`);

    // Step D: 训练 → 不达标则让 GPT-5.4 换更大架构重试
    const minAccuracy = 0.85;
    let currentArch = archConfig.model_arch;
    let currentTrainConfig = trainConfig;
    let previousAttempt = null;
    const maxRetries = this.selector.candidates.length;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      this._log(distillId, `🏃 D${attempt > 0 ? ` (第${attempt + 1}次尝试)` : ''}. 训练 ${currentArch}...`);

      const trainResult = await this.trainer.train({
        taskType,
        modelArch: currentArch,
        trainingData,
        labels: finalLabels,
        epochs: currentTrainConfig.epochs || parseInt(process.env.TRAIN_EPOCHS) || 5,
        batchSize: currentTrainConfig.batch_size || 16,
        learningRate: currentTrainConfig.learning_rate || (currentArch === 'fasttext' ? 0.1 : 2e-5),
        maxLength: currentTrainConfig.max_length || 128,
        valSplit: parseFloat(process.env.TRAIN_VAL_SPLIT) || 0.2,
      });

      const accuracy = trainResult.metrics.accuracy;
      this._log(distillId, `   训练完成! Accuracy: ${accuracy?.toFixed(4)}`);

      if (accuracy >= minAccuracy) {
        // 达标，注册工具
        const toolName = `${taskType}_${currentArch}`;
        this._log(distillId, `🔧 E. 注册工具: ${toolName}`);

        await this.toolRegistry.register({
          name: toolName,
          description: `自动蒸馏的${description}工具 (${currentArch}, acc=${accuracy.toFixed(4)})`,
          taskType,
          modelArch: currentArch,
          modelPath: trainResult.modelPath,
          onnxPath: trainResult.onnxPath,
          accuracy,
          config: {
            labels: finalLabels,
            max_length: currentTrainConfig.max_length || 128,
            threshold: parseFloat(process.env.PREDICT_CONFIDENCE_THRESHOLD) || 0.8,
          },
        });

        this._log(distillId, `✅ 蒸馏完成: ${toolName}`);
        this._broadcast('distill_complete', {
          id: distillId, toolName, accuracy,
          modelSize: trainResult.metrics.model_size_mb,
        });

        return { toolName, metrics: trainResult.metrics, modelPath: trainResult.modelPath, onnxPath: trainResult.onnxPath };
      }

      // 不达标，让 GPT-5.4 选更大的架构
      this._log(distillId, `⚠️ ${currentArch} 准确率 ${accuracy.toFixed(4)} 不达标，让 GPT-5.4 重新选型...`);
      previousAttempt = { arch: currentArch, accuracy: accuracy.toFixed(4) };

      try {
        const retryConfig = await this.selector.selectModelArch(description, [], previousAttempt);
        if (retryConfig.model_arch === currentArch) {
          this._log(distillId, `   GPT-5.4 仍推荐 ${currentArch}，终止重试`);
          break;
        }
        currentArch = retryConfig.model_arch;
        currentTrainConfig = retryConfig.training_config || {};
        this._log(distillId, `   GPT-5.4 建议升级到: ${currentArch}`);
      } catch {
        this._log(distillId, `   GPT-5.4 重新选型失败，终止重试`);
        break;
      }
    }

    throw new Error(`所有候选架构均未达到 ${minAccuracy} 准确率`);
  }

  /**
   * 根据用户自然语言意图进化
   * 例如："在处理最前面加一个恶意识别，如果是恶意的，直接返回'请好好说话'"
   */
  async evolveWithIntent(intentText) {
    const intentId = uuidv4();
    this._broadcast('intent_start', { id: intentId, intent: intentText });
    this._log(intentId, `🎯 收到进化意图: "${intentText}"`);

    // Step 1: GPT-5.4 分析意图
    this._log(intentId, '🧠 Step 1: GPT-5.4 分析进化意图...');
    const currentCode = this.loopManager.getCurrentCode();
    const currentTools = this.toolRegistry.list();

    let plan;
    try {
      plan = await this.intent.analyzeIntent(intentText, currentTools, currentCode);
    } catch (err) {
      this._log(intentId, `   GPT-5.4 分析失败: ${err.message}，使用原始意图直接生成代码`);
      plan = {};
    }

    const summary = plan.summary || plan.description || intentText;
    const needsModel = plan.needs_model === true;
    const loopInstruction = plan.loop_instruction || plan.instruction || plan.code_instruction || intentText;

    this._log(intentId, `   分析结果: ${summary}`);
    this._log(intentId, `   需要训练模型: ${needsModel ? '是' : '否'}`);
    if (plan.reason) this._log(intentId, `   理由: ${plan.reason}`);

    // Step 2: 如果需要小模型，先蒸馏
    let newToolName = null;
    if (needsModel && plan.model_task) {
      const mt = plan.model_task;
      this._log(intentId, `\n🏭 Step 2: 蒸馏 "${mt.task_type}" (${mt.labels?.join(', ')})...`);
      try {
        const distillResult = await this.distillTask({
          taskType: mt.task_type,
          description: mt.description,
          labels: mt.labels,
          dataCount: parseInt(process.env.TRAIN_DATA_COUNT) || 5000,
        });
        newToolName = distillResult.toolName;
        this._log(intentId, `   蒸馏完成: ${newToolName} (acc=${distillResult.metrics.accuracy?.toFixed(4)})`);
      } catch (err) {
        this._log(intentId, `   ⚠️ 蒸馏失败: ${err.message}，将尝试纯代码实现`);
      }
    } else {
      this._log(intentId, '\n📝 Step 2: 无需训练模型，跳过蒸馏');
    }

    // Step 3: Claude 根据意图 + 指令生成新 loop.js
    this._log(intentId, '\n🔧 Step 3: Claude 生成新流程代码...');
    const allTools = this.toolRegistry.list();

    const fullInstruction = `用户原始意图：「${intentText}」\n\n具体实现指令：${loopInstruction}`;

    const newCode = await this.claude.generateLoopCodeWithIntent(
      currentCode, allTools, fullInstruction, `用户意图: ${summary}`
    );

    // Step 4: 双缓冲替换
    this._log(intentId, '🔄 Step 4: 双缓冲替换...');
    try {
      const result = await this.loopManager.updateLoop(newCode, `意图进化: ${summary}`);
      this._log(intentId, `✅ Loop 已更新到 v${result.version}`);
    } catch (err) {
      this._log(intentId, `❌ Loop 更新失败: ${err.message}`);
      throw err;
    }

    this._saveEvolutionLog(intentId, { intent: intentText, summary, plan }, null, 'intent_evolve', 'success');
    this._broadcast('intent_complete', { id: intentId, summary, version: this.loopManager.getCurrentVersion() });
    this._log(intentId, `\n✅ 意图进化完成!`);

    return { status: 'success', plan, newToolName, version: this.loopManager.getCurrentVersion() };
  }

  /**
   * 分析请求日志
   */
  async _analyzePatterns() {
    const db = getDB();
    const minRequests = parseInt(process.env.EVOLVE_MIN_REQUESTS) || 50;

    const totalCount = db.prepare('SELECT COUNT(*) as count FROM request_logs').get().count;
    if (totalCount < minRequests) {
      return { patterns: [], summary: `请求数 ${totalCount} 不足 ${minRequests}，暂不分析` };
    }

    const existingTools = this.toolRegistry.list().map(t => t.taskType);
    const logs = db.prepare(`
      SELECT user_input, task_type, tool_used, confidence
      FROM request_logs
      WHERE tool_used = 'gemini_direct' OR tool_used = 'gemini_fallback'
      ORDER BY created_at DESC
      LIMIT 500
    `).all();

    if (logs.length < 20) {
      return { patterns: [], summary: '纯 LLM 请求不足 20 条' };
    }

    const analysis = await this.claude.analyzePatterns(logs);

    if (analysis.patterns) {
      analysis.patterns = analysis.patterns.filter(p => {
        if (existingTools.includes(p.task_type)) {
          return false;
        }
        const threshold = parseFloat(process.env.EVOLVE_PATTERN_THRESHOLD) || 0.3;
        return p.frequency >= threshold;
      });
    }

    return analysis;
  }

  /**
   * 使用 Claude 生成新版 loop.js 并执行双缓冲切换
   */
  async _updateLoopCode(newTools, evolveId) {
    const currentCode = this.loopManager.getCurrentCode();
    const allTools = this.toolRegistry.list();

    const newCode = await this.claude.generateLoopCode(
      currentCode,
      allTools,
      `自动进化: 新增 ${newTools.map(t => t.name).join(', ')} 工具`
    );

    const reason = `自进化 [${evolveId.substring(0, 8)}]: 集成 ${newTools.map(t => t.name).join(', ')}`;

    try {
      const result = await this.loopManager.updateLoop(newCode, reason);
      this._log(evolveId, `🔄 Loop 已更新到 v${result.version}`);
      return result;
    } catch (err) {
      this._log(evolveId, `⚠️ Loop 更新失败: ${err.message}，保持当前版本`);
      throw err;
    }
  }

  _broadcast(type, data) {
    broadcast({ type: `evolver_${type}`, ...data });
  }

  _log(id, message) {
    const timestamp = new Date().toLocaleTimeString('zh-CN');
    const fullMsg = `[${timestamp}] ${message}`;
    console.log(`[Evolver] ${message}`);
    broadcast({ type: 'evolver_log', id, message: fullMsg });
  }

  _saveEvolutionLog(evolveId, analysis, decision, action, result) {
    try {
      const db = getDB();
      db.prepare(`
        INSERT INTO evolution_logs (id, analysis, decision, action, result)
        VALUES (?, ?, ?, ?, ?)
      `).run(
        evolveId,
        analysis ? JSON.stringify(analysis) : null,
        decision ? JSON.stringify(decision) : null,
        action,
        result,
      );
    } catch (err) {
      console.error('[Evolver] 保存日志失败:', err.message);
    }
  }
}

module.exports = { Evolver };
