const { v4: uuidv4 } = require('uuid');
const { getDB } = require('./db');
const { broadcast } = require('./ws');
const { GeminiService } = require('./services/gemini');
const { ClaudeService } = require('./services/claude');
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
   */
  async distillTask(config) {
    const { taskType, description, labels, modelArch, dataCount } = config;
    const distillId = uuidv4();

    this._broadcast('distill_start', { id: distillId, taskType, modelArch });
    this._log(distillId, `🏭 开始蒸馏: ${taskType} → ${modelArch}`);

    // Step A: 使用 Claude 选择/确认模型架构
    this._log(distillId, '🤖 A. 确认模型架构...');
    let archConfig;
    try {
      archConfig = await this.claude.selectModelArch(description, []);
      this._log(distillId, `   推荐: ${archConfig.model_arch} (预估大小: ${archConfig.estimated_size_mb}MB)`);
    } catch {
      archConfig = {
        model_arch: modelArch,
        labels,
        training_config: {
          epochs: parseInt(process.env.TRAIN_EPOCHS) || 5,
          batch_size: 16,
          learning_rate: modelArch === 'fasttext' ? 0.1 : 2e-5,
          max_length: 128,
        },
      };
    }

    const finalArch = archConfig.model_arch || modelArch;
    const finalLabels = archConfig.labels || labels;
    const trainConfig = archConfig.training_config || {};

    // Step B: 使用 Gemini 生成训练数据
    this._log(distillId, `📝 B. 使用 Gemini 生成 ${dataCount} 条训练数据...`);
    const trainingData = await this.gemini.generateTrainingData(description, finalLabels, dataCount);
    this._log(distillId, `   生成了 ${trainingData.length} 条数据`);

    if (trainingData.length < 100) {
      throw new Error(`训练数据不足 (${trainingData.length} < 100)，无法训练可靠模型`);
    }

    // Step C: 使用 Gemini 对数据进行验证标注（质量控制）
    this._log(distillId, '🏷️ C. 验证标注质量...');
    const sampleSize = Math.min(50, Math.floor(trainingData.length * 0.1));
    const sample = trainingData.slice(0, sampleSize);
    const verified = await this.gemini.labelBatch(
      sample.map(d => d.text),
      description,
      finalLabels
    );

    let matchCount = 0;
    for (let i = 0; i < Math.min(verified.length, sample.length); i++) {
      if (verified[i]?.label === sample[i].label) matchCount++;
    }
    const consistency = matchCount / Math.max(verified.length, 1);
    this._log(distillId, `   标注一致性: ${(consistency * 100).toFixed(1)}%`);

    // Step D: 训练模型
    this._log(distillId, `🏃 D. 开始 ${finalArch} 训练...`);
    const trainResult = await this.trainer.train({
      taskType,
      modelArch: finalArch,
      trainingData,
      labels: finalLabels,
      epochs: trainConfig.epochs || parseInt(process.env.TRAIN_EPOCHS) || 5,
      batchSize: trainConfig.batch_size || 16,
      learningRate: trainConfig.learning_rate || 2e-5,
      maxLength: trainConfig.max_length || 128,
      valSplit: parseFloat(process.env.TRAIN_VAL_SPLIT) || 0.2,
    });

    this._log(distillId, `   训练完成! Accuracy: ${trainResult.metrics.accuracy?.toFixed(4)}`);

    // Step E: 检查模型质量
    const minAccuracy = 0.85;
    if (trainResult.metrics.accuracy < minAccuracy) {
      this._log(distillId, `⚠️ 模型准确率 ${trainResult.metrics.accuracy.toFixed(4)} 低于阈值 ${minAccuracy}，放弃此工具`);
      throw new Error(`模型准确率不达标: ${trainResult.metrics.accuracy.toFixed(4)} < ${minAccuracy}`);
    }

    // Step F: 注册为工具
    const toolName = `${taskType}_${finalArch}`;
    this._log(distillId, `🔧 E. 注册工具: ${toolName}`);

    await this.toolRegistry.register({
      name: toolName,
      description: `自动蒸馏的${description}工具 (${finalArch}, acc=${trainResult.metrics.accuracy.toFixed(4)})`,
      taskType,
      modelArch: finalArch,
      modelPath: trainResult.modelPath,
      onnxPath: trainResult.onnxPath,
      accuracy: trainResult.metrics.accuracy,
      config: {
        labels: finalLabels,
        max_length: trainConfig.max_length || 128,
        threshold: parseFloat(process.env.PREDICT_CONFIDENCE_THRESHOLD) || 0.8,
      },
    });

    this._log(distillId, `✅ 蒸馏完成: ${toolName}`);
    this._broadcast('distill_complete', {
      id: distillId,
      toolName,
      accuracy: trainResult.metrics.accuracy,
      modelSize: trainResult.metrics.model_size_mb,
    });

    return {
      toolName,
      metrics: trainResult.metrics,
      modelPath: trainResult.modelPath,
      onnxPath: trainResult.onnxPath,
    };
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
