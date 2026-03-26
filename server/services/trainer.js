const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const { getDB } = require('../db');
const { broadcast } = require('../ws');

class Trainer {
  constructor() {
    this.pythonPath = process.env.PYTHON_PATH || 'python';
    this.trainingDir = path.join(__dirname, '..', '..', 'training');
    this.dataDir = path.join(__dirname, '..', '..', 'data');
    this.activeJobs = new Map();
  }

  /**
   * 启动训练任务
   * @param {object} config - 训练配置
   * @param {string} config.taskType - 任务类型
   * @param {string} config.modelArch - 模型架构 (tinybert|minilm|fasttext|distilbert)
   * @param {Array} config.trainingData - [{text, label}]
   * @param {Array} config.labels - 标签列表
   * @param {number} config.epochs - 训练轮数
   * @param {number} config.batchSize - 批大小
   * @param {number} config.learningRate - 学习率
   * @param {number} config.maxLength - 最大序列长度
   * @param {string} config.toolId - 关联的工具ID
   */
  async train(config) {
    const jobId = uuidv4();
    const db = getDB();

    db.prepare(`
      INSERT INTO training_jobs (id, tool_id, task_type, model_arch, status, data_count, epochs)
      VALUES (?, ?, ?, ?, 'preparing', ?, ?)
    `).run(jobId, config.toolId || null, config.taskType, config.modelArch, config.trainingData.length, config.epochs);

    const jobDir = path.join(this.dataDir, 'models', jobId);
    fs.mkdirSync(jobDir, { recursive: true });

    try {
      this._appendLog(jobId, `准备训练数据: ${config.trainingData.length} 条`);
      await this._prepareData(jobId, jobDir, config);

      this._appendLog(jobId, `启动 ${config.modelArch} 训练 (epochs: ${config.epochs})`);
      this._updateStatus(jobId, 'training');

      const result = await this._runTraining(jobId, jobDir, config);

      this._updateStatus(jobId, 'completed');
      db.prepare('UPDATE training_jobs SET metrics = ?, updated_at = datetime(\'now\') WHERE id = ?')
        .run(JSON.stringify(result.metrics), jobId);

      this._appendLog(jobId, `训练完成! Accuracy: ${result.metrics.accuracy?.toFixed(4)}`);

      return {
        jobId,
        modelPath: result.modelPath,
        onnxPath: result.onnxPath,
        metrics: result.metrics,
        labelsPath: path.join(jobDir, 'labels.json'),
      };
    } catch (err) {
      this._updateStatus(jobId, 'failed');
      db.prepare("UPDATE training_jobs SET error = ?, updated_at = datetime('now') WHERE id = ?")
        .run(err.message, jobId);
      this._appendLog(jobId, `训练失败: ${err.message}`);
      throw err;
    }
  }

  async _prepareData(jobId, jobDir, config) {
    const dataPath = path.join(jobDir, 'train_data.json');
    fs.writeFileSync(dataPath, JSON.stringify(config.trainingData, null, 2));

    const labelsPath = path.join(jobDir, 'labels.json');
    fs.writeFileSync(labelsPath, JSON.stringify(config.labels));

    const db = getDB();
    const insertStmt = db.prepare(
      'INSERT INTO training_data (id, job_id, input_text, label, source) VALUES (?, ?, ?, ?, ?)'
    );
    db.transaction(() => {
      for (const item of config.trainingData) {
        insertStmt.run(uuidv4(), jobId, item.text, item.label, 'gemini');
      }
    })();
  }

  async _runTraining(jobId, jobDir, config) {
    const scriptMap = {
      tinybert: 'train_transformer.py',
      minilm: 'train_transformer.py',
      distilbert: 'train_transformer.py',
      fasttext: 'train_fasttext.py',
    };

    const script = path.join(this.trainingDir, scriptMap[config.modelArch] || 'train_transformer.py');

    const modelNameMap = {
      tinybert: 'huawei-noah/TinyBERT_General_4L_312D',
      minilm: 'microsoft/MiniLM-L6-H384-uncased',
      distilbert: 'distilbert-base-uncased',
    };

    const args = [
      script,
      '--data_path', path.join(jobDir, 'train_data.json'),
      '--output_dir', jobDir,
      '--model_name', modelNameMap[config.modelArch] || config.modelArch,
      '--num_labels', String(config.labels.length),
      '--epochs', String(config.epochs || 5),
      '--batch_size', String(config.batchSize || 16),
      '--learning_rate', String(config.learningRate || 2e-5),
      '--max_length', String(config.maxLength || 128),
      '--val_split', String(config.valSplit || 0.2),
    ];

    return new Promise((resolve, reject) => {
      const proc = spawn(this.pythonPath, args, {
        cwd: this.trainingDir,
        env: { ...process.env },
      });

      this.activeJobs.set(jobId, proc);
      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => {
        const line = data.toString().trim();
        stdout += line + '\n';

        if (line.startsWith('METRICS_JSON:')) {
          // 解析指标
        } else if (line.startsWith('MODEL_PATH:') || line.startsWith('ONNX_PATH:')) {
          // 解析路径
        } else {
          this._appendLog(jobId, line);
        }

        broadcast({ type: 'training_log', jobId, message: line });
      });

      proc.stderr.on('data', (data) => {
        const line = data.toString().trim();
        stderr += line + '\n';
        if (line && !line.startsWith('Some weights') && !line.includes('FutureWarning')) {
          broadcast({ type: 'training_log', jobId, message: `[stderr] ${line}` });
        }
      });

      proc.on('close', (code) => {
        this.activeJobs.delete(jobId);

        if (code !== 0) {
          return reject(new Error(`训练进程退出码 ${code}\n${stderr.slice(-500)}`));
        }

        let metrics = {};
        let modelPath = null;
        let onnxPath = null;

        const metricsMatch = stdout.match(/METRICS_JSON:(.+)/);
        if (metricsMatch) {
          try { metrics = JSON.parse(metricsMatch[1]); } catch {}
        }

        const modelMatch = stdout.match(/MODEL_PATH:(.+)/);
        if (modelMatch) modelPath = modelMatch[1].trim();

        const onnxMatch = stdout.match(/ONNX_PATH:(.+)/);
        if (onnxMatch) onnxPath = onnxMatch[1].trim();

        if (!modelPath) {
          const possiblePaths = [
            path.join(jobDir, 'model.onnx'),
            path.join(jobDir, 'pytorch_model.bin'),
            path.join(jobDir, 'model.bin'),
          ];
          modelPath = possiblePaths.find(p => fs.existsSync(p));
        }

        if (!onnxPath) {
          const onnxFile = path.join(jobDir, 'model.onnx');
          if (fs.existsSync(onnxFile)) onnxPath = onnxFile;
        }

        resolve({ metrics, modelPath, onnxPath });
      });

      proc.on('error', (err) => {
        this.activeJobs.delete(jobId);
        reject(new Error(`启动训练进程失败: ${err.message}`));
      });
    });
  }

  _updateStatus(jobId, status) {
    const db = getDB();
    db.prepare("UPDATE training_jobs SET status = ?, updated_at = datetime('now') WHERE id = ?")
      .run(status, jobId);
    broadcast({ type: 'training_status', jobId, status });
  }

  _appendLog(jobId, message) {
    const db = getDB();
    const job = db.prepare('SELECT log FROM training_jobs WHERE id = ?').get(jobId);
    const timestamp = new Date().toLocaleTimeString('zh-CN');
    const newLog = (job?.log || '') + `[${timestamp}] ${message}\n`;
    db.prepare('UPDATE training_jobs SET log = ? WHERE id = ?').run(newLog, jobId);
  }

  abort(jobId) {
    const proc = this.activeJobs.get(jobId);
    if (proc) {
      proc.kill('SIGTERM');
      this.activeJobs.delete(jobId);
      this._updateStatus(jobId, 'aborted');
    }
  }
}

module.exports = { Trainer };
