const fs = require('fs');
const path = require('path');
const { getDB } = require('./db');
const { broadcast } = require('./ws');

class LoopManager {
  constructor() {
    this.loopPath = path.join(__dirname, 'loop.js');
    this.nextPath = path.join(__dirname, 'loop_next.js');
    this.versionsDir = path.join(__dirname, '..', 'data', 'loop-versions');
    this.currentModule = null;
    this.version = '1.0.0';

    fs.mkdirSync(this.versionsDir, { recursive: true });
    this._loadCurrent();
    this._saveInitialVersion();
  }

  _loadCurrent() {
    this._clearCache(this.loopPath);
    try {
      this.currentModule = require(this.loopPath);
      this._loopMtime = fs.statSync(this.loopPath).mtimeMs;
      console.log(`[LoopManager] 已加载 loop.js`);
    } catch (err) {
      console.error(`[LoopManager] 加载 loop.js 失败:`, err.message);
      this.currentModule = { process: this._fallbackProcess };
    }
  }

  _clearCache(filePath) {
    const resolved = require.resolve(filePath);
    delete require.cache[resolved];
  }

  _fallbackProcess(request, context) {
    return context.gemini.chat(request.input).then(output => ({
      output,
      tool_used: 'gemini_fallback',
      confidence: 1.0,
      metadata: { version: 'fallback' },
    }));
  }

  _saveInitialVersion() {
    try {
      const db = getDB();
      const existing = db.prepare('SELECT id FROM loop_versions WHERE version = ?').get('1.0.0');
      if (!existing) {
        const code = fs.readFileSync(this.loopPath, 'utf-8');
        db.prepare('INSERT INTO loop_versions (version, code, reason, active) VALUES (?, ?, ?, 1)')
          .run('1.0.0', code, '初始版本：所有请求直接转发到 Gemini');
        this._saveToFile(code, '1.0.0', '初始版本');
      }
    } catch (err) {
      console.error('[LoopManager] 保存初始版本失败:', err.message);
    }
  }

  _formatTime(date = new Date()) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}_${pad(date.getHours())}-${pad(date.getMinutes())}-${pad(date.getSeconds())}`;
  }

  _saveToFile(code, version, reason) {
    const timeStr = this._formatTime();
    const fileName = `loop_v${version}_${timeStr}.js`;
    const filePath = path.join(this.versionsDir, fileName);
    const header = `// version: ${version}\n// time: ${timeStr}\n// reason: ${reason}\n\n`;
    fs.writeFileSync(filePath, header + code, 'utf-8');
    console.log(`[LoopManager] 版本文件已保存: ${fileName}`);
    return fileName;
  }

  async executeLoop(request, context) {
    try {
      const mtime = fs.statSync(this.loopPath).mtimeMs;
      if (!this.currentModule || !this.currentModule.process || mtime !== this._loopMtime) {
        console.log(`[LoopManager] 检测到 loop.js 变更，重新加载`);
        this._loadCurrent();
      }
    } catch {}
    return this.currentModule.process(request, context);
  }

  /**
   * 双缓冲更新 loop.js
   * 1. 将新代码写入 loop_next.js
   * 2. 验证新代码可加载且有 process 函数
   * 3. 备份当前 loop.js 到版本目录
   * 4. 替换 loop.js
   * 5. 重新加载
   */
  async updateLoop(newCode, reason) {
    const newVersion = this._nextVersion();

    fs.writeFileSync(this.nextPath, newCode, 'utf-8');

    try {
      this._clearCache(this.nextPath);
      const nextModule = require(this.nextPath);

      if (typeof nextModule.process !== 'function') {
        throw new Error('新代码缺少 process 函数导出');
      }

      const testResult = await nextModule.process(
        { input: '__validation_test__', type: 'test' },
        {
          gemini: { chat: async () => '[test ok]' },
          tools: { predict: async () => ({ label: 'test', confidence: 1.0 }) },
          log: () => {},
        }
      );

      if (!testResult || typeof testResult.output === 'undefined') {
        throw new Error('新代码 process 函数返回格式无效');
      }
    } catch (err) {
      fs.unlinkSync(this.nextPath);
      throw new Error(`Loop 验证失败: ${err.message}`);
    }

    this._saveToFile(fs.readFileSync(this.loopPath, 'utf-8'), this.version, `被 v${newVersion} 替换前的备份`);

    fs.copyFileSync(this.nextPath, this.loopPath);
    fs.unlinkSync(this.nextPath);

    const savedFile = this._saveToFile(newCode, newVersion, reason);

    const db = getDB();
    db.prepare('UPDATE loop_versions SET active = 0 WHERE active = 1').run();
    db.prepare('INSERT INTO loop_versions (version, code, reason, active) VALUES (?, ?, ?, 1)')
      .run(newVersion, newCode, reason);

    this.version = newVersion;
    this._loadCurrent();

    broadcast({
      type: 'loop_updated',
      version: newVersion,
      reason,
    });

    console.log(`[LoopManager] Loop 已更新到 v${newVersion}: ${reason}`);
    return { version: newVersion, file: savedFile };
  }

  rollback(versionId) {
    const db = getDB();
    const target = db.prepare('SELECT * FROM loop_versions WHERE id = ?').get(versionId);
    if (!target) throw new Error(`版本 ${versionId} 不存在`);

    this._saveToFile(fs.readFileSync(this.loopPath, 'utf-8'), this.version, `回滚到 v${target.version} 前的备份`);

    fs.writeFileSync(this.loopPath, target.code, 'utf-8');
    db.prepare('UPDATE loop_versions SET active = 0 WHERE active = 1').run();
    db.prepare('UPDATE loop_versions SET active = 1 WHERE id = ?').run(versionId);

    this._saveToFile(target.code, target.version, `回滚恢复`);

    this.version = target.version;
    this._loadCurrent();

    broadcast({ type: 'loop_rollback', version: target.version });
    console.log(`[LoopManager] 已回滚到 v${target.version}`);
    return { version: target.version };
  }

  getCurrentVersion() {
    return this.version;
  }

  getCurrentCode() {
    return fs.readFileSync(this.loopPath, 'utf-8');
  }

  getVersionHistory() {
    const db = getDB();
    return db.prepare('SELECT id, version, reason, active, created_at FROM loop_versions ORDER BY id DESC').all();
  }

  _nextVersion() {
    const parts = this.version.split('.').map(Number);
    parts[2]++;
    if (parts[2] >= 100) { parts[1]++; parts[2] = 0; }
    if (parts[1] >= 100) { parts[0]++; parts[1] = 0; }
    return parts.join('.');
  }
}

module.exports = { LoopManager };
