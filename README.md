# TinyBERT Pipeline — 自迭代 LLM 驱动的 Agent 服务系统

一个能**自我进化**的 AI Agent 服务框架。系统初始状态仅靠大模型（Gemini）处理所有请求，随着运行积累经验或管理员主动指令，自动完成**能力抽取 → 小模型蒸馏 → 工具注册 → 流程重写**的完整闭环，逐步将高频重复任务下沉到毫秒级推理的轻量模型，同时保留大模型兜底能力。

## 核心理念

```
用户请求 → loop.js（元流程）→ 小模型前置分流 → 大模型兜底
                ↑
        系统自动重写（Claude 生成代码，双缓冲热替换）
```

- **loop.js** 是可被 AI 动态重写的核心处理流程，支持热加载
- **小模型工具** 由系统自动训练、注册，插入到 loop.js 中做前置分流
- **大模型** 始终作为最终兜底，处理小模型无法覆盖的长尾请求

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Vue 3 前端 (Vite)                     │
│  📊 概览  💬 对话  🔧 工具  🧬 进化  🔄 流程             │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP + WebSocket
┌────────────────────────┴────────────────────────────────┐
│                   Node.js 服务端                         │
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ loop.js  │  │ LoopManager  │  │   ToolRegistry   │  │
│  │ (元流程)  │  │ (热加载/版本) │  │  (工具注册/推理)  │  │
│  └────┬─────┘  └──────────────┘  └────────┬─────────┘  │
│       │                                    │            │
│  ┌────┴────────────────────────────────────┴────────┐  │
│  │                  Evolver (进化引擎)                │  │
│  │  意图分析 → 选型 → 数据生成 → 训练 → 注册 → 代码生成 │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─────────────── LLM 服务层 ──────────────────────┐   │
│  │  Gemini 3.1 Pro   │ 用户请求处理、数据验证       │   │
│  │  Gemini 2.5 Flash │ 训练数据生成（高并发）        │   │
│  │  Claude Opus 4.6  │ loop.js 代码生成/重写        │   │
│  │  GPT-5.4          │ 意图分析、任务模式+架构选型   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────── 小模型推理层 ─────────────────────────┐   │
│  │  ONNX Runtime  │ Transformer 模型推理 (<50ms)    │   │
│  │  FastText       │ 词袋模型推理 (<1ms)            │   │
│  │  支持: classify / ner / similarity / regression  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─ Python 训练层 ─┐  ┌─ SQLite ─┐  ┌─ WebSocket ─┐   │
│  │ PyTorch + HF    │  │ 持久化    │  │ 实时推送     │   │
│  └─────────────────┘  └──────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 支持的微调任务类型

| 任务模式 | 说明 | 数据格式 | 推理返回值 |
|---------|------|---------|-----------|
| **classify** | 文本分类 | `{text, label}` | `{label, confidence, probabilities}` |
| **ner** | 命名实体识别 | `{text, entities: [{start, end, label}]}` | `{entities: [{label, start, end, text}], entity_count}` |
| **similarity** | 句子对关系 | `{text_a, text_b, label}` | `{label, confidence}` 或 `{score}` |
| **regression** | 文本打分 | `{text, score}` | `{score, confidence}` |

GPT-5.4 会根据进化意图自动判断最合适的任务模式。

## 支持的模型架构

| 架构 | 中文模型 | 大小 | 适用场景 | 支持的任务 |
|------|---------|------|---------|-----------|
| **fasttext** | — | <2MB | 关键词级分类 | classify |
| **tinybert** | hfl/rbt3 (3层) | 15-50MB | 语义分类、意图识别 | 全部 |
| **minilm** | hfl/rbt4 (4层) | 20-80MB | 复杂多分类、NER | 全部 |
| **distilbert** | hfl/rbt6 (6层) | 100MB+ | 长文本、高精度 | 全部 |

## 快速开始

### 环境要求

- **Node.js** >= 18
- **Python** >= 3.10（训练用）
- **npm** >= 9

### 安装

```bash
# 克隆项目
git clone <repo-url> tinybert-pipeline
cd tinybert-pipeline

# 一键安装所有依赖（Node.js + 前端 + Python）
npm run setup

# 或手动分步安装
npm install
cd client && npm install && cd ..
pip install -r training/requirements.txt
```

### 配置

复制环境变量模板并填入 API Key：

```bash
cp .env.example .env
```

编辑 `.env`，至少配置以下项：

```env
# 处理用户请求的大模型
GEMINI_API_BASE=https://openrouter.ai/api/v1
GEMINI_API_KEY=sk-or-v1-your-key
GEMINI_MODEL=google/gemini-3.1-pro-preview

# 训练数据生成（推荐 Flash 系列，速度快成本低）
DATAGEN_API_KEY=sk-or-v1-your-key
DATAGEN_MODEL=google/gemini-2.5-flash

# 代码生成（重写 loop.js）
CLAUDE_API_KEY=sk-or-v1-your-key
CLAUDE_MODEL=anthropic/claude-opus-4.6

# 意图分析 + 模型选型
INTENT_API_KEY=sk-or-v1-your-key
SELECTOR_API_KEY=sk-or-v1-your-key

# Python 路径（Windows 用户需要写完整路径）
PYTHON_PATH=python
```

### 启动

```bash
# 同时启动后端 + 前端开发服务器
npm run dev

# 或分开启动
npm run dev:server   # 后端 → http://localhost:3000
npm run dev:client   # 前端 → http://localhost:5173
```

打开浏览器访问 `http://localhost:5173`。

## 使用方式

### 1. 对话

在「对话」页面直接与 Agent 交流。初始状态所有请求都转发给 Gemini 大模型处理。

### 2. 进化

在「进化」页面用自然语言描述想让系统增加的能力，例如：

- `"在处理最前面加一个意图识别，如果是恶意，则直接返回：请好好说话"`
- `"识别用户输入中的人名和地名，提取出来"`
- `"判断用户的两个问题是否在问同一件事"`
- `"给用户评论打一个 0~1 的情绪分数"`

系统会自动执行完整的进化流水线：

```
用户意图 → GPT-5.4 意图分析
         → GPT-5.4 选择 task_mode + model_arch
         → Flash 生成训练数据（高并发，实时保存）
         → 数据质量验证
         → Python 训练模型（ONNX 导出）
         → 注册为工具
         → Claude 生成新 loop.js（双缓冲替换）
```

### 3. 查看工具

在「工具」页面查看所有已注册的小模型工具，包括任务类型、准确率、模型架构等信息。

### 4. 查看流程

在「流程」页面查看当前 loop.js 的代码和版本历史。每次进化都会保存版本快照。

## 项目结构

```
tinybert-pipeline/
├── server/                    # Node.js 后端
│   ├── index.js               # Express 入口
│   ├── db.js                  # SQLite 数据库初始化
│   ├── ws.js                  # WebSocket 实时推送
│   ├── loop.js                # 元流程（可被 AI 重写）
│   ├── loop-manager.js        # loop.js 热加载、版本管理、双缓冲
│   ├── evolver.js             # 进化引擎（编排完整流水线）
│   ├── routes/
│   │   ├── chat.js            # 对话 API
│   │   ├── admin.js           # 管理 API（进化触发、任务恢复）
│   │   └── config.js          # 配置查询 API
│   └── services/
│       ├── gemini.js          # Gemini 服务（对话 + 数据生成）
│       ├── claude.js          # Claude 服务（代码生成）
│       ├── intent.js          # GPT-5.4 意图分析服务
│       ├── selector.js        # GPT-5.4 模型选型服务
│       ├── trainer.js         # 训练编排（调度 Python 子进程）
│       ├── predictor.js       # 推理引擎（ONNX / FastText）
│       └── tool-registry.js   # 工具注册表
├── training/                  # Python 训练脚本
│   ├── train_transformer.py   # 文本分类训练 + ONNX 导出
│   ├── train_ner.py           # NER 训练 + ONNX 导出
│   ├── train_similarity.py    # 句子对/相似度训练 + ONNX 导出
│   ├── train_regression.py    # 回归/打分训练 + ONNX 导出
│   ├── train_fasttext.py      # FastText 轻量分类器训练
│   └── requirements.txt       # Python 依赖
├── client/                    # Vue 3 前端
│   └── src/
│       ├── App.vue            # 布局 + 侧边栏
│       ├── views/
│       │   ├── Dashboard.vue  # 概览
│       │   ├── Chat.vue       # 对话
│       │   ├── Tools.vue      # 工具管理
│       │   ├── Evolution.vue  # 进化控制台
│       │   └── Loop.vue       # 流程查看
│       ├── stores/app.js      # Pinia 状态管理
│       └── api.js             # API 封装
├── data/                      # 运行时数据（自动创建）
│   ├── models/                # 训练产出的模型文件
│   ├── training-data/         # 生成的训练数据 (JSONL)
│   └── loop-versions/         # loop.js 历史版本快照
├── .env.example               # 环境变量模板
├── package.json
└── README.md
```

## 进化流水线详解

```
                    ┌──────────────────┐
                    │  用户自然语言意图  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
           Step 1   │  GPT-5.4 意图分析 │  判断是否需要训练模型
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
           Step 2A  │  GPT-5.4 智能选型 │  选择 task_mode + model_arch
                    │  (classify/ner/   │
                    │  similarity/      │
                    │  regression)      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
           Step 2B  │  Flash 数据生成   │  高并发、实时保存、可断点续传
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
           Step 2C  │   数据质量验证    │  分类→标注一致性 / NER→实体覆盖率
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
           Step 2D  │   Python 模型训练  │  PyTorch → ONNX 导出
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
           Step 2E  │   注册为工具      │  写入 SQLite + 加载到内存
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
           Step 3   │  Claude 生成代码  │  根据工具 task_mode 生成正确调用
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
           Step 4   │  双缓冲热替换     │  写入 temp → 验证 → 替换 loop.js
                    └────────┬─────────┘
                             │
                        ✅ 完成
```

每一步都会保存状态到数据库，进化失败后可从断点继续。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 发送对话消息 |
| GET | `/api/status` | 系统状态 |
| GET | `/api/config` | 系统配置 |
| GET | `/api/admin/tools` | 工具列表 |
| GET | `/api/admin/loop` | 当前 loop.js 代码 |
| GET | `/api/admin/loop/versions` | loop.js 版本历史 |
| POST | `/api/admin/evolve/intent` | 提交进化意图 |
| GET | `/api/admin/evolve/tasks` | 未完成的进化任务 |
| POST | `/api/admin/evolve/resume/:id` | 恢复进化任务 |

WebSocket 连接到 `ws://localhost:3000`，实时接收：
- `loop_log` — loop.js 执行日志
- `evolve_progress` — 进化进度
- `tool_registered` — 工具注册通知

## 关键设计

### 双缓冲热替换

loop.js 更新流程：写入临时文件 → 语法校验 → require() 验证 → 执行 `__validation_test__` → 通过后原子替换 → 热重载生效。服务全程不中断。

### 断点续传

训练数据生成实时追加保存为 JSONL 文件。进化任务的每一步都持久化到 SQLite。如果中途失败（API 超时、训练 OOM 等），可以从失败步骤恢复继续。

### LLM 分工

| 模型 | 职责 | 为什么选它 |
|------|------|-----------|
| Gemini 3.1 Pro | 处理用户请求 | 多模态、长上下文、性价比高 |
| Gemini 2.5 Flash | 批量生成训练数据 | 速度快、成本极低 |
| Claude Opus 4.6 | 生成/重写 loop.js | 代码质量最高 |
| GPT-5.4 | 意图分析、架构选型 | 推理和结构化输出最强 |

### 最小够用原则

模型选型遵循"能用小的就不用大的"：

```
FastText (<2MB, <1ms) → TinyBERT (15MB, ~20ms) → MiniLM (40MB, ~30ms) → DistilBERT (100MB+, ~50ms)
```

只有当任务确实需要语义理解时才上 Transformer。

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | 3000 | 服务端口 |
| `GEMINI_MODEL` | gemini-3.1-pro-preview | 用户请求处理模型 |
| `DATAGEN_MODEL` | gemini-2.5-flash | 数据生成模型 |
| `DATAGEN_CONCURRENCY` | 20 | 数据生成并发数 |
| `CLAUDE_MODEL` | claude-opus-4.6 | 代码生成模型 |
| `INTENT_MODEL` | gpt-5.4 | 意图分析模型 |
| `SELECTOR_MODEL` | gpt-5.4 | 架构选型模型 |
| `MODEL_CANDIDATES` | fasttext,tinybert,minilm,distilbert | 候选架构列表 |
| `PREDICT_CONFIDENCE_THRESHOLD` | 0.8 | 小模型置信度阈值 |
| `TRAIN_DATA_COUNT` | 5000 | 训练数据生成数量 |
| `TRAIN_EPOCHS` | 5 | 训练轮数 |
| `MODEL_TINYBERT` | hfl/rbt3 | TinyBERT 中文预训练模型 |
| `MODEL_MINILM` | hfl/rbt4 | MiniLM 中文预训练模型 |
| `MODEL_DISTILBERT` | hfl/rbt6 | DistilBERT 中文预训练模型 |

## License

MIT
