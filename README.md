# SoulNPC-Agent

**基于 LangGraph 的多模态认知-情感游戏角色 Agent 系统**

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](Dockerfile)

SoulNPC 是一个轻量级认知-情感游戏角色 Agent 系统。基于 LangGraph 状态图编排引擎，将事件解析、记忆 RAG、情感计算、Tool Calling 与角色化台词生成解耦为独立节点，支持条件分支与循环执行。从数据构建、模型微调（QLoRA，AutoDL 云端）到 Agent Runtime 的完整链路。

---

## 架构概览

```
                    ┌──────────────────────────────┐
                    │      LangGraph StateGraph      │
                    │  (条件分支 + 循环 + 并行执行)   │
                    └──────────────────────────────┘
                                     │
    ┌────────────┬──────────┬────────┼────────┬──────────┐
    ▼            ▼          ▼        ▼        ▼          ▼
┌───────┐  ┌──────────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐
│ Event │  │  Memory  │ │ Emo- │ │ Rela-│ │Tool  │ │ Dialogue │
│Parser │  │   RAG    │ │ tion │ │ tion │ │Call- │ │   Gen    │
│       │  │ (加权召回)│ │ Vect │ │ Vect │ │ ing  │ │(rule/LoRA)│
└───────┘  └──────────┘ │(6维) │ │(6维) │ └──────┘ └──────────┘
                        └──────┘ └──────┘
                            │
                    ┌───────┴───────┐
                    │  Langfuse     │
                    │  全链路追踪    │
                    │  Bad Case 收集 │
                    └───────────────┘
```

### 核心模块

| 模块 | 说明 | 技术栈 |
|------|------|--------|
| **Orchestration** | LangGraph 状态图编排，条件分支与并行执行 | LangGraph, StateGraph |
| **Tool Calling** | NPC 工具调用（情感计算、关系评估、知识检索、台词变体生成） | Function Calling, Tool Schema |
| **Memory RAG** | 分层记忆系统：短期记忆（窗口截断）+ 长期记忆（加权语义 RAG） | TF-IDF, 多维评分 |
| **State Management** | 6维情感向量 + 6维关系向量持续追踪 | CognitiveState, StateStore |
| **Observability** | Langfuse 全链路追踪、Prompt 管理、Bad Case 标注 | Langfuse, Local Tracing |
| **API Server** | FastAPI + SSE 流式输出，生产级部署 | FastAPI, Uvicorn, Docker |

---

## 快速开始

### 本地运行

```bash
# 1. 克隆项目
git clone https://github.com/chn9692619/SoulNPC-NPC-agent-.git
cd SoulNPC-NPC-agent-

# 2. 安装依赖
pip install -r requirements.txt

# 3. Gradio 工作台
python app/main.py
# 访问 http://127.0.0.1:7860

# 4. FastAPI 服务
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
# 访问 http://127.0.0.1:8000/docs
```

### Docker 部署

```bash
docker build -t soulnpc-agent .
docker run -p 8000:8000 -p 7860:7860 soulnpc-agent
```

### API 使用

```bash
# 发送玩家事件
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"player_event": "玩家走近吧台，低声说：来一杯最烈的酒", "thread_id": "session-1"}'

# SSE 流式执行
curl -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"player_event": "玩家拔出剑指向NPC", "thread_id": "session-2"}'

# 健康检查
curl http://localhost:8000/health
```

---

## Tool Calling 示例

```python
from src.tools.npc_tools import invoke_tool

# 计算情感强度
result = invoke_tool("calculate_emotion_intensity", emotion_vector={
    "valence": 0.6, "arousal": 0.8, "dominance": 0.3,
    "safety": -0.2, "stress": 0.5, "curiosity": 0.4
})
# => {"dominant_emotion": "arousal", "emotional_state": "excited", ...}

# 评估关系变化
result = invoke_tool("assess_relationship_change",
    relation_before={"trust": 0.5, "intimacy": 0.3},
    relation_after={"trust": 0.7, "intimacy": 0.4}
)
# => {"most_significant_change": "trust", "overall_stability": "evolving"}
```

---

## AutoDL 云端训练

训练部分通过 AutoDL 云端 GPU 完成，支持：

- **QLoRA 微调**: 基于 Qwen2.5-1.5B-Instruct，4-bit 量化，LoRA Rank 8
- **SFT/DPO 双模式**: 自动化数据管线生成结构化样本
- **Paramiko 远程控制**: SSH 连接、文件上传、训练启动、日志监控

详见 `docs/AUTODL_TRAINING.md`。

---

## 项目结构

```
SoulNPC-Agent/
├── app/                    # Gradio 工作台
│   └── main.py             # 主入口（8个Tab）
├── src/
│   ├── orchestration/      # LangGraph 编排引擎 ✨
│   │   └── langgraph_engine.py
│   ├── tools/              # Tool Calling 系统 ✨
│   │   ├── __init__.py     # 4个NPC工具 + Schema定义
│   │   └── npc_tools.py
│   ├── agent_runtime/      # Agent 推理运行时
│   │   ├── runtime_engine.py
│   │   ├── event_parser.py
│   │   ├── persistent_memory.py
│   │   ├── state_store.py
│   │   └── prompt_builder.py
│   ├── observability/      # Langfuse 可观测性 ✨
│   │   ├── __init__.py     # 追踪器 + Bad Case管理
│   │   └── tracer.py
│   ├── api/                # FastAPI 服务 ✨
│   │   ├── __init__.py     # REST端点 + SSE流式
│   │   └── server.py
│   ├── character/          # 角色系统（认知状态、事件模型）
│   ├── memory/             # 加权记忆存储
│   ├── data_generation/    # 数据工厂（SFT/DPO导出）
│   ├── llm/                # LLM 客户端
│   └── modeling/           # 模型训练/推理
├── scripts/                # 训练脚本（LoRA SFT/DPO）
├── configs/                # 角色配置 + 训练配置
├── docs/                   # 技术文档
├── Dockerfile              # Docker 部署 ✨
└── requirements.txt        # 依赖（含 LangGraph/Langfuse/FastAPI）
```

✨ = v2.1 新增模块

---

## 技术细节

详见：
- `docs/ARCHITECTURE.md` — 整体架构设计
- `docs/AGENT_RUNTIME_TECHNICAL.md` — Agent Runtime 技术细节
- `docs/AUTODL_TRAINING.md` — 云端训练指南
