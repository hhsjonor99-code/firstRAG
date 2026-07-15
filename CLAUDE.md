# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本仓库是 **本地知识库 RAG 系统** 的开发目录。当前状态：项目初始化阶段，仅有需求文档 (`prompt.txt`) 与依赖清单 (`requirements.txt`)，尚未生成任何源代码。所有架构与实现决策需依据 `prompt.txt` 中的明确要求。

## 核心功能要求

依据 `prompt.txt`：

1. **文档上传**：支持 PDF、Word、TXT、Markdown
2. **文档解析**：自动解析并提取关键信息
3. **智能问答**：根据用户问题从文档中检索相关内容
4. **引用来源**：支持定位到具体段落或句子
5. **多轮对话**：支持上下文关联的多轮问答

## 技术栈

| 层级 | 选项 |
|------|------|
| 前端 | HTML + CSS + JavaScript **或** Streamlit |
| 后端 | FastAPI **或** Streamlit |
| 向量数据库 | NumPy、faiss-cpu |
| 框架 | LangChain（langchain / langchain-community / langchain-text-splitters） |
| Embedding | `Qwen/Qwen3-Embedding-4B`（SiliconFlow 接口） |
| 对话模型 | `MiniMax-M3`（MiniMax 接口） |

UI 风格：简洁、直观、用户友好；需包含「本地知识库」「智能问答」「引用来源」「多轮对话」四个界面。

## 外部 API 配置（关键）

### Qwen3 Embedding（SiliconFlow）
- 端点：`https://api.siliconflow.cn/v1/embeddings`
- 模型：`Qwen/Qwen3-Embedding-4B`
- 鉴权：Header `Authorization: Bearer <API_KEY>`

调用示例（来自 `prompt.txt`）：
```python
import requests
response = requests.post(
    "https://api.siliconflow.cn/v1/embeddings",
    headers={"Authorization": "<API_KEY>", "Content-Type": "application/json"},
    json={"input": "Hello, world!", "model": "Qwen/Qwen3-Embedding-4B"},
)
```

### MiniMax 对话模型
- BaseURL：`https://api.minimaxi.com/v1`
- 模型：`MiniMax-M3`（OpenAI 兼容客户端）
- 重要：`extra_body={"reasoning_split": True}` —— 思考内容会分离到 `reasoning_details` 字段，需在解析响应时同时读取 `reasoning_details[0]['text']` 与 `content`。

调用示例（来自 `prompt.txt`）：
```python
from openai import OpenAI
client = OpenAI(base_url="https://api.minimaxi.com/v1", api_key="<API_KEY>")
response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    extra_body={"reasoning_split": True},
)
# response.choices[0].message.reasoning_details[0]['text']  -> 思考
# response.choices[0].message.content                       -> 正文
```

### API Key 约定
用户已声明：**项目开发到核心或重要部分时，需先询问 API Key 再调用模型**。在调用 `Qwen3-Embedding-4B` 或 `MiniMax-M3` 之前，必须先获取 Key；Key 不得硬编码进代码，建议通过 `.env`（`python-dotenv` 已在依赖中）加载。

## 环境与依赖

- Python 环境：conda 环境名 `qc`（依据 `requirements.txt` 注释）
- 安装命令：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
- 关键依赖分组（见 `requirements.txt`）：
  - 服务端：`streamlit`、`fastapi`、`uvicorn[standard]`、`python-multipart`
  - 文档解析：`python-docx`、`pypdf`、`markdown`、`beautifulsoup4`
  - 向量：`numpy`、`faiss-cpu`
  - LangChain：`langchain`、`langchain-community`、`langchain-text-splitters`
  - 模型客户端：`openai`、`requests`
  - 工具：`python-dotenv`、`pydantic`、`pydantic-settings`

## 常见开发命令

> 以下命令基于选定的技术栈组合（FastAPI + Streamlit 任一）—— 在源码生成前为预想命令：

```bash
# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动 FastAPI 后端（典型）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 启动 Streamlit 前端/单进程应用
streamlit run app.py

# 运行测试
pytest
```

具体脚本以代码生成后的 `README` / `pyproject.toml` 为准。

## 架构规划（待实现）

依据 `prompt.txt` 的功能拆解，预期系统模块：

- **文档加载与解析层**：按扩展名分发到 `pypdf` / `python-docx` / `markdown` / `txt` 解析器，统一产出 `Document` 列表（含文本与元数据）。
- **分块（Chunking）层**：基于 LangChain `text_splitters`（如 `RecursiveCharacterTextSplitter`），保留 `chunk_id`、来源文档、起止位置，用于后续引用定位。
- **嵌入与索引层**：调用 Qwen3 Embedding 得到向量，使用 `faiss-cpu` 构建 IndexFlatIP/IVF；同时保留元数据索引以便回查 chunk 原文。
- **检索层**：Query 向量化 → Top-K 相似度检索 → 返回 chunk 文本 + 来源信息。
- **问答与对话层**：拼装 system prompt（角色 + 检索上下文 + 引用格式要求）→ 调用 MiniMax-M3（开启 `reasoning_split`）→ 解析 `reasoning_details` 与 `content`。
- **会话管理层**：维护多轮对话历史（按 session id 隔离），系统需判断「当前问题是否需要调用知识库」（可在 prompt 中加入意图判定）。
- **API 层（FastAPI 路线）**：路由建议 `POST /upload`、`POST /chat`、`GET /citation/{chunk_id}`；请求/响应使用 `pydantic` 模型。
- **前端层**：本地知识库页（展示分块示例）、问答页（输入 + 流式/非流式回答 + 引用列表）。

## 决策点（开发过程中需向用户确认）

`prompt.txt` 明确要求：开发到核心或重要部分时给用户选项以决定项目走向。**不要自行锁定以下决策**，应在动手前与用户确认：

- 前端 / 后端技术组合（Streamlit 单体 vs FastAPI + 前端分离）
- 分块策略与 chunk 大小 / 重叠
- 向量索引类型（Flat / IVF / HNSW）与 Top-K 数值
- 是否启用流式输出（MiniMax 是否支持 stream + reasoning_split）
- 多轮历史的持久化方式（内存 / 文件 / SQLite）

## 数据资产

- `国家基本药物目录（2026年版）(OCR).docx`：作为 RAG 系统的初始示例语料，可用于开发期的解析、分块、检索端到端测试。