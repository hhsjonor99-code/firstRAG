# firstRAG

本地文档知识库 RAG 系统。第一版基于 Streamlit 单体应用，支持 PDF / DOCX / TXT / Markdown 文档的上传、解析、Embedding、检索与多轮问答。

完整开发计划见 [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)。

## 环境要求

- 操作系统：Windows / macOS / Linux
- Python：3.11
- Conda 环境（推荐）：`py311`

```bash
conda activate py311
```

## 安装

### 1. 运行时核心依赖（第一版必需）

```bash
python -m pip install -r requirements.txt
```

### 2. 开发与测试依赖（可选）

```bash
python -m pip install -r requirements-dev.txt
```

### 3. API 依赖（第二阶段启用 FastAPI 时安装，**第一版不要安装**）

```bash
python -m pip install -r requirements-api.txt
```

### 国内网络环境（可选镜像）

如下载速度较慢，可临时使用镜像（**不推荐作为项目要求固化**）：

```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 配置

复制 `.env.example` 为 `.env` 并填入 API Key：

```bash
cp .env.example .env
```

需要的环境变量：

- `SILICONFLOW_API_KEY` —— SiliconFlow Embedding API Key
- `MINIMAX_API_KEY` —— MiniMax LLM API Key

> 注意：缺少 API Key 时，文档解析、分块与本地索引功能仍可使用；Embedding 与 LLM 调用会在使用时给出明确错误提示。

## 运行

第一版启动 Streamlit：

```bash
streamlit run app.py
```

> 阶段 1 尚未实现业务逻辑，`app.py` 将在阶段 8 创建。

## 测试

```bash
python -m pytest tests/ -v
```

## 环境自检

```bash
python scripts/check_env.py
```

## 目录结构（阶段 1）

```
firstRAG/
├─ requirements.txt          # 运行时核心依赖
├─ requirements-api.txt      # 可选 API 依赖（第二阶段）
├─ requirements-dev.txt      # 开发与测试依赖
├─ .env.example              # 环境变量示例
├─ .gitignore
├─ config/                   # 配置与日志
├─ rag/                      # 业务模块（数据模型等）
├─ storage/                  # 运行时数据
├─ scripts/                  # 自检脚本
├─ tests/                    # 单元测试
└─ docs/PROJECT_PLAN.md      # 完整开发计划
```

## 风险与边界

- 第一版**仅支持带文本层的 PDF**；扫描 PDF 不支持 OCR，会给出明确提示。
- LLM 第一版使用标准 OpenAI 兼容流式接口；`reasoning_split` 等扩展字段在核心稳定后单独验证。
- 不在本地加载 Embedding 模型，依赖远程 API。