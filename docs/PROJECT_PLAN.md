# firstRAG 本地知识库 RAG 系统 — 开发计划 v2

> 本文档是仓库内项目计划副本，与外部计划文件保持一致。如有调整以仓库内版本为准。

## Context

`E:\Coding\claud code\firstRAG` 当前仅含三份文件：`prompt.txt`（产品需求）、`requirements.txt`（依赖清单）、一份"国家基本药物目录（2026 年版）(OCR).docx"（示例语料）。本计划为该仓库从零搭建 **本地文档知识库 RAG 系统** 的架构与分阶段实施方案。

**约束（已与用户对齐）**：
- 第一版**只使用 Streamlit 单体**；FastAPI 仅作可选依赖预留
- Embedding 走 **SiliconFlow 远程 API**（不在本地加载模型）
- LLM 使用 **MiniMax-M3**（OpenAI 兼容接口）
- **第一版不启用** MiniMax `reasoning_split` 扩展字段，先用标准 content 流式
- 本轮仅完成规划，**不安装依赖、不写业务代码**

---

## 1. 当前环境（已核验）

| 项 | 状态 |
|---|---|
| OS | Windows |
| Conda 环境 | `py311`（`D:\ProgramData\miniconda3\envs\py311`） |
| Python | 3.11.15 |
| pip | 26.1.2 |
| 命令约定 | `python -m pip`、`python -m pytest` |

**关键依赖（已装可导入）**：streamlit 1.59.2 / faiss-cpu 1.14.3 / numpy 2.4.6 / langchain 1.3.13 + langchain-community 0.4.2 + langchain-text-splitters 1.1.2 / pypdf 6.14.2 / python-docx 1.2.0 / markdown 3.10.2 / beautifulsoup4 4.15.0 / openai 2.45.0 / requests 2.34.2 / python-dotenv 1.2.2 / pydantic 2.13.4 / pydantic-settings 2.14.2 / python-multipart 0.0.32 / tenacity 9.1.4。

**未装（符合要求）**：sentence-transformers / transformers / accelerate / torch。

**冒烟测试已通过**：faiss 余弦检索、python-docx 段落写入、openai SDK 实例化均 OK。

**示例 docx 真实类型**：`Microsoft Word 2007+`（ZIP 容器），扩展名 `.docx` 真实有效；文件名带 `(OCR)` 后缀，需在 parser 层容错。

---

## 2. 依赖划分（按用户意见重构）

不再把所有依赖塞进单文件。**拆为 4 个文件**：

### `requirements.txt`（运行时核心依赖，第一版必需）
```
streamlit
pypdf
python-docx
markdown
beautifulsoup4
numpy
faiss-cpu
langchain-text-splitters
openai
requests
tenacity              # 重试：显式直接依赖，不依赖透传
python-dotenv
pydantic
pydantic-settings
```
- **不固化任何镜像地址**；镜像（如清华源）仅在 README 中作为可选提示。
- `langchain` 主包与 `langchain-community` **首版不引入**，避免版本分裂风险；只用 `langchain-text-splitters` 提供 `RecursiveCharacterTextSplitter`。

### `requirements-api.txt`（可选，第二阶段启用 FastAPI 时安装）
```
fastapi
uvicorn[standard]
python-multipart
```

### `requirements-dev.txt`（开发与测试依赖）
```
pytest
pytest-cov
# reportlab 仅在需要生成 PDF fixture 时启用；优先使用仓库内固定的小型 PDF 二进制 fixture，避免引入
# reportlab
```

### 拆分原则
- 运行时不依赖 dev 依赖；测试时 `pip install -r requirements.txt -r requirements-dev.txt`。
- API 依赖明确分离，避免 Streamlit 单体应用误启动 FastAPI。

---

## 3. Settings 设计（API Key 延迟校验）

`config/settings.py`：

```python
class Settings(BaseSettings):
    # Embedding —— Key 可为 None，使用时才校验
    siliconflow_api_key: Optional[str] = None
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    siliconflow_embedding_dimensions: int = 1024
    siliconflow_embedding_batch_size: int = 16
    siliconflow_timeout: int = 60

    # LLM —— Key 可为 None，使用时才校验
    minimax_api_key: Optional[str] = None
    minimax_base_url: str = "https://api.minimaxi.com/v1"
    minimax_model: str = "MiniMax-M3"

    # 切分/检索
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 5

    # 上传限制
    max_upload_mb: int = 20
    max_history_turns: int = 10

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

**延迟校验**：
- `Settings()` 构造时**不抛错**；缺 Key 也能正常加载。
- `SiliconFlowEmbeddingProvider.__init__` / `LLMClient.__init__` 中调用 `settings.require_siliconflow_key()`：
  ```python
  def require_siliconflow_key(self) -> str:
      if not self.siliconflow_api_key:
          raise MissingAPIKeyError(
              "缺少环境变量 SILICONFLOW_API_KEY；"
              "文档解析、分块、本地索引可继续使用，但 Embedding 与检索需要此 Key。"
          )
      return self.siliconflow_api_key
  ```
- `MissingAPIKeyError` 异常消息**明确指出环境变量名**。
- 无 Key 时，`scripts/check_env.py` 给出 WARNING 而非 ERROR；UI 在调用 Embedding/LLM 时才阻塞并提示。

---

## 4. 日志安全

- 全局 logging 配置统一在 `config/logging.py`。
- 严禁在日志中输出 API Key 的任何形式：
  - **完整 Key**：禁止
  - **末四位**：禁止
  - **前缀/哈希**：禁止
  - **Key 字段名**：仅在错误信息中可出现（如 `SILICONFLOW_API_KEY 未设置`），但**不拼接任何 Key 字符**
- 在 `siliconflow_embeddings.py` 内部对 headers 做白名单日志：只记录 `Authorization 头存在/缺失` 与 HTTP 状态码。

---

## 5. FAISS 向量归一化与校验

`vector_store.py` 内部封装：
```python
def _prepare_vectors(vectors: np.ndarray | list[list[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)        # 强制 float32
    if arr.ndim != 2:
        raise ValueError(f"向量必须是 2D，得到 ndim={arr.ndim}")
    if arr.shape[1] != self.dim:
        raise ValueError(f"向量维度 {arr.shape[1]} 与索引维度 {self.dim} 不一致")
    if not np.isfinite(arr).all():
        raise ValueError("向量包含 NaN 或 Inf")
    if arr.shape[0] == 0:
        raise ValueError("向量为空，禁止写入索引")
    faiss.normalize_L2(arr)                            # 不使用未经确认的 copy=False
    return arr
```
- 检索前对 query 向量做同样校验。
- 校验失败抛 `VectorStoreError`，调用方在 UI 上明确提示。

---

## 6. 索引持久化设计（含 manifest.json 与原子保存）

### 6.1 目录结构
```
storage/
├─ uploads/{document_id}{ext}         # 仅入库文件
├─ indexes/
│  ├─ faiss.index                     # FAISS 二进制索引
│  ├─ chunks.jsonl                    # 每行 DocumentChunk JSON
│  ├─ documents.jsonl                 # 每行 DocumentInfo JSON
│  └─ manifest.json                   # 索引元数据
└─ metadata/                          # 预留（如导出/调试用）
```

### 6.2 manifest.json 字段
```json
{
  "embedding_provider": "siliconflow",
  "embedding_model": "Qwen/Qwen3-Embedding-4B",
  "embedding_dimensions": 1024,
  "chunk_size": 800,
  "chunk_overlap": 120,
  "index_type": "IndexFlatIP",
  "created_at": "2026-07-15T10:30:00Z"
}
```

### 6.3 启动加载校验
`VectorStore.load()` 流程：
1. 检查 `faiss.index` / `chunks.jsonl` / `documents.jsonl` / `manifest.json` 是否齐全；任一缺失 → 空索引启动。
2. 读取 manifest 与当前 `Settings` 对比以下字段：
   - `embedding_provider` 必须匹配
   - `embedding_model` 必须匹配
   - `embedding_dimensions` 必须等于 `settings.siliconflow_embedding_dimensions`
   - `chunk_size` 必须等于 `settings.chunk_size`
   - `chunk_overlap` 必须等于 `settings.chunk_overlap`
   - `index_type` 必须等于 `IndexFlatIP`
3. **任一不一致**：抛出 `IndexIncompatibleError`，错误消息**明确列出差异字段与期望值/实际值**，并在 UI 提示「索引已废弃，请重建」。**不自动删除旧索引**（避免误删）。
4. 加载后校验 `index.ntotal == len(chunks)`；不一致 → `IndexCorruptedError`，停止加载并提示「索引损坏，请重建」。

### 6.4 原子保存（避免写入中断导致损坏）
所有写操作通过临时文件 + `os.replace`：
```python
def _atomic_write_bytes(target: Path, data: bytes):
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)        # 原子替换（同分区）
```

写入顺序（`save()`）：
1. 写 `faiss.index` 临时 → replace
2. 写 `chunks.jsonl` 临时 → replace
3. 写 `documents.jsonl` 临时 → replace
4. 写 `manifest.json` 临时 → replace

任一步骤失败 → 已成功的临时文件清理 + 抛错（不留下半成品索引）。

---

## 7. 文档入库流程（调整后顺序）

```
1. 读取上传字节（Streamlit UploadedFile.read()）
2. 校验：
   - 扩展名白名单（.pdf / .docx / .txt / .md）
   - 大小 ≤ max_upload_mb
   - 文件头魔数（PDF: %PDF- / DOCX: PK\x03\x04 实际是 ZIP / TXT: 不强校验 / MD: UTF-8 可解码）
3. 计算 SHA256（基于字节）
4. 检查重复：documents.jsonl 中是否存在同 file_hash → 命中则跳过入库并提示已存在
5. 不重复时：
   - 生成 document_id = uuid4().hex
   - 解析原始文件名 → 仅保存到 DocumentInfo.original_file_name
   - 磁盘文件名 = {document_id}{原扩展名（小写）}，写入 storage/uploads/
   - 解析（parsers）→ 切分（splitter）→ embedding（SiiliconFlow）→ 归一化 → 入库
   - 检查解析结果非空；**PDF 解析后若无文本**，提示「可能为扫描 PDF，当前版本不支持 OCR」，不入库
6. 保存 DocumentInfo / DocumentChunk / faiss.index / manifest
```

**空内容文件静默入库被禁止**：
- 解析后 chunk 列表为空 → 抛 `EmptyDocumentError`；删除已写入的 uploads 文件；UI 明确提示原因。

---

## 8. PDF 边界（明确）

- 第一版 **仅支持带文本层的 PDF**。
- 解析策略：`pypdf.PdfReader` 遍历每页 `extract_text()`；按页累积文本。
- 检测条件：`if sum(len(p) for p in pages_text) == 0` → 抛 `UnsupportedScannedPDFError("可能为扫描 PDF，当前版本暂不支持 OCR")`。
- 不引入 pypdfium2 / pdfplumber / ocrmypdf；OCR 作为未来扩展。

---

## 9. 引用编号生成与校验（程序生成）

引用编号**完全由程序控制**，禁止依赖 LLM 自报：

### 9.1 检索端编号
```python
def to_retrieved(ranked: list[tuple[DocumentChunk, float]]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=chk, score=score, citation_id=f"S{i+1}")
        for i, (chk, score) in enumerate(ranked)
    ]
```

### 9.2 Prompt 注入
把编号映射写进 system prompt：
```
你必须且只能引用以下片段，使用 [S1]..[S{k}] 编号：
[S1] {filename} | 页码: {page} | 段落: {paragraph}
[S2] ...
```

### 9.3 输出校验（LLM 防虚构）
```python
import re
ALLOWED = {rc.citation_id for rc in retrieved}
cited = set(re.findall(r"\[S\d+\]", llm_output))
illegal = cited - ALLOWED
if illegal:
    logger.warning("LLM 引用了未提供的编号: %s，已过滤", illegal)
    for tag in illegal:
        llm_output = llm_output.replace(tag, "")    # 过滤，不展示
```
- **禁止让 LLM 在回答中重新声明文件名/页码/编号** —— 这些信息只通过引用面板渲染。
- 引用面板数据源 = `RetrievedChunk`，每条渲染：编号 / 文件名 / 页码或段落或行号 / heading / 相似度分数 / 折叠原文。

---

## 10. LLM 客户端（第一版简化）

- 使用 `openai.OpenAI(base_url=settings.minimax_base_url, api_key=...)`。
- **第一版不传 `extra_body={"reasoning_split": True}`**。
- 流式：`client.chat.completions.create(..., stream=True)`，逐 chunk 读取 `delta.content`。
- 非流式：`response.choices[0].message.content`。
- `reasoning_split` 验证推迟到核心稳定后单独进行。

---

## 11. 完整阶段计划（8 阶段）

> 每阶段**前一阶段全部验收通过**后才进入下一阶段。

### 阶段 1：脚手架（可执行最小实现 + 基础测试）
**交付物**：
- 4 个依赖文件：`requirements.txt` / `requirements-api.txt` / `requirements-dev.txt`
- `README.md` / `.env.example` / `.gitignore` / `docs/PROJECT_PLAN.md`
- `config/__init__.py` / `config/settings.py` / `config/logging.py`
- `rag/__init__.py` / `rag/models.py`（DocumentInfo / DocumentChunk / RetrievedChunk / ChatMessage 完整 Pydantic 模型）
- `scripts/check_env.py`（可执行：列出 Python 版本、解释器路径、关键包导入结果、未装包、API Key 状态 WARNING/OK）
- `tests/__init__.py` / `tests/test_settings.py` / `tests/test_models.py`
- `storage/uploads/.gitkeep` / `storage/indexes/.gitkeep` / `storage/metadata/.gitkeep`

**验收**：
- `python scripts/check_env.py` 全绿；缺 Key 时 WARNING 而非 ERROR
- `pytest tests/test_settings.py tests/test_models.py` 全绿
- `from config.settings import Settings` 在缺 Key 时不抛错
- `Settings().require_siliconflow_key()` 在缺 Key 时抛 `MissingAPIKeyError` 并明确指出变量名

### 阶段 2：文档解析 + 分块
**交付物**：`rag/parsers.py`（4 种格式；DOCX 容错 OCR 噪声；PDF 无文本层抛 `UnsupportedScannedPDFError`） / `rag/splitter.py` / `tests/fixtures/` / `tests/test_parsers.py` / `tests/test_splitter.py`

**验收**：pytest 全绿；解析示例 docx 后 chunk metadata 包含 `heading` / `paragraph_number`；空内容 PDF → 抛 `UnsupportedScannedPDFError`；空内容文档 → 抛 `EmptyDocumentError`

### 阶段 3：SiliconFlow Embedding 封装
**交付物**：`rag/embedding_provider.py` / `rag/siliconflow_embeddings.py` / `scripts/test_siliconflow_embedding.py` / `tests/test_siliconflow_embeddings.py`

**验收**：真实调用返回 `(N, 1024)` 向量且 L2 范数=1；mock 测试触发重试；日志无 Key 字符

### 阶段 4：向量库与持久化（含 manifest + 原子保存）
**交付物**：`rag/vector_store.py` / `tests/test_vector_store.py`

**验收**：add/search/delete/clear/rebuild/save/load 全绿；维度/dtype/NaN/空结果校验抛错；manifest 不一致抛 `IndexIncompatibleError`；index.ntotal ≠ len(chunks) 抛 `IndexCorruptedError`；写入中断无 `.tmp` 残留

### 阶段 5：检索服务 + Prompt 构造
**交付物**：`rag/retriever.py` / `rag/prompt_builder.py` / `tests/test_retriever.py` / `tests/test_prompt_builder.py`

**验收**：检索返回 S1..Sk 编号；prompt 中含「只能依据 [S#]」「禁止虚构文件名/页码/编号」；非法编号过滤

### 阶段 6：LLM 客户端（第一版简化）
**交付物**：`rag/llm_client.py` / `scripts/test_minimax_connection.py`

**验收**：流式 / 非流式调用通；缺 Key → `MissingAPIKeyError` 消息含 `MINIMAX_API_KEY`；日志无 Key 字符

### 阶段 7：ChatService 串联
**交付物**：`rag/chat_service.py`

**验收**：单轮问答端到端通过；多轮历史只用于改写；citations 数量 = top_k；非法编号被过滤并 warn

### 阶段 8：Streamlit UI
**交付物**：`app.py`

**验收**：4 Tab 全部可用；流式输出；引用面板；删除/清空；索引不兼容提示

---

## 12. 主要技术风险（v2 更新）

1. **LangChain 版本分裂**：本机 `langchain 1.3.13` + `langchain-classic 1.0.8` 同时存在。首版**仅依赖 `langchain-text-splitters`**，不引入主链式 API，降低风险。
2. **OCR docx 噪声**：示例 docx 带 `(OCR)` 后缀可能含乱码，parser 对单字符 / 全空白段需过滤。
3. **索引不兼容处理**：若用户中途改 `chunk_size` 等参数，旧索引必须明确失效而非默默重建（避免历史回答引用消失）。
4. **原子保存的 tmp 残留**：进程崩溃可能留下 `.tmp` 文件；启动时清理 `<= 1 小时` 内的孤儿 tmp 文件，**不删除可能正在写入的 tmp**。
5. **大文档 embedding 耗时**：UI 需有进度提示。
6. **历史泄漏**：LLM 容易把历史回答当事实；prompt 硬约束 + 引用编号程序化双重保险。
7. **FAISS 内存**：`IndexFlatIP` ≈ `N × dim × 4` 字节；万级文档 ≈ 几百 MB，首版不构成问题。
8. **路径安全**：中文文件名 + 空格 → `pathlib.Path.name` + `re.sub(r'[^\w一-鿿.-]', '_', name)`。

---

## 13. 端到端验证（阶段 8 完成后）

1. `pip install -r requirements.txt -r requirements-dev.txt`
2. `streamlit run app.py`
3. 上传"国家基本药物目录（2026 年版）.docx"
4. 知识库 Tab 显示入库成功（chunk 数 > 0）
5. 问答 Tab 提问"目录包含哪些类别药品？" → 含 `[S1]..[S5]`，每个引用可展开原文
6. 追问"化学药品部分呢？" → 历史被改写，引用仍来自当前文档
7. 删除该文档 → 知识库 Tab 已清空 → 再问 → 提示"未找到充分依据"
8. 关闭并重启 → 知识库 Tab 内容仍在（持久化生效）
9. 故意把 `chunk_size` 改成 1000 → 启动报错「索引不兼容，embedding_model/embedding_dimensions/chunk_size 不一致，请重建」，旧索引文件未被删除