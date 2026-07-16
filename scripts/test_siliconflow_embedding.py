"""SiliconFlow Embedding 真实 API 手工测试脚本。

运行：
    python scripts/test_siliconflow_embedding.py

脚本仅发起 2 条极短、不敏感的测试文本，**不会**上传任何真实业务文档；
只输出概要指标，不输出原始向量，也不输出 Key。

输出字段：
- 是否连接成功（OK / FAIL）
- 模型名称
- 向量数量
- 向量维度
- dtype
- 每个向量的 L2 范数
- 请求耗时（秒）

退出码：
- 0 = 成功
- 1 = 配置错误（缺 Key）
- 2 = API 调用失败
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 让脚本可直接运行：把项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging import setup_logging  # noqa: E402
from config.settings import Settings, MissingAPIKeyError  # noqa: E402

from rag.embedding_provider import EmbeddingError  # noqa: E402
from rag.siliconflow_embeddings import (  # noqa: E402
    EmbeddingConfigurationError,
    SiliconFlowEmbeddingProvider,
)


# 两条短测试文本（与示例 docx 主题相关但不敏感；不是真实患者数据）
TEST_TEXTS = [
    "阿莫西林是一种抗菌药物。",
    "高血压患者需要定期监测血压。",
]


def _print_section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def main() -> int:
    setup_logging(level="INFO")

    _print_section("SiliconFlow Embedding 真实 API 测试")
    print("此脚本仅用于手工验证 Embedding 是否可用；不输出原始向量。")

    settings = Settings()

    _print_section("配置")
    print(f"  base_url       : {settings.siliconflow_base_url}")
    print(f"  model          : {settings.siliconflow_embedding_model}")
    print(f"  expected_dim   : {settings.siliconflow_embedding_dimensions}")
    print(f"  batch_size     : {settings.siliconflow_embedding_batch_size}")
    print(f"  timeout (sec)  : {settings.siliconflow_timeout}")

    if not settings.has_siliconflow_key():
        print()
        print("  [FAIL] 未检测到 SILICONFLOW_API_KEY。")
        print("         请先在 .env 中设置该变量，或通过环境变量注入。")
        return 1

    try:
        # 延迟校验 Key（明确给出错误时打印不含 Key 的提示）
        settings.require_siliconflow_key()
    except MissingAPIKeyError as exc:
        print(f"  [FAIL] Key 校验失败：{exc}")
        return 1

    provider = SiliconFlowEmbeddingProvider(settings=settings)

    _print_section("请求")
    print(f"  文本数   : {len(TEST_TEXTS)}")
    print(f"  文本预览 : 第 1 条 {len(TEST_TEXTS[0])} 字，第 2 条 {len(TEST_TEXTS[1])} 字")
    print("  正在调用 API...")

    started = time.perf_counter()
    try:
        vectors = provider.embed_documents(TEST_TEXTS)
    except EmbeddingConfigurationError as exc:
        print(f"  [FAIL] 配置错误：{exc}")
        return 1
    except EmbeddingError as exc:
        print(f"  [FAIL] API 调用失败：{type(exc).__name__}: {exc}")
        print("         可能原因：Key 无效、网络问题、配额耗尽或服务端异常。")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] 未预期异常：{type(exc).__name__}: {exc}")
        return 2
    elapsed = time.perf_counter() - started

    _print_section("结果")
    print(f"  连接状态 : OK")
    print(f"  模型     : {provider.model_name}")
    print(f"  向量数量 : {vectors.shape[0]}")
    print(f"  维度     : {vectors.shape[1]}")
    print(f"  dtype    : {vectors.dtype}")
    print(f"  耗时     : {elapsed:.3f} 秒")
    norms = __import__("numpy").linalg.norm(vectors, axis=1)
    print("  L2 范数  :")
    for i, n in enumerate(norms):
        print(f"    [S{i+1}]  {n:.6f}")

    if not __import__("numpy").allclose(norms, 1.0, atol=1e-3):
        print("  [WARN] L2 范数偏离 1；建议检查归一化逻辑。")
    if vectors.shape != (len(TEST_TEXTS), settings.siliconflow_embedding_dimensions):
        print(
            f"  [WARN] shape={vectors.shape} 与期望"
            f" ({len(TEST_TEXTS)}, {settings.siliconflow_embedding_dimensions}) 不一致。"
        )

    _print_section("结论")
    print("  Embedding 真实 API 测试通过。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)