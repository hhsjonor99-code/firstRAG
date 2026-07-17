"""MiniMax LLM 真实 API 手工连接测试脚本（v1.1：含推理内容隔离）。

运行：
    python scripts/test_minimax_connection.py

脚本同时测试非流式与流式调用；只使用不敏感的极简测试消息，
**不会**上传任何真实业务文档；只输出概要指标，不输出 Key、请求头
或完整响应体；**不**输出内部思考过程。

输出字段（按 v1.1 规范）：
- 连接状态（OK / FAIL）
- 模型名称
- 是否检测到独立 reasoning 字段
- 是否检测到嵌入式推理标签
- 最终用户可见答案（非流式）
- 最终答案长度
- 流式最终用户可见答案
- 流式片段数量
- 两种调用耗时
- 流式与非流式最终答案是否语义一致

退出码：
- 0 = 成功
- 1 = 配置错误（缺 Key）
- 2 = API 调用失败
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

# 让脚本可直接运行：把项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging import setup_logging  # noqa: E402
from config.settings import MissingAPIKeyError, Settings  # noqa: E402

from rag.llm_client import (  # noqa: E402
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    LLMResponseFormatError,
    MiniMaxLLMClient,
    ReasoningContentFilter,
    sanitize_model_content,
)


# 不敏感测试消息（不包含任何业务或隐私数据）
TEST_MESSAGES = [
    {"role": "system", "content": "你是一个连接测试助手，请简洁回答。"},
    {"role": "user", "content": "请只回复“连接成功”。"},
]

# 已知的「独立」推理字段名（不同模型可能不同）
INDEPENDENT_REASONING_FIELDS = (
    "reasoning",
    "reasoning_content",
    "analysis",
    "thinking",
    "thought",
)

# 嵌入式标签检测
EMBEDDED_TAGS = ("<think>", "<analysis>")


def _print_section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def _safe_classify_error(exc: Exception) -> str:
    """把异常归类为安全的错误类型字符串。"""
    if isinstance(exc, LLMAuthenticationError):
        return "鉴权失败（401/403）"
    if isinstance(exc, LLMRateLimitError):
        return "触发限流（429）"
    if isinstance(exc, LLMTimeoutError):
        return "请求超时"
    if isinstance(exc, LLMConnectionError):
        return "网络连接错误"
    if isinstance(exc, LLMServerError):
        return "服务端错误（5xx）"
    if isinstance(exc, LLMConfigurationError):
        return "配置错误（缺 Key 或参数错误）"
    if isinstance(exc, LLMError):
        return f"LLM 错误（{type(exc).__name__}）"
    return f"未预期异常（{type(exc).__name__}）"


def _normalize_for_compare(text: str) -> str:
    """用于语义一致性比较：去空白、统一小写。"""
    return re.sub(r"\s+", "", text or "").strip().lower()


def _detect_independent_reasoning(message) -> bool:
    """判断 message 上是否存在独立的 reasoning 类字段。"""
    for attr in INDEPENDENT_REASONING_FIELDS:
        if getattr(message, attr, None) is not None:
            return True
    return False


def _detect_embedded_tags(content: str) -> bool:
    """判断 content 中是否含嵌入式推理标签。"""
    if not content:
        return False
    return any(tag in content for tag in EMBEDDED_TAGS)


def _stream_raw_chunks(client: MiniMaxLLMClient) -> tuple[str, int]:
    """低层级流式：直接调用 SDK，不做客户端层 sanitize，合并 raw content。

    :returns: (raw_merged_content, chunk_count_with_content)
    """
    resp_iter = client._client.chat.completions.create(  # type: ignore[attr-defined]
        model=client.model_name,
        messages=TEST_MESSAGES,
        stream=True,
    )
    pieces: list[str] = []
    chunk_count = 0
    for chunk in resp_iter:
        chunk_count += 1
        if not getattr(chunk, "choices", None):
            continue
        try:
            delta = chunk.choices[0].delta
        except (IndexError, AttributeError, TypeError):
            continue
        c = getattr(delta, "content", None)
        if isinstance(c, str) and c:
            pieces.append(c)
    return "".join(pieces), chunk_count


def main() -> int:
    setup_logging(level="INFO")

    _print_section("MiniMax LLM 真实 API 连接测试")
    print(
        "此脚本仅用于手工验证 LLM 是否可用；不输出 Key、请求头、"
        "内部思考过程或完整响应体。"
    )

    settings = Settings()

    _print_section("配置（仅模型/端点）")
    print(f"  base_url       : {settings.minimax_base_url}")
    print(f"  model          : {settings.minimax_model}")

    if not settings.has_minimax_key():
        print()
        print("  [FAIL] 未检测到 MINIMAX_API_KEY。")
        print("         请先在 .env 中设置该变量，或通过环境变量注入。")
        return 1

    try:
        settings.require_minimax_key()
    except MissingAPIKeyError as exc:
        print(f"  [FAIL] Key 校验失败：{exc}")
        return 1

    # 构造客户端：捕获 SOCKS / 代理依赖等环境错误，给出明确提示
    try:
        client = MiniMaxLLMClient(settings=settings)
    except ImportError as exc:
        print()
        print("  [FAIL] 客户端构造失败：缺少可选依赖（ImportError）。")
        print(f"         错误类型：{type(exc).__name__}")
        print("         常见原因：环境配置了 SOCKS 代理，但 httpx[socks] 未安装。")
        print("         修复方式：")
        print("           1) pip install \"httpx[socks]\"")
        print("           2) 或取消设置 ALL_PROXY / SOCKS5_PROXY 等环境变量后重试")
        print("         出于安全考虑，脚本不输出完整异常消息与任何环境变量。")
        return 1
    except LLMConfigurationError as exc:
        print(f"  [FAIL] 配置错误：{exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print()
        print(f"  [FAIL] 客户端构造失败：{type(exc).__name__}")
        print("         出于安全考虑，脚本不输出完整异常消息。")
        return 2

    # ------------------------------------------------------------------
    # 1. 非流式调用：客户端层 complete() 已经 sanitize
    # ------------------------------------------------------------------
    _print_section("1. 非流式（客户端层 complete，已 sanitize）")
    print("  正在调用 API...")
    started = time.perf_counter()
    non_stream_text: str = ""
    try:
        non_stream_text = client.complete(TEST_MESSAGES)
    except LLMResponseFormatError as exc:
        print(f"  [WARN] 响应被 sanitize 后为空：{type(exc).__name__}")
    except LLMConfigurationError as exc:
        print(f"  [FAIL] 配置错误：{exc}")
        return 1
    except LLMError as exc:
        elapsed = time.perf_counter() - started
        print(f"  [FAIL] API 调用失败：{_safe_classify_error(exc)}")
        print(f"         耗时 {elapsed:.3f}s")
        return 2
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        print(f"  [FAIL] 未预期异常：{_safe_classify_error(exc)}")
        print(f"         耗时 {elapsed:.3f}s")
        return 2
    non_stream_elapsed = time.perf_counter() - started

    # ------------------------------------------------------------------
    # 2. 低层级非流式：用于检测独立 reasoning 字段与嵌入式标签
    #    （脚本**不**打印 raw content；只输出结构信息）
    # ------------------------------------------------------------------
    _print_section("2. 响应结构检测（不打印原文）")
    raw_non_stream = None
    has_independent = False
    has_embedded = False
    try:
        raw_non_stream = client._client.chat.completions.create(  # type: ignore[attr-defined]
            model=client.model_name,
            messages=TEST_MESSAGES,
            stream=False,
        )
        msg = raw_non_stream.choices[0].message
        raw_content = getattr(msg, "content", "") or ""
        has_independent = _detect_independent_reasoning(msg)
        has_embedded = _detect_embedded_tags(raw_content)
        # 不输出 raw_content
        print(f"  独立 reasoning 字段 : {'是' if has_independent else '否'}")
        print(f"  嵌入式推理标签      : {'是' if has_embedded else '否'}")
    except LLMError as exc:
        print(f"  [WARN] 结构检测失败：{_safe_classify_error(exc)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] 结构检测异常：{type(exc).__name__}")

    # ------------------------------------------------------------------
    # 3. 流式调用：客户端层 stream() 已经 sanitize
    # ------------------------------------------------------------------
    _print_section("3. 流式（客户端层 stream，已 sanitize）")
    print("  正在调用 API...")
    started = time.perf_counter()
    stream_pieces: list[str] = []
    try:
        for piece in client.stream(TEST_MESSAGES):
            stream_pieces.append(piece)
    except LLMResponseFormatError as exc:
        print(f"  [WARN] 响应被 sanitize 后为空：{type(exc).__name__}")
    except LLMConfigurationError as exc:
        print(f"  [FAIL] 配置错误：{exc}")
        return 1
    except LLMError as exc:
        elapsed = time.perf_counter() - started
        print(f"  [FAIL] API 调用失败：{_safe_classify_error(exc)}")
        print(f"         耗时 {elapsed:.3f}s")
        return 2
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        print(f"  [FAIL] 未预期异常：{_safe_classify_error(exc)}")
        print(f"         耗时 {elapsed:.3f}s")
        return 2
    stream_elapsed = time.perf_counter() - started
    stream_merged = "".join(stream_pieces)

    # ------------------------------------------------------------------
    # 4. 流式原始内容：仅用于验证流式 chunk 数（不打印原文）
    # ------------------------------------------------------------------
    raw_stream_chunks = 0
    try:
        _raw_merged, raw_stream_chunks = _stream_raw_chunks(client)
    except Exception:  # noqa: BLE001
        # 不影响主流程
        pass

    # ------------------------------------------------------------------
    # 5. 输出最终用户可见结果
    # ------------------------------------------------------------------
    _print_section("4. 最终用户可见结果（已 sanitize）")
    print("  非流式最终答案：")
    print(f"    长度    : {len(non_stream_text)} 字符")
    print(f"    内容    : {non_stream_text}")
    print("  流式最终答案：")
    print(f"    片段数  : {len(stream_pieces)}")
    print(f"    长度    : {len(stream_merged)} 字符")
    print(f"    内容    : {stream_merged}")
    print(f"  耗时    : 非流式 {non_stream_elapsed:.3f}s, 流式 {stream_elapsed:.3f}s")
    print(f"  原始流式 chunk 数（参考）: {raw_stream_chunks}")

    # 语义一致性
    consistent = _normalize_for_compare(non_stream_text) == _normalize_for_compare(
        stream_merged
    )
    print(f"  流式 / 非流式语义一致 : {'是' if consistent else '否'}")

    # ------------------------------------------------------------------
    # 6. 结论
    # ------------------------------------------------------------------
    _print_section("5. 结论")
    if not non_stream_text and not stream_merged:
        print("  [WARN] 两种调用均未返回非空用户可见内容，请检查 Key 权限或模型。")
        return 2
    if not consistent:
        print("  [WARN] 流式与非流式结果不一致，可能受模型随机性影响。")
        # 不视作 FAIL
    print("  MiniMax LLM 真实 API 测试通过（推理内容已隔离）。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
