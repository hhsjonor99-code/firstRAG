"""firstRAG 统一日志配置。

安全策略：
- 严禁在任何日志输出中包含 API Key 的任何形式（完整、末四位、前缀、哈希）。
- 提供 :class:`APIKeyRedactionFilter` 自动过滤已知的 Key 字段名。
- 默认输出到 stderr；调用 :func:`setup_logging` 后即可在项目各处使用标准 logging。
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Iterable

# 这些字段名如果在日志中出现，对应值会被替换为 [REDACTED]。
# 注意：仅匹配字段名匹配 + 紧跟等号/冒号/引号的场景，避免误杀普通字符串。
_API_KEY_FIELD_NAMES = (
    "SILICONFLOW_API_KEY",
    "MINIMAX_API_KEY",
    "siliconflow_api_key",
    "minimax_api_key",
    "Authorization",
    "authorization",
    "api_key",
    "apikey",
    "api-key",
)

# 形如 KEY="..." / KEY=... / KEY: ... / KEY=sk-xxx 的赋值都会被过滤
_REDACTION_RE = re.compile(
    r"(" + "|".join(re.escape(n) for n in _API_KEY_FIELD_NAMES) + r")\s*([:=])\s*([^\s,;}\]]+)"
)


class APIKeyRedactionFilter(logging.Filter):
    """对日志记录中的 API Key 字段值进行替换。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - std API
        try:
            msg = record.getMessage()
            new_msg = _REDACTION_RE.sub(
                lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", msg
            )
            if new_msg != msg:
                # 修改 args 以便 %-format 不会再注入原始 Key
                record.msg = new_msg
                record.args = ()
        except Exception:  # pragma: no cover - 防御性
            pass
        return True


def setup_logging(
    level: int | str = logging.INFO,
    stream=None,
    extra_filters: Iterable[logging.Filter] | None = None,
) -> None:
    """初始化项目日志。

    - 默认级别 INFO；可通过参数覆盖。
    - 输出到 stderr（便于 Streamlit 等捕获）。
    - 自动附加 :class:`APIKeyRedactionFilter`。
    """
    root = logging.getLogger()
    root.setLevel(level)

    # 清理已有 handler，避免重复输出
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(APIKeyRedactionFilter())
    if extra_filters:
        for f in extra_filters:
            handler.addFilter(f)

    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """获取项目内统一 logger。"""
    return logging.getLogger(f"firstrag.{name}")