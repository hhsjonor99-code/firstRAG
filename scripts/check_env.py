"""firstRAG 环境与依赖自检脚本。

运行：
    python scripts/check_env.py

输出内容：
- Python 版本与解释器路径
- Conda 环境（若可识别）
- 关键包导入结果（OK / FAIL）
- 未装包清单（如有）
- API Key 状态（OK / WARNING）
- 退出码：0 = 全绿或仅 KEY 缺失警告；非 0 = 关键依赖缺失
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# 允许以脚本方式直接运行：把项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings, MissingAPIKeyError  # noqa: E402
from config.logging import setup_logging, get_logger  # noqa: E402

logger = get_logger("check_env")


# 关键运行时依赖（与 requirements.txt 对齐）
RUNTIME_PACKAGES = [
    "streamlit",
    "pypdf",
    "docx",                # python-docx 的 import 名
    "markdown",
    "bs4",                 # beautifulsoup4 的 import 名
    "numpy",
    "faiss",
    "langchain_text_splitters",
    "openai",
    "requests",
    "tenacity",
    "dotenv",              # python-dotenv 的 import 名
    "pydantic",
    "pydantic_settings",
]

# 可选依赖（缺失仅 WARNING）
OPTIONAL_PACKAGES = [
    ("fastapi", "FastAPI（第二阶段）"),
    ("uvicorn", "Uvicorn（第二阶段）"),
    ("multipart", "python-multipart（第二阶段）"),
]


def _print_section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def check_python_env() -> dict:
    info = {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "prefix": sys.prefix,
    }
    return info


def check_packages() -> tuple[list[str], list[tuple[str, str, str]]]:
    """返回 (ok_list, fail_list)，fail 项含 (import_name, version_or_err, label)。"""
    ok: list[str] = []
    fail: list[tuple[str, str, str]] = []
    for name in RUNTIME_PACKAGES:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            ok.append(f"{name}=={ver}")
        except Exception as e:  # noqa: BLE001
            fail.append((name, str(e), "runtime"))
    for name, label in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            ok.append(f"{name}=={ver} (optional)")
        except Exception:  # noqa: BLE001
            # 可选包缺失仅记录，不影响退出码
            ok.append(f"{name}: not installed (optional: {label})")
    return ok, fail


def check_keys() -> tuple[list[str], list[str]]:
    settings = Settings()
    warnings: list[str] = []
    oks: list[str] = []
    if settings.has_siliconflow_key():
        oks.append("SILICONFLOW_API_KEY: 已设置")
    else:
        warnings.append("SILICONFLOW_API_KEY: 未设置（Embedding / 检索功能不可用）")
    if settings.has_minimax_key():
        oks.append("MINIMAX_API_KEY: 已设置")
    else:
        warnings.append("MINIMAX_API_KEY: 未设置（LLM 问答功能不可用）")
    return oks, warnings


def check_settings_delay_validation() -> list[str]:
    """验证 Settings 延迟校验行为。"""
    notes: list[str] = []
    settings = Settings()
    # 构造不抛错
    notes.append("Settings() 构造成功（未因缺 Key 抛错）")

    # 缺 Key 时抛错且消息含变量名
    if not settings.has_siliconflow_key():
        try:
            settings.require_siliconflow_key()
            notes.append("[FAIL] 期望抛 MissingAPIKeyError，但未抛")
        except MissingAPIKeyError as e:
            if "SILICONFLOW_API_KEY" in str(e):
                notes.append("require_siliconflow_key(): 缺 Key 时正确抛错并包含变量名")
            else:
                notes.append(f"[FAIL] 错误消息未包含变量名: {e}")
    if not settings.has_minimax_key():
        try:
            settings.require_minimax_key()
            notes.append("[FAIL] 期望抛 MissingAPIKeyError，但未抛")
        except MissingAPIKeyError as e:
            if "MINIMAX_API_KEY" in str(e):
                notes.append("require_minimax_key(): 缺 Key 时正确抛错并包含变量名")
            else:
                notes.append(f"[FAIL] 错误消息未包含变量名: {e}")
    return notes


def main() -> int:
    setup_logging(level="WARNING")  # 自检脚本只把日志用于内部异常

    _print_section("Python 环境")
    info = check_python_env()
    for k, v in info.items():
        print(f"  {k:12s}: {v}")

    _print_section("关键依赖")
    ok, fail = check_packages()
    for line in ok:
        print(f"  OK    {line}")
    for name, err, label in fail:
        print(f"  FAIL  {name} ({label}): {err}")

    _print_section("API Key 状态")
    oks, warnings = check_keys()
    for line in oks:
        print(f"  OK       {line}")
    for line in warnings:
        print(f"  WARNING  {line}")

    _print_section("Settings 延迟校验")
    for note in check_settings_delay_validation():
        print(f"  {note}")

    _print_section("结论")
    if fail:
        print(f"  共 {len(fail)} 个关键依赖缺失，请先 `pip install -r requirements.txt`")
        return 2
    if warnings:
        print(f"  关键依赖全部就绪；{len(warnings)} 个 API Key 缺失（WARNING，可后续设置）")
        return 0
    print("  全部 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())