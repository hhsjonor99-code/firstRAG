"""Settings 行为测试。"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

# 让脚本可直接运行：把项目根目录加入 sys.path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import MissingAPIKeyError, Settings  # noqa: E402


def _new_settings(**overrides) -> Settings:
    """构造一个不读 .env 的 Settings，便于单测。"""
    with mock.patch.dict(os.environ, {}, clear=True):
        return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_settings_construct_without_keys_does_not_raise():
    """缺 Key 时 Settings() 构造不应抛错。"""
    s = _new_settings()
    assert s.siliconflow_api_key is None
    assert s.minimax_api_key is None


def test_settings_defaults():
    s = _new_settings()
    assert s.siliconflow_base_url == "https://api.siliconflow.cn/v1"
    assert s.siliconflow_embedding_model == "Qwen/Qwen3-Embedding-4B"
    assert s.siliconflow_embedding_dimensions == 1024
    assert s.siliconflow_embedding_batch_size == 16
    assert s.siliconflow_timeout == 60
    assert s.minimax_base_url == "https://api.minimaxi.com/v1"
    assert s.minimax_model == "MiniMax-M3"
    assert s.chunk_size == 800
    assert s.chunk_overlap == 120
    assert s.retrieval_top_k == 5
    assert s.max_upload_mb == 20
    assert s.max_history_turns == 10


def test_require_siliconflow_key_missing():
    s = _new_settings()
    with pytest.raises(MissingAPIKeyError) as ei:
        s.require_siliconflow_key()
    assert "SILICONFLOW_API_KEY" in str(ei.value)


def test_require_siliconflow_key_empty_string():
    s = _new_settings(siliconflow_api_key="   ")
    with pytest.raises(MissingAPIKeyError):
        s.require_siliconflow_key()


def test_require_siliconflow_key_present():
    s = _new_settings(siliconflow_api_key="sk-test-1234")
    assert s.require_siliconflow_key() == "sk-test-1234"


def test_require_minimax_key_missing():
    s = _new_settings()
    with pytest.raises(MissingAPIKeyError) as ei:
        s.require_minimax_key()
    assert "MINIMAX_API_KEY" in str(ei.value)


def test_require_minimax_key_present():
    s = _new_settings(minimax_api_key="mm-test-5678")
    assert s.require_minimax_key() == "mm-test-5678"


def test_has_keys_helpers():
    s = _new_settings()
    assert s.has_siliconflow_key() is False
    assert s.has_minimax_key() is False
    s2 = _new_settings(siliconflow_api_key="x", minimax_api_key="y")
    assert s2.has_siliconflow_key() is True
    assert s2.has_minimax_key() is True


def test_extra_env_vars_are_ignored():
    """未在 Settings 中声明的环境变量应被忽略，不抛错。"""
    with mock.patch.dict(os.environ, {"SOME_RANDOM_VAR": "foo"}, clear=True):
        s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.siliconflow_api_key is None


def test_get_settings_caches_singleton():
    from config.settings import get_settings

    # 清理缓存
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b