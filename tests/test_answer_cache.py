# -*- coding: utf-8 -*-
"""tests/test_answer_cache.py — 回答缓存（多轮对话标记恢复）单元测试。

运行: python -m pytest tests/test_answer_cache.py -v
"""

import time

import pytest

from core import answer_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    answer_cache.clear()
    yield
    answer_cache.clear()


def test_store_lookup_roundtrip():
    """登记后按发出内容找回原始标记文本。"""
    answer_cache.store("渲染后文本（含行内图片）",
                       "原文 [COMPOSITE:reaction]...[/COMPOSITE]")
    assert answer_cache.lookup("渲染后文本（含行内图片）") == \
        "原文 [COMPOSITE:reaction]...[/COMPOSITE]"


def test_lookup_miss_returns_none():
    assert answer_cache.lookup("从未见过的文本") is None
    assert answer_cache.lookup("") is None
    assert answer_cache.lookup(None) is None


def test_reasoning_stripped_on_store():
    """[REASONING] 思考块不进入缓存（过程而非结论，用户裁定）。"""
    answer_cache.store(
        "渲染后文本",
        "前言 [REASONING]先数原子编号：0=C 1=O[/REASONING] 正文 "
        "[STRUCT:CCO] 结尾")
    restored = answer_cache.lookup("渲染后文本")
    assert "REASONING" not in restored
    assert "数原子编号" not in restored
    assert "[STRUCT:CCO]" in restored and "前言" in restored and "结尾" in restored


def test_identical_content_not_stored():
    """无渲染产物的纯文本回答（content == raw）不缓存。"""
    answer_cache.store("纯文本回答", "纯文本回答")
    assert answer_cache.lookup("纯文本回答") is None


def test_store_lookup_full_meta():
    """lookup_full 返回 (原始标记, meta)；无 meta 存默认 {}。"""
    answer_cache.store("渲染C", "[STRUCT:CCO]",
                       meta={"failed": True, "reason": "化学校验：不守恒"})
    raw, meta = answer_cache.lookup_full("渲染C")
    assert raw == "[STRUCT:CCO]"
    assert meta == {"failed": True, "reason": "化学校验：不守恒"}
    answer_cache.store("渲染D", "[STRUCT:CCO]")
    raw2, meta2 = answer_cache.lookup_full("渲染D")
    assert raw2 == "[STRUCT:CCO]" and meta2 == {}
    # lookup（旧接口）仍只回标记文本
    assert answer_cache.lookup("渲染C") == "[STRUCT:CCO]"


def test_ttl_expiry(monkeypatch):
    """超过 TTL 的条目失效并清除。"""
    answer_cache.store("内容A", "标记A")
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 25 * 3600)
    assert answer_cache.lookup("内容A") is None


def test_lru_eviction(monkeypatch):
    """超出容量上限时淘汰最久未访问的条目。"""
    monkeypatch.setattr(answer_cache, "_MAX_ENTRIES", 3)
    for i in range(4):
        answer_cache.store(f"内容{i}", f"标记{i}")
    assert answer_cache.lookup("内容0") is None       # 最旧的被淘汰
    assert answer_cache.lookup("内容3") == "标记3"     # 最新的保留
