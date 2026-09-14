# -*- coding: utf-8 -*-
"""tests/test_capabilities.py — 端点能力表（学习 + 记忆）单元测试。

能力表**不写死厂商映射**，
只记录"行为判定"的结论，维度到**被接受的取值集合**。
"""

import pytest

from core import capabilities
from core.config import EFFORT_HIGH, EFFORT_LOW, EFFORT_MAX, EFFORT_MEDIUM


@pytest.fixture(autouse=True)
def _clean():
    capabilities.reset_for_tests()
    yield
    capabilities.reset_for_tests()


# ---------------------------------------------------------------- 键

def test_key_contains_host_model_and_key_fingerprint():
    k = capabilities.make_key("https://open.bigmodel.cn/api/paas/v4",
                              "glm-5.3-flash", "sk-secret-xyz")
    assert "open.bigmodel.cn" in k
    assert "glm-5.3-flash" in k
    assert "sk-secret-xyz" not in k          # ★ 绝不出现密钥明文


def test_key_isolated_by_host_and_key():
    a = capabilities.make_key("https://a.example/v1", "m", "sk-1")
    b = capabilities.make_key("https://b.example/v1", "m", "sk-1")   # 换 host
    c = capabilities.make_key("https://a.example/v1", "m", "sk-2")   # 换 key
    d = capabilities.make_key("https://a.example/v1", "m2", "sk-1")  # 换 model
    assert len({a, b, c, d}) == 4


def test_key_without_api_key_is_anonymous_bucket():
    assert capabilities.make_key("https://a.example/v1", "m", "") == \
           capabilities.make_key("https://a.example/v1", "m", "")


# ---------------------------------------------------------------- toggle 学习

def test_force_on_learned_from_disabled_rejection():
    """GLM 病例：`thinking=disabled` 被拒 + 摘掉后成功 → 端点强制思考。"""
    k = capabilities.make_key("https://glm.example/v1", "glm-5.3-flash", "sk-x")
    capabilities.note_field_rejected(k, "thinking", "disabled")
    cap = capabilities.get(k)
    assert cap.toggle == capabilities.TOGGLE_FORCE_ON
    assert cap.allows_off() is False
    assert cap.allows_on() is True


def test_unsupported_learned_from_enabled_rejection():
    """纯文本模型：`thinking=enabled` 被拒 + 摘掉后成功 → 不支持思考。"""
    k = capabilities.make_key("https://plain.example/v1", "gpt-4o", "sk-x")
    capabilities.note_field_rejected(k, "thinking", "enabled")
    cap = capabilities.get(k)
    assert cap.toggle == capabilities.TOGGLE_UNSUPPORTED
    assert cap.allows_on() is False


def test_both_when_disabled_accepted():
    k = capabilities.make_key("https://ds.example/v1", "deepseek-flash", "sk-x")
    capabilities.note_accepted(k, "thinking", "disabled")
    cap = capabilities.get(k)
    assert cap.toggle == capabilities.TOGGLE_BOTH
    assert cap.allows_off() is True


def test_unknown_by_default():
    k = capabilities.make_key("https://new.example/v1", "brand-new", "sk-x")
    cap = capabilities.get(k)
    assert cap.toggle == capabilities.TOGGLE_UNKNOWN
    assert cap.allows_off() is None      # 未知 → 交给请求去试
    assert cap.allows_on() is None


# ---------------------------------------------------------------- effort 取值级

def test_effort_value_removed_on_rejection():
    """★ Gemini 病例：拒 `medium`、接受 `low`/`high`（取值级别的不支持）。"""
    k = capabilities.make_key(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3.7-flash", "sk-x")
    capabilities.note_field_rejected(k, "reasoning_effort", EFFORT_MEDIUM)
    cap = capabilities.get(k)
    assert cap.supports_effort(EFFORT_MEDIUM) is False
    assert cap.supports_effort(EFFORT_LOW) is True
    assert cap.supports_effort(EFFORT_HIGH) is True


def test_best_effort_at_most_prefers_lower():
    """用户选 medium 但端点只认 low/high → 就近换到 low（不超过用户意图）。"""
    k = capabilities.make_key("https://g.example/v1", "gm", "sk-x")
    capabilities.note_field_rejected(k, "reasoning_effort", EFFORT_MEDIUM)
    assert capabilities.get(k).best_effort_at_most(EFFORT_MEDIUM) == EFFORT_LOW


def test_best_effort_at_most_falls_back_up_when_all_lower_rejected():
    """low/medium 都被拒但 high 可用 → 退到 high（总比不发参数更接近意图）。"""
    k = capabilities.make_key("https://g2.example/v1", "gm2", "sk-x")
    capabilities.note_field_rejected(k, "reasoning_effort", EFFORT_LOW)
    capabilities.note_field_rejected(k, "reasoning_effort", EFFORT_MEDIUM)
    assert capabilities.get(k).best_effort_at_most(EFFORT_MEDIUM) == EFFORT_HIGH


def test_best_effort_returns_none_when_all_rejected():
    k = capabilities.make_key("https://g3.example/v1", "gm3", "sk-x")
    for v in (EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_MAX):
        capabilities.note_field_rejected(k, "reasoning_effort", v)
    cap = capabilities.get(k)
    assert cap.best_effort_at_most(EFFORT_HIGH) is None
    assert cap.supports_effort(EFFORT_LOW) is False


def test_effort_all_is_optimistic_before_any_rejection():
    k = capabilities.make_key("https://new2.example/v1", "m", "sk-x")
    assert capabilities.get(k).supports_effort(EFFORT_MAX) is True


# ---------------------------------------------------------------- 其他字段

def test_max_tokens_rejection_recorded():
    k = capabilities.make_key("https://mt.example/v1", "m", "sk-x")
    capabilities.note_field_rejected(k, "max_tokens", None)
    assert capabilities.get(k).max_tokens_ok is False


def test_observed_thinking_only_upgrades():
    """观察到思考内容后不会被后续"没看到"改判。"""
    k = capabilities.make_key("https://ob.example/v1", "m", "sk-x")
    capabilities.note_success(k, observed_thinking=True)
    assert capabilities.get(k).observed_thinking is True
    capabilities.note_success(k, observed_thinking=False)
    assert capabilities.get(k).observed_thinking is True


# ---------------------------------------------------------------- 生命周期

def test_peek_does_not_create():
    k = capabilities.make_key("https://peek.example/v1", "m", "sk-x")
    assert capabilities.peek(k) is None
    assert capabilities.snapshot() == {}


def test_reset_clears_table():
    k = capabilities.make_key("https://r.example/v1", "m", "sk-x")
    capabilities.note_field_rejected(k, "thinking", "disabled")
    assert capabilities.snapshot()
    capabilities.reset_for_tests()
    assert capabilities.snapshot() == {}


def test_ttl_expiry_treated_as_new(monkeypatch):
    """TTL 过期后按新记录对待（自愈：只是重新学一次）。"""
    k = capabilities.make_key("https://ttl.example/v1", "m", "sk-x")
    capabilities.note_field_rejected(k, "thinking", "disabled")
    # 把时间推到 TTL 之后
    monkeypatch.setattr(capabilities, "_now",
                        lambda: __import__("time").time()
                        + capabilities.TTL_SECONDS + 10)
    assert capabilities.peek(k) is None
    assert capabilities.get(k).toggle == capabilities.TOGGLE_UNKNOWN


def test_lru_eviction_keeps_table_bounded():
    """超出 LRU 上限时淘汰最旧条目（表不会无界增长——缺陷 D）。"""
    keys = [capabilities.make_key(f"https://h{i}.example/v1", "m", "sk-x")
            for i in range(capabilities.LRU_LIMIT + 50)]
    for k in keys:
        capabilities.note_field_rejected(k, "thinking", "disabled")
    assert len(capabilities.snapshot()) <= capabilities.LRU_LIMIT + 1


def test_describe_shape_for_frontend():
    k = capabilities.make_key("https://d.example/v1", "m", "sk-x")
    capabilities.note_field_rejected(k, "reasoning_effort", EFFORT_MEDIUM)
    d = capabilities.get(k).describe()
    assert d["supports_effort"] is True
    assert EFFORT_MEDIUM not in d["effort_values"]
    assert EFFORT_LOW in d["effort_values"]
    assert "toggle" in d and "observed_thinking" in d
