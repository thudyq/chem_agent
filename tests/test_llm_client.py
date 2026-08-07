# -*- coding: utf-8 -*-
"""core/llm_client.py 思考参数渐进降级链与回退模型测试（不访问网络）。

运行: python -m pytest tests/test_llm_client.py -v
"""

import types

import pytest

import core.llm_client as lc


def _cfg(**kw):
    base = dict(
        api_key="k", base_url="http://x", model_name="m",
        is_configured=True, thinking_mode="", reasoning_effort="",
        fallback_model_name="", temperature=0.2, max_tokens=100,
        timeout=1, retries=2, max_concurrent=1,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture
def fake_env(monkeypatch):
    """替换 llm_client 全局 settings 的入口，并跳过 backoff 的 sleep。"""
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)

    def _use(cfg):
        monkeypatch.setattr(lc, "settings", types.SimpleNamespace(llm=cfg))
    return _use


def _stream_recorder(monkeypatch, outcomes):
    """伪造 _stream_chat：按序返回 (content, finish_reason, reasoning_chars)，
    记录每次调用的 payload 快照。"""
    calls = []

    def fake(url, headers, payload, on_piece=None):
        calls.append(dict(payload))
        return outcomes[min(len(calls), len(outcomes)) - 1]

    monkeypatch.setattr(lc, "_stream_chat", fake)
    return calls


class TestThinkingStages:
    def test_disabled_single_stage(self):
        assert lc._thinking_stages(_cfg(thinking_mode="disabled")) == [
            ("disabled", None)]

    def test_enabled_low_degrades_to_disabled(self):
        assert lc._thinking_stages(
            _cfg(thinking_mode="enabled", reasoning_effort="low")) == [
            ("enabled", "low"), ("disabled", None)]

    def test_enabled_high_steps_down(self):
        assert lc._thinking_stages(
            _cfg(thinking_mode="enabled", reasoning_effort="high")) == [
            ("enabled", "high"), ("enabled", "low"), ("disabled", None)]

    def test_unset_means_api_default_then_degrades(self):
        assert lc._thinking_stages(_cfg()) == [
            (None, None), ("enabled", "low"), ("disabled", None)]

    def test_enabled_without_effort(self):
        assert lc._thinking_stages(_cfg(thinking_mode="enabled")) == [
            ("enabled", None), ("enabled", "low"), ("disabled", None)]


class TestAskLlmDegradation:
    def test_reasoning_only_response_degrades_to_disabled(
            self, fake_env, monkeypatch):
        fake_env(_cfg(thinking_mode="enabled", reasoning_effort="low"))
        calls = _stream_recorder(
            monkeypatch, [(None, "stop", 5000), ("答案", "stop", 0)])
        assert lc.ask_llm("q", system_prompt="x", retries=2) == "答案"
        assert len(calls) == 2
        assert calls[0]["thinking"] == {"type": "enabled"}
        assert calls[0]["reasoning_effort"] == "low"
        assert calls[1]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in calls[1]

    def test_fallback_model_forces_thinking_disabled(
            self, fake_env, monkeypatch):
        fake_env(_cfg(thinking_mode="enabled", reasoning_effort="low",
                      fallback_model_name="m2"))
        calls = _stream_recorder(
            monkeypatch,
            [(None, "stop", 5000), (None, "stop", 3000), ("答案", "stop", 0)])
        assert lc.ask_llm("q", system_prompt="x", retries=2) == "答案"
        assert [c["model"] for c in calls] == ["m", "m", "m2"]
        assert calls[-1]["thinking"] == {"type": "disabled"}

    def test_disabled_config_single_stage_retries(
            self, fake_env, monkeypatch):
        fake_env(_cfg(thinking_mode="disabled"))
        calls = _stream_recorder(
            monkeypatch, [(None, "stop", 100), ("答案", "stop", 0)])
        assert lc.ask_llm("q", system_prompt="x", retries=2) == "答案"
        assert len(calls) == 2
        assert all(c.get("thinking") == {"type": "disabled"} for c in calls)

    def test_all_stages_fail_returns_none(self, fake_env, monkeypatch):
        fake_env(_cfg(thinking_mode="enabled", reasoning_effort="low"))
        _stream_recorder(monkeypatch, [(None, "stop", 5000)])
        assert lc.ask_llm("q", system_prompt="x", retries=1) is None
