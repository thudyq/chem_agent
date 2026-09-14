# -*- coding: utf-8 -*-
"""tests/test_llm_client.py — core/llm_client.py 测试：思考参数映射、降级链、摘字段行为判定（不访问网络）。

运行: python -m pytest tests/test_llm_client.py -v
"""

import types

import pytest

import core.llm_client as lc
from core import capabilities, credentials
from core.config import EFFORT_HIGH, EFFORT_LOW, EFFORT_MAX, EFFORT_MEDIUM


def _cfg(**kw):
    """完整的 llm 配置替身（ask_llm 现在会读 thinking_default/effort_default）。"""
    base = dict(
        api_key="k", base_url="http://x", model_name="m",
        is_configured=True, thinking_default="on", effort_default="low",
        fallback_model_name="", temperature=0.2, max_tokens=100,
        timeout=1, retries=2, max_concurrent=1,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.fixture
def fake_env():
    """替换凭证层基准配置 + 清空能力表 + 跳过 backoff 的 sleep。

    注：`llm_client` 不再 import `settings`（配置全部经 `core.credentials`），
    因此测试只通过 `set_base_settings_for_tests` 注入替身。
    """
    lc.time.sleep = lambda s: None          # 跳过 backoff（模块属性，退出即随进程恢复）
    capabilities.reset_for_tests()

    def _use(cfg):
        credentials.set_base_settings_for_tests(
            types.SimpleNamespace(llm=cfg, vision=credentials.base_settings().vision))
    yield _use
    credentials.set_base_settings_for_tests(None)
    credentials._reset_semaphores_for_tests()
    capabilities.reset_for_tests()


def _stream_recorder(monkeypatch, outcomes):
    """伪造 _stream_chat：按序返回 (content, finish_reason, reasoning_chars,
    observed_reasoning)，并记录每次调用的 payload 快照。"""
    calls = []

    def fake(url, headers, payload, on_piece=None):
        calls.append(dict(payload))
        content, finish, rchars = outcomes[min(len(calls), len(outcomes)) - 1]
        return content, finish, rchars, bool(rchars)

    monkeypatch.setattr(lc, "_stream_chat", fake)
    return calls


def _http_error_recorder(monkeypatch, plan):
    """伪造 `_stream_chat`：按 plan 逐步返回（可模拟 400 → 摘字段 → 成功）。

    每一步二选一：
      (status_int, body_str)                  → 抛 `_HttpFailure`（HTTP 错误）
      (content, finish_reason, rchars)        → 正常返回（第三位是整数）
    最后一步会被重复使用（超出 plan 后一直沿用）。
    """
    calls = []

    def fake(url, headers, payload, on_piece=None):
        calls.append(dict(payload))
        step = plan[min(len(calls), len(plan)) - 1]
        if len(step) == 2:
            raise lc._HttpFailure(step[0], step[1])
        content, finish, rchars = step
        return content, finish, rchars, bool(rchars)

    monkeypatch.setattr(lc, "_stream_chat", fake)
    return calls


class TestEffortStages:
    """降级链：严格单调下降（只降不升）。"""

    def test_from_max(self):
        assert lc._effort_stages(True, EFFORT_MAX) == [
            (True, "max"), (True, "high"), (True, "medium"), (True, "low"),
            (False, None)]

    def test_from_high(self):
        assert lc._effort_stages(True, EFFORT_HIGH) == [
            (True, "high"), (True, "medium"), (True, "low"), (False, None)]

    def test_from_medium(self):
        assert lc._effort_stages(True, EFFORT_MEDIUM) == [
            (True, "medium"), (True, "low"), (False, None)]

    def test_from_low(self):
        assert lc._effort_stages(True, EFFORT_LOW) == [
            (True, "low"), (False, None)]

    def test_off_has_single_stage(self):
        assert lc._effort_stages(False, None) == [(False, None)]

    def test_unknown_effort_starts_at_low(self):
        assert lc._effort_stages(True, "nonsense") == [(True, "low"), (False, None)]


class TestPayloadMapping:
    """(开关, 强度) → 请求字段（二维映射）。"""

    def _payload(self, on, effort, dropped=None):
        from core.capabilities import Capability
        return lc._build_payload("m", [], on, effort, 32768, 0.2,
                                 Capability(), dropped or set())

    def test_off_sends_disabled_and_temperature(self):
        p = self._payload(False, None)
        assert p["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in p
        assert p["temperature"] == 0.2        # 关思考时才发 temperature

    def test_on_sends_enabled_and_effort_without_temperature(self):
        for eff in (EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_MAX):
            p = self._payload(True, eff)
            assert p["thinking"] == {"type": "enabled"}
            assert p["reasoning_effort"] == eff
            assert "temperature" not in p     # 思考模式下无效

    def test_dropped_fields_are_omitted(self):
        p = self._payload(True, EFFORT_LOW, dropped={"thinking", "max_tokens"})
        assert "thinking" not in p
        assert "max_tokens" not in p
        assert p["reasoning_effort"] == "low"   # 保留仍可用的档位


class TestBehaviorJudgement:
    """★ 核心：400 判定只看"摘掉后能否成功"，不解析错误文本。"""

    def test_chinese_error_still_strips_field(self, fake_env, monkeypatch):
        """GLM 病例：中文报错、不含英文字段名，仍须摘字段并成功。"""
        fake_env(_cfg())
        glm_error = ('{"error":{"code":"1210","message":"该模型始终思考，'
                     '不支持关闭思考；请使用 low、high 或 max。"}}')
        calls = _http_error_recorder(monkeypatch, [
            (400, glm_error),                 # 第一次：拒绝 thinking=disabled
            ("答案", "stop", 0),              # 摘掉 thinking 后成功
        ])
        out = lc.ask_llm("q", system_prompt="x", thinking="disabled", retries=1)
        assert out == "答案"
        assert calls[0].get("thinking") == {"type": "disabled"}
        assert "thinking" not in calls[1]      # 摘掉了
        # 能力表学到了"不能关"
        from core.capabilities import make_key
        cap = capabilities.peek(make_key("http://x", "m", "k"))
        assert cap is not None and cap.allows_off() is False

    def test_effort_field_preserved_when_thinking_rejected(
            self, fake_env, monkeypatch):
        """GLM 提示"请使用 low/high/max"→ 摘 thinking 时**保留 reasoning_effort**。"""
        fake_env(_cfg())
        calls = _http_error_recorder(monkeypatch, [
            (400, '{"error":{"message":"该模型始终思考，不支持关闭思考"}}'),
            ("答案", "stop", 0),
        ])
        # 用户选"开 + 高"：端点拒 thinking=enabled 时，effort 必须留着
        out = lc.ask_llm("q", system_prompt="x", thinking="enabled",
                         effort="high", retries=1)
        assert out == "答案"
        assert calls[0]["thinking"] == {"type": "enabled"}
        assert calls[1].get("reasoning_effort") == "high"   # ★ 保留档位
        assert "thinking" not in calls[1]

    def test_real_error_not_masked(self, fake_env, monkeypatch):
        """真错误（额度不足）：摘字段重试仍失败 → 原样报错，不当作档位问题。"""
        fake_env(_cfg())
        _http_error_recorder(monkeypatch, [
            (400, '{"error":{"message":"Insufficient Balance"}}'),
            (400, '{"error":{"message":"Insufficient Balance"}}'),
            (400, '{"error":{"message":"Insufficient Balance"}}'),
            (400, '{"error":{"message":"Insufficient Balance"}}'),
        ])
        out = lc.ask_llm("q", system_prompt="x", retries=1)
        assert out is None

    def test_capability_memo_prevents_relearning(self, fake_env, monkeypatch):
        """能力表已记住"不能关"后，第二次请求**直接不发** thinking。"""
        fake_env(_cfg())
        # 第一次：学到 force_on
        _http_error_recorder(monkeypatch, [
            (400, "该模型始终思考"), ("答案", "stop", 0)])
        lc.ask_llm("q", system_prompt="x", thinking="disabled", retries=1)
        # 第二次：应直接不发 thinking（无 400）
        calls2 = _stream_recorder(monkeypatch, [("答案2", "stop", 0)])
        out = lc.ask_llm("q", system_prompt="x", thinking="disabled", retries=1)
        assert out == "答案2"
        assert "thinking" not in calls2[0]


class TestDegradation:
    def test_reasoning_only_content_degrades(self, fake_env, monkeypatch):
        """只想不答 → 降档重试（不再换模型）。"""
        fake_env(_cfg(thinking_default="on", effort_default="high"))
        calls = _stream_recorder(
            monkeypatch, [(None, "stop", 5000), ("答案", "stop", 0)])
        assert lc.ask_llm("q", system_prompt="x", retries=2) == "答案"
        assert len(calls) == 2
        assert calls[0]["reasoning_effort"] == "high"
        assert calls[1]["reasoning_effort"] == "medium"   # 降到下一档

    def test_length_truncation_always_degrades(self, fake_env, monkeypatch):
        """★ 缺陷 F：finish_reason=length 一律降档（不浪费同档重试）。"""
        fake_env(_cfg(thinking_default="on", effort_default="low"))
        calls = _stream_recorder(
            monkeypatch, [("短", "length", 100), ("答案", "stop", 0)])
        assert lc.ask_llm("q", system_prompt="x", retries=3) == "答案"
        assert len(calls) == 2, "同预算重试无意义，应直接降档"
        assert calls[0]["thinking"] == {"type": "enabled"}
        assert calls[1]["thinking"] == {"type": "disabled"}   # low → 关

    def test_all_stages_fail_returns_none(self, fake_env, monkeypatch):
        fake_env(_cfg(thinking_default="on", effort_default="high"))
        _stream_recorder(monkeypatch, [(None, "stop", 5000)])
        assert lc.ask_llm("q", system_prompt="x", retries=1) is None

    def test_no_model_switch_ever(self, fake_env, monkeypatch):
        """单模型：整个降级链里 model 名恒为同一个（不再有回退/升级模型）。"""
        fake_env(_cfg(thinking_default="on", effort_default="max"))
        calls = _stream_recorder(
            monkeypatch, [(None, "stop", 9000), (None, "stop", 9000),
                          (None, "stop", 9000), ("答案", "stop", 0)])
        lc.ask_llm("q", system_prompt="x", retries=1)
        assert {c["model"] for c in calls} == {"m"}


class TestResultMetadata:
    """LLMResult：实际档位与"静默改写"提示（供诚实上报）。"""

    def test_effort_effective_reported(self, fake_env, monkeypatch):
        fake_env(_cfg(thinking_default="on", effort_default="low"))
        _stream_recorder(monkeypatch, [("答案", "stop", 0)])
        res = lc.ask_llm("q", system_prompt="x", return_result=True)
        assert res.text == "答案"
        assert res.effort_requested == "low"
        assert res.effort_effective == "low"
        assert res.downgraded is False
        assert res.notice == ""

    def test_forced_thinking_reports_notice(self, fake_env, monkeypatch):
        """用户选关、端点强制思考 → 必须给出提示。"""
        fake_env(_cfg())
        _http_error_recorder(monkeypatch, [
            (400, "该模型始终思考，不支持关闭思考"), ("答案", "stop", 0)])
        res = lc.ask_llm("q", system_prompt="x", thinking="disabled",
                         retries=1, return_result=True)
        assert res.text == "答案"
        assert res.downgraded is True
        assert res.notice, "应给出用户可见提示"
        assert "思考" in res.notice

    def test_downgrade_due_to_failure_reported(self, fake_env, monkeypatch):
        fake_env(_cfg(thinking_default="on", effort_default="high"))
        _stream_recorder(monkeypatch, [(None, "stop", 5000), ("答案", "stop", 0)])
        res = lc.ask_llm("q", system_prompt="x", retries=2, return_result=True)
        assert res.text == "答案"
        assert res.effort_effective == "medium"       # 实际降了一档
        assert res.downgraded is True

    def test_text_return_mode_still_works(self, fake_env, monkeypatch):
        fake_env(_cfg())
        _stream_recorder(monkeypatch, [("答案", "stop", 0)])
        assert lc.ask_llm("q", system_prompt="x") == "答案"


class TestCredentialResolution:
    """统一凭证解析：model=None 时用**用户指定的模型**，而不是服务器的。"""

    def test_user_model_wins(self, fake_env, monkeypatch):
        fake_env(_cfg(model_name="server-model"))
        calls = _stream_recorder(monkeypatch, [("答案", "stop", 0)])
        with credentials.user_credentials({"api_key": "sk-user",
                                           "model": "user-model"}):
            lc.ask_llm("q", system_prompt="x")
        assert calls[0]["model"] == "user-model"

    def test_server_model_used_without_credentials(self, fake_env, monkeypatch):
        fake_env(_cfg(model_name="server-model"))
        calls = _stream_recorder(monkeypatch, [("答案", "stop", 0)])
        lc.ask_llm("q", system_prompt="x")
        assert calls[0]["model"] == "server-model"
