# -*- coding: utf-8 -*-
"""tests/test_credentials.py — 每请求模型凭证与参数（BYOK）单元测试。

覆盖：请求头字段规范化 / 密钥指纹与脱敏 / contextvar 覆盖与还原（含子线程
传播语义）/ 主模型与思考参数合并（开关 × 强度 × 最大输出）/ 视觉组规则
（留空跟主模型、独立模型、绝不继承服务器视觉配置）/ 用户端点地址 SSRF 校验 /
按凭证分桶的并发信号量。
"""

import dataclasses
import threading

import pytest

from core import credentials
from core.config import settings


@pytest.fixture(autouse=True)
def _clean_state():
    credentials._reset_semaphores_for_tests()
    yield
    credentials._reset_semaphores_for_tests()
    credentials.set_base_settings_for_tests(None)


@pytest.fixture
def base_cfg():
    """注入基准配置替身（模拟服务器 .env），退出自动恢复。"""
    def _apply(llm=None, vision=None):
        base = dataclasses.replace(settings, llm=llm or settings.llm,
                                   vision=vision or settings.vision)
        credentials.set_base_settings_for_tests(base)
        return base

    yield _apply
    credentials.set_base_settings_for_tests(None)
    credentials._reset_semaphores_for_tests()


# ---------------------------------------------------------------- 规范化

def test_normalize_keeps_known_fields_only():
    out = credentials.normalize({
        "api_key": "  sk-abc  ",
        "base_url": "https://api.deepseek.com/v1/",
        "model": "deepseek-flash",
        "thinking": "off",
        "effort": "high",
        "max_tokens": "16384",
        "evil_field": "x",              # 未知键必须被丢弃
        "vision_api_key": "",           # 空值丢弃
    })
    assert out == {"api_key": "sk-abc",
                   "base_url": "https://api.deepseek.com/v1",
                   "model": "deepseek-flash",
                   "thinking": "off", "effort": "high", "max_tokens": "16384"}


def test_normalize_none_and_empty():
    assert credentials.normalize(None) == {}
    assert credentials.normalize({}) == {}


# ---------------------------------------------------------------- 脱敏

def test_redact_never_leaks_full_secret():
    key = "sk-1234567890abcdefghij"
    masked = credentials.redact(key)
    assert key not in masked
    assert masked.startswith("sk-123")
    assert masked.endswith("ghij")
    assert credentials.redact("") == "(空)"
    assert credentials.redact(None) == "(空)"
    assert credentials.redact("short") == "s***"


def test_fingerprint_stable_and_isolated():
    a = {"api_key": "sk-aaa", "base_url": "https://a.example/v1"}
    b = {"api_key": "sk-bbb", "base_url": "https://a.example/v1"}
    c = {"api_key": "sk-aaa", "base_url": "https://b.example/v1"}
    assert credentials.fingerprint(a) == credentials.fingerprint(dict(a))
    assert credentials.fingerprint(a) != credentials.fingerprint(b)
    assert credentials.fingerprint(a) != credentials.fingerprint(c)
    assert "sk-aaa" not in credentials.fingerprint(a)
    assert credentials.fingerprint(None) == "default"
    assert credentials.fingerprint({}) == "default"


# ---------------------------------------------------------------- 覆盖作用域

def test_user_credentials_scoped_and_restored():
    assert credentials.current() == {}
    with credentials.user_credentials({"api_key": "sk-1"}):
        assert credentials.current()["api_key"] == "sk-1"
        with credentials.user_credentials({"api_key": "sk-2"}):
            assert credentials.current()["api_key"] == "sk-2"
        assert credentials.current()["api_key"] == "sk-1"
    assert credentials.current() == {}


def test_user_credentials_isolated_between_threads():
    """并发关键：两个线程各自的覆盖不得互相可见（contextvars 语义）。"""
    seen = {}
    barrier = threading.Barrier(2)

    def worker(name, key):
        with credentials.user_credentials({"api_key": key}):
            barrier.wait(timeout=5)
            seen[name] = credentials.current().get("api_key")

    t1 = threading.Thread(target=worker, args=("a", "sk-a"))
    t2 = threading.Thread(target=worker, args=("b", "sk-b"))
    t1.start(); t2.start(); t1.join(5); t2.join(5)
    assert seen == {"a": "sk-a", "b": "sk-b"}


def test_child_thread_does_not_inherit_without_copy_context():
    """**重要前提**：新线程不会自动继承 contextvars。

    凡在子线程跑管线的入口必须显式 `contextvars.copy_context().run(...)`，
    否则线程内退回服务器 .env 配置（BYOK 失效）。已传播的入口：
    `api._sse_stream`、`web_api._sse_stream`、`app._run_with_timeout`。
    """
    box = {}

    def child():
        box["key"] = credentials.current().get("api_key")

    with credentials.user_credentials({"api_key": "sk-parent"}):
        t = threading.Thread(target=child)
        t.start(); t.join(5)
    assert box["key"] is None


def test_child_thread_inherits_with_copy_context():
    """显式传播后子线程能看到父线程的凭证（SSE 工作线程的写法）。"""
    import contextvars

    box = {}

    def child():
        box["key"] = credentials.current().get("api_key")

    with credentials.user_credentials({"api_key": "sk-parent"}):
        ctx = contextvars.copy_context()
        t = threading.Thread(target=lambda: ctx.run(child))
        t.start(); t.join(5)
    assert box["key"] == "sk-parent"


def test_apply_credentials_sets_without_reset():
    """`apply_credentials` 只设不重置（生成器里不能配对 reset，见其 docstring）。"""
    credentials.apply_credentials({"api_key": "sk-a", "model": "m-a"})
    assert credentials.current()["api_key"] == "sk-a"
    credentials.apply_credentials({"api_key": "sk-b"})
    assert credentials.current()["api_key"] == "sk-b"


def test_context_copy_isolates_set_from_sibling_copy():
    """**流式病例的根因**：`set` 只影响当前 Context，不影响"再拷贝一份"。

    Starlette 把 `StreamingResponse` 的生成器放进线程池**逐块**迭代，每次
    `next()` 都从请求任务上下文重新拷贝一份 Context。因此在第 1 次 `next()`
    里 `apply_credentials()` 写下的值，到第 N 次 `next()`（新拷贝）里并不存在。
    结论：凭证必须在**工作线程函数内部**设置，而不是在生成器迭代处设置。

    用一份全新 Context 当"请求任务上下文"，避免依赖测试线程的环境状态。
    """
    import contextvars

    def scenario():
        # 第 1 次迭代：拷贝请求上下文后在其中 set
        first = contextvars.copy_context()
        first.run(lambda: credentials.apply_credentials({"api_key": "sk-user"}))
        assert first.run(lambda: credentials.current().get("api_key")) == "sk-user"

        # 第 2 次迭代：从同一个请求上下文**重新拷贝**，看不到上一次的 set
        second = contextvars.copy_context()
        assert second.run(lambda: credentials.current().get("api_key")) is None

    contextvars.Context().run(scenario)


# ---------------------------------------------------------------- 主模型与思考参数

def test_llm_config_without_override_is_env_config():
    assert credentials.llm_config() is settings.llm


def test_thinking_setting_defaults_from_env():
    assert credentials.thinking_setting() in ("on", "off")
    assert credentials.effort_setting() in ("low", "medium", "high", "max")
    assert credentials.max_tokens_setting() >= 1


def test_user_overrides_model_and_thinking(base_cfg):
    import dataclasses
    base_cfg(llm=dataclasses.replace(settings.llm, thinking_default="on",
                                     effort_default="high", max_tokens=8192))
    with credentials.user_credentials({
            "api_key": "sk-user", "base_url": "https://u.example/v1",
            "model": "user-model", "thinking": "off", "max_tokens": "16384"}):
        assert credentials.user_model() == "user-model"
        assert credentials.thinking_setting() == "off"
        assert credentials.max_tokens_setting() == 16384
        cfg = credentials.llm_config()
        assert cfg.api_key == "sk-user"
        assert cfg.model_name == "user-model"
        assert cfg.base_url == "https://u.example/v1"
        assert cfg.max_tokens == 16384
        # temperature 等未涉及的项保持基准配置
        assert cfg.temperature == settings.llm.temperature


def test_partial_user_override_falls_back_to_base(base_cfg):
    """只填 Key：端点/模型/思考参数都用基准配置。"""
    with credentials.user_credentials({"api_key": "sk-user"}):
        cfg = credentials.llm_config()
    assert cfg.api_key == "sk-user"
    assert cfg.base_url == settings.llm.base_url
    assert cfg.model_name == settings.llm.model_name
    assert cfg.thinking_default == settings.llm.thinking_default


def test_effort_default_used_when_user_only_sets_thinking(base_cfg):
    import dataclasses
    base_cfg(llm=dataclasses.replace(settings.llm, effort_default="medium"))
    with credentials.user_credentials({"api_key": "sk-user", "thinking": "on"}):
        assert credentials.effort_setting() == "medium"


def test_effort_alias_normalized():
    with credentials.user_credentials({"api_key": "sk-u", "effort": "xhigh"}):
        assert credentials.effort_setting() == "high"
    with credentials.user_credentials({"api_key": "sk-u", "effort": "ultra"}):
        assert credentials.effort_setting() == "max"


def test_bad_max_tokens_falls_back():
    with credentials.user_credentials({"api_key": "sk-u", "max_tokens": "abc"}):
        assert credentials.max_tokens_setting() == settings.llm.max_tokens


# ---------------------------------------------------------------- 视觉组

def test_vision_without_override_is_env_config():
    assert credentials.vision_config() is settings.vision


def test_vision_falls_back_to_main_model(base_cfg):
    """用户只填主模型 → 视觉就用主模型（DeepSeek V4.1 自带视觉）。"""
    base_cfg(vision=dataclasses.replace(settings.vision, model_name="",
                                        base_url="", api_key=""))
    with credentials.user_credentials({"api_key": "sk-user", "model": "deepseek-flash",
                                       "base_url": "https://api.deepseek.com/v1"}):
        v = credentials.vision_config()
    assert v.model_name == "deepseek-flash"
    assert v.api_key == "sk-user"
    assert v.base_url == "https://api.deepseek.com/v1"
    assert credentials.vision_is_main() is True
    # 与主模型相同时，思考参数留空 = 复用主模型设置（由调用方取值）
    assert v.thinking == ""


def test_vision_independent_model(base_cfg):
    """用户填了独立视觉模型 → 用它；未给端点/Key 时沿用用户自己的主模型。"""
    with credentials.user_credentials({
            "api_key": "sk-user", "model": "kimi-k2",
            "base_url": "https://api.moonshot.cn/v1",
            "vision_model": "glm-5.3-flash"}):
        v = credentials.vision_config()
    assert v.model_name == "glm-5.3-flash"
    assert v.api_key == "sk-user"
    assert v.base_url == "https://api.moonshot.cn/v1"
    assert credentials.vision_is_main() is False
    assert v.thinking == "off"      # 独立模型默认关思考（识图要快要省）


def test_vision_independent_endpoint_and_key(base_cfg):
    with credentials.user_credentials({
            "api_key": "sk-user", "model": "kimi-k2",
            "base_url": "https://api.moonshot.cn/v1",
            "vision_model": "glm-5.3-flash",
            "vision_base_url": "https://open.bigmodel.cn/api/paas/v4",
            "vision_api_key": "sk-vision"}):
        v = credentials.vision_config()
    assert v.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert v.api_key == "sk-vision"


def test_vision_never_inherits_server_vision(base_cfg):
    """★ 安全约定：用户路径绝不继承服务器的视觉模型/端点/计费。"""
    import dataclasses
    base_cfg(vision=dataclasses.replace(
        settings.vision, model_name="server-vision-model",
        base_url="https://server.example/v1", api_key="sk-server-vision"))
    with credentials.user_credentials({"api_key": "sk-user", "model": "m1",
                                       "base_url": "https://user.example/v1"}):
        v = credentials.vision_config()
    assert v.model_name == "m1"                     # 用主模型，而不是 server-vision-model
    assert v.api_key == "sk-user"
    assert v.base_url == "https://user.example/v1"


def test_vision_explicit_thinking_override(base_cfg):
    with credentials.user_credentials({
            "api_key": "sk-u", "model": "m1", "vision_model": "v1",
            "vision_thinking": "on", "vision_effort": "high"}):
        v = credentials.vision_config()
    assert v.thinking == "on"
    assert v.effort == "high"


def test_vision_unconfigured_without_main_credentials(base_cfg):
    """用户在视觉/主模型凭证都不全时 → 判为未配置（不静默用服务器配置）。"""
    import dataclasses
    base_cfg(llm=dataclasses.replace(settings.llm, api_key="", model_name=""),
             vision=dataclasses.replace(settings.vision, model_name="srv",
                                        base_url="https://s/v1", api_key="sk-srv"))
    with credentials.user_credentials({"thinking": "on"}):
        assert credentials.vision_config().is_configured is False


# ---------------------------------------------------------------- base_url 校验

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/v1",
    "http://localhost:11434/v1",
    "http://10.0.0.5/v1",
    "http://192.168.1.10:8080/v1",
    "http://172.16.3.4/v1",
    "http://169.254.169.254/latest/meta-data",   # 云元数据
    "file:///etc/passwd",
    "not a url",
    "ftp://example.com/v1",
])
def test_client_host_allowed_blocks(url):
    ok, reason = credentials.client_host_allowed(url)
    assert ok is False, f"{url} 应被拒绝"
    assert reason


@pytest.mark.parametrize("url", [
    "https://api.deepseek.com/v1",
    "https://open.bigmodel.cn/api/paas/v4",
    "",
])
def test_client_host_allowed_accepts_public(url):
    ok, reason = credentials.client_host_allowed(url)
    assert ok is True, f"{url} 应被放行（原因：{reason}）"


def test_client_host_allowed_private_opt_in(monkeypatch):
    monkeypatch.setenv("WEB_ALLOW_PRIVATE_BASE_URL", "1")
    ok, _ = credentials.client_host_allowed("http://127.0.0.1:11434/v1")
    assert ok is True


# ---------------------------------------------------------------- 并发信号量

def test_semaphore_shared_per_credential_and_distinct_across_credentials():
    with credentials.user_credentials({"api_key": "sk-a"}):
        sem_a1 = credentials.llm_semaphore()
        sem_a2 = credentials.llm_semaphore()
    with credentials.user_credentials({"api_key": "sk-b"}):
        sem_b = credentials.llm_semaphore()
    assert sem_a1 is sem_a2
    assert sem_a1 is not sem_b


def test_semaphore_default_bucket_without_credentials():
    assert credentials.llm_semaphore() is credentials.llm_semaphore()


def test_session_is_reused():
    assert credentials.session() is credentials.session()
