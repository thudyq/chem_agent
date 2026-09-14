# -*- coding: utf-8 -*-
"""tests/test_app_hardening.py — R9/R10/R11/R13 四组应用层加固。

| 项 | 内容 |
|---|---|
| R9 | CORS 不再 `allow_origins=["*"]`，只放行本机调试 + 自己的公开地址 |
| R10 | 统一安全响应头（禁 iframe 套框 / nosniff / Referrer-Policy） |
| R11 | 会话 id 必须是**服务端下发过**的（客户端自造的忽略并换新） |
| R13 | 慢活（管线、视觉识图、LaTeX 编译、起标题）必须离开事件循环 |

★ R13 的核心用例是**行为**验证：一边跑一个 0.6 秒的慢请求，一边发一个快请求，
断言快请求**没有被挡住**（修复前它要等满 0.6 秒）。源码级接线检查只是补充。
"""

import re
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api
from core import credentials, web_api

_ROOT = Path(__file__).resolve().parents[1]
SID = "a" * 32
KEY_HEADERS = {"X-Chem-Api-Key": "sk-test-key-123456",
               "X-Chem-Model": "deepseek-flash",
               "X-Chem-Base-Url": "https://api.deepseek.com/v1"}


# ---------------------------------------------------------------- 公共夹具

@pytest.fixture
def answered(monkeypatch):
    """管线 mock：记录收到的 question，返回固定文本。"""
    seen = {}

    def fake_pq(question, **kw):
        seen["question"] = question
        return "苯的结构式如下：C6H6"

    monkeypatch.setattr(web_api, "process_question", fake_pq)
    monkeypatch.setattr(web_api, "diaglog", type("D", (), {
        "log_request": staticmethod(lambda *a, **k: None)})())
    monkeypatch.setattr(web_api, "answer_cache", type("C", (), {
        "store": staticmethod(lambda *a, **k: None)})())
    return seen


@pytest.fixture
def web(tmp_path, monkeypatch, answered):
    """独立 app（只装 web router），会话目录落 tmp_path。"""
    app = FastAPI()
    app.include_router(web_api.router)
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", tmp_path / "web_sessions")
    monkeypatch.setattr(web_api, "RATE_LIMIT_PER_MINUTE", 0)
    web_api._reset_rate_limit_for_tests()
    web_api._session_dir(SID)          # R11：登记这个 id（= 服务端下发过）
    with TestClient(app) as client:
        yield client
    web_api._reset_rate_limit_for_tests()


def _payload(session_id=SID, stream=False):
    return {"messages": [{"role": "user", "content": "画出苯"}],
            "stream": stream, "session_id": session_id}


# ---------------------------------------------------------------- R9 CORS

class TestCorsOrigins:
    def test_default_has_no_wildcard(self):
        """默认绝不能是 `*`（那正是 R9 要修的）。"""
        origins = api._cors_origins()
        assert "*" not in origins
        assert "http://127.0.0.1:8000" in origins

    def test_includes_own_public_base_url(self, monkeypatch):
        monkeypatch.setattr(api, "settings", type("S", (), {
            "service": type("V", (), {"public_base_url": "https://chem.example"})()})())
        monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
        assert "https://chem.example" in api._cors_origins()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CORS_ALLOW_ORIGINS",
                           "https://a.example, https://b.example/")
        origins = api._cors_origins()
        assert origins == ["https://a.example", "https://b.example"]

    def test_preflight_blocked_for_foreign_origin(self):
        """外站发预检 → 不能拿到 access-control-allow-origin。"""
        r = TestClient(api.app).options(
            "/api/chat", headers={"Origin": "https://evil.example",
                                  "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}

    def test_preflight_allowed_for_own_origin(self):
        r = TestClient(api.app).options(
            "/api/chat", headers={"Origin": "http://127.0.0.1:8000",
                                  "Access-Control-Request-Method": "POST"})
        assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"


# ---------------------------------------------------------------- R10 安全响应头

class TestSecurityHeaders:
    @pytest.mark.parametrize("path", ["/chat", "/api/web-config"])
    def test_headers_present(self, path):
        r = TestClient(api.app).get(path)
        assert r.headers.get("x-frame-options") == "DENY"
        assert "frame-ancestors 'none'" in r.headers.get("content-security-policy", "")
        assert r.headers.get("x-content-type-options") == "nosniff"
        assert r.headers.get("referrer-policy") == "no-referrer"

    def test_only_minimal_csp_is_used(self):
        """只加**最小 CSP**：完整资源 CSP 会与内联 script/style 的单文件页面冲突。"""
        csp = TestClient(api.app).get("/chat").headers["content-security-policy"]
        assert "default-src" not in csp and "script-src" not in csp

    def test_headers_on_error_responses_too(self):
        r = TestClient(api.app).get("/api/session/nonexistent/aaaa.png")
        assert r.status_code == 404
        assert r.headers.get("x-frame-options") == "DENY"


# ---------------------------------------------------------------- R11 会话 id 归属

class TestSessionOwnership:
    def test_unknown_id_is_replaced(self, web):
        """客户端自造的 id → 服务端换一个新的（不再照抄）。"""
        r = web.post("/api/chat", json=_payload(session_id="b" * 32),
                     headers=KEY_HEADERS)
        assert r.status_code == 200
        assert r.json()["session_id"] != "b" * 32
        assert re.fullmatch(r"[0-9a-f]{32}", r.json()["session_id"])

    def test_registered_id_is_kept(self, web):
        """服务端下发过的 id → 照常沿用（前端第二次请求走的就是这条路）。"""
        r = web.post("/api/chat", json=_payload(), headers=KEY_HEADERS)
        assert r.json()["session_id"] == SID

    def test_selfmade_id_does_not_squat_a_directory(self, web):
        """自造 id 不能凭空建出目录（否则等于可以抢占别人的号码）。"""
        web.post("/api/chat", json=_payload(session_id="c" * 32),
                 headers=KEY_HEADERS)
        assert not (web_api._SESSIONS_DIR / ("c" * 32)).exists()

    def test_first_request_then_reuse_roundtrip(self, web):
        """模拟前端真实流程：第一次不带 id → 拿到 id → 第二次带回来仍被接受。"""
        first = web.post("/api/chat", json=_payload(session_id=""),
                         headers=KEY_HEADERS).json()["session_id"]
        second = web.post("/api/chat", json=_payload(session_id=first),
                          headers=KEY_HEADERS).json()["session_id"]
        assert first == second

    def test_seq_invalid_format_still_replaced(self, web):
        r = web.post("/api/chat", json=_payload(session_id="../../etc"),
                     headers=KEY_HEADERS)
        assert r.json()["session_id"] != "../../etc"

    def test_restart_does_not_invalidate_existing_ids(self, web):
        """判据是"目录已存在" → 重启后老会话仍可用（纯内存集合做不到这点）。"""
        # SID 的目录由 fixture 预先建好，等价于"上一个进程下发过"
        assert (web_api._SESSIONS_DIR / SID).is_dir()
        r = web.post("/api/chat", json=_payload(), headers=KEY_HEADERS)
        assert r.json()["session_id"] == SID


# ---------------------------------------------------------------- R13 事件循环不被阻塞

class TestEventLoopNotBlocked:
    def _fast_request_while_slow_is_held(self, web, monkeypatch, slow_attr):
        """慢请求**确实还在跑**时，快请求必须能返回。

        ★ 用 Event 精确控制时序，**不用 sleep 竞速**：
        最初写的是"慢函数 sleep 0.6 秒 + 断言快请求 < 0.4 秒"，结果在压测里
        12 轮抖 1 次（`assert 1.08 < 0.4`）——那是**测试自身不稳**，不是产品回归。
        现在改成：慢函数一进入就 `entered.set()` 并阻塞在 `release` 上，我们等到
        `entered` 再发快请求，断言它**在 release 之前**返回；`Timer` 兜底放行，
        所以真回归时不会把测试挂死，而是以清晰的断言失败收场。
        """
        entered = threading.Event()
        release = threading.Event()

        def slow(*a, **k):
            entered.set()
            release.wait(10)                  # 一直占着这次请求，直到我们放行
            return "" if slow_attr == "_prepare_answer" else "慢答案"

        monkeypatch.setattr(web_api, slow_attr, slow)
        slow_done = threading.Event()

        def run_slow():
            try:
                web.post("/api/chat", json=_payload(), headers=KEY_HEADERS)
            finally:
                slow_done.set()

        t = threading.Thread(target=run_slow)
        t.start()
        watchdog = threading.Timer(10.0, release.set)   # 兜底：绝不挂死
        watchdog.start()
        try:
            assert entered.wait(5), "慢请求没能进入（测试自身的问题）"
            t0 = time.time()
            web.get("/api/web-config")        # 快请求
            elapsed = time.time() - t0
            assert not slow_done.is_set(), "慢请求已结束，这次断言不成立"
            assert elapsed < 1.0, \
                f"事件循环被 {slow_attr} 阻塞了（快请求耗时 {elapsed:.2f}s）"
        finally:
            release.set()
            watchdog.cancel()
            t.join(5)

    def test_fast_request_not_blocked_by_slow_pipeline(self, web, monkeypatch):
        """★ R13 核心：慢请求在跑时，另一个请求**不该等它**。

        修复前：`process_question` 同步跑在事件循环里 → 快请求会一直等到它结束。
        """
        self._fast_request_while_slow_is_held(web, monkeypatch,
                                              "process_question")

    def test_fast_request_not_blocked_by_slow_latex_compile(self, web, monkeypatch):
        """LaTeX 编译（`_prepare_answer`）同样必须在事件循环之外。"""
        self._fast_request_while_slow_is_held(web, monkeypatch, "_prepare_answer")

    def test_threadpool_keeps_user_credentials(self, web, monkeypatch):
        """★ 线程池必须看得到用户凭证（contextvar 会随 context 拷贝过去）。

        这是"网页填了 A、实际用 .env 的 B"事故的同类风险：
        把活儿挪进线程池时，凭证作用域绝不能丢。
        """
        seen = {}

        def fake_build(text, images, session_dir):
            seen["key"] = credentials.llm_config().api_key
            return "问题：" + text

        monkeypatch.setattr(web_api, "_build_question", fake_build)
        r = web.post("/api/chat", json=_payload(), headers=KEY_HEADERS)
        assert r.status_code == 200
        assert seen["key"] == "sk-test-key-123456", "线程池里读不到用户凭证"

    def test_title_endpoint_also_offloads(self, web, monkeypatch):
        monkeypatch.setattr(web_api, "summarize_title", lambda q: "标题")
        r = web.post("/api/title", json={"question": "画出苯"},
                     headers=KEY_HEADERS)
        assert r.status_code == 200 and r.json()["title"] == "标题"


def test_slow_calls_use_threadpool():
    """接线检查：六处慢活都必须走 `run_in_threadpool`（漏一处就前功尽弃）。"""
    web_src = (_ROOT / "core/web_api.py").read_text(encoding="utf-8")
    for pat in (r"await run_in_threadpool\(_build_question",
                r"await run_in_threadpool\(\s*\n\s*process_question",
                r"await run_in_threadpool\(_prepare_answer",
                r"await run_in_threadpool\(summarize_title"):
        assert re.search(pat, web_src), f"web_api 缺少接线: {pat}"
    api_src = (_ROOT / "api.py").read_text(encoding="utf-8")
    for pat in (r"await run_in_threadpool\(\s*\n\s*_build_question",
                r"await run_in_threadpool\(\s*\n\s*process_question",
                r"await run_in_threadpool\(build_attachments"):
        assert re.search(pat, api_src), f"api.py 缺少接线: {pat}"
