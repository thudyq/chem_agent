# -*- coding: utf-8 -*-
"""tests/test_byok_end_to_end.py — BYOK 端到端验收（真实 HTTP → mock 上游）。

与 `test_web_api.py` 的分工
--------------------------
`test_web_api.py` 把 `process_question` **整个 mock 掉**，只验请求头→凭证
解析那一段；本文件**不 mock 管线**，让请求真的走完 `process_question` →
`llm_client.ask_llm` → HTTP，落在一个本机 mock 上游上，直接观察
**上游到底收到了哪个 model / 哪把 Key**。

为什么必须有这一层
------------------
用户实测：网页面板填 `deepseek-flash`，"测试连接"显示正确，但一提问
日志里走的却是服务器 `.env` 的 `gemini-3.7-flash`。根因在凭证的 Context
作用域（SSE 生成器跨迭代时 ContextVar 会丢，详见 Test-Method §21.5-5）——
**mock 掉管线的用例看不见这一类问题**，因为链路根本没走到真正读凭证的地方。
本文件是那个病例的直接回归锚点。

不联网、不花钱：上游是本机 `HTTPServer`。
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core import credentials, web_api

USER_KEY = "sk-user-byok-key"
USER_MODEL = "deepseek-flash"
SERVER_MODEL = "server-env-model"
SERVER_KEY = "sk-server-env-key"

# 不带任何标记的纯文本答案：让管线只发**一次**上游调用（不做标记校验/修正/
# PubChem 兜底），断言才干净。
ANSWER = "苯（Benzene）是最基本的芳香族化合物，分子式为 C6H6。"


class _QuietHTTPServer(HTTPServer):
    """mock 上游服务器：把"对端正常断开连接"当成噪声忽略掉。

    背景：`protocol_version = "HTTP/1.1"` 会**keep-alive**，
    处理器读完一个请求后会继续等**下一个请求行**；而客户端是产品里那个模块级
    `requests.Session`（连接池），测试跑完/回收时会把 socket 直接关掉 —— 服务端
    阻塞中的 `readline` 于是抛 `ConnectionResetError`（Windows 上是 WinError 10054）。

    `socketserver` 默认把这类异常当作"处理请求时出错"，打印**整段 traceback**
    （见 `BaseServer.handle_error`）。它**不影响测试结果**，但会插进 pytest 输出里
    看着像真失败 —— 实测 3 次里出现 1 次，属间歇性噪声。

    对端断开不是错误，静默；**其它异常照常打印**（不能因为消噪而掩盖真问题）。
    """

    def handle_error(self, request, client_address):
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError,
                            BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class _Upstream:
    """本机 mock 上游：记录每次请求收到的 model / Authorization / 路径。"""

    def __init__(self):
        self.calls = []
        calls = self.calls

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):                      # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                calls.append(SimpleNamespace(
                    model=body.get("model"),
                    auth=self.headers.get("Authorization", ""),
                    stream=bool(body.get("stream")),
                    path=self.path,
                ))
                if body.get("stream"):
                    frame = json.dumps(
                        {"choices": [{"index": 0, "delta": {"content": ANSWER},
                                      "finish_reason": "stop"}]},
                        ensure_ascii=False)
                    data = f"data: {frame}\n\n".encode("utf-8")
                    done = b"data: [DONE]\n\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                    self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    return
                payload = json.dumps({"choices": [{"index": 0, "message":
                    {"role": "assistant", "content": ANSWER},
                    "finish_reason": "stop"}]}, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):              # 静音
                pass

        self._srv = _QuietHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._srv.server_port}/v1"

    def stop(self):
        self._srv.shutdown()
        self._srv.server_close()


@pytest.fixture
def upstream(monkeypatch):
    """启动 mock 上游，并把**服务器侧**配置也指向它。

    服务器侧指向 mock 有两个作用：① `/v1`（清小搭）用例能观察到它发的是什么，
    不再真的打到 `.env` 里的线上端点（否则会烧用户额度）；② 任何"本该用用户
    凭证却回退到服务器配置"的泄漏都会被 mock 原样记录下来。
    """
    up = _Upstream()
    # 面板允许填内网地址（仅本进程；产品侧默认禁止，见 docs/deploy.md）
    monkeypatch.setenv("WEB_ALLOW_PRIVATE_BASE_URL", "1")

    import dataclasses
    from core.config import settings
    server_llm = dataclasses.replace(settings.llm, model_name=SERVER_MODEL,
                                     api_key=SERVER_KEY, base_url=up.base_url)
    credentials.set_base_settings_for_tests(
        dataclasses.replace(settings, llm=server_llm))
    try:
        yield up
    finally:
        credentials.set_base_settings_for_tests(None)
        up.stop()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """完整 app（`/api/chat` + `/v1/*`），会话目录落 tmp_path，不写诊断日志。"""
    import api

    monkeypatch.setattr(web_api, "_SESSIONS_DIR", tmp_path / "web_sessions")
    monkeypatch.setattr(web_api, "RATE_LIMIT_PER_MINUTE", 0)
    web_api._reset_rate_limit_for_tests()
    # R11 之后客户端给的 session_id 必须"服务端下发过"（= 会话目录已存在）；
    # 这些用例沿用固定 id，先把它登记上。
    web_api._session_dir("a" * 32)
    # 不往仓库 data/ 写诊断与回答缓存（保持测试无副作用）
    monkeypatch.setattr(api, "diaglog", SimpleNamespace(log_request=lambda *a, **k: None))
    monkeypatch.setattr(web_api, "diaglog",
                        SimpleNamespace(log_request=lambda *a, **k: None))
    monkeypatch.setattr(web_api, "answer_cache", SimpleNamespace(store=lambda *a, **k: None))
    monkeypatch.setattr("core.answer_cache.store", lambda *a, **k: None)
    with TestClient(api.app) as c:
        yield c
    web_api._reset_rate_limit_for_tests()


def _web_headers(model=USER_MODEL, key=USER_KEY, base_url=None, **extra):
    h = {"X-Chem-Api-Key": key, "X-Chem-Model": model}
    if base_url:
        h["X-Chem-Base-Url"] = base_url
    h.update(extra)
    return h


def _chat_body(text="苯是什么？", stream=False, session_id="a" * 32):
    return {"messages": [{"role": "user", "content": text}],
            "stream": stream, "session_id": session_id}


# ────────────────────────────────────────────────────────── 网页入口（BYOK）

def test_non_stream_uses_user_model_and_key(client, upstream):
    r = client.post("/api/chat",
                    json=_chat_body(),
                    headers=_web_headers(base_url=upstream.base_url))
    assert r.status_code == 200
    assert upstream.calls, "根本没有发出上游调用"
    assert {c.model for c in upstream.calls} == {USER_MODEL}
    assert all(USER_KEY in c.auth for c in upstream.calls), \
        "上游收到的不是用户的 Key"


def test_stream_uses_user_model_and_key(client, upstream):
    """★ 用户实测病例的直接回归：**流式**也必须用网页里填的模型/密钥。

    （`_sse_stream` 是生成器，凭证必须在 SSE 工作线程函数内部落地，见
    Test-Method §21.5-5。把 `apply_credentials` 挪回生成器迭代处即变红。）
    """
    r = client.post("/api/chat",
                    json=_chat_body(stream=True),
                    headers=_web_headers(base_url=upstream.base_url))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in r.text, "SSE 流没有正常收尾"
    assert upstream.calls, "流式路径根本没有发出上游调用"
    assert {c.model for c in upstream.calls} == {USER_MODEL}, \
        f"流式走错了模型：{[c.model for c in upstream.calls]}（服务器是 {SERVER_MODEL}）"
    assert all(USER_KEY in c.auth for c in upstream.calls)
    assert all(c.stream for c in upstream.calls)


def test_concurrent_streams_do_not_mix_credentials(client, upstream):
    """两个并发流式请求各自带不同模型 → 上游必须分别收到各自的模型。"""
    results = {}

    def ask(tag):
        r = client.post(
            "/api/chat", json=_chat_body(stream=True),
            headers=_web_headers(model=f"model-{tag}", key=f"sk-{tag}",
                                 base_url=upstream.base_url))
        results[tag] = r.status_code

    threads = [threading.Thread(target=ask, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)

    assert results == {"a": 200, "b": 200}
    seen = {c.model for c in upstream.calls}
    assert seen == {"model-a", "model-b"}, f"并发请求串了凭证：{seen}"


def test_missing_user_key_is_rejected_without_touching_upstream(client, upstream):
    """没填 Key → 400，且**一次上游调用都不该发**（不得偷偷用服务器 Key）。"""
    r = client.post("/api/chat", json=_chat_body(), headers={})
    assert r.status_code == 400
    assert upstream.calls == []


# ────────────────────────────────────────────────────── 清小搭入口（服务器侧）

def test_v1_uses_server_env_config(client, upstream):
    """`/v1` 没有设置面板，本就该用服务器 `.env` 的模型与 Key。"""
    import api
    r = client.post("/v1/chat/completions", json=_chat_body(),
                    headers={"Authorization": f"Bearer {api.SERVICE_KEY}"})
    assert r.status_code == 200
    assert upstream.calls, "根本没有发出上游调用"
    assert {c.model for c in upstream.calls} == {SERVER_MODEL}
    assert all(SERVER_KEY in c.auth for c in upstream.calls)


def test_v1_history_isolation_via_cache(client, upstream):
    """回归：跑到第二次请求（带历史）时模型依旧不漂移。"""
    r1 = client.post("/api/chat", json=_chat_body("苯是什么？"),
                     headers=_web_headers(base_url=upstream.base_url))
    assert r1.status_code == 200
    assert r1.json()["content"]

    answer = r1.json()["content"]
    r2 = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "苯是什么？"},
                     {"role": "assistant", "content": answer},
                     {"role": "user", "content": "那甲苯呢？"}],
        "stream": True, "session_id": "a" * 32,
    }, headers=_web_headers(base_url=upstream.base_url))
    assert r2.status_code == 200
    assert {c.model for c in upstream.calls} == {USER_MODEL}


# ─────────────────────────────────────── mock 上游的噪声（排查）

def test_quiet_server_suppresses_client_disconnect(capsys):
    """★ 客户端断开（keep-alive 连接被关掉）不该打印 traceback。

    这条是**确定性**验证：直接构造 `handle_error` 的两种输入，而不是靠"跑很多次
    看有没有噪声"（那个是间歇性的，实测 3 次里 1 次）。
    """
    class _H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

    srv = _QuietHTTPServer(("127.0.0.1", 0), _H)
    try:
        try:
            raise ConnectionResetError(10054, "远程主机强迫关闭了一个现有的连接。")
        except ConnectionResetError:
            srv.handle_error(None, ("127.0.0.1", 12345))
        assert capsys.readouterr().err == "", "对端断开是噪声，不该打印"

        # 但不能把真异常也吞掉
        try:
            raise ValueError("真错误")
        except ValueError:
            srv.handle_error(None, ("127.0.0.1", 12345))
        err = capsys.readouterr().err
        assert "ValueError" in err and "真错误" in err
    finally:
        srv.server_close()
