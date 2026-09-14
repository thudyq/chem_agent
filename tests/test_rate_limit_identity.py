# -*- coding: utf-8 -*-
"""tests/test_rate_limit_identity.py — 限流用的"客户端身份"（安全审查 R4）。

背景：限流按客户端 IP 分桶。`_client_ip()` 过去**自己解析** `X-Forwarded-For`
并取**第一个**地址 —— 而那个头客户端完全可以伪造，于是每个请求换一个假 IP
就得到一个全新的桶，"每 IP 20 次/分钟"形同虚设（R5 假 key 打慢端点、R6 塞满
磁盘都靠它放大）。

正确做法是**不要自己解析**：uvicorn 的 `ProxyHeadersMiddleware` 只在直连对端是
受信代理时才采信该头，并按"从右往左找第一个不受信地址"取真实客户端，然后写回
`scope["client"]`。应用只读 `request.client` 即可。

这里锁三件事：
1. 伪造 `X-Forwarded-For` **不能**刷出新限流桶；
2. `_client_ip()` 只看 `request.client`，不看请求头；
3. 部署配置（nginx 模板 + 部署文档）里 `X-Forwarded-For` 必须是**覆盖**写法。
"""

from pathlib import Path

import pytest
from starlette.requests import Request

from core import web_api

KEY_HEADERS = {"X-Chem-Api-Key": "sk-test-key-123456",
               "X-Chem-Model": "deepseek-flash",
               "X-Chem-Base-Url": "https://api.deepseek.com/v1"}
_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def web(tmp_path, monkeypatch):
    """独立 app（只装 web router）+ mock 管线 + 会话目录指向 tmp_path。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(web_api.router)
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", tmp_path / "web_sessions")
    web_api._reset_rate_limit_for_tests()
    web_api._session_dir("a" * 32)      # R11：让用例里的固定 id 成为"已下发"
    with TestClient(app) as client:
        yield client
    web_api._reset_rate_limit_for_tests()


@pytest.fixture
def answered(monkeypatch):
    """管线 mock：只记录是否被调用（本文件不关心内容）。"""
    seen = {}

    def fake_pq(question, **kw):
        seen["question"] = question
        return "苯的结构式如下：C6H6"

    monkeypatch.setattr(web_api, "process_question", fake_pq)
    monkeypatch.setattr(web_api, "diaglog", type("D", (), {
        "log_request": staticmethod(lambda *a, **k: None)})())
    monkeypatch.setattr(web_api, "build_attachments", lambda *a, **k: [])
    return seen


def _req(client=("1.2.3.4", 1234), xff=None):
    """最小可用的 Starlette Request（只带我们关心的字段）。"""
    headers = []
    if xff is not None:
        headers.append((b"x-forwarded-for", xff.encode("latin1")))
    scope = {"type": "http", "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/api/chat", "raw_path": b"/api/chat",
             "query_string": b"", "root_path": "", "headers": headers,
             "client": client, "server": ("testserver", 80)}
    return Request(scope)


def _payload():
    return {"messages": [{"role": "user", "content": "画出苯"}],
            "stream": False, "session_id": "a" * 32}


# ---------------------------------------------------------------- 1. 伪造头不能刷出新桶

def test_spoofed_xff_does_not_reset_rate_limit(web, answered, monkeypatch):
    """★ R4 核心回归：每次都换一个伪造 IP，第 3 次必须仍然被限流。

    修复前：`_client_ip` 取 XFF 第一个地址 → 每请求一个新桶 → 全部 200。
    """
    monkeypatch.setattr(web_api, "RATE_LIMIT_PER_MINUTE", 2)
    web_api._reset_rate_limit_for_tests()
    codes = [web.post("/api/chat", json=_payload(),
                      headers={**KEY_HEADERS, "X-Forwarded-For": f"10.1.2.{i}"}
                      ).status_code
             for i in range(3)]
    assert codes[:2] == [200, 200]
    assert codes[2] == 429, "伪造 X-Forwarded-For 不得绕过限流"


def test_same_client_still_limited(web, answered, monkeypatch):
    """正常路径不能被改坏：同一客户端连续超限仍然 429。"""
    monkeypatch.setattr(web_api, "RATE_LIMIT_PER_MINUTE", 1)
    web_api._reset_rate_limit_for_tests()
    assert web.post("/api/chat", json=_payload(),
                    headers=KEY_HEADERS).status_code == 200
    assert web.post("/api/chat", json=_payload(),
                    headers=KEY_HEADERS).status_code == 429


# ---------------------------------------------------------------- 2. 只认净化后的 client

def test_client_ip_ignores_forwarded_header():
    """请求头里的 IP 链完全被忽略（那是客户端可控的输入）。"""
    assert web_api._client_ip(_req(xff="9.9.9.9")) == "1.2.3.4"
    assert web_api._client_ip(_req(xff="9.9.9.9, 8.8.8.8")) == "1.2.3.4"
    assert web_api._client_ip(_req(xff="not-an-ip")) == "1.2.3.4"


def test_client_ip_handles_missing_and_non_ip_client():
    assert web_api._client_ip(_req(client=None)) == "unknown"
    # TestClient / unix socket 这类非 IP 对端标识：原样返回，不抛异常
    assert web_api._client_ip(_req(client=("testclient", 50000))) == "testclient"


def test_client_ip_normalizes_ipv6():
    assert web_api._client_ip(_req(client=("2001:0db8::1", 1))) == "2001:db8::1"


# ---------------------------------------------------------------- 3. 部署配置不得用追加写法

@pytest.mark.parametrize("rel", ["deploy/nginx-chem-agent.conf", "DEPLOY.md"])
def test_forwarded_for_must_overwrite_not_append(rel):
    """★ 配置层的同一条：`$proxy_add_x_forwarded_for` 是**追加**，会把客户端
    伪造的值留在最前面。审查时两边都写着这个写法，属于"文档教你踩坑"。"""
    text = (_ROOT / rel).read_text(encoding="utf-8")
    directives = [ln.strip() for ln in text.splitlines()
                  if ln.strip().startswith("proxy_set_header X-Forwarded-For")]
    assert directives, f"{rel} 里应当有 X-Forwarded-For 指令"
    for d in directives:
        # nginx 的 `#` 注释可以跟在指令行尾；说明文字里提到那个写法是**好事**
        # （正是在警告不要用），所以只检查指令本身，不检查注释。
        body = d.split("#", 1)[0].strip()
        assert "$proxy_add_x_forwarded_for" not in body, f"{rel}: 不能用追加写法 → {d}"
        assert "$remote_addr" in body, f"{rel}: 应当覆盖为真实对端地址 → {d}"


def test_systemd_unit_trusts_only_loopback_proxy():
    """uvicorn 只在直连对端是受信代理时才采信 XFF；单元里应显式写死。"""
    text = (_ROOT / "deploy/chem-agent.service").read_text(encoding="utf-8")
    assert "--forwarded-allow-ips=127.0.0.1" in text
