# -*- coding: utf-8 -*-
"""tests/test_inflight_cap.py — 出站调用的**在途上限**（安全审查 R5）。

背景：`/api/chat` 只校验 Key **是否存在**（假 key 随便填），而端点地址由用户
指定。攻击者用假 key + 一台"接受连接但永不回包"的公网服务器，就能让每个请求
占住一个 worker 直到读超时。原来的并发闸门是**按凭证指纹分桶**的 —— 假 key
要多少有多少，于是每个请求都拿到一个全新的桶，对服务器**毫无保护**。

本文件锁住：
1. **全局在途上限**：服务器同时处理多少出站调用有上界；
2. **按 IP 在途上限**：单个 IP 不能把全局额度吃光、把其他访客饿死；
3. **不泄漏槽位**：任何失败路径（包括按 IP 拒绝）都必须归还全局槽位；
4. 两道闸门真的挂在**所有**出站调用上（主模型 + 视觉）；
5. 超时拆成（建连, 读取），不可达端点快速失败；
6. 限满时给用户的是可操作的提示（不是"请检查你的设置"）。
"""

import contextlib
from pathlib import Path

import pytest

import core.llm_client as lc
import utils.ocr_utils as ocr
from core import credentials, web_api

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """不等待 + 清空在途状态 + 清掉 request_ip（contextvar 会跨用例残留）。"""
    monkeypatch.setattr(credentials, "INFLIGHT_WAIT_SECONDS", 0)
    credentials.set_request_ip(None)
    credentials.reset_inflight_for_tests()
    yield
    credentials.set_request_ip(None)
    credentials.reset_inflight_for_tests()


# ---------------------------------------------------------------- 1. 全局上限

def test_global_cap_raises_server_busy(monkeypatch):
    monkeypatch.setattr(credentials, "GLOBAL_MAX_INFLIGHT", 1)
    credentials.reset_inflight_for_tests()
    with credentials.inflight_guard():
        with pytest.raises(credentials.ServerBusy):
            with credentials.inflight_guard():
                pass


def test_global_cap_zero_means_disabled(monkeypatch):
    """0 = 关闭这道闸门（本地/测试用）。"""
    monkeypatch.setattr(credentials, "GLOBAL_MAX_INFLIGHT", 0)
    credentials.reset_inflight_for_tests()
    with contextlib.ExitStack() as st:
        for _ in range(50):
            st.enter_context(credentials.inflight_guard())


def test_slot_released_after_normal_use(monkeypatch):
    monkeypatch.setattr(credentials, "GLOBAL_MAX_INFLIGHT", 1)
    credentials.reset_inflight_for_tests()
    for _ in range(3):                     # 反复进出不该把额度漏光
        with credentials.inflight_guard():
            pass


def test_slot_released_after_body_raises(monkeypatch):
    """调用方在槽位里抛异常也必须归还（否则一次报错就永久少一个额度）。"""
    monkeypatch.setattr(credentials, "GLOBAL_MAX_INFLIGHT", 1)
    credentials.reset_inflight_for_tests()
    with pytest.raises(ValueError):
        with credentials.inflight_guard():
            raise ValueError("boom")
    with credentials.inflight_guard():
        pass


# ---------------------------------------------------------------- 2. 按 IP 上限

def test_per_ip_cap_blocks_same_ip(monkeypatch):
    monkeypatch.setattr(credentials, "GLOBAL_MAX_INFLIGHT", 0)
    monkeypatch.setattr(credentials, "PER_IP_MAX_INFLIGHT", 1)
    credentials.reset_inflight_for_tests()
    credentials.set_request_ip("1.1.1.1")
    with credentials.inflight_guard():
        with pytest.raises(credentials.ServerBusy):
            with credentials.inflight_guard():
                pass


def test_per_ip_cap_is_per_ip(monkeypatch):
    """别的 IP 不受同一个 IP 占满的影响 —— 这正是加这道闸门的目的。"""
    monkeypatch.setattr(credentials, "GLOBAL_MAX_INFLIGHT", 0)
    monkeypatch.setattr(credentials, "PER_IP_MAX_INFLIGHT", 1)
    credentials.reset_inflight_for_tests()
    credentials.set_request_ip("1.1.1.1")
    with credentials.inflight_guard():
        credentials.set_request_ip("2.2.2.2")      # 换一个访客
        with credentials.inflight_guard():
            pass


def test_per_ip_skipped_when_ip_unknown(monkeypatch):
    """Streamlit / CLI 路径没有 IP → 只受全局闸门约束，不该被判忙。"""
    monkeypatch.setattr(credentials, "GLOBAL_MAX_INFLIGHT", 0)
    monkeypatch.setattr(credentials, "PER_IP_MAX_INFLIGHT", 1)
    credentials.reset_inflight_for_tests()
    credentials.set_request_ip(None)
    with contextlib.ExitStack() as st:
        for _ in range(10):
            st.enter_context(credentials.inflight_guard())


def test_per_ip_counter_is_cleaned_up(monkeypatch):
    """IP 计数不能只增不减（否则长期运行会内存泄漏）。"""
    monkeypatch.setattr(credentials, "GLOBAL_MAX_INFLIGHT", 0)
    monkeypatch.setattr(credentials, "PER_IP_MAX_INFLIGHT", 2)
    credentials.reset_inflight_for_tests()
    for i in range(20):
        credentials.set_request_ip(f"10.0.0.{i}")
        with credentials.inflight_guard():
            pass
    assert credentials._ip_inflight == {}


def test_global_slot_not_leaked_when_per_ip_rejects(monkeypatch):
    """★ 按 IP 拒绝发生在**拿到全局槽位之后**，必须把它还回去。

    不还的话，被拒绝的请求每次都会永久吃掉一个全局额度 —— 攻击者只要一直
    触发按 IP 拒绝，就能把服务器的全局并发额度慢慢漏到 0（比原漏洞更糟）。
    """
    monkeypatch.setattr(credentials, "GLOBAL_MAX_INFLIGHT", 2)
    monkeypatch.setattr(credentials, "PER_IP_MAX_INFLIGHT", 1)
    credentials.reset_inflight_for_tests()
    credentials.set_request_ip("1.1.1.1")
    with credentials.inflight_guard():
        for _ in range(5):
            with pytest.raises(credentials.ServerBusy):
                with credentials.inflight_guard():
                    pass
    credentials.set_request_ip(None)
    with contextlib.ExitStack() as st:     # 全局额度应当仍是满的（2）
        st.enter_context(credentials.inflight_guard())
        st.enter_context(credentials.inflight_guard())


# ------------------------------------------------------ 3. llm_slot 的组合

def test_llm_slot_includes_both_gates(monkeypatch):
    entered = []

    @contextlib.contextmanager
    def fake_bucket():
        entered.append("bucket")
        yield

    monkeypatch.setattr(credentials, "llm_semaphore", fake_bucket)
    monkeypatch.setattr(credentials, "GLOBAL_MAX_INFLIGHT", 1)
    credentials.reset_inflight_for_tests()
    with credentials.llm_slot():
        entered.append("slot")
        with pytest.raises(credentials.ServerBusy):   # 全局闸门仍然生效
            with credentials.llm_slot():
                pass
    assert entered == ["bucket", "slot"]


def test_call_sites_use_the_gates():
    """所有出站调用点都必须走这两道闸门（漏掉一处 = 那处就是攻击入口）。"""
    llm = (_ROOT / "core/llm_client.py").read_text(encoding="utf-8")
    assert "with credentials.llm_slot()" in llm
    assert "with credentials.llm_semaphore()" not in llm, "主模型必须换成 llm_slot"
    ocr_src = (_ROOT / "utils/ocr_utils.py").read_text(encoding="utf-8")
    assert "with credentials.inflight_guard()" in ocr_src, "视觉调用也要受闸门约束"


# ---------------------------------------------------------------- 4. 超时拆分

class _Resp:
    def __init__(self, status=200, text=""):
        self.status_code = status
        self.text = text
        self.content = b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self, decode_unicode=False):
        return iter([])


def test_stream_chat_uses_split_timeout(monkeypatch):
    seen = {}

    class _S:
        def post(self, url, **kw):
            seen.update(kw)
            return _Resp(200)

    monkeypatch.setattr(lc.credentials, "session", lambda: _S())
    lc._stream_chat("https://api.example.com/v1/chat/completions", {}, {})
    assert seen["timeout"] == (lc.CONNECT_TIMEOUT, lc.READ_TIMEOUT)
    assert seen["timeout"][0] < seen["timeout"][1], "建连超时必须短于读取超时"
    assert seen["timeout"][0] <= 15, "黑洞地址要在十几秒内失败，不能干等"


def test_vision_uses_split_timeout(monkeypatch):
    seen = {}

    def fake_post(url, **kw):
        seen.update(kw)
        return _Resp(302, "moved")

    monkeypatch.setattr(ocr.requests, "post", fake_post)
    ocr._describe_once("https://vision.example/v1/chat/completions", {}, {}, "m")
    assert seen["timeout"] == ocr.VISION_TIMEOUT
    assert seen["timeout"] == (ocr.VISION_CONNECT_TIMEOUT, ocr.VISION_READ_TIMEOUT)


# ------------------------------------------------------------ 5. 用户可见提示

def test_friendly_error_explains_busy():
    msg = web_api._friendly_error("服务器繁忙：同时在处理的请求过多，请稍后重试")
    assert "稍后重试" in msg
    assert "不是你的设置问题" in msg, "限流/限满不是用户配置错了，别让他去改设置"


def test_json_path_returns_503_when_busy(monkeypatch, tmp_path):
    """非流式路径：限满时给 503 + 可操作文案，而不是 500。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(web_api.router)
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", tmp_path / "web_sessions")
    monkeypatch.setattr(web_api, "RATE_LIMIT_PER_MINUTE", 0)
    web_api._reset_rate_limit_for_tests()
    web_api._session_dir("a" * 32)      # R11：让用例里的固定 id 成为"已下发"

    def busy(*a, **k):
        raise credentials.ServerBusy()

    monkeypatch.setattr(web_api, "process_question", busy)
    with TestClient(app) as client:
        r = client.post("/api/chat",
                        json={"messages": [{"role": "user", "content": "苯"}],
                              "stream": False, "session_id": "a" * 32},
                        headers={"X-Chem-Api-Key": "sk-x",
                                 "X-Chem-Base-Url": "https://api.deepseek.com/v1"})
    assert r.status_code == 503
    assert "稍后重试" in r.json()["detail"]
