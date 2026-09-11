# -*- coding: utf-8 -*-
"""tests/test_web_api.py — 公开网页（BYOK）后端测试。

全部 mock 管线与附件编译（不调真实 LLM、不写系统 Temp —— 会话目录用
`data/web_sessions/`，临时文件用 pytest tmp_path 并 monkeypatch 到项目内，
沙箱环境下同样可跑）。
"""

import json
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import answer_cache, credentials, web_api

KEY_HEADERS = {"X-Chem-Api-Key": "sk-test-key-123456",
               "X-Chem-Model": "deepseek-flash",
               "X-Chem-Base-Url": "https://api.deepseek.com/v1"}


@pytest.fixture
def web(tmp_path, monkeypatch):
    """独立 app（只装 web router）+ mock 管线 + 会话目录指向 tmp_path。"""
    app = FastAPI()
    app.include_router(web_api.router)
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", tmp_path / "web_sessions")
    monkeypatch.setattr(web_api, "RATE_LIMIT_PER_MINUTE", 0)   # 默认关闭限流
    web_api._reset_rate_limit_for_tests()
    with TestClient(app) as client:
        yield client
    web_api._reset_rate_limit_for_tests()


@pytest.fixture
def answered(monkeypatch):
    """管线与附件全部 mock：记录收到的 question/凭证/生效配置，返回固定标记文本。"""
    seen = {}

    def fake_pq(question, max_corrections=2, history=None,
                progress_callback=None, correction_callback=None,
                diagnostics=None, responses=None, thinking=None, effort=None,
                max_tokens=None, **kw):
        seen["question"] = question
        seen["history"] = history
        seen["credential"] = credentials.current()
        # 请求作用域内解析出的生效配置（出了 with 块凭证就 reset 了，必须在此取）
        seen["effective_model"] = credentials.llm_config().model_name
        seen["thinking"] = thinking
        seen["effort"] = effort
        seen["max_tokens"] = max_tokens
        if responses is not None:
            responses.append("[STRUCT:c1ccccc1,label=苯]")
        return "苯的结构式如下：\n\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"

    monkeypatch.setattr(web_api, "process_question", fake_pq)
    monkeypatch.setattr(web_api, "diaglog", type("D", (), {
        "log_request": staticmethod(lambda *a, **k: None)})())
    seen["answer_cache"] = answer_cache
    return seen


def _payload(text="画出苯", stream=False, session_id="a" * 32, **kw):
    body = {"messages": [{"role": "user", "content": text}],
            "stream": stream, "session_id": session_id}
    body.update(kw)
    return body


# ---------------------------------------------------------------- 公共配置

def test_web_config_has_no_secrets(web):
    r = web.get("/api/web-config")
    assert r.status_code == 200
    data = r.json()
    assert data["byok"] is True
    assert data["default_base_url"].startswith("https://")
    assert data["default_model"]
    # 不得泄漏服务器 .env 里的任何密钥/端点
    from core.config import settings
    blob = json.dumps(data, ensure_ascii=False)
    if settings.llm.api_key:
        assert settings.llm.api_key not in blob
    if settings.service.api_key:
        assert settings.service.api_key not in blob


def test_index_page_served(web):
    for path in ("/chat", "/web"):
        r = web.get(path)
        assert r.status_code == 200
        assert "Chem_Agent" in r.text
        assert "sessionStorage" in r.text


# ---------------------------------------------------------------- 凭证校验

def test_missing_key_rejected(web):
    r = web.post("/api/chat", json=_payload())
    assert r.status_code == 400
    assert "X-Chem-Api-Key" in r.json()["detail"]


def test_private_base_url_rejected(web):
    r = web.post("/api/chat", json=_payload(),
                 headers={"X-Chem-Api-Key": "sk-x",
                          "X-Chem-Base-Url": "http://127.0.0.1:11434/v1"})
    assert r.status_code == 400
    assert "端点地址" in r.json()["detail"]


def test_bad_json_body(web):
    r = web.post("/api/chat", content="not-json",
                 headers={**KEY_HEADERS, "Content-Type": "application/json"})
    assert r.status_code == 400


def test_question_too_long(web, monkeypatch):
    monkeypatch.setattr(web_api, "_MAX_QUESTION_CHARS", 10)
    r = web.post("/api/chat", json=_payload(text="苯" * 50), headers=KEY_HEADERS)
    assert r.status_code == 400
    assert "过长" in r.json()["detail"]


# ---------------------------------------------------------------- 限流

def test_rate_limit_returns_429(web, monkeypatch, answered):
    monkeypatch.setattr(web_api, "RATE_LIMIT_PER_MINUTE", 2)
    web_api._reset_rate_limit_for_tests()
    # 固定客户端 IP（TestClient 的 client 可能为 None/Host，需显式可控）
    monkeypatch.setattr(web_api, "_client_ip", lambda request: "203.0.113.9")
    codes = [web.post("/api/chat", json=_payload(), headers=KEY_HEADERS).status_code
             for _ in range(3)]
    assert codes[0] == 200 and codes[1] == 200
    assert codes[2] == 429


def test_rate_limit_disabled_by_zero(web, answered):
    # fixture 已设 RATE_LIMIT_PER_MINUTE=0
    codes = [web.post("/api/chat", json=_payload(), headers=KEY_HEADERS).status_code
             for _ in range(5)]
    assert codes == [200] * 5


# ---------------------------------------------------------------- 非流式

def test_non_stream_returns_content_and_session(web, answered):
    r = web.post("/api/chat", json=_payload(), headers=KEY_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == "a" * 32
    assert "苯的结构式如下" in data["content"]
    assert answered["credential"]["api_key"] == "sk-test-key-123456"
    assert answered["credential"]["model"] == "deepseek-flash"


def test_non_stream_generates_session_when_missing(web, answered):
    body = _payload()
    body.pop("session_id")
    r = web.post("/api/chat", json=body, headers=KEY_HEADERS)
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert len(sid) == 32 and all(c in "0123456789abcdef" for c in sid)


def test_invalid_session_id_is_replaced(web, answered):
    r = web.post("/api/chat", json=_payload(session_id="../../etc"),
                 headers=KEY_HEADERS)
    assert r.status_code == 200
    assert r.json()["session_id"] != "../../etc"


def test_credentials_do_not_leak_after_request(web, answered):
    web.post("/api/chat", json=_payload(), headers=KEY_HEADERS)
    assert credentials.current() == {}


def test_user_model_wins_over_server_model(web, monkeypatch, answered):
    """★ 用户实测病例回归：网页填的模型不得被服务器 .env 的模型顶掉。

    （单模型后服务器不再有"升级路由"，但"用户模型优先"这条契约必须保持。）
    """
    import dataclasses
    from core.config import settings as real
    base = dataclasses.replace(
        real, llm=dataclasses.replace(real.llm, model_name="server-model"))
    credentials.set_base_settings_for_tests(base)
    try:
        r = web.post("/api/chat", json=_payload(text="画出苯的硝化反应机理"),
                     headers={**KEY_HEADERS, "X-Chem-Model": "deepseek-chat"})
        assert r.status_code == 200
        assert answered["credential"]["model"] == "deepseek-chat"
        assert answered["effective_model"] == "deepseek-chat"
    finally:
        credentials.set_base_settings_for_tests(None)


def test_thinking_params_forwarded_from_headers(web, answered):
    """X-Chem-Thinking / X-Chem-Effort / X-Chem-Max-Tokens 透传到管线。"""
    headers = {**KEY_HEADERS, "X-Chem-Thinking": "off",
               "X-Chem-Effort": "high", "X-Chem-Max-Tokens": "12345"}
    r = web.post("/api/chat", json=_payload(), headers=headers)
    assert r.status_code == 200
    assert answered["thinking"] == "off"
    assert answered["effort"] == "high"
    assert answered["max_tokens"] == 12345


def test_invalid_max_tokens_header_ignored(web, answered):
    """非法的 max_tokens 头被忽略（None → 用调用点默认），不报错。"""
    r = web.post("/api/chat", json=_payload(),
                 headers={**KEY_HEADERS, "X-Chem-Max-Tokens": "abc"})
    assert r.status_code == 200
    assert answered["max_tokens"] is None


def test_obsolete_model_headers_ignored(web, answered):
    """旧的 fallback/upgrade 头已废弃：发了也不该报错（避免旧前端 400）。"""
    r = web.post("/api/chat", json=_payload(),
                 headers={**KEY_HEADERS, "X-Chem-Upgrade-Model": "x",
                          "X-Chem-Fallback-Model": "y"})
    assert r.status_code == 200


def _frames(text):
    """SSE 文本 → 帧对象列表（`[DONE]` 记为 {"_done": True}）。"""
    out = []
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload == "[DONE]":
                    out.append({"_done": True})
                else:
                    out.append(json.loads(payload))
    return out


def test_stream_frame_sequence(web, answered):
    r = web.post("/api/chat", json=_payload(stream=True), headers=KEY_HEADERS)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = _frames(r.text)
    assert frames[-1] == {"_done": True}
    deltas = [f["choices"][0]["delta"] for f in frames if "choices" in f]
    assert deltas[0].get("role") == "assistant"
    assert any("reasoning" in d for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    assert "苯的结构式如下" in content
    stops = [f for f in frames if "choices" in f
             and f["choices"][0].get("finish_reason") == "stop"]
    assert stops and stops[-1]["choices"][0]["delta"].get("session_id") == "a" * 32


def test_stream_credentials_reach_worker_thread(web, answered):
    """★ 用户实测病例回归：**流式**也必须全程用网页里填的模型/端点/密钥。

    根因（20260830 定位）：`_sse_stream` 是生成器，Starlette 把它放进线程池
    **逐块**迭代，每次 `next()` 都从请求任务上下文重新拷贝一份 Context ——
    在生成器迭代处 `apply_credentials()` 写下的凭证，传不到后面启动工作线程
    的那一次迭代。于是工作线程读到空凭证 → 静默回退服务器 `.env`
    （用户看到的现象：网页填 `deepseek-flash`、日志却走 `.env` 的 gemini）。
    修法：凭证在 `work()` **内部**设置（整个 `work()` 跑在同一个 Context 里）。
    """
    import dataclasses
    from core.config import settings as real
    base = dataclasses.replace(
        real, llm=dataclasses.replace(real.llm, model_name="server-model"))
    credentials.set_base_settings_for_tests(base)
    try:
        r = web.post("/api/chat", json=_payload(stream=True), headers=KEY_HEADERS)
        assert r.status_code == 200
        assert _frames(r.text)[-1] == {"_done": True}      # 流没被弄坏
        assert answered["credential"].get("model") == "deepseek-flash"
        assert answered["effective_model"] == "deepseek-flash"
    finally:
        credentials.set_base_settings_for_tests(None)


def test_stream_thinking_params_reach_worker_thread(web, answered):
    """流式路径的思考三件套同样透传（与 requests 头一致）。"""
    headers = {**KEY_HEADERS, "X-Chem-Thinking": "off",
               "X-Chem-Effort": "high", "X-Chem-Max-Tokens": "12345"}
    r = web.post("/api/chat", json=_payload(stream=True), headers=headers)
    assert r.status_code == 200
    assert answered["thinking"] == "off"
    assert answered["effort"] == "high"
    assert answered["max_tokens"] == 12345


def test_stream_credentials_do_not_leak_after_request(web, answered):
    """流式请求结束后不得把用户凭证留在当前 Context 里。"""
    web.post("/api/chat", json=_payload(stream=True), headers=KEY_HEADERS)
    assert credentials.current() == {}


def test_concurrent_streams_keep_credentials_isolated(web, monkeypatch):
    """BYOK 核心保证：两个**并发**流式请求不得互相串模型/密钥。

    凭证是在 SSE 工作线程里用 `apply_credentials()`（只设不重置）落地的，
    这依赖"每个请求一份独立的 Context 拷贝"。本用例让两个请求在管线里真实
    重叠（Barrier），确认各自只看到自己的模型。
    """
    import threading

    seen, lock = [], threading.Lock()
    both_inside = threading.Barrier(2, timeout=15)

    def fake_pq(question, max_corrections=2, history=None,
                progress_callback=None, correction_callback=None,
                diagnostics=None, responses=None, thinking=None,
                effort=None, max_tokens=None, **kw):
        model = credentials.current().get("model")
        with lock:
            seen.append(model)
        both_inside.wait()          # 两个请求都进到管线，制造真实重叠
        return f"答案来自 {model}"

    monkeypatch.setattr(web_api, "process_question", fake_pq)
    monkeypatch.setattr(web_api, "diaglog", type("D", (), {
        "log_request": staticmethod(lambda *a, **k: None)})())

    codes = {}

    def ask(model):
        r = web.post("/api/chat", json=_payload(stream=True),
                     headers={**KEY_HEADERS, "X-Chem-Model": model})
        codes[model] = r.status_code

    threads = [threading.Thread(target=ask, args=(m,))
               for m in ("model-a", "model-b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert codes == {"model-a": 200, "model-b": 200}
    assert sorted(seen) == ["model-a", "model-b"], "并发请求串了凭证"


# ---------------------------------------------------------------- 附件与图片

def test_answer_compiles_tikz_to_session_image(web, answered, monkeypatch):
    """回答里的 TikZ → 编译 PNG → 就地替换为会话级图片 URL。"""
    from core import attachments
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 32

    def fake_build(answer, public_base, dir_path=None, max_bytes=None,
                   max_files=None):
        blocks = attachments.extract_code_blocks(answer)
        assert blocks, "应识别出 TikZ 代码块"
        dir_path.mkdir(parents=True, exist_ok=True)
        name = "b" * 32 + ".png"
        (dir_path / name).write_bytes(png)
        return [{"fileUrl": f"{public_base}/files/{name}",
                 "fileName": "化学图示-1.png", "fileType": "image",
                 "mimeType": "image/png", "fileSize": len(png)}]

    monkeypatch.setattr(web_api, "build_attachments", fake_build)
    r = web.post("/api/chat", json=_payload(), headers=KEY_HEADERS)
    content = r.json()["content"]
    assert "![化学图示-1](" in content
    assert f"/api/session/{'a' * 32}/{'b' * 32}.png" in content
    assert "tikzpicture" not in content          # 裸代码已替换
    img = content.split("![化学图示-1](", 1)[1].split(")", 1)[0]
    assert img == f"/api/session/{'a' * 32}/{'b' * 32}.png", \
        "图示 URL 必须是同源相对路径（不得拼服务器 PUBLIC_BASE_URL）"


def test_session_image_url_is_origin_relative_and_fetchable(web, answered,
                                                             monkeypatch):
    """★ 回归（20260910）：图示 URL 是同源相对路径，且真能取回 PNG。

    以前 `_prepare_answer` 把服务器 `.env` 的 `PUBLIC_BASE_URL` 当图片前缀，
    于是**本机起服务**时浏览器会去请求那台公网机器上的
    `/api/session/<本地会话id>/<图>`：那台机器跑的是旧版应用（实测该路由与
    `/api/web-config` 都是 404），而且图**根本没落在它上面**（图写在本机
    `_SESSIONS_DIR`）——所以每个化学图示都裂成 alt 文本。相对路径让浏览器
    必然回到"它此刻正在访问的这个源"，同时避免反代下 scheme 推断成 http
    触发的混合内容拦截。

    这里用**真实**的 `build_attachments`（只把 LaTeX 编译换成假 PNG 字节），
    因此连"前缀被丢弃""文件真的落盘""相对 URL 真能 200 取回"一起覆盖。
    """
    from core import attachments
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
    monkeypatch.setattr(attachments, "compile_tikz_to_png",
                        lambda code, **kw: png)

    r = web.post("/api/chat", json=_payload(), headers=KEY_HEADERS)
    assert r.status_code == 200
    content = r.json()["content"]
    assert "![化学图示-1](" in content, f"没生成图片引用: {content!r}"
    img = content.split("![化学图示-1](", 1)[1].split(")", 1)[0]

    assert img.startswith("/api/session/"), f"应是同源相对路径: {img!r}"
    assert "://" not in img, \
        f"图示 URL 不得带 scheme/host（那会指向服务器 .env 的公网地址）: {img!r}"

    # 真取一次：同一 app 上必须能拿到这张 PNG
    got = web.get(img)
    assert got.status_code == 200, f"{img} 取不到（{got.status_code}）"
    assert got.headers["content-type"] == "image/png"
    assert got.content == png


def test_serve_session_attachment(web, answered, monkeypatch):
    sid, name = "c" * 32, "d" * 32 + ".png"
    d = web_api._SESSIONS_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    r = web.get(f"/api/session/{sid}/{name}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


@pytest.mark.parametrize("sid,name", [
    ("../..", "d" * 32 + ".png"),
    ("c" * 32, "not-a-png.txt"),
    ("c" * 32, "../../secret.png"),
    ("CC" * 32, "d" * 32 + ".png"),
])
def test_serve_session_attachment_rejects_bad_names(web, sid, name):
    r = web.get(f"/api/session/{sid}/{name}")
    assert r.status_code in (404, 422)


def test_image_uses_main_model_when_no_vision_configured(web, answered,
                                                         monkeypatch):
    """★ 新语义（§4.9）：用户没填视觉模型时**用主模型识图**，而不是判"未配置"。

    原生多模态模型（deepseek-flash / Gemini / GLM 视觉版）自带视觉，
    这正是本次重构要消除的"必须再申请一把别家 Key"的误解。
    """
    import utils.ocr_utils as ocr
    seen = {}

    def fake_describe(path):
        seen["called"] = True
        return {"type": "结构式", "content": "苯，SMILES: c1ccccc1",
                "smiles_ok": True, "downgraded": False}

    monkeypatch.setattr(ocr, "describe_image", fake_describe)
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "这是什么分子"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
        "stream": False, "session_id": "a" * 32}
    r = web.post("/api/chat", json=body, headers=KEY_HEADERS)
    assert r.status_code == 200
    assert seen.get("called") is True, "应直接用主模型识图"
    assert "c1ccccc1" in answered["question"]


def test_image_recognition_failure_is_reported(web, answered, monkeypatch):
    """视觉模型可用但识别失败 → 明确告知主模型"图片不可用"（不静默丢弃）。

    注：`_require_credentials` 会给"只填 Key"的请求补上默认模型/端点
    （BYOK 的默认值），因此`未配置视觉模型`分支只在**连主模型凭证都没有**
    时出现（见 `test_image_uses_main_model_when_no_vision_configured` 与
    credentials 层用例）。这里验证"配置了但识别失败"的提示。
    """
    import utils.ocr_utils as ocr
    monkeypatch.setattr(ocr, "describe_image", lambda path: None)
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "这是什么分子"},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
        "stream": False, "session_id": "a" * 32}
    r = web.post("/api/chat", json=body, headers=KEY_HEADERS)
    assert r.status_code == 200
    assert "识别失败" in answered["question"]
    assert "重新上传" in answered["question"] or "文字描述" in answered["question"]


def test_image_recognized_with_vision_config(web, answered, monkeypatch):
    import utils.ocr_utils as ocr
    monkeypatch.setattr(ocr, "describe_image",
                        lambda path: {"type": "结构式", "content": "苯，SMILES: c1ccccc1",
                                      "smiles_ok": True, "downgraded": False})
    headers = {**KEY_HEADERS, "X-Chem-Vision-Model": "glm-4.6v"}
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "这是什么分子"},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64," + "A" * 40}}]}],
        "stream": False, "session_id": "a" * 32}
    r = web.post("/api/chat", json=body, headers=headers)
    assert r.status_code == 200
    assert "c1ccccc1" in answered["question"]
    assert "识别可能有误" in answered["question"]
    # 上传的临时图已删除（不留用户数据）
    uploads = web_api._SESSIONS_DIR / ("a" * 32) / "uploads"
    assert not uploads.exists() or not list(uploads.iterdir())


def test_image_vision_failure_is_reported_to_model(web, answered, monkeypatch):
    import utils.ocr_utils as ocr
    monkeypatch.setattr(ocr, "describe_image", lambda path: None)
    headers = {**KEY_HEADERS, "X-Chem-Vision-Model": "glm-4.6v"}
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "看图"},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64," + "A" * 40}}]}],
        "stream": False, "session_id": "a" * 32}
    web.post("/api/chat", json=body, headers=headers)
    assert "识别失败" in answered["question"]


def test_http_image_url_is_ssrf_checked(web, answered, monkeypatch):
    """内网图片 URL 必须被拒（不得让服务端代取内网资源）。"""
    called = {"n": 0}

    def fake_get(*a, **k):
        called["n"] += 1
        raise AssertionError("不应发起网络请求")

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    headers = {**KEY_HEADERS, "X-Chem-Vision-Model": "glm-4.6v"}
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "看图"},
        {"type": "image_url",
         "image_url": {"url": "http://169.254.169.254/latest/meta-data"}}]}],
        "stream": False, "session_id": "a" * 32}
    web.post("/api/chat", json=body, headers=headers)
    assert called["n"] == 0
    assert "下载/解码失败" in answered["question"]


# ---------------------------------------------------------------- 历史

def test_history_keeps_user_text_and_summarizes_assistant(web, answered):
    body = {"messages": [
        {"role": "user", "content": "苯的结构式"},
        {"role": "assistant", "content": "苯是 [STRUCT:c1ccccc1]"},
        {"role": "user", "content": "它有什么用途"}],
        "stream": False, "session_id": "a" * 32}
    web.post("/api/chat", json=body, headers=KEY_HEADERS)
    history = answered["history"]
    assert [h["role"] for h in history] == ["user", "assistant"]
    assert history[0]["content"] == "苯的结构式"
    assert "STRUCT" not in history[1]["content"]      # 去锚定：不回喂标记


# ---------------------------------------------------------------- 附件配额回收

def _mkpng(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_prune_noop_when_within_quota(tmp_path, monkeypatch):
    """★ 常态：没超配额就**一个都不删**（这是与旧 24h TTL 最本质的区别）。"""
    root = tmp_path / "web_sessions"
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", root)
    a = _mkpng(root / "s1" / "a.png", 100)
    b = _mkpng(root / "s2" / "b.png", 100)
    import os
    stale = time.time() - 365 * 24 * 3600          # 一年前的文件也不删
    os.utime(a, (stale, stale))

    assert web_api.prune_web_attachments(max_bytes=10_000, max_files=100) == 0
    assert a.is_file() and b.is_file()
    assert (root / "s1").is_dir() and (root / "s2").is_dir()


def test_prune_deletes_oldest_across_sessions(tmp_path, monkeypatch):
    """超配额 → 跨会话按 mtime 删最旧的；新的留着；空目录收尾掉。"""
    root = tmp_path / "web_sessions"
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", root)
    import os
    old = _mkpng(root / "s1" / "old.png", 1000)
    mid = _mkpng(root / "s2" / "mid.png", 1000)
    new = _mkpng(root / "s3" / "new.png", 1000)
    t0 = time.time() - 3 * 3600
    for p, off in ((old, 0), (mid, 3600), (new, 2 * 3600)):
        os.utime(p, (t0 + off, t0 + off))

    # 配额只容得下 2 个 → 应删掉最旧的 old.png
    removed = web_api.prune_web_attachments(max_bytes=0, max_files=2)
    assert removed == 1
    assert not old.exists()
    assert mid.is_file() and new.is_file()
    assert not (root / "s1").is_dir()              # 空目录被收尾
    assert (root / "s2").is_dir() and (root / "s3").is_dir()


def test_prune_by_bytes(tmp_path, monkeypatch):
    root = tmp_path / "web_sessions"
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", root)
    import os
    a = _mkpng(root / "s1" / "a.png", 800)
    b = _mkpng(root / "s2" / "b.png", 800)
    t0 = time.time() - 3 * 3600
    os.utime(a, (t0, t0))
    os.utime(b, (t0 + 60, t0 + 60))

    removed = web_api.prune_web_attachments(max_bytes=900, max_files=0)
    assert removed == 1
    assert not a.exists() and b.is_file()


def test_prune_protects_brand_new_files(tmp_path, monkeypatch):
    """刚写入（_PRUNE_MIN_AGE 内）的文件不删 —— 可能正被某个请求引用。

    宁超额、不删新图：删完仍超配额就保持现状（与 /v1 的 keep 同一取舍）。
    """
    root = tmp_path / "web_sessions"
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", root)
    fresh = _mkpng(root / "s1" / "fresh.png", 5000)

    removed = web_api.prune_web_attachments(max_bytes=100, max_files=0)
    assert removed == 0
    assert fresh.is_file()


def test_prune_disabled_when_limits_zero(tmp_path, monkeypatch):
    root = tmp_path / "web_sessions"
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", root)
    import os
    old = _mkpng(root / "s1" / "old.png", 5000)
    stale = time.time() - 3 * 3600
    os.utime(old, (stale, stale))

    assert web_api.prune_web_attachments(max_bytes=0, max_files=0) == 0
    assert old.is_file()


def test_web_config_exposes_attachment_quota(web):
    """网页配置暴露的是**配额**，不再是 TTL（前端不再有"24 小时后图会没"的概念）。"""
    d = web.get("/api/web-config").json()
    assert d["attachment_max_bytes"] == web_api.WEB_ATTACHMENT_MAX_BYTES
    assert d["attachment_max_files"] == web_api.WEB_ATTACHMENT_MAX_FILES
    assert "session_ttl_hours" not in d


# ---------------------------------------------------------------- 图示源码 / AI 标题

def test_stream_stop_frame_carries_latex_sources(web, answered):
    """★ stop 帧必须带回 TikZ 原文 —— 正文里的代码块已被替换成图片 URL，
    前端"查看图示 LaTeX 源码"面板只能从这里拿。"""
    r = web.post("/api/chat", json=_payload(stream=True), headers=KEY_HEADERS)
    frames = _frames(r.text)
    stops = [f for f in frames if "choices" in f
             and f["choices"][0].get("finish_reason") == "stop"]
    latex = stops[-1]["choices"][0]["delta"].get("latex")
    assert isinstance(latex, list) and len(latex) == 1
    assert latex[0].startswith("\\begin{tikzpicture}")
    assert latex[0].endswith("\\end{tikzpicture}")


def test_json_path_carries_latex_sources(web, answered):
    r = web.post("/api/chat", json=_payload(), headers=KEY_HEADERS)
    d = r.json()
    assert len(d["latex"]) == 1 and "tikzpicture" in d["latex"][0]
    # 正文里是图片引用，不是裸代码
    assert "tikzpicture" not in d["content"]


def test_latex_sources_helper_is_total():
    """取源码是"锦上添花"，任何输入都不该抛。"""
    from core import web_api
    assert web_api.latex_sources("") == []
    assert web_api.latex_sources("没有代码块") == []
    assert web_api.latex_sources(None) == []


def test_title_uses_user_key_and_returns_llm_title(web, monkeypatch):
    """★ BYOK：起标题必须用**用户自己的** key/模型，且标记为用户作用域调用。"""
    seen = {}
    from core import credentials

    def fake_ask_llm(prompt, system_prompt=None, max_tokens=None, thinking=None,
                     user_scoped=False, **kw):
        seen["key"] = (credentials.current() or {}).get("api_key")
        seen["model"] = credentials.llm_config().model_name
        seen["scoped"] = user_scoped
        seen["thinking"] = thinking
        seen["max_tokens"] = max_tokens
        return "苯的结构式"

    monkeypatch.setattr("core.llm_client.ask_llm", fake_ask_llm)
    r = web.post("/api/title", json={"question": "画出苯的结构式，并说明它的分子式"},
                 headers=KEY_HEADERS)
    assert r.status_code == 200
    assert r.json()["title"] == "苯的结构式"
    assert seen["key"] == "sk-test-key-123456"          # 用户自己的 key
    assert seen["model"] == "deepseek-flash"            # 用户填的模型
    assert seen["scoped"] is True                       # 不许回退服务器模型
    assert seen["thinking"] == "disabled"               # 起标题不用思考
    assert seen["max_tokens"] == 64


def test_title_falls_back_to_truncation_on_failure(web, monkeypatch):
    """上游失败 → 回退为提问前 12 字，且**仍然返回 200**（不影响对话）。"""
    def boom(*a, **k):
        raise RuntimeError("upstream down")

    monkeypatch.setattr("core.llm_client.ask_llm", boom)
    q = "画出苯的结构式并说明分子式"
    r = web.post("/api/title", json={"question": q}, headers=KEY_HEADERS)
    assert r.status_code == 200
    assert r.json()["title"] == q[:12]
    assert len(r.json()["title"]) == 12


def test_title_cleans_quotes_and_limits_length(web, monkeypatch):
    monkeypatch.setattr("core.llm_client.ask_llm",
                        lambda *a, **k: '“苯环的结构与芳香性说明”')
    r = web.post("/api/title", json={"question": "苯"}, headers=KEY_HEADERS)
    assert r.json()["title"] == "苯环的结构与芳香性说明"[:12]


def test_title_requires_user_key(web):
    """没有 Key 直接 400 —— 绝不用服务器 .env 的 key 替用户起标题。"""
    r = web.post("/api/title", json={"question": "苯"},
                 headers={"X-Chem-Model": "m"})
    assert r.status_code == 400
    assert "API Key" in r.json()["detail"]


def test_title_empty_question_makes_no_llm_call(web, monkeypatch):
    called = []
    monkeypatch.setattr("core.llm_client.ask_llm",
                        lambda *a, **k: called.append(1))
    r = web.post("/api/title", json={}, headers=KEY_HEADERS)
    assert r.status_code == 200
    assert r.json()["title"] == ""
    assert called == []


# ---------------------------------------------------------------- 错误友好化

@pytest.mark.parametrize("raw,expect", [
    ("HTTP 401 unauthorized", "API Key 无效"),
    ("invalid_api_key", "API Key 无效"),
    ("model_not_found: no such model", "模型名或端点地址不正确"),
    ("402 insufficient balance", "额度不足"),
    ("429 rate limit exceeded", "限流"),
    ("Read timed out", "超时"),
    ("KeyError: 'x'", "生成回答时出现错误"),
])
def test_friendly_error_mapping(raw, expect):
    assert expect in web_api._friendly_error(raw)


def test_friendly_error_does_not_echo_raw():
    assert "KeyError" not in web_api._friendly_error("KeyError: 'x'")


def test_pipeline_failure_text_is_rewritten_for_web(web, monkeypatch):
    """管线内部文案（"请检查 .env"）对网页用户是误导，必须换成设置面板指引。"""
    monkeypatch.setattr(web_api, "process_question",
                        lambda *a, **k: web_api._PIPELINE_FAIL_TEXT)
    monkeypatch.setattr(web_api, "diaglog", type("D", (), {
        "log_request": staticmethod(lambda *a, **k: None)})())
    r = web.post("/api/chat", json=_payload(), headers=KEY_HEADERS)
    content = r.json()["content"]
    assert ".env" not in content
    assert "设置里的 API Key" in content


def test_answer_without_pipeline_failure_untouched(web, monkeypatch):
    monkeypatch.setattr(web_api, "process_question",
                        lambda *a, **k: "正常回答，无需改写。")
    monkeypatch.setattr(web_api, "diaglog", type("D", (), {
        "log_request": staticmethod(lambda *a, **k: None)})())
    r = web.post("/api/chat", json=_payload(), headers=KEY_HEADERS)
    assert r.json()["content"] == "正常回答，无需改写。"
