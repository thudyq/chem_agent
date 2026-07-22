# -*- coding: utf-8 -*-
"""tests/test_api.py — 清小搭接入服务（api.py）单元测试。

process_question 全部 mock，不调用真实 LLM。
运行: python -m pytest tests/test_api.py -v
"""

import json

import pytest
from fastapi.testclient import TestClient

import api

pytestmark = pytest.mark.skipif(not hasattr(api, "app"), reason="api 应用不可用")

TEST_KEY = "sk-test-key"
AUTH = {"Authorization": f"Bearer {TEST_KEY}"}
FAKE_ANSWER = "苯的结构式为 [STRUCT:c1ccccc1]，分子式 C6H6。"


@pytest.fixture(autouse=True)
def _mock_pipeline(monkeypatch):
    monkeypatch.setattr(api, "SERVICE_KEY", TEST_KEY)
    monkeypatch.setattr(api, "process_question", lambda q: FAKE_ANSWER)
    monkeypatch.setattr(api, "build_attachments", lambda answer, base: [])
    yield


@pytest.fixture
def client():
    return TestClient(api.app)


def _chat_payload(**overrides):
    payload = {"messages": [{"role": "user", "content": "苯是什么？"}]}
    payload.update(overrides)
    return payload


def _parse_sse(text: str):
    """把 SSE 响应体解析为 (frames, done) 列表与终止哨兵标志。"""
    frames, done = [], False
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data:"):
            continue
        data = block[len("data:"):].strip()
        if data == "[DONE]":
            done = True
        else:
            frames.append(json.loads(data))
    return frames, done


def test_models_requires_auth(client):
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_models_ok(client):
    resp = client.get("/v1/models", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["data"][0]["object"] == "model"


def test_chat_requires_auth(client):
    assert client.post("/v1/chat/completions", json=_chat_payload()).status_code == 401


def test_chat_non_stream(client):
    resp = client.post("/v1/chat/completions", json=_chat_payload(), headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == FAKE_ANSWER
    assert choice["finish_reason"] == "stop"
    usage = data["usage"]
    assert set(usage) == {"prompt_tokens", "completion_tokens", "total_tokens"}


def test_chat_max_tokens_one_accepted(client):
    """探测会发 max_tokens:1，必须能接受。"""
    resp = client.post("/v1/chat/completions",
                       json=_chat_payload(max_tokens=1), headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"]


def test_chat_stream_string_false_is_non_stream(client):
    """stream 严格按布尔解析：字符串 "false" 不视为流式。"""
    resp = client.post("/v1/chat/completions",
                       json=_chat_payload(stream="false"), headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")


def test_chat_stream_sse(client):
    resp = client.post("/v1/chat/completions",
                       json=_chat_payload(stream=True), headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames, done = _parse_sse(resp.text)
    assert done, "缺少 data: [DONE] 终止哨兵"
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert frames[0]["choices"][0]["finish_reason"] is None
    # content 帧按序拼出完整回答
    content = "".join(
        f["choices"][0]["delta"]["content"]
        for f in frames if "content" in f["choices"][0]["delta"]
    )
    assert content == FAKE_ANSWER
    # 最后一帧为 stop 帧，finish_reason 白名单内，usage 合并在 stop 帧
    last = frames[-1]
    assert last["choices"][0]["delta"] == {}
    assert last["choices"][0]["finish_reason"] == "stop"
    assert "usage" in last
    for f in frames:
        assert f["choices"][0]["finish_reason"] in (None, "stop", "length",
                                                    "tool_calls", "content_filter",
                                                    "function_call")


def test_chat_stream_error_fallback(client, monkeypatch):
    """管线抛异常时：stop 帧 + error 字段，finish_reason 不为 error。"""
    def _boom(q):
        raise RuntimeError("upstream boom")
    monkeypatch.setattr(api, "process_question", _boom)
    resp = client.post("/v1/chat/completions",
                       json=_chat_payload(stream=True), headers=AUTH)
    assert resp.status_code == 200
    frames, done = _parse_sse(resp.text)
    assert done
    last = frames[-1]
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["error"]["type"] == "upstream_error"


def test_chat_multimodal_image(client, monkeypatch, tmp_path):
    """content 数组中的 image_url 经识别后并入问题。"""
    captured = {}
    monkeypatch.setattr(api, "_fetch_image_to_temp", lambda url, tmp: "x.png")
    import utils.ocr_utils as ocr
    monkeypatch.setattr(ocr, "image_to_smiles", lambda p: "c1ccccc1")
    monkeypatch.setattr(api, "process_question",
                        lambda q: captured.setdefault("q", q) or FAKE_ANSWER)
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "这是什么分子？"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}
    resp = client.post("/v1/chat/completions", json=payload, headers=AUTH)
    assert resp.status_code == 200
    assert "这是什么分子？" in captured["q"]
    assert "c1ccccc1" in captured["q"]


def test_extract_question_format_variants():
    """多模态格式变体：image_url 字符串形式 / input_image / 纯字符串 part。"""
    # image_url 为字符串 + content 混入纯字符串
    text, images, _, _ = api._extract_question([{"role": "user", "content": [
        "纯文本部分",
        {"type": "image_url", "image_url": "https://x/y.png"},
    ]}])
    assert text == "纯文本部分"
    assert images == ["https://x/y.png"]
    # input_image 类型
    text, images, _, _ = api._extract_question([{"role": "user", "content": [
        {"type": "input_text", "text": "t"},
        {"type": "input_image", "image_url": {"url": "data:image/png;base64,BB"}},
    ]}])
    assert text == "t"
    assert images == ["data:image/png;base64,BB"]
    # 非法结构不崩溃
    assert api._extract_question("not-a-list") == ("", [], [], [])
    assert api._extract_question([{"role": "user", "content": [123, None]}]) == ("", [], [], [])


def test_extract_question_audio_and_file():
    """input_audio 与 file part 按文档字段解析。"""
    text, images, audios, files = api._extract_question([{"role": "user", "content": [
        {"type": "text", "text": "处理这些"},
        {"type": "input_audio", "input_audio": {"url": "https://oss/voice.mp3", "format": "mp3"}},
        {"type": "file", "file": {"url": "https://oss/note.txt", "filename": "note.txt"}},
        {"type": "file", "file": {"file_id": "fid-1", "filename": "doc.pdf"}},
    ]}])
    assert text == "处理这些"
    assert audios == [("https://oss/voice.mp3", "mp3")]
    assert files == [("https://oss/note.txt", "", "note.txt"),
                     ("", "fid-1", "doc.pdf")]


def test_audio_input_graceful_note(client, monkeypatch):
    """音频输入：显式提示暂不支持，不静默丢弃。"""
    captured = {}
    monkeypatch.setattr(api, "process_question",
                        lambda q: captured.setdefault("q", q) or FAKE_ANSWER)
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"url": "https://oss/v.mp3", "format": "mp3"}},
    ]}]}
    resp = client.post("/v1/chat/completions", json=payload, headers=AUTH)
    assert resp.status_code == 200
    assert "暂不支持音频输入" in captured["q"]


def test_file_input_txt_inlined(client, monkeypatch):
    """文本类文件（txt/md/csv）按 URL 下载并内联内容。"""
    captured = {}
    monkeypatch.setattr(api, "_download_text", lambda url: "苯的熔点为 5.5℃")
    monkeypatch.setattr(api, "process_question",
                        lambda q: captured.setdefault("q", q) or FAKE_ANSWER)
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "总结这份文档"},
        {"type": "file", "file": {"url": "https://oss/note.txt", "filename": "note.txt"}},
    ]}]}
    resp = client.post("/v1/chat/completions", json=payload, headers=AUTH)
    assert resp.status_code == 200
    assert "苯的熔点为 5.5℃" in captured["q"]
    assert "note.txt" in captured["q"]


def test_file_input_unsupported_type(client, monkeypatch):
    """不支持的文件类型与仅 file_id：显式提示，不静默丢弃。"""
    captured = {}
    monkeypatch.setattr(api, "process_question",
                        lambda q: captured.setdefault("q", q) or FAKE_ANSWER)
    payload = {"messages": [{"role": "user", "content": [
        {"type": "file", "file": {"url": "https://oss/a.pdf", "filename": "a.pdf"}},
        {"type": "file", "file": {"file_id": "fid-1", "filename": "b.docx"}},
    ]}]}
    resp = client.post("/v1/chat/completions", json=payload, headers=AUTH)
    assert resp.status_code == 200
    assert "暂不支持解析该类型文件" in captured["q"]
    assert "file_id" in captured["q"]


FAKE_ATTACHMENTS = [{
    "fileUrl": "https://host/files/" + "a" * 32 + ".png",
    "fileName": "化学图示-1.png",
    "fileType": "image",
    "mimeType": "image/png",
    "fileSize": 100,
}]


def test_x_soda_attachments_non_stream(client, monkeypatch):
    """有附件时非流式响应顶层带 x_soda.attachments；无附件时不带。"""
    monkeypatch.setattr(api, "build_attachments",
                        lambda answer, base: FAKE_ATTACHMENTS)
    resp = client.post("/v1/chat/completions", json=_chat_payload(), headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["x_soda"]["attachments"][0]["fileType"] == "image"
    assert data["x_soda"]["attachments"][0]["fileUrl"].startswith("https://")

    monkeypatch.setattr(api, "build_attachments", lambda answer, base: [])
    resp = client.post("/v1/chat/completions", json=_chat_payload(), headers=AUTH)
    assert "x_soda" not in resp.json()


def test_x_soda_attachments_stream(client, monkeypatch):
    """流式：x_soda 挂在 stop 帧（与 usage 同帧），增量帧不带。"""
    monkeypatch.setattr(api, "build_attachments",
                        lambda answer, base: FAKE_ATTACHMENTS)
    resp = client.post("/v1/chat/completions",
                       json=_chat_payload(stream=True), headers=AUTH)
    frames, done = _parse_sse(resp.text)
    assert done
    last = frames[-1]
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["x_soda"]["attachments"][0]["mimeType"] == "image/png"
    assert "usage" in last
    for f in frames[:-1]:
        assert "x_soda" not in f


def test_serve_attachment(client, tmp_path, monkeypatch):
    """/files/{name}：合法文件可下载；非法名与不存在的文件 404。"""
    import core.attachments as att_mod
    name = "b" * 32 + ".png"
    real_dir = api.settings.service.attachment_dir
    real_dir.mkdir(parents=True, exist_ok=True)
    target = real_dir / name
    target.write_bytes(b"\x89PNG fake")
    try:
        resp = client.get(f"/files/{name}")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/png")
        assert resp.content == b"\x89PNG fake"
    finally:
        target.unlink(missing_ok=True)
    assert client.get(f"/files/{'c' * 32}.png").status_code == 404
    assert client.get("/files/..%2F..%2Fetc%2Fpasswd").status_code in (404, 422)
    assert client.get("/files/not-a-valid-name.png").status_code == 404


def test_image_without_vision_config(client, monkeypatch):
    """收到图片但未配置视觉模型：回答中明确说明原因，不静默忽略。"""
    import types
    monkeypatch.setattr(api, "settings", types.SimpleNamespace(
        vision=types.SimpleNamespace(is_configured=False),
        service=types.SimpleNamespace(public_base_url="", attachment_dir=None,
                                      attachment_ttl=0),
    ))
    captured = {}
    monkeypatch.setattr(api, "process_question",
                        lambda q: captured.setdefault("q", q) or FAKE_ANSWER)
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}
    resp = client.post("/v1/chat/completions", json=payload, headers=AUTH)
    assert resp.status_code == 200
    assert "VISION_MODEL" in captured["q"]
