# -*- coding: utf-8 -*-
"""tests/test_upload_limits.py — 上传图片的大小与真实类型约束（安全审查 R6）。

背景：图片是前端 `FileReader.readAsDataURL` **原样 base64** 内联发送的，而后端
过去**不做任何大小检查**（`_DATA_URL_RE` 不限制长度、解码后直接 `write_bytes`），
唯一的约束是反代的 `client_max_body_size`。于是：
* 直接构造 JSON 就能塞进几百 MB → 内存 + 磁盘被打满；
* `data:image/png;base64,<任意字节>` 都会被当成图片落盘（不看内容）；
* http(s) 那条 URL 由用户指定，`requests.get` 默认**把整个响应读进内存**再写盘。

本文件锁住三件事：
1. **大小**：单张 8MB、单次合计 16MB（解码后），在**解码/落盘之前**就拒绝；
2. **真实类型**：按魔数判断，扩展名也跟着内容走（声明 png 实际 jpg 不再写错）；
3. **回收公平**：先按**单会话**上限在内部消化，再走全局最旧优先 —— 削弱
   "一个人猛传图就能把别人的附件挤掉"。
"""

import base64
import re
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from core import web_api

_ROOT = Path(__file__).resolve().parents[1]

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
JPG = b"\xff\xd8\xff\xe0" + b"0" * 64
GIF = b"GIF89a" + b"0" * 64
BMP = b"BM" + b"0" * 64
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"0" * 64
NOT_IMAGE = b"hello, I am not an image at all"


def _data_url(data: bytes, mime="image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


# ---------------------------------------------------------------- 1. 大小估算

@pytest.mark.parametrize("n", [0, 1, 2, 3, 100, 1023, 1024, 65536])
def test_b64_size_estimate_matches_decode(n):
    payload = base64.b64encode(b"x" * n).decode()
    assert web_api._b64_decoded_size(payload) == n


def test_budget_allows_normal_images():
    web_api._check_image_budget([_data_url(PNG), _data_url(JPG)])   # 不抛


def test_budget_rejects_single_too_big(monkeypatch):
    monkeypatch.setattr(web_api, "_MAX_IMAGE_BYTES", 1024)
    with pytest.raises(HTTPException) as ei:
        web_api._check_image_budget([_data_url(b"x" * 2048)])
    assert ei.value.status_code == 400
    assert "单张上限" in ei.value.detail


def test_budget_rejects_total_too_big(monkeypatch):
    monkeypatch.setattr(web_api, "_MAX_IMAGE_BYTES", 0)          # 只测合计
    monkeypatch.setattr(web_api, "_MAX_TOTAL_IMAGE_BYTES", 1000)
    with pytest.raises(HTTPException) as ei:
        web_api._check_image_budget([_data_url(b"x" * 600),
                                     _data_url(b"x" * 600)])
    assert ei.value.status_code == 400
    assert "单次上限" in ei.value.detail


def test_budget_boundary_is_inclusive(monkeypatch):
    """正好等于上限要放行（别把边界做成"少一个字节也拒"）。"""
    monkeypatch.setattr(web_api, "_MAX_IMAGE_BYTES", 1000)
    monkeypatch.setattr(web_api, "_MAX_TOTAL_IMAGE_BYTES", 2000)
    web_api._check_image_budget([_data_url(b"x" * 1000),
                                 _data_url(b"x" * 1000)])


def test_budget_ignores_http_urls():
    """http(s) 图片的大小只能下载时才知道，估算阶段不该误判。"""
    web_api._check_image_budget(["https://example.com/a.png"])


# ---------------------------------------------------------------- 2. 真实类型

@pytest.mark.parametrize("data,kind", [(PNG, "png"), (JPG, "jpg"), (GIF, "gif"),
                                       (BMP, "bmp"), (WEBP, "webp")])
def test_sniff_image_recognizes_real_formats(data, kind):
    assert web_api._sniff_image(data) == kind


@pytest.mark.parametrize("data", [b"", NOT_IMAGE, b"RIFF\x00\x00\x00\x00AVI ",
                                  b"<html></html>"])
def test_sniff_image_rejects_non_images(data):
    assert web_api._sniff_image(data) is None


def test_save_uses_sniffed_extension_not_declared_mime(tmp_path):
    """声明是 png、实际是 jpg → 按**内容**存成 .jpg（不再写错扩展名）。"""
    out = web_api._save_upload_image(_data_url(JPG, mime="image/png"), 0, tmp_path)
    assert out and out.endswith(".jpg")
    assert Path(out).read_bytes() == JPG


def test_save_rejects_non_image_and_writes_nothing(tmp_path):
    out = web_api._save_upload_image(_data_url(NOT_IMAGE), 0, tmp_path)
    assert out is None
    assert list(tmp_path.iterdir()) == [], "非图片内容不得落盘"


def test_save_rejects_oversized_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(web_api, "_MAX_IMAGE_BYTES", 64)
    out = web_api._save_upload_image(_data_url(PNG + b"x" * 4096), 0, tmp_path)
    assert out is None
    assert list(tmp_path.iterdir()) == []


def test_save_accepts_normal_image(tmp_path):
    out = web_api._save_upload_image(_data_url(PNG), 0, tmp_path)
    assert out and out.endswith(".png")
    assert Path(out).read_bytes() == PNG


# ---------------------------------------------------------------- 3. http(s) 边下边卡

class _StreamResp:
    """替身响应：`iter_content` 能吐出比上限多得多的数据。"""

    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks
        self.read = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, size=65536):
        for c in self._chunks:
            self.read += 1
            yield c


def test_http_download_aborts_when_over_limit(tmp_path, monkeypatch):
    """★ 不能"先整包读进内存、再判断太大" —— 必须边下边卡。"""
    import requests

    monkeypatch.setattr(web_api, "_validate_download_url", lambda u: True)
    monkeypatch.setattr(web_api, "_MAX_IMAGE_BYTES", 1000)
    chunks = [b"x" * 256] * 100          # 共 25KB，远超 1000
    resp = _StreamResp(chunks)
    monkeypatch.setattr(requests, "get", lambda *a, **k: resp)

    out = web_api._save_upload_image("https://example.com/a.png", 0, tmp_path)
    assert out is None
    assert resp.read < len(chunks), "应当在读到上限时就中断，而不是读完全部"
    assert list(tmp_path.iterdir()) == []


def test_http_download_rejects_non_image(tmp_path, monkeypatch):
    import requests

    monkeypatch.setattr(web_api, "_validate_download_url", lambda u: True)
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: _StreamResp([NOT_IMAGE]))
    assert web_api._save_upload_image("https://example.com/a.png", 0,
                                      tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_http_download_uses_streaming(tmp_path, monkeypatch):
    """必须带 stream=True（否则 requests 先把整包读进内存，白卡）。"""
    import requests

    seen = {}

    def fake_get(url, **kw):
        seen.update(kw)
        return _StreamResp([PNG])

    monkeypatch.setattr(web_api, "_validate_download_url", lambda u: True)
    monkeypatch.setattr(requests, "get", fake_get)
    out = web_api._save_upload_image("https://example.com/a.png", 0, tmp_path)
    assert seen.get("stream") is True
    assert out and out.endswith(".png")


# ---------------------------------------------------------------- 4. 端点接线（真的会 400）

@pytest.fixture
def web(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(web_api.router)
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", tmp_path / "web_sessions")
    monkeypatch.setattr(web_api, "RATE_LIMIT_PER_MINUTE", 0)
    web_api._reset_rate_limit_for_tests()
    web_api._session_dir("a" * 32)      # R11：让用例里的固定 id 成为"已下发"
    with TestClient(app) as client:
        yield client
    web_api._reset_rate_limit_for_tests()


def _vision_payload(url):
    return {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": url}}]}],
        "stream": False, "session_id": "a" * 32}


HEADERS = {"X-Chem-Api-Key": "sk-x", "X-Chem-Vision-Model": "glm-4.6v",
           "X-Chem-Base-Url": "https://api.deepseek.com/v1"}


def test_endpoint_rejects_oversized_image(web, monkeypatch):
    """★ 绕开前端直接构造 JSON → 必须 400（这是"前端拦不住"的那条路）。"""
    monkeypatch.setattr(web_api, "_MAX_IMAGE_BYTES", 1024)
    r = web.post("/api/chat", json=_vision_payload(_data_url(b"x" * 5000)),
                 headers=HEADERS)
    assert r.status_code == 400
    assert "单张上限" in r.json()["detail"]


def test_endpoint_rejects_oversized_total(web, monkeypatch):
    monkeypatch.setattr(web_api, "_MAX_IMAGE_BYTES", 0)
    monkeypatch.setattr(web_api, "_MAX_TOTAL_IMAGE_BYTES", 1000)
    body = _vision_payload(_data_url(b"x" * 600))
    body["messages"][0]["content"].append(
        {"type": "image_url", "image_url": {"url": _data_url(b"x" * 600)}})
    r = web.post("/api/chat", json=body, headers=HEADERS)
    assert r.status_code == 400
    assert "单次上限" in r.json()["detail"]


# ---------------------------------------------------------------- 5. 回收：先会话内，再全局

def _mkpng(path, size, age=3 * 3600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    import os
    t = time.time() - age
    os.utime(path, (t, t))
    return path


def test_session_cap_prunes_within_session_only(tmp_path, monkeypatch):
    """★ 一个会话超额 → 只删它自己的旧图，别的会话一根毛都不动。"""
    root = tmp_path / "web_sessions"
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", root)
    monkeypatch.setattr(web_api, "WEB_SESSION_MAX_BYTES", 1000)
    a1 = _mkpng(root / "s1" / "a1.png", 600, age=5 * 3600)
    a2 = _mkpng(root / "s1" / "a2.png", 600, age=4 * 3600)
    b1 = _mkpng(root / "s2" / "b1.png", 600, age=9 * 3600)   # 最旧，但在小会话里

    removed = web_api.prune_web_attachments(max_bytes=10_000, max_files=0)
    assert removed == 1
    assert not a1.exists(), "应当删掉超额会话里最旧的那个"
    assert a2.exists()
    assert b1.exists(), "★ 别的会话不该被牵连（全局配额根本没超）"


def test_session_cap_keeps_fresh_files(tmp_path, monkeypatch):
    """单会话回收同样受 `_PRUNE_MIN_AGE` 保护（可能正被某个请求引用）。"""
    root = tmp_path / "web_sessions"
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", root)
    monkeypatch.setattr(web_api, "WEB_SESSION_MAX_BYTES", 100)
    fresh = _mkpng(root / "s1" / "fresh.png", 500, age=1)      # 刚写的
    assert web_api.prune_web_attachments(max_bytes=10_000, max_files=0) == 0
    assert fresh.exists()


def test_global_quota_still_works(tmp_path, monkeypatch):
    """加了单会话那道之后，全局回收不能失效（回归护栏）。"""
    root = tmp_path / "web_sessions"
    monkeypatch.setattr(web_api, "_SESSIONS_DIR", root)
    monkeypatch.setattr(web_api, "WEB_SESSION_MAX_BYTES", 10_000_000)
    old = _mkpng(root / "s1" / "old.png", 600, age=5 * 3600)
    new = _mkpng(root / "s2" / "new.png", 600, age=4 * 3600)
    removed = web_api.prune_web_attachments(max_bytes=0, max_files=1)
    assert removed == 1 and not old.exists() and new.exists()


# ---------------------------------------------------------------- 6. 前后端上限不许漂移

def test_frontend_and_backend_limits_match():
    """前端提示的 8MB/16MB 必须与后端常量一致（否则用户会撞上"前端放行、后端拒绝"）。"""
    html = (_ROOT / "web/index.html").read_text(encoding="utf-8")
    one = re.search(r"const MAX_ONE = (\d+) \* 1024 \* 1024", html)
    all_ = re.search(r"MAX_ALL = (\d+) \* 1024 \* 1024", html)
    assert one and all_, "前端应当有 MAX_ONE / MAX_ALL 两个常量"
    assert int(one.group(1)) * 1024 * 1024 == web_api._MAX_IMAGE_BYTES
    assert int(all_.group(1)) * 1024 * 1024 == web_api._MAX_TOTAL_IMAGE_BYTES
