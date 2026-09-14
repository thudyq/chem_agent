# -*- coding: utf-8 -*-
"""tests/test_outbound_redirects.py — 出站请求一律不跟随重定向（安全审查 R3）。

背景：端点地址与图片 URL 在进入管线前都会过 SSRF 校验（`client_host_allowed` /
`_validate_download_url`），但 **302 之后跳到哪不受那个校验管** —— 只要跟随，
校验就等于作废：一个公网域名可以先通过校验，再 302 到 `169.254.169.254`
（云元数据）或内网管理面板。所以每一处"带着用户凭证/代表服务端出站"的请求都
必须显式 `allow_redirects=False`，并把 3xx 当失败。

本文件把**四个出站点**逐个锁住（全部不打网络）：
`core/llm_client._stream_chat`（主模型）、`utils.ocr_utils._describe_once`（视觉）、
`core.web_api._save_upload_image`（网页图片）、`api._download_text`（/v1 附件）。
"""

import pytest
import requests

import api
import core.llm_client as lc
import core.web_api as web_api
import utils.ocr_utils as ocr


class _Resp:
    def __init__(self, status, content=b"", text=""):
        self.status_code = status
        self.content = content
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Session:
    """替身 `requests.Session`：只记录参数，返回一个 302。"""

    def __init__(self, resp):
        self.resp = resp
        self.seen = {}

    def post(self, url, **kw):
        self.seen["url"] = url
        self.seen.update(kw)
        return self.resp


# ---------------------------------------------------------------- 主模型调用

def test_llm_call_disables_redirects(monkeypatch):
    sess = _Session(_Resp(302, text="moved"))
    monkeypatch.setattr(lc.credentials, "session", lambda: sess)
    with pytest.raises(lc._HttpFailure) as ei:
        lc._stream_chat("https://api.example.com/v1/chat/completions", {}, {})
    assert sess.seen["allow_redirects"] is False
    assert ei.value.status == 302
    assert "重定向" in str(ei.value)


def test_llm_http_failure_hint_only_for_3xx():
    assert "重定向" not in str(lc._HttpFailure(401, "bad key"))
    assert "重定向" not in str(lc._HttpFailure(500, "boom"))
    assert "重定向" in str(lc._HttpFailure(301, ""))


# ---------------------------------------------------------------- 视觉模型调用

def test_vision_call_disables_redirects(monkeypatch):
    seen = {}

    def fake_post(url, **kw):
        seen["url"] = url
        seen.update(kw)
        return _Resp(302, text="moved")

    monkeypatch.setattr(ocr.requests, "post", fake_post)
    desc, retryable = ocr._describe_once(
        "https://vision.example/v1/chat/completions",
        {"Authorization": "Bearer sk-x"}, {"model": "m"}, "m")
    assert seen["allow_redirects"] is False
    assert desc is None
    assert retryable is False, "端点跳转是配置问题，重试没有意义"


# ---------------------------------------------------------------- 网页图片下载

def test_web_image_download_disables_redirects(monkeypatch, tmp_path):
    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        seen.update(kw)
        return _Resp(302, content=b"internal-stuff", text="moved")

    # web_api 里是函数内 `import requests`，模块上没有该属性 → 打全局
    monkeypatch.setattr(requests, "get", fake_get)
    out = web_api._save_upload_image("https://example.com/a.png", 0, tmp_path)
    assert seen["allow_redirects"] is False
    assert out is None, "302 不能被当成下载成功"


# ---------------------------------------------------------------- /v1 附件下载

def test_v1_download_disables_redirects(monkeypatch):
    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        seen.update(kw)
        return _Resp(302, content=b"internal-stuff", text="moved")

    monkeypatch.setattr(api, "_validate_download_url", lambda u: True)
    monkeypatch.setattr(api.requests, "get", fake_get)
    assert api._download_text("https://example.com/a.txt") is None
    assert seen["allow_redirects"] is False


def test_v1_image_download_disables_redirects(monkeypatch, tmp_path):
    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        seen.update(kw)
        return _Resp(302, content=b"internal-stuff", text="moved")

    monkeypatch.setattr(api, "_validate_download_url", lambda u: True)
    monkeypatch.setattr(api.requests, "get", fake_get)
    assert api._fetch_image_to_temp("https://example.com/a.png",
                                    str(tmp_path)) is None
    assert seen["allow_redirects"] is False


# ---------------------------------------------------------------- 用户可见提示

@pytest.mark.parametrize("raw", ["HTTP 301: moved", "HTTP 302: <html>",
                                 "HTTP 307: T", "http 308: y"])
def test_friendly_error_explains_redirect(raw):
    msg = web_api._friendly_error(raw)
    assert "重定向" in msg and "接口地址" in msg


def test_friendly_error_untouched_for_other_codes():
    assert "重定向" not in web_api._friendly_error("HTTP 404: model_not_found")
    assert "重定向" not in web_api._friendly_error("HTTP 429: rate limit")
