# -*- coding: utf-8 -*-
"""utils/ocr_utils.describe_image 单元测试（mock 视觉 API，不访问网络）。

运行: python -m pytest tests/test_ocr_utils.py -v
"""

import types

import pytest

import utils.ocr_utils as ocr


def _fake_response(content: str, status: int = 200):
    resp = types.SimpleNamespace(status_code=status, text="")
    resp.json = lambda: {"choices": [{"message": {"content": content}}]}
    return resp


@pytest.fixture
def fake_vision(monkeypatch, tmp_path):
    """配置视觉模型 + 测试图片 + 可控响应；返回 (state, 图片路径)。"""
    monkeypatch.setattr(ocr, "settings", types.SimpleNamespace(vision=types.SimpleNamespace(
        api_key="k", base_url="http://v", model_name="glm", is_configured=True)))
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG")
    state = {"content": "类型：结构式\n内容：苯环，SMILES: c1ccccc1", "status": 200}

    def fake_post(url, headers=None, json=None, timeout=None):
        return _fake_response(state["content"], state["status"])

    monkeypatch.setattr(ocr.requests, "post", fake_post)
    return state, str(img)


def test_describe_well_formed(fake_vision):
    state, img = fake_vision
    desc = ocr.describe_image(img)
    assert desc["type"] == "结构式"
    assert "c1ccccc1" in desc["content"]


def test_describe_malformed_falls_back(fake_vision):
    """视觉输出不带 类型/内容 格式：整体作为 content，type=未分类。"""
    state, img = fake_vision
    state["content"] = "图中是一个苯环"
    desc = ocr.describe_image(img)
    assert desc["type"] == "未分类"
    assert desc["content"] == "图中是一个苯环"


def test_describe_empty_content_returns_none(fake_vision):
    """思考过长/空输出 → None（不回退 reasoning_content）。"""
    state, img = fake_vision
    state["content"] = ""
    assert ocr.describe_image(img) is None


def test_describe_http_error_returns_none(fake_vision):
    state, img = fake_vision
    state["status"] = 400
    assert ocr.describe_image(img) is None


def test_describe_unconfigured_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr, "settings", types.SimpleNamespace(
        vision=types.SimpleNamespace(is_configured=False)))
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG")
    assert ocr.describe_image(str(img)) is None
