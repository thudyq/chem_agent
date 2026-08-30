# -*- coding: utf-8 -*-
"""utils/ocr_utils.describe_image 单元测试（mock 视觉 API，不访问网络）。

运行: python -m pytest tests/test_ocr_utils.py -v
"""

import types

import pytest
import requests

import utils.ocr_utils as ocr


def _fake_response(content: str, status: int = 200):
    resp = types.SimpleNamespace(status_code=status, text="")
    resp.json = lambda: {"choices": [{"message": {"content": content}}]}
    return resp


@pytest.fixture
def fake_vision(monkeypatch):
    """配置视觉模型 + mock 读图 + 可控响应；返回 (state, 图片路径)。

    读图层（_read_image_b64）整体 mock 掉——沙箱禁止测试进程写文件
    （含项目内临时目录），describe_image 只依赖 base64 字符串。
    """
    monkeypatch.setattr(ocr, "settings", types.SimpleNamespace(vision=types.SimpleNamespace(
        api_key="k", base_url="http://v", model_name="glm", is_configured=True)))
    monkeypatch.setattr(ocr, "_read_image_b64", lambda path: "aGVsbG8=")
    state = {"content": "类型：结构式\n内容：苯环，SMILES: c1ccccc1", "status": 200}

    def fake_post(url, headers=None, json=None, timeout=None):
        return _fake_response(state["content"], state["status"])

    monkeypatch.setattr(ocr.requests, "post", fake_post)
    return state, "x.png"


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


def test_describe_unconfigured_returns_none(monkeypatch):
    """未配置视觉模型 → None（在读图之前即返回，无需真实图片）。"""
    monkeypatch.setattr(ocr, "settings", types.SimpleNamespace(
        vision=types.SimpleNamespace(is_configured=False)))
    assert ocr.describe_image("no-such-file.png") is None


# ---------------- B1：结构式 SMILES RDKit 硬校验（20260826） ----------------

def test_describe_structure_good_smiles_ok(fake_vision):
    """结构式内容含合法 SMILES（c1ccccc1）→ smiles_ok=True。"""
    state, img = fake_vision  # 默认 "苯环，SMILES: c1ccccc1"
    desc = ocr.describe_image(img)
    assert desc["smiles_ok"] is True


def test_describe_structure_bad_smiles_flagged(fake_vision):
    """结构式内容含不可解析 SMILES（XYZABC）→ smiles_ok=False（供 B3 示警）。"""
    state, img = fake_vision
    state["content"] = "类型：结构式\n内容：SMILES: XYZABC"
    desc = ocr.describe_image(img)
    assert desc["smiles_ok"] is False


def test_describe_non_structure_smiles_ok(fake_vision):
    """非结构式（机理图）→ 无可核验 SMILES 断言 → smiles_ok=True。"""
    state, img = fake_vision
    state["content"] = "类型：机理图\n内容：苯环 π 进攻 SO3 的 S"
    desc = ocr.describe_image(img)
    assert desc["smiles_ok"] is True


def test_describe_structure_text_desc_ok(fake_vision):
    """结构式但无法确定、用文字描述 → 无 SMILES token → smiles_ok=True。"""
    state, img = fake_vision
    state["content"] = "类型：结构式\n内容：苯环连一个硝基（无法确定 SMILES）"
    desc = ocr.describe_image(img)
    assert desc["smiles_ok"] is True


# ---------------- 重试容错（连接不稳定，20260818） ----------------

def test_describe_retries_then_succeeds(fake_vision, monkeypatch):
    """网络异常（连接不稳定）→ 重试成功（默认最多 3 次尝试）。"""
    state, img = fake_vision
    calls = {"n": 0}

    def flaky_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ConnectionError("boom")
        return _fake_response(state["content"], state["status"])

    monkeypatch.setattr(ocr, "_RETRY_DELAY", 0)
    monkeypatch.setattr(ocr.requests, "post", flaky_post)
    desc = ocr.describe_image(img)
    assert desc and "c1ccccc1" in desc["content"]
    assert calls["n"] == 2  # 第 1 次失败 + 第 2 次成功


def test_describe_all_attempts_fail(fake_vision, monkeypatch):
    """持续网络异常 → 重试耗尽返回 None（默认 3 次尝试）。"""
    _, img = fake_vision
    calls = {"n": 0}

    def boom_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(ocr, "_RETRY_DELAY", 0)
    monkeypatch.setattr(ocr.requests, "post", boom_post)
    assert ocr.describe_image(img) is None
    assert calls["n"] == 3  # 初始 1 次 + 重试 2 次


def test_describe_5xx_retried(fake_vision, monkeypatch):
    """5xx 服务端错误 → 重试；耗尽返回 None。"""
    _, img = fake_vision
    calls = {"n": 0}

    def err_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return _fake_response("", 500)

    monkeypatch.setattr(ocr, "_RETRY_DELAY", 0)
    monkeypatch.setattr(ocr.requests, "post", err_post)
    assert ocr.describe_image(img) is None
    assert calls["n"] == 3


def test_describe_400_not_retried(fake_vision, monkeypatch):
    """400（模型不支持视觉，配置性错误）不重试，1 次即返回。"""
    _, img = fake_vision
    calls = {"n": 0}

    def bad_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return _fake_response("", 400)

    monkeypatch.setattr(ocr.requests, "post", bad_post)
    assert ocr.describe_image(img) is None
    assert calls["n"] == 1


def test_describe_empty_content_retried(fake_vision, monkeypatch):
    """200 但 content 空（瞬时抖动）→ 重试；重试成功则返回。"""
    state, img = fake_vision
    calls = {"n": 0}

    def flaky_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_response("")
        return _fake_response(state["content"], state["status"])

    monkeypatch.setattr(ocr, "_RETRY_DELAY", 0)
    monkeypatch.setattr(ocr.requests, "post", flaky_post)
    desc = ocr.describe_image(img)
    assert desc and "c1ccccc1" in desc["content"]
    assert calls["n"] == 2


def test_describe_max_attempts_param(fake_vision, monkeypatch):
    """max_attempts=1 → 失败不重试。"""
    _, img = fake_vision
    calls = {"n": 0}

    def boom_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(ocr.requests, "post", boom_post)
    assert ocr.describe_image(img, max_attempts=1) is None
    assert calls["n"] == 1


# ---------------- _extract_description（20260828 思考回退收敛） ----------------
def test_extract_description_structured():
    """含"类型/内容"两行 → 正常解析（哪怕前面有分析文字）。"""
    d = ocr._extract_description("前置分析。\n类型：结构式\n内容：这是苯环。")
    assert d["type"] == "结构式"
    assert d["content"] == "这是苯环。"


def test_extract_description_draft_converges():
    """无两行格式的思考草稿 → 收敛标注，不整段透传（防止污染下游）。"""
    d = ocr._extract_description("Wait, let me look. Actually the radical is on...")
    assert d["type"] == "未分类_草稿"
    assert d["content"].startswith("Wait")


def test_extract_description_long_truncated():
    """超长草稿截断到 max_len 上限。"""
    d = ocr._extract_description("x" * 800)
    assert len(d["content"]) <= 601   # 600 + "…"
    assert d["content"].endswith("…")


def test_extract_description_empty():
    """空文本 → 空 content。"""
    assert ocr._extract_description("") == {"type": "未分类", "content": ""}


# ---------------- _best_effort_extract（20260828 思考链抢救） ----------------
def test_best_effort_extract_keeps_useful():
    """思考草稿里含结构/文字线索 → 尽量抢救，去掉无用的"Let me look"等。"""
    d = ocr._best_effort_extract(
        'Let me look. The image reads: "请按稳定性排序"。'
        'Structure 1 is a cyclopentane ring with radical on C. SMILES: C1CCCC1')
    assert d["type"] == "未分类_草稿"
    assert "SMILES: C1CCCC1" in d["content"]
    assert "Let me look" not in d["content"]


def test_best_effort_extract_fallback_whole():
    """无关键词草稿 → 兜底整段（不空白）。"""
    d = ocr._best_effort_extract("Wait, actually looking at the coordinates ...")
    assert d["type"] == "未分类_草稿"
    assert d["content"].startswith("Wait")


def test_best_effort_extract_empty():
    """空 → 空 content。"""
    assert ocr._best_effort_extract("") == {"type": "未分类", "content": ""}
