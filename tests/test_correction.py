# -*- coding: utf-8 -*-
"""P2 渲染反馈闭环测试：渲染/校验失败时回传 LLM 自动修正。

验证四个行为：
1. 无失败不触发重试（ask_llm 仅调用 1 次）；
2. 校验失败 → 修正重试 → 采用修正版；
3. 渲染失败 → 修正重试 → 采用修正版；
4. 修正仍失败 → 降级输出且不无限重试（最多 max_corrections 次）。
"""

import pytest

from app import _build_correction_prompt, process_question
from renderers import registry


@pytest.fixture
def flawed_renderers(monkeypatch):
    """STRUCT 渲染器：c1ccccc1 成功，其余（合法但模拟内部失败）返回失败串。"""

    def render_struct(smiles, label=None):
        if smiles == "c1ccccc1":
            return "RENDERED:c1ccccc1"
        return f"（结构渲染失败：无法为「{smiles}」生成结构式）"

    original = dict(registry.RENDERER_REGISTRY)
    registry.RENDERER_REGISTRY["STRUCT"] = render_struct
    yield
    registry.RENDERER_REGISTRY.clear()
    registry.RENDERER_REGISTRY.update(original)


def test_no_failure_no_retry(fake_rdkit, fake_renderers, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or "苯是 [STRUCT:c1ccccc1]。")
    result = process_question("画苯")
    assert len(calls) == 1, "无失败不应触发修正重试"
    assert "RENDERED:c1ccccc1" in result


def test_correction_after_validation_failure(fake_rdkit, fake_renderers,
                                             monkeypatch):
    calls = []
    answers = [
        "苯是 [STRUCT:XYZABC]。",      # 非法 SMILES → 校验拦截 → 触发修正
        "苯是 [STRUCT:c1ccccc1]。",     # 修正版
    ]
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or answers.pop(0))
    result = process_question("画苯")
    assert len(calls) == 2, "应触发一次修正重试"
    assert "RENDERED:c1ccccc1" in result
    assert "无效 SMILES" not in result
    assert "[STRUCT:XYZABC]" not in result


def test_correction_after_render_failure(fake_rdkit, flawed_renderers,
                                         monkeypatch):
    calls = []
    answers = [
        "看 [STRUCT:CCl]。",            # CCl 合法但渲染器失败 → 触发修正
        "看 [STRUCT:c1ccccc1]。",       # 修正版
    ]
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or answers.pop(0))
    result = process_question("画苯")
    assert len(calls) == 2
    assert "RENDERED:c1ccccc1" in result
    assert "结构渲染失败" not in result


def test_correction_exhausted_degrades(fake_rdkit, fake_renderers,
                                       monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or "苯是 [STRUCT:XYZABC]。")
    result = process_question("画苯")
    assert len(calls) == 2, "最多修正一次，不应无限重试"
    assert "图示无法渲染" in result


def test_correction_prompt_contains_failure_info():
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = "苯是 [STRUCT:XYZABC]。"
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    prompt = _build_correction_prompt("画苯", text, [(tag, vr.reason)])
    assert "[STRUCT:XYZABC]" in prompt
    assert "无效 SMILES" in prompt
    assert "画苯" in prompt
    assert "修正要求" in prompt
