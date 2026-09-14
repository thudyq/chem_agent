# -*- coding: utf-8 -*-
"""tests/test_struct_rewrite.py — 手术式结构重写测试。

背景：修正循环对 SMILES/label 级错误成功率不稳（锚定效应——修正 prompt
含错误答案原文）。结构重写把箭头重写的成功模式推广到结构错误：给 LLM
"名称 + 约束 + 上下文"的去锚定微任务（不含错误 SMILES），重写产物保留
原组件 id/mode/系数，整个标记校验通过才采纳。

验证：
1. _is_struct_fixable 分类（SMILES/label 类放行，守恒/箭头类不拦）；
2. 顶层 STRUCT 与 COMPOSITE 组件的端到端重写（含去锚定断言）；
3. 重写产物保留组件 id（被 MECHARROW 引用的组件不断链）；
4. 重写失败回退常规修正。
"""

import pytest

from app import (process_question, _is_struct_fixable, _rewrite_struct_smiles)
from core.prompt_manager import load_struct_rewrite_prompt

STRUCT_PROMPT = load_struct_rewrite_prompt()

_BAD_CATION = "中间体是 [STRUCT:C[C+](C)CC,label=2-丁基碳正离子]。"
_GOOD_CATION_STRUCT = "[STRUCT:CC[CH+]C,label=2-丁基碳正离子]"


def _struct_tag(text=None):
    from core.tag_parser import parse_tags
    return parse_tags(text or _BAD_CATION)[0]


@pytest.mark.parametrize("text,reason,expected", [
    ("[STRUCT:XYZABC]", "无效 SMILES「XYZABC」", True),
    ("[STRUCT:CCO]", "化学校验：label「2-丁醇」与 SMILES 不一致——应为 4 个碳", True),
    ("[STRUCT:CCO]", "label 标注「质子化」但 SMILES 中没有带正电的杂原子", True),
    ("[STRUCT:CC]", "label 含「自由基」但 SMILES 没有任何带单电子的原子", True),
    ("[STRUCT:CCO]", "化学校验：第 1 步两侧原子不守恒（C2H6O vs C4H10O）", False),
    ("[STRUCT:CCO]", "MECHARROW 源端点「a:0-1」…没有成键", False),
    ("[STRUCT:CCO]", "label 过长（30 > 24 字符）", False),
])
def test_is_struct_fixable_toplevel(text, reason, expected):
    assert _is_struct_fixable(_struct_tag(text), reason) is expected


def test_is_struct_fixable_composite_component():
    """COMPOSITE：错误须定位到具体组件且为 SMILES/label 类。"""
    text = ("[COMPOSITE:reaction][STRUCT:CC(C)(C)Br,id=r0][ARROW:type=single]"
            "[STRUCT:C[C+](C)CC,id=cat][/COMPOSITE]")
    tag = _struct_tag(text)
    assert _is_struct_fixable(tag, "组件 cat: 无效 SMILES「…」") is True
    assert _is_struct_fixable(
        tag, "组件 cat: 化学校验：label「叔丁基正离子」与 SMILES 不一致") is True
    assert _is_struct_fixable(tag, "组件 cat: 系数格式错误「…」") is False
    assert _is_struct_fixable(
        tag, "化学校验：第 1 步两侧原子不守恒（…）") is False


def test_struct_rewrite_end_to_end(monkeypatch):
    """顶层 STRUCT：label 与 SMILES 不符 → 去锚定微任务重写 → 渲染通过。
    断言微任务 prompt 含名称与约束、不含错误 SMILES（去锚定）。"""
    pytest.importorskip("rdkit")
    calls = []

    def fake_ask(q, system_prompt=None, **k):
        calls.append({"q": q, "sp": system_prompt})
        if system_prompt == STRUCT_PROMPT:
            return _GOOD_CATION_STRUCT
        return _BAD_CATION

    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", fake_ask)
    result = process_question("画出 2-溴丁烷 SN1 的中间体", max_corrections=1)
    assert len(calls) == 2                       # 主生成 + 手术重写
    surg = calls[1]
    assert surg["sp"] == STRUCT_PROMPT
    assert "2-丁基碳正离子" in surg["q"]          # 名称在场
    assert "应为 4 个碳" in surg["q"]             # 约束在场
    assert "C[C+](C)CC" not in surg["q"]          # 错误 SMILES 不在场（去锚定）
    assert "tikzpicture" in result                # 重写后渲染成功
    assert "无法渲染" not in result


def test_struct_rewrite_component_preserves_id(monkeypatch):
    """COMPOSITE 组件级重写：保留 id/全部属性（该组件被 MECHARROW 引用）。
    病例：叔丁基正离子写成 5 碳（label 不符）+ 自愈箭头指向它。"""
    pytest.importorskip("rdkit")
    bad = ("[COMPOSITE:reaction]"
           "[STRUCT:CC(C)(C)Br,label=叔丁基溴,id=r0]"
           "[ARROW:type=single,慢]"
           "[STRUCT:C[C+](C)CC,label=叔丁基正离子,id=cat][PLUS]"
           "[STRUCT:[Br-],label=Br-,id=br]"
           "[MECHARROW:r0:1-4>r0:4]"
           "[/COMPOSITE]")
    monkeypatch.setattr(
        "app.ask_llm",
        lambda *a, **k: "[STRUCT:C[C+](C)C,label=叔丁基正离子]")
    tag = _struct_tag(bad)
    from core.tag_validator import validate_tag
    vr = validate_tag(tag)
    assert not vr.ok and "组件 cat" in vr.reason
    new_raw = _rewrite_struct_smiles("介绍 SN1 机理", bad, tag, vr.reason)
    assert new_raw is not None
    assert "id=cat" in new_raw                     # id 保留（引用不断链）
    assert "C[C+](C)CC" not in new_raw             # 错误 SMILES 已替换
    assert "[MECHARROW:r0:1-4>r0:4]" in new_raw    # 箭头原样保留


def test_struct_rewrite_fallback_to_correction(monkeypatch):
    """重写输出非法 → 同轮回退常规部分修正，修正成功。"""
    pytest.importorskip("rdkit")
    calls = []

    def fake_ask(q, system_prompt=None, **k):
        calls.append({"q": q, "sp": system_prompt})
        if system_prompt == STRUCT_PROMPT:
            return "我写不出来。"                     # 无标记 → 重写失败
        if "渲染失败的标记及原因" in q:
            return _GOOD_CATION_STRUCT
        return _BAD_CATION

    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", fake_ask)
    result = process_question("画出 2-溴丁烷 SN1 的中间体", max_corrections=1)
    assert len(calls) == 3                          # 主生成 + 手术(失败) + 常规修正
    assert "tikzpicture" in result
