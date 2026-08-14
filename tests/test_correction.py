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


@pytest.fixture(autouse=True)
def no_synrbl(monkeypatch):
    """本文件测试聚焦 P2 修正闭环，不涉 SynRBL——屏蔽真实导入（~23s）。

    SynRBL 配平兜底的行为由 tests/test_rxn_balancer.py 单独覆盖。
    """
    monkeypatch.setattr(
        "utils.rxn_balancer._load_synrbl", lambda: False)


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
    result = process_question("画苯", max_corrections=1)
    assert len(calls) == 1, "无失败不应触发修正重试"
    assert "RENDERED:c1ccccc1" in result


def test_correction_after_validation_failure(fake_rdkit, fake_renderers,
                                             monkeypatch):
    calls = []
    answers = [
        "苯是 [STRUCT:XYZABC]。",      # 非法 SMILES → 校验拦截 → 触发修正
        "苯是 [STRUCT:c1ccccc1]。",     # 修正版
    ]
    # 禁用 PubChem 翻译（避免消耗 ask_llm 调用序列）
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or answers.pop(0))
    result = process_question("画苯", max_corrections=1)
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
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or answers.pop(0))
    result = process_question("画苯", max_corrections=1)
    assert len(calls) == 2
    assert "RENDERED:c1ccccc1" in result
    assert "结构渲染失败" not in result


def test_correction_exhausted_degrades(fake_rdkit, fake_renderers,
                                       monkeypatch):
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or "苯是 [STRUCT:XYZABC]。")
    result = process_question("画苯", max_corrections=1)
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


def test_correction_prompt_chem_guidance():
    """化学校验失败时修正 prompt 给出守恒修正提示（氧化/脱氢补物种）。"""
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = "乙醇氧化：[REACTION:CCO|CC=O|Cu, Δ]"
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    assert vr.reason.startswith("化学校验：")
    prompt = _build_correction_prompt("乙醇氧化成乙醛", text, [(tag, vr.reason)])
    assert "两侧原子" in prompt                      # 失败原因
    assert "补全缺失的具体反应物/生成物" in prompt   # 修正指导
    assert "催化剂" in prompt and "箭头条件" in prompt  # 辅助试剂归位规则
    assert "[O]" in prompt and "[H]" in prompt        # 占位符禁止提示
    assert "2b 箭头补足" in prompt                    # 2b 方案提示


def test_correction_prompt_invalid_smiles_guidance():
    """无效 SMILES（配离子写法错误）时修正 prompt 给出具体改法
    （Drawbacks 十：银镜反应 [Ag(NH3)2]OH / NH3 裸写）。"""
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = "[REACTION:CC=O;2[Ag(NH3)2]OH|CC(=O)[O-];2Ag;3NH3;H2O|Δ]"
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    prompt = _build_correction_prompt("银镜反应", text, [(tag, vr.reason)])
    assert "[Ag(NH3)2]OH" in prompt                  # 失败标记原文
    assert "[Ag]([NH3])[NH3]" in prompt              # 配离子拆分改法
    assert "NH3" in prompt and "裸写" in prompt       # 氨写法提醒
    assert "降级为文字描述" in prompt                 # 降级策略


def test_correction_prompt_inorganic_salt_guidance():
    """无机盐/含氧酸盐 SMILES 非法时修正 prompt 给出离子式改法
    （KMnO4 写成 K[Mn](=O)(=O)=O / KMn(=O)=O——金属与中心原子无直接键）。"""
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = "[REACTION:CCO;K[Mn](=O)(=O)=O|CC=O|]"
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    assert "无效 SMILES" in vr.reason or "化学校验" in vr.reason
    prompt = _build_correction_prompt("乙醇被高锰酸钾氧化", text,
                                      [(tag, vr.reason)])
    assert "K[Mn](=O)(=O)=O" in prompt              # 失败标记原文
    assert "[K+].[O-][Mn](=O)(=O)=O" in prompt      # 离子式改法
    assert "离子式" in prompt                        # 引导策略


def test_retry_succeeds_after_smiles_fix(fake_rdkit, fake_renderers,
                                         monkeypatch):
    """修正重试成功闭环：首次无效 SMILES 校验拦截 → 修正版渲染成功。

    注：fake_rdkit 的 _FakeMol 无元素计数，ARROW/REACTION 的守恒校验
    在 fake 下必然拦截——用 [STRUCT:XYZABC]（无效 SMILES，校验明确
    拦截）触发首次失败，验证"改简单标记后重试成功"的闭环。"""
    calls = []
    answers = [
        # 首次：无效 SMILES（配离子/复杂物种写错的一类）→ 校验拦截
        "乙醛氧化：[STRUCT:XYZABC]",
        # 修正版：简单结构渲染成功
        "乙醛氧化：[STRUCT:CCO]",
    ]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or answers.pop(0))
    result = process_question("写出乙醛发生银镜反应的化学方程式。", max_corrections=1)
    assert len(calls) == 2, "首次失败应触发一次修正重试"
    assert "RENDERED:CCO" in result                # 修正版已渲染
    assert "无法渲染" not in result                # 无降级提示
