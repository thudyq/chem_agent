# -*- coding: utf-8 -*-
"""tests/test_arrow_rewrite.py — 手术式箭头重写测试。

背景：回放实验证明——纯机理箭头类失败（编号/方向/配对）时，
给 LLM"固定骨架 + 原子编号地图"单独补写 MECHARROW（查表代替数编号），
5 个箭头类案例 15/15 通过；STRUCT 化学/守恒级错误地图无效（对照组）。
本文件验证：
1. 原子地图生成（显式 id / 自动编号 / BLOCK 嵌套 / 系数 / 化学式组件）；
2. _is_arrow_fixable 失败分类（箭头类放行、STRUCT/守恒/label 类拦截）；
3. 端到端：箭头失败 → 手术重写（专用 prompt + 地图）→ 渲染通过；
4. 手术失败回退常规部分修正；守恒类失败不触发手术重写；
5. 常规修正 prompt 对 COMPOSITE 失败附原子编号地图。
"""

import pytest

from app import (process_question, _build_correction_prompt,
                 _is_arrow_fixable, _rewrite_composite_arrows)
from core.prompt_manager import load_mech_arrow_prompt

# 跨步箭头错误（乙醇质子化：etoh:2>pro:2 指向产物侧）——自动修复不覆盖
# （端点是原子不是键），属纯箭头类失败
_BAD_PROTONATION = (
    "第一步：质子化。乙醇的羟基氧结合 H+。\n\n"
    "[COMPOSITE:reaction]\n"
    "[STRUCT:CCO,label=乙醇,id=etoh][PLUS][STRUCT:[H+],label=H+,id=h]\n"
    "[ARROW:type=reversible]\n"
    "[STRUCT:CC[OH2+],label=质子化乙醇,id=pro]\n"
    "[MECHARROW:etoh:2>pro:2]\n"
    "[/COMPOSITE]"
)
_FIXED_PROTONATION = (
    "[COMPOSITE:reaction]\n"
    "[STRUCT:CCO,label=乙醇,id=etoh][PLUS][STRUCT:[H+],label=H+,id=h]\n"
    "[ARROW:type=reversible]\n"
    "[STRUCT:CC[OH2+],label=质子化乙醇,id=pro]\n"
    "[MECHARROW:etoh:2>h:0]\n"
    "[/COMPOSITE]"
)


def _dispatch(answers, calls):
    """按调用特征分发 mock 回答：手术重写（专用系统提示）/ 常规修正 /
    主生成。answers: {"surgical":..., "correction":..., "main":...}"""
    mech_prompt = load_mech_arrow_prompt()

    def fake_ask(q, system_prompt=None, **k):
        calls.append({"q": q, "sp": system_prompt})
        if system_prompt and system_prompt == mech_prompt:
            return answers.get("surgical")
        if "渲染失败的标记及原因" in q:
            return answers.get("correction")
        return answers.get("main")

    return fake_ask


# ---------------------------------------------------------------- 原子地图生成


def test_atom_maps_basic():
    """显式 id 组件：原子地图 + 可引用的键 + 显式 H 提示。"""
    pytest.importorskip("rdkit")
    from core.tag_parser import parse_tags
    from core.tag_validator import build_component_atom_maps
    text = ("[COMPOSITE:reaction][STRUCT:CCO,label=乙醇,id=etoh][PLUS]"
            "[STRUCT:[H+],label=H+,id=h][ARROW:type=single]"
            "[STRUCT:CC[OH2+],label=质子化乙醇,id=pro][/COMPOSITE]")
    maps = build_component_atom_maps(parse_tags(text)[0])
    assert "组件 id=etoh（label=乙醇）" in maps
    assert "原子地图：0=C，1=C，2=O" in maps
    assert "可引用的键：0-1、1-2" in maps
    assert "显式 H：无" in maps                       # ox 需改写的提示
    assert "组件 id=h（label=H+）" in maps
    assert "显式 H：有（序号 0）" in maps             # 游离 H+ 可引用


def test_atom_maps_block_coeff_and_formula():
    """BLOCK 内组件自动编号 b{N}r{N}（与校验器同口径）；系数前缀剥离；
    化学式文本组件注明不能作端点。"""
    pytest.importorskip("rdkit")
    from core.tag_parser import parse_tags
    from core.tag_validator import build_component_atom_maps
    text = ("[COMPOSITE:reaction][STRUCT:2KMnO4,id=k][BLOCK]"
            "[STRUCT:C1=CC=CC=C1][ARROW:type=resonance]"
            "[STRUCT:C1C=CC=CC=1][/BLOCK][/COMPOSITE]")
    maps = build_component_atom_maps(parse_tags(text)[0])
    assert "化学式文本「KMnO4」" in maps              # 系数 2 已剥离
    assert "不能作为机理箭头端点" in maps
    assert "组件 id=b1r1" in maps                     # BLOCK 内自动编号
    assert "组件 id=b2r2" in maps
    assert "5-0" in maps                              # 闭环键在可引用键列表


def test_atom_maps_auto_id_and_non_composite():
    """无 id 组件自动编号 r0/r1；非 COMPOSITE 返回空串。"""
    pytest.importorskip("rdkit")
    from core.tag_parser import parse_tags
    from core.tag_validator import build_component_atom_maps
    text = ("[COMPOSITE:reaction][STRUCT:CCl][PLUS][STRUCT:[OH-]]"
            "[ARROW:type=single][STRUCT:CO][/COMPOSITE]")
    maps = build_component_atom_maps(parse_tags(text)[0])
    assert "组件 id=r0" in maps and "组件 id=r1" in maps
    assert build_component_atom_maps(parse_tags("[STRUCT:CCO]")[0]) == ""


# ---------------------------------------------------------------- 失败分类


def _composite_tag(with_mech=True):
    from core.tag_parser import parse_tags
    mech = "[MECHARROW:r0:0>r1:0]" if with_mech else ""
    text = (f"[COMPOSITE:reaction][STRUCT:CCl,id=r0][PLUS]"
            f"[STRUCT:[OH-],id=r1][ARROW:type=single][STRUCT:CO,id=r2]"
            f"{mech}[/COMPOSITE]")
    return parse_tags(text)[0]


@pytest.mark.parametrize("reason,expected", [
    # 端点错误消息尾部含"…需先在 SMILES 中把该 H 写成显式 [H]…"指引——
    # 拦截词若用"SMILES"单词会误伤此类（实测回归）
    ("MECHARROW 源端点「ox:2-3」键端点「2-3」引用原子 2 与 3 之间的键，"
     "但该分子中这两原子没有成键（原子 2=C 连接 [1=C]；可直接引用的键：2-1；"
     "该分子没有显式 H 原子——若意图引用 X—H 键（如脱质子），需先在 SMILES 中"
     "把该 H 写成显式 [H]（如 CC([H])CC），显式 H 参与编号后再引用）", True),
    ("MECHARROW「etoh:2>pro:2」跨越主反应箭头", True),
    ("MECHARROW「ar:0-1>so3:0」π 电子应进攻亲电中心，而非电负性的中性氧原子", True),
    ("质子转移缺配对箭头：「ox:4-5>ox:4」是脱质子", True),
    ("SN2 进攻位点错误：「nu:2>pro:0」的终点碳未连离去基团", True),
    ("MECHARROW 成键空白位配对错误：空白位「cl2:0+me:0」有 1 根鱼钩", True),
    ("化学校验：第 1 步两侧原子不守恒（C2H6O vs C4H10O", False),      # 守恒
    ("组件 x: 无效 SMILES「XYZABC」", False),                          # SMILES
    ("组件 ether: label 标注「质子化」但 SMILES 中没有带正电的杂原子", False),  # label
    ("（COMPOSITE 渲染失败：未知错误）", False),                        # 渲染失败
])
def test_is_arrow_fixable_classification(reason, expected):
    tag = _composite_tag(with_mech=True)
    assert _is_arrow_fixable(tag, reason) is expected


def test_is_arrow_fixable_requires_mecharrow():
    """无 MECHARROW 的 COMPOSITE（如纯共振块）不走手术重写。"""
    tag = _composite_tag(with_mech=False)
    assert _is_arrow_fixable(tag, "MECHARROW 源端点…") is False


def test_is_arrow_fixable_requires_composite():
    """非 COMPOSITE 标记（顶层 STRUCT 等）不走手术重写。"""
    from core.tag_parser import parse_tags
    tag = parse_tags("[STRUCT:CCO]")[0]
    assert _is_arrow_fixable(tag, "MECHARROW 源端点…") is False


# ---------------------------------------------------------------- 端到端


def test_surgical_rewrite_end_to_end(monkeypatch):
    """纯箭头失败 → 手术重写（专用 prompt + 地图 + 上文叙述）→ 渲染通过，
    不触发常规修正。"""
    pytest.importorskip("rdkit")
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", _dispatch({
        "main": _BAD_PROTONATION,
        "surgical": _FIXED_PROTONATION,
        "correction": "（不应被调用）",
    }, calls))
    result = process_question("乙醇被质子酸质子化的过程", max_corrections=1)
    assert len(calls) == 2                          # 主生成 + 手术重写
    surg = calls[1]
    assert surg["sp"] == load_mech_arrow_prompt()   # 专用系统提示
    assert "原子编号地图" in surg["q"]               # 注入地图
    assert "原子地图：0=C，1=C，2=O" in surg["q"]
    assert "骨架（照抄，仅补写 MECHARROW 行）" in surg["q"]
    assert "[MECHARROW" not in surg["q"].split("骨架")[1].split("组件原子编号地图")[0]
    assert "第一步：质子化" in surg["q"]             # 上文叙述作语境
    assert "tikzpicture" in result                   # 重写后渲染成功
    assert "无法渲染" not in result


def test_surgical_fallback_to_full_correction(monkeypatch):
    """手术重写输出非法 → 同轮回退常规部分修正，修正成功。"""
    pytest.importorskip("rdkit")
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", _dispatch({
        "main": _BAD_PROTONATION,
        "surgical": "抱歉，我无法确定电子流向。",      # 无标记 → 手术失败
        "correction": _FIXED_PROTONATION,
    }, calls))
    result = process_question("乙醇被质子酸质子化的过程", max_corrections=1)
    assert len(calls) == 3                          # 主生成 + 手术 + 常规修正
    assert calls[1]["sp"] == load_mech_arrow_prompt()
    assert "修正要求" in calls[2]["q"]               # 常规修正 prompt
    assert "tikzpicture" in result


def test_surgical_not_triggered_for_conservation(monkeypatch):
    """守恒类失败（STRUCT 骨架不可信）不触发手术重写，直接常规修正。"""
    pytest.importorskip("rdkit")
    bad = ("乙醇脱水：[COMPOSITE:reaction][STRUCT:CCO,id=a,label=乙醇]"
           "[ARROW:type=single,浓H2SO4][STRUCT:CCOCC,id=b,label=乙醚]"
           "[/COMPOSITE]")  # 单→单 C 当量不等（2 vs 4）→ 化学校验拦截
    fixed = ("[COMPOSITE:reaction][STRUCT:2CCO,id=a,label=乙醇]"
             "[ARROW:type=single,浓H2SO4, 140℃]"
             "[STRUCT:CCOCC,id=b,label=乙醚][PLUS][STRUCT:O,id=w,label=水]"
             "[/COMPOSITE]")
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", _dispatch({
        "main": bad, "surgical": "（不应被调用）", "correction": fixed,
    }, calls))
    result = process_question("乙醇脱水生成乙醚", max_corrections=1)
    assert len(calls) == 2                          # 主生成 + 常规修正（无手术）
    assert "修正要求" in calls[1]["q"]
    assert calls[1]["sp"] != load_mech_arrow_prompt()
    assert "tikzpicture" in result


def test_surgical_attempted_once_per_tag(monkeypatch):
    """手术重写对同一标记只尝试一次：重写非法 → 常规修正原样重犯 →
    逃生降级（不无限烧调用）。"""
    pytest.importorskip("rdkit")
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", _dispatch({
        "main": _BAD_PROTONATION,
        "surgical": "无法确定。",
        "correction": _BAD_PROTONATION,             # 常规修正原样重犯
    }, calls))
    result = process_question("乙醇被质子酸质子化的过程", max_corrections=2)
    # 主生成 1 + 手术 1 + 常规修正 1（修正后 fingerprint 相同 → 逃生）
    assert len(calls) == 3
    assert sum(1 for c in calls
               if c["sp"] == load_mech_arrow_prompt()) == 1  # 手术只试一次
    assert "无法渲染" in result or "已省略" in result  # 降级


def test_correction_prompt_includes_atom_maps():
    """常规修正 prompt：COMPOSITE 失败附组件原子编号地图。"""
    pytest.importorskip("rdkit")
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    tag = parse_tags(_BAD_PROTONATION)[0]
    vr = validate_tag(tag)
    assert not vr.ok
    prompt = _build_correction_prompt("乙醇被质子酸质子化的过程",
                                      _BAD_PROTONATION, [(tag, vr.reason)])
    assert "原子编号地图" in prompt
    assert "原子地图：0=C，1=C，2=O" in prompt        # etoh/pro 的地图
    assert "显式 H：有（序号 0）" in prompt            # [H+] 组件


def test_rewrite_rejects_invalid_llm_output():
    """_rewrite_composite_arrows：LLM 输出的重写块自身校验不过时返回 None
    （不把坏块换进原文）。"""
    pytest.importorskip("rdkit")
    from core.tag_parser import parse_tags
    tag = parse_tags(_BAD_PROTONATION)[0]
    bad_reply = _FIXED_PROTONATION.replace(
        "[MECHARROW:etoh:2>h:0]", "[MECHARROW:etoh:99>h:0]")  # 越界
    import app as app_mod
    orig_ask = app_mod.ask_llm
    app_mod.ask_llm = lambda *a, **k: bad_reply
    try:
        assert _rewrite_composite_arrows("乙醇被质子酸质子化的过程",
                                         _BAD_PROTONATION, tag) is None
    finally:
        app_mod.ask_llm = orig_ask


# ---------------------------------------------------------------- 电子流模拟结论注入手术重写

# 均裂两根鱼钩都归同一原子（模式规则不查，模拟器兜底拦截）
_SIM_FAIL = (
    "[COMPOSITE:reaction]"
    "[STRUCT:ClCl,label=Cl2,id=cl2]"
    "[ARROW:type=single,hν]"
    "[STRUCT:[Cl],label=Cl·,id=cl1][PLUS][STRUCT:[Cl],label=Cl·,id=cl2b]"
    "[MECHARROW:cl2:0-1>>cl2:0][MECHARROW:cl2:0-1>>cl2:0]"
    "[/COMPOSITE]"
)
_FIXED_HOMOLYSIS = _SIM_FAIL.replace(
    "[MECHARROW:cl2:0-1>>cl2:0][MECHARROW:cl2:0-1>>cl2:0]",
    "[MECHARROW:cl2:0-1>>cl2:0][MECHARROW:cl2:0-1>>cl2:1]")


def test_surgical_includes_sim_conclusion(monkeypatch):
    """电子流模拟失败 → 手术重写 prompt 注入模拟结论（含推得的实际
    结构与未推出的声明产物），引导 LLM 针对性修正。"""
    pytest.importorskip("rdkit")
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", _dispatch({
        "main": f"氯气在光照下均裂：\n\n{_SIM_FAIL}",
        "surgical": _FIXED_HOMOLYSIS,
        "correction": "（不应被调用）",
    }, calls))
    result = process_question("氯气在光照下的裂解机理", max_corrections=1)
    assert len(calls) == 2                          # 主生成 + 手术重写
    surg = calls[1]
    assert surg["sp"] == load_mech_arrow_prompt()
    assert "电子流模拟" in surg["q"]                 # 模拟结论注入
    assert "不成立" in surg["q"]                      # 含模拟失败原因（双自由基超价）
    assert "tikzpicture" in result                   # 重写后渲染成功


def test_surgical_no_sim_section_for_pattern_errors(monkeypatch):
    """对照：非模拟类箭头错误（跨步）不注入模拟结论段落。"""
    pytest.importorskip("rdkit")
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", _dispatch({
        "main": _BAD_PROTONATION,
        "surgical": _FIXED_PROTONATION,
        "correction": "（不应被调用）",
    }, calls))
    process_question("乙醇被质子酸质子化的过程", max_corrections=1)
    assert len(calls) == 2
    assert "电子流模拟" not in calls[1]["q"]          # 无模拟结论段落


def test_correction_prompt_sim_guidance():
    """常规修正 prompt：电子流模拟类失败附定向修正指引（优先改箭头，
    预期产物可参考模拟推得结构）。"""
    pytest.importorskip("rdkit")
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    tag = parse_tags(_SIM_FAIL)[0]
    vr = validate_tag(tag)
    assert not vr.ok and "电子流模拟" in vr.reason
    prompt = _build_correction_prompt("氯气光解", _SIM_FAIL, [(tag, vr.reason)])
    assert "优先" in prompt and "推出声明产物" in prompt
    assert "预期产物" in prompt


# ---------------------------------------------------------------- 管线集成：diff 反推 / 错侧翻转的证据序

_Q17_WRONG = (
    "脱质子恢复芳香性：\n\n"
    "[COMPOSITE:reaction]\n"
    "[STRUCT:BrC([H])1C=CC=C[CH+]1,label=σ 络合物,id=sigma]\n"
    "[ARROW:type=single]\n"
    "[STRUCT:BrC1=CC=CC=C1,label=溴苯][PLUS][STRUCT:[H+],label=H+]\n"
    "[MECHARROW:sigma:1-2>sigma:1][MECHARROW:sigma:7-1>sigma:7]\n"
    "[/COMPOSITE]"
)

_CORPUS_FLIP = (
    "第二步：亲核取代：\n\n"
    "[COMPOSITE:reaction]\n"
    "[STRUCT:CC[OH2+],label=质子化乙醇,id=pro][PLUS][STRUCT:CCO,label=乙醇,id=nu]\n"
    "[ARROW:type=single]\n"
    "[STRUCT:CCOCC,label=乙醚][PLUS][STRUCT:O,label=水][PLUS][STRUCT:[H+],label=H+]\n"
    "[MECHARROW:nu:2>pro:1][MECHARROW:pro:1-2>pro:2]\n"
    "[/COMPOSITE]"
)


def test_pipeline_diff_autofix_without_llm(monkeypatch):
    """模拟不一致 → 图 diff 反推确定性修复（零额外 LLM 调用）——
    修正循环只有主生成一次调用。"""
    pytest.importorskip("rdkit")
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or _Q17_WRONG)
    result = process_question("苯的溴代机理", max_corrections=1)
    assert len(calls) == 1                       # 反推免 LLM 修正调用
    assert "tikzpicture" in result               # 修复后渲染成功
    assert "无法渲染" not in result


def test_pipeline_flip_after_surgical_fails(monkeypatch):
    """模拟不一致 + 反推不适用（多物种合并）+ 手术重写失败 → 翻转产物
    （乙醚+H+ 合并为模拟推得的质子化乙醚），箭头不动。"""
    pytest.importorskip("rdkit")
    calls = []

    def fake_ask(q, system_prompt=None, **k):
        calls.append({"q": q, "sp": system_prompt})
        if system_prompt == load_mech_arrow_prompt():
            return "我改不动这组箭头。"          # 手术重写失败
        return _CORPUS_FLIP

    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", fake_ask)
    result = process_question("乙醇生成乙醚的机理", max_corrections=1)
    assert len(calls) == 2                       # 主生成 + 手术（翻转免 LLM）
    assert "tikzpicture" in result
    assert "无法渲染" not in result
