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

    def render_struct(smiles, label=None, mode="skeleton", subs="",
                      bond="", angle="", charge=""):
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


def test_diagnostics_resolved_after_correction(fake_rdkit, fake_renderers,
                                               monkeypatch):
    """修正救回：diagnostics 记录失败轮（round=0），resolved=True。"""
    calls = []
    answers = ["苯是 [STRUCT:XYZABC]。", "苯是 [STRUCT:c1ccccc1]。"]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(a[0] if a else None) or answers.pop(0))
    diag = []
    result = process_question("画苯", max_corrections=1, diagnostics=diag)
    assert len(diag) == 1
    assert diag[0]["round"] == 0
    assert diag[0]["resolved"] is True
    assert "无效 SMILES" in diag[0]["reason"]          # 技术细节留给后端
    assert diag[0]["friendly"].startswith("（结构式图示无法渲染，已省略")  # 前端友好版
    assert "RENDERED:c1ccccc1" in result


def test_diagnostics_include_final_round_unresolved(fake_rdkit,
                                                    fake_renderers,
                                                    monkeypatch):
    """修正救不回：diagnostics 含最后一轮（round=1，即 max），resolved=False
    ——后端拿到完整失败反馈（含最后一次），前端只见友好降级。"""
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm",
        lambda *a, **k: calls.append(a[0] if a else None) or "苯是 [STRUCT:XYZABC]。")
    diag = []
    result = process_question("画苯", max_corrections=1, diagnostics=diag)
    assert len(calls) == 2
    assert len(diag) == 2                       # 两轮失败都被记录（含最后一次）
    assert [d["round"] for d in diag] == [0, 1]
    assert all(d["resolved"] is False for d in diag)
    assert all(d["raw"] == "[STRUCT:XYZABC]" for d in diag)
    # 前端友好、后端拿技术细节
    assert "图示无法渲染" in result
    assert "无效 SMILES" not in result
    assert all("无效 SMILES" in d["reason"] for d in diag)


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
    text = ("乙醇氧化：[COMPOSITE:reaction][STRUCT:CCO,id=a][PLUS]"            "[STRUCT:O,id=w][ARROW:type=single,Cu, Δ]"            "[STRUCT:CC=O,id=b][/COMPOSITE]")
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    assert vr.reason.startswith("化学校验：")
    prompt = _build_correction_prompt("乙醇氧化成乙醛", text, [(tag, vr.reason)])
    assert "两侧原子" in prompt                      # 失败原因
    assert "补物种或调系数" in prompt               # 修正指导
    assert "辅助试剂" in prompt and "箭头条件" in prompt  # 辅助试剂归位规则
    assert "[O]" in prompt and "[H]" in prompt        # 占位符禁止提示
    assert "氧化剂" in prompt                        # 氧化配平规则


def test_correction_prompt_invalid_smiles_guidance():
    """无效 SMILES 时修正 prompt 给出具体改法（20260821 更新：配离子
    走分子式轨道——[Ag(NH3)2]+ 直写合法，不再建议拆分组分）。"""
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = ("[COMPOSITE:reaction][STRUCT:CC=O,id=a][PLUS]"
            "[STRUCT:[Ag(NH3)2OH],id=b][ARROW:type=single,Δ]"
            "[STRUCT:CC(=O)[O-],id=c][/COMPOSITE]")
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    assert "无效 SMILES" in vr.reason
    prompt = _build_correction_prompt("银镜反应", text, [(tag, vr.reason)])
    assert "[Ag(NH3)2OH]" in prompt                  # 失败标记原文
    assert "[Ag(NH3)2]+" in prompt                   # 配离子分子式改法
    assert "分子式" in prompt                         # 分子式轨道指引
    assert "省略或文字描述" in prompt                 # 降级策略


def test_correction_prompt_inorganic_salt_guidance():
    """无机盐/含氧酸盐 SMILES 非法时修正 prompt 给出分子式/离子式改法
    （KMnO4 写成 K[Mn](=O)(=O)=O——金属与中心原子无直接键）。"""
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = ("[COMPOSITE:reaction][STRUCT:CCO,id=a][PLUS]"
            "[STRUCT:K[Mn](=O)(=O)=O,id=k][ARROW:type=single]"
            "[STRUCT:CC=O,id=b][/COMPOSITE]")
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    assert "无效 SMILES" in vr.reason or "化学校验" in vr.reason
    prompt = _build_correction_prompt("乙醇被高锰酸钾氧化", text,
                                      [(tag, vr.reason)])
    assert "K[Mn](=O)(=O)=O" in prompt              # 失败标记原文
    assert "[K+].[O-][Mn](=O)(=O)=O" in prompt      # 离子式改法
    assert "分子式" in prompt                        # 分子式轨道指引


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


# ---------- flash 首跑 + 失败升级 pro 路由 ----------

def _enable_route(monkeypatch, upgrade_model="deepseek-v4-pro"):
    """启用路由：替换 app.settings.llm.upgrade_model_name。"""
    import types
    monkeypatch.setattr(
        "app.settings",
        types.SimpleNamespace(
            llm=types.SimpleNamespace(upgrade_model_name=upgrade_model)))


def test_route_keyword_direct_upgrade(fake_rdkit, fake_renderers,
                                      monkeypatch):
    """命中难题关键词（如"机理"）→ 跳过主模型首跑，直接升级模型。"""
    _enable_route(monkeypatch)
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm",
        lambda *a, **k: calls.append(k) or "SN1 机理是 [STRUCT:c1ccccc1]。")
    diag = []
    result = process_question("介绍 SN1 反应的机理", max_corrections=1,
                              diagnostics=diag)
    assert len(calls) == 1, "命中关键词应直接 pro（无 flash 首跑）"
    assert calls[0].get("model") == "deepseek-v4-pro"
    assert "RENDERED:c1ccccc1" in result
    assert not diag                              # pro 一遍过，无失败诊断


def test_route_keyword_custom_list(fake_rdkit, fake_renderers, monkeypatch):
    """自定义 UPGRADE_KEYWORDS 生效；未命中词汇仍走主模型首跑。"""
    import types
    monkeypatch.setattr(
        "app.settings",
        types.SimpleNamespace(
            llm=types.SimpleNamespace(
                upgrade_model_name="deepseek-v4-pro",
                upgrade_keywords=("卤代",))),)  # 仅"卤代"算难题
    calls = []
    answers = ["苯是 [STRUCT:XYZABC]。", "苯是 [STRUCT:c1ccccc1]。"]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or answers.pop(0))
    # "画苯"不含"卤代" → 主模型首跑 → 失败升级 pro
    process_question("画苯", max_corrections=1)
    assert [c.get("model") for c in calls] == [None, "deepseek-v4-pro"]


def test_route_pass_no_upgrade(fake_rdkit, fake_renderers, monkeypatch):
    """主模型一遍过 → 不触发升级（ask_llm 仅 1 次，model 不覆盖）。"""
    _enable_route(monkeypatch)
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm",
        lambda *a, **k: calls.append(k) or "苯是 [STRUCT:c1ccccc1]。")
    diag = []
    result = process_question("画苯", max_corrections=1, diagnostics=diag)
    assert len(calls) == 1
    assert calls[0].get("model") is None          # 主模型默认配置
    assert not diag                                # 无失败 → 无诊断
    assert "RENDERED:c1ccccc1" in result


def test_route_upgrade_on_failure(fake_rdkit, fake_renderers, monkeypatch):
    """主模型失败 → 升级模型只做部分修正（不重跑全文），修正后一遍过。"""
    _enable_route(monkeypatch)
    calls = []
    answers = [
        "苯是 [STRUCT:XYZABC]。",      # flash 首跑失败（不做 flash 修正）
        "[STRUCT:c1ccccc1]",            # pro 部分修正只输出修正标记
    ]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or answers.pop(0))
    diag = []
    result = process_question("画苯", max_corrections=1, diagnostics=diag)
    assert len(calls) == 2, "flash 失败应直接升级 pro 部分修正"
    assert calls[0].get("model") is None           # flash（默认配置）
    assert calls[1].get("model") == "deepseek-v4-pro"  # 升级 pro
    assert calls[1].get("thinking") == "disabled"  # 修正调用关思考（非全文重跑）
    assert "RENDERED:c1ccccc1" in result
    # 诊断：flash 失败在 main 与 upgrade（修正前重新校验）各记一条，最终被解决
    assert len(diag) == 2
    assert [d["stage"] for d in diag] == ["main", "upgrade"]
    assert all(d["resolved"] is True for d in diag)


def test_route_upgrade_then_correction(fake_rdkit, fake_renderers,
                                       monkeypatch):
    """flash 失败 → 升级 pro 部分修正仍失败（相同错误）→ P3 逃生不再烧轮次。"""
    _enable_route(monkeypatch)
    calls = []
    answers = [
        "苯是 [STRUCT:XYZABC]。",      # flash 首跑失败
        "[STRUCT:XYZABC]",              # pro 第 1 次修正原样重犯（同 fingerprint）
    ]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or answers.pop(0))
    result = process_question("画苯", max_corrections=2)
    # P3：修正后失败原因与上轮完全相同 → 跳过剩余修正轮次（20260821）
    assert len(calls) == 2
    assert calls[0].get("model") is None
    assert calls[1].get("model") == "deepseek-v4-pro"
    assert calls[1].get("thinking") == "disabled"   # 修正调用关思考
    assert "RENDERED" not in result                  # 未救回 → 降级文本


def test_route_correction_continues_on_new_error(fake_rdkit, fake_renderers,
                                                 monkeypatch):
    """修正后失败原因**变化**（fingerprint 不同）→ 不触发 P3，继续修正救回。"""
    _enable_route(monkeypatch)
    calls = []
    answers = [
        "苯是 [STRUCT:XYZABC]。",      # flash 首跑失败
        "[STRUCT:XYZABD]",              # pro 第 1 次修正仍失败（不同原因串）
        "[STRUCT:c1ccccc1]",            # pro 第 2 次修正成功
    ]
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or answers.pop(0))
    result = process_question("画苯", max_corrections=2)
    assert len(calls) == 3
    assert calls[2].get("model") == "deepseek-v4-pro"
    assert "RENDERED:c1ccccc1" in result


def test_route_upgrade_unresolved(fake_rdkit, fake_renderers, monkeypatch):
    """flash 失败 → 升级 pro 仍失败且修正救不回 → 降级；诊断含 stage。"""
    _enable_route(monkeypatch)
    calls = []
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr(
        "app.ask_llm",
        lambda *a, **k: calls.append(k) or "苯是 [STRUCT:XYZABC]。")
    diag = []
    result = process_question("画苯", max_corrections=1, diagnostics=diag)
    assert "图示无法渲染" in result
    assert "无效 SMILES" not in result               # 前端友好
    stages = [d["stage"] for d in diag]
    assert "main" in stages and "upgrade" in stages  # 两阶段失败都记录
    assert all(d["resolved"] is False for d in diag)


# ---------- 部分降级：COMPOSITE 仅 MECHARROW 报错时剔除箭头保留分子 ----------

_BAD_MECH_COMPOSITE = (
    "[COMPOSITE:reaction]"
    "[STRUCT:CCl,id=r0,label=CH3Cl][ARROW:type=single]"
    "[STRUCT:CO,id=p0,label=CH3OH]"
    "[MECHARROW:r0:9>r0:0]"   # 9 越界（该分子只有 2 个重原子）→ 仅 MECHARROW 报错
    "[/COMPOSITE]"
)


def test_partial_render_without_mecharrows():
    """部分降级：仅 MECHARROW 报错的 COMPOSITE 剔除箭头后仍渲染出分子图。"""
    pytest.importorskip("rdkit")
    from app import _partial_render_composite_without_mecharrows
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tags

    tags = parse_tags(_BAD_MECH_COMPOSITE)
    _, invalid = validate_tags(tags)
    assert len(invalid) == 1
    assert "MECHARROW" in invalid[0].reason
    out = _partial_render_composite_without_mecharrows(invalid[0].tag)
    assert out is not None
    assert out.startswith("\\begin{tikzpicture}")
    assert "\\begin{scope}" in out          # 分子组件保留
    assert "Stealth[length=2.5mm]" in out  # 主反应箭头保留


def test_partial_render_not_for_non_mecharrow():
    """非 MECHARROW 错误（STRUCT 无效）不触发部分渲染（整体降级）。"""
    pytest.importorskip("rdkit")
    from app import _partial_render_composite_without_mecharrows
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tags

    text = ("[COMPOSITE:reaction][STRUCT:XYZABC,id=r0][ARROW:type=single]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:0>r0:0][/COMPOSITE]")
    tags = parse_tags(text)
    _, invalid = validate_tags(tags)
    assert "MECHARROW" not in invalid[0].reason
    assert _partial_render_composite_without_mecharrows(invalid[0].tag) is None


def test_mecharrow_only_degrades_partially(monkeypatch):
    """集成：仅 MECHARROW 报错且修正耗尽 → 输出分子图 + 「反应箭头无法渲染」
    提示（而非整图省略）。"""
    pytest.importorskip("rdkit")
    bad = f"机理：{_BAD_MECH_COMPOSITE}"
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", lambda *a, **k: bad)
    result = process_question("画机理", max_corrections=1)
    assert "反应箭头无法渲染，已省略" in result
    assert "\\begin{tikzpicture}" in result           # 分子图保留
    assert "（复合图图示无法渲染" not in result        # 未整体降级


def test_non_mecharrow_still_whole_degrade(monkeypatch):
    """非 MECHARROW 错误修正耗尽 → 仍整体降级（行为不变）。"""
    pytest.importorskip("rdkit")
    bad = "苯是 [STRUCT:XYZABC]。"
    monkeypatch.setattr("app._translate_name_zh2en", lambda n: None)
    monkeypatch.setattr("app.ask_llm", lambda *a, **k: bad)
    result = process_question("画苯", max_corrections=1)
    assert "图示无法渲染" in result
    assert "无效 SMILES" not in result                # 前端友好（无技术细节）


def test_autofix_endpoint_skips_llm_correction(monkeypatch):
    """P1 端到端：唯一候选端点自动修复——主生成后直接渲染，零修正调用。"""
    pytest.importorskip("rdkit")
    calls = []
    bad_tag = ("[COMPOSITE:reaction]"
               "[STRUCT:O=[N+]([O-])C([H])1C=CC=C[CH+]1,id=sg]"
               "[PLUS][STRUCT:O=[N+]([O-])[O-],id=no3]"
               "[ARROW:type=single][STRUCT:O=[N+]([O-])C1=CC=CC=C1,id=nb]"
               "[PLUS][STRUCT:O=[N+]([O-])O,id=hno3]"
               "[MECHARROW:no3:2>sg:4,sg:3-6>sg:3][/COMPOSITE]")  # 3-6 应为 3-4
    monkeypatch.setattr(
        "app.ask_llm", lambda *a, **k: calls.append(k) or f"机理：{bad_tag}")
    result = process_question("磺化脱质子", max_corrections=2)
    assert len(calls) == 1                     # 端点自动修复，无 LLM 修正
    assert "渲染失败" not in result and "无法渲染" not in result
    assert "tikzpicture" in result


def test_correction_prompt_dynamic_sections():
    """P2：修正要求按失败类型裁剪——mech 失败只给端点指引，不掺守恒讲座。"""
    pytest.importorskip("rdkit")
    from core.tag_parser import parse_tags
    from core.tag_validator import validate_tag
    text = ("[COMPOSITE:reaction][STRUCT:CC(C)(C)[OH2+],id=p]"
            "[MECHARROW:p:3-4>p:4][/COMPOSITE]")
    tag = parse_tags(text)[0]
    vr = validate_tag(tag)
    assert "MECHARROW" in vr.reason
    prompt = _build_correction_prompt("脱质子", text, [(tag, vr.reason)])
    assert "直接改写" in prompt                  # mech 指引在场
    assert "禁用 [O]/[H] 占位符" not in prompt      # 守恒段落被裁掉
    assert "配体括号写全" not in prompt             # SMILES 段落被裁掉
