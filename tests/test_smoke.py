# -*- coding: utf-8 -*-
"""渲染器冒烟测试：核心渲染器对代表性输入产出合法完整 TikZ。

替代端到端视觉验证——快速确认每个渲染器链路可运行、输出结构完整
（\\begin{tikzpicture} 与 \\end{tikzpicture} 闭合、无错误串），
不深入具体视觉细节（那是单元测试的职责）。

覆盖：STRUCT / REACTION / COMPOSITE（reaction_mech/energy/row/resonance）
/ ENERGY / NEWMAN / STEREO / LEWIS / CHARGE / HBOND / RETRO + 完整管线注入。
"""

import pytest

rdkit = pytest.importorskip("rdkit", reason="rdkit 未安装，跳过冒烟测试")

from core.tag_injector import inject_tags_into_text
from core.tag_parser import parse_tags
from core.tag_validator import validate_tags
from renderers.registry import render_tag


def _render(text):
    """parse → validate → 渲染全部合法标记，返回 (输出列表, 校验失败数)。"""
    tags = parse_tags(text)
    valid, invalid = validate_tags(tags)
    outs = []
    for t in valid:
        out = render_tag(t)
        if out is not None:
            outs.append(out)
    return outs, len(invalid)


def _assert_ok(out, *needles):
    """冒烟断言：TikZ 完整闭合、无错误串、含关键元素。"""
    assert out.startswith("\\begin{tikzpicture}") or out.startswith("\\chemfig"), \
        f"输出应以 tikzpicture/chemfig 开头: {out[:60]}"
    assert out.endswith("\\end{tikzpicture}") or out.startswith("\\chemfig"), \
        "输出应以 \\end{tikzpicture} 闭合"
    assert "渲染失败" not in out and "渲染失败" not in out, \
        f"不应含错误串: {out[:120]}"
    for nd in needles:
        assert nd in out, f"缺少关键元素 {nd!r}"


def test_smoke_struct():
    outs, bad = _render("[STRUCT:c1ccccc1,label=苯]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "苯")


def test_smoke_struct_no_lone_pairs():
    """顶层 STRUCT 不画孤对电子（键线式规范第 3 条，20260818 回归锚点）。

    此前 render_structure 漏传 show_lone_pairs=False（molecule_scope_lines
    默认 True）→ [STRUCT:CCl] 的 Cl 画出 3 对孤对电子点（6 个 \\fill）。
    对照：LEWIS 仍画（test_smoke_lewis 断言 \\fill）。
    """
    outs, bad = _render("[STRUCT:CCl]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "{Cl}")
    assert "\\fill" not in outs[0], "顶层 STRUCT 不应有孤对电子点"


def test_smoke_reaction():
    outs, bad = _render("[REACTION:CCO;CCO|CCOCC;O|H2SO4]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "very thick")


def test_smoke_composite_reaction_mech():
    text = ("[COMPOSITE:reaction_mech]"
            "[STRUCT:CCl,label=CH3Cl][PLUS][STRUCT:[OH-],id=nu,label=OH-]"
            "[RXNARROW:SN2][STRUCT:CO,label=CH3OH][PLUS][STRUCT:[Cl-],label=Cl-]"
            "[MECHARROW:nu:0>r0:0][MECHARROW:r0:0-1>r0:1]"
            "[/COMPOSITE]")
    outs, bad = _render(text)
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "\\begin{scope}[shift=", "red")


def test_smoke_composite_energy():
    text = ("[COMPOSITE:energy][ENERGY:0,108,-20]"
            "[STRUCT:CCl.[OH-],label=反应物,at=0]"
            "[STRUCT:CCl.[OH-],label=过渡态,at=1]"
            "[STRUCT:CO.[Cl-],label=产物,at=2]"
            "[/COMPOSITE]")
    outs, bad = _render(text)
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "plot coordinates", "Ea $\\approx$")


def test_smoke_composite_row():
    text = ("[COMPOSITE:row][STRUCT:C=C,label=乙烯][RXNARROW:H2O / H+]"
            "[STRUCT:CCO,label=乙醇][/COMPOSITE]")
    outs, bad = _render(text)
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "very thick")


def test_smoke_composite_resonance():
    text = ("[COMPOSITE:row][STRUCT:C1=CC=CC=C1]"
            "[RESARROW][STRUCT:C1C=CC=CC=1][/COMPOSITE]")
    outs, bad = _render(text)
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "\\leftrightarrow")


def test_smoke_ethanol_ether_mechanism():
    """乙醇→乙醚 SN2 机理（质子化物种）冒烟：验证渲染器能画完整机理图。

    用正确的质子化物种标记（乙基氧鎓离子 CC[OH2+]、质子化乙醚 CC[OH+]CC），
    覆盖：多组分 + 电荷 + 孤对电子 + 断键箭头。替代端到端（不调用 LLM）。
    """
    text = ("[COMPOSITE:reaction_mech]"
            "[STRUCT:CCO,label=乙醇,id=nu][PLUS]"
            "[STRUCT:CC[OH2+],label=乙基氧鎓离子,id=pe]"
            "[RXNARROW:H2SO4,140°C]"
            "[STRUCT:CC[OH+]CC,label=质子化乙醚,id=ps][PLUS]"
            "[STRUCT:O,label=水,id=w]"
            "[MECHARROW:nu:2>pe:1][MECHARROW:pe:1-2>pe:2]"
            "[/COMPOSITE]")
    outs, bad = _render(text)
    assert len(outs) == 1 and bad == 0, [r.reason for r in bad]
    _assert_ok(outs[0], "\\begin{scope}[shift=", "red",
               "H$_{2}$O", "H$_2$SO$_4$")


def test_smoke_energy():
    outs, bad = _render("[ENERGY:0,108,-20]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "plot coordinates")


def test_smoke_newman():
    outs, bad = _render("[NEWMAN:CC,60]")
    assert len(outs) == 1 and bad == 0
    # NEWMAN 不用 scope 封装（直接绝对坐标），关键元素是中心圆与取代基键
    _assert_ok(outs[0], "circle")


def test_smoke_newman_with_bond():
    """NEWMAN 三参数（指定投影键 a-b）：渲染成功。"""
    outs, bad = _render("[NEWMAN:CC,0-1,60]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "circle")


def test_smoke_stereo():
    outs, bad = _render("[STEREO:C[C@H](O)C(=O)O]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0])


def test_smoke_stereo_with_label():
    outs, bad = _render("[STEREO:C[C@H](O)C(=O)O,label=(R)-乳酸]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "\\node[below]")


def test_smoke_xh_toplevel():
    outs, bad = _render("[XH:CC(=O)O|3]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "{H}")


def test_smoke_bond_toplevel():
    outs, bad = _render("[BOND:CCC=O|1-2]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "very thick, red")


def test_smoke_lewis():
    outs, bad = _render("[LEWIS:O]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "\\fill")


def test_smoke_struct_mode_lewis_equals_old_tag():
    """分子家族统一：STRUCT mode=lewis 与旧 [LEWIS:] 输出一致。"""
    outs_new, _ = _render("[STRUCT:O, mode=lewis, label=水]")
    outs_old, _ = _render("[LEWIS:O,label=水]")
    assert len(outs_new) == 1 and len(outs_old) == 1
    assert outs_new[0] == outs_old[0]
    _assert_ok(outs_new[0], "\\fill", "\\node[below]")


def test_smoke_struct_mode_newman_equals_old_tag():
    """STRUCT mode=newman（bond/angle 参数）与旧 [NEWMAN:...] 输出一致。"""
    outs_new, _ = _render("[STRUCT:CC, mode=newman, bond=0-1, angle=60]")
    outs_old, _ = _render("[NEWMAN:CC,0-1,60]")
    assert len(outs_new) == 1 and len(outs_old) == 1
    assert outs_new[0] == outs_old[0]
    _assert_ok(outs_new[0], "circle")


def test_smoke_composite_mode_lewis():
    """容器内 mode=lewis 组件显示孤对电子点（Lewis 式分子进 COMPOSITE）。"""
    outs, bad = _render(
        "[COMPOSITE:row][STRUCT:O, mode=lewis, id=w, label=水][/COMPOSITE]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "\\fill")


def test_smoke_reaction_layout():
    """大一统架构 reaction 布局：ARROW 序列渲染（正向 + 可逆 + 附件结构式）。"""
    outs, bad = _render(
        "[COMPOSITE:reaction][STRUCT:CCl,id=A,label=CH3Cl]"
        "[STRUCT:[OH-],id=E,arrow][STRUCT:[Cl-],id=F,arrow]"
        "[ARROW:type=reversible,sup=+E;-F]"
        "[STRUCT:CO,id=B,label=CH3OH][/COMPOSITE]")
    assert len(outs) == 1 and bad == 0, [r.reason for r in bad]
    _assert_ok(outs[0], "\\begin{scope}[shift=")
    # 附件结构式（副反应物/副产物）：OH- 电荷圈、Cl- 标签存在；⇌ 令牌剥离
    assert "{$-$}" in outs[0] and "{Cl}" in outs[0]
    assert "⇌" not in outs[0]


def test_smoke_reaction_block():
    """reaction 布局 [BLOCK] 共振块渲染（块 scope + ↔）。"""
    outs, bad = _render(
        "[COMPOSITE:reaction][STRUCT:c1ccccc1,id=A,label=苯]"
        "[ARROW:type=single,条件]"
        "[BLOCK][STRUCT:C1=CC=CC=C1,id=B1]"
        "[ARROW:type=resonance]"
        "[STRUCT:C1C=CC=CC=1,id=B2][/BLOCK]"
        "[ARROW:type=single,条件]"
        "[STRUCT:O=[N+]([O-])c1ccccc1,id=C,label=硝基苯][/COMPOSITE]")
    assert len(outs) == 1 and bad == 0, [r.reason for r in bad]
    _assert_ok(outs[0], "\\leftrightarrow")


def test_smoke_reaction_retro():
    """reaction 布局逆合成箭头（⇒ 双线杆 + 开放折线尖）。"""
    outs, bad = _render(
        "[COMPOSITE:reaction][STRUCT:O=Cc1ccccc1,id=T,label=苯甲醛]"
        "[ARROW:type=retro,formylation][STRUCT:c1ccccc1,id=P,label=苯]"
        "[/COMPOSITE]")
    assert len(outs) == 1 and bad == 0
    assert "\\node[font=\\large]" not in outs[0]   # 非共振
    # 双线杆（两条平行线 y=±0.05）存在
    assert outs[0].count("0.05") >= 2 and outs[0].count("-0.05") >= 1


def test_smoke_reaction_block_mecharrow_and_brackets():
    """BLOCK 内 MECHARROW（共振式间转化）+ 方括号 []。"""
    outs, bad = _render(
        "[COMPOSITE:reaction][STRUCT:c1ccccc1,id=A,label=苯]"
        "[ARROW:type=single,条件]"
        "[BLOCK][STRUCT:C1=CC=CC=C1,id=b1]"
        "[ARROW:type=resonance]"
        "[STRUCT:C1C=CC=CC=1,id=b2]"
        "[MECHARROW:b1:0>b2:1]"
        "[/BLOCK]"
        "[ARROW:type=single,条件]"
        "[STRUCT:O=[N+]([O-])c1ccccc1,id=C,label=硝基苯][/COMPOSITE]")
    assert len(outs) == 1 and bad == 0, [r.reason for r in bad]
    _assert_ok(outs[0], "\\leftrightarrow")
    # 方括号：左/右竖线（块 bbox 左右缘的裸 draw 竖线）
    assert "\\draw" in outs[0]
    # 块内机理箭头（红色）存在
    assert "red" in outs[0]


def test_smoke_lewis_with_label():
    outs, bad = _render("[LEWIS:O,label=水]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "\\node[below]")


def test_smoke_charge():
    outs, bad = _render("[CHARGE:OCC|0:δ-,1:δ+]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "delta")


def test_smoke_struct_bond_charge_params():
    """20260821：STRUCT 参数化标注（bond=/charge=）——单分子键突出 + δ± 节点。"""
    outs, bad = _render("[STRUCT:CCC=O, bond=1-2, charge=0:+,3:-]")
    assert len(outs) == 1 and bad == 0, [r.reason for r in bad]
    _assert_ok(outs[0], "very thick, red")   # bond= 键突出
    _assert_ok(outs[0], "delta")             # charge= 部分电荷（0:+ → δ+）


def test_smoke_hbond():
    # HBOND 仅容器内：SMILES 显式 H（[H]OCCO 的 0 号）画氢 + HBOND 画 teal 点
    outs, bad = _render(
        "[COMPOSITE:row][STRUCT:[H]OCCO,id=diol]"
        "[HBOND:diol:0>diol:4][/COMPOSITE]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0], "teal")


def test_smoke_hbond_toplevel_rejected():
    """顶层 HBOND 已移除：校验明确拒绝（仅支持容器内）。"""
    _, bad = _render("[HBOND:OCCO|0-3]")
    assert bad == 1


def test_smoke_retro():
    outs, bad = _render(
        "[RETRO:CC(=O)c1ccccc1,c1ccccc1,Friedel-Crafts acylation]")
    assert len(outs) == 1 and bad == 0
    _assert_ok(outs[0])


def test_smoke_pipeline_inject():
    """完整管线冒烟：解析 → 校验 → 渲染 → 注入（无 LLM）。"""
    text = "苯的结构：[STRUCT:c1ccccc1,label=苯] 和 [REACTION:CCO;CCO|CCOCC;O|H2SO4]。"
    tags = parse_tags(text)
    valid, invalid = validate_tags(tags)
    assert len(invalid) == 0, [r.reason for r in invalid]
    rendered = {}
    for t in valid:
        out = render_tag(t)
        if out is not None:
            rendered[t.raw] = out
    result = inject_tags_into_text(text, tags, rendered)
    assert "[STRUCT:" not in result and "[REACTION:" not in result, "标记应被替换"
    assert "\\begin{tikzpicture}" in result
    assert "苯的结构" in result and "和" in result
