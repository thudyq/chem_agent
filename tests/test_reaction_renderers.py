# -*- coding: utf-8 -*-
"""tests/test_reaction_renderers.py — REACTION / ARROW / RETRO 渲染器（布局引擎迁移）单元测试。

运行: python -m pytest tests/test_reaction_renderers.py -v
"""

import re

import pytest

from renderers.arrow import render_arrow
from renderers.reaction import render_reaction
from renderers.retro import render_retro

rdkit = pytest.importorskip("rdkit", reason="rdkit 未安装，跳过渲染测试")


def test_reaction_basic_layout():
    """REACTION：多反应物 + 多产物 + 条件，组件 scope 化、加号与主箭头齐全。"""
    out = render_reaction("c1ccccc1;[O-][N+](=O)O",
                          "O=[N+]([O-])c1ccccc1;O", "H2SO4, Δ")
    assert out.startswith("\\begin{tikzpicture}")
    assert out.count("\\begin{scope}[shift=") == 4     # 4 个分子组件
    assert len(re.findall(r"\\node at \([-\d.]+,[-\d.]+\) \{\$\+\$\}", out)) == 2   # 两侧各一个加号
    assert out.count("\\draw[->, very thick]") == 1
    assert "H$_2$SO$_4$, $\\Delta$" in out          # 条件自动下标（HNO3 已在反应物，不重复）
    assert "浓HNO$_3$" not in out                       # 催化剂条件不重复已写物种
    assert "\\fill" not in out                         # 普通方程式不画孤对电子


def test_reaction_no_overlap():
    """REACTION 布局：组件按序排列且不重叠（全局包围盒比较）。"""
    out = render_reaction("CCl;[OH-]", "CO;[Cl-]", "SN2")
    scopes = re.findall(
        r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\]", out)
    xs = sorted(float(sx) for sx, _ in scopes)
    assert len(xs) == 4
    assert all(b > a for a, b in zip(xs, xs[1:]))       # 严格递增、无重叠


def test_reaction_error_paths():
    """REACTION 错误路径：空反应物 / 空产物 / 无效 SMILES。"""
    assert "反应物不能为空" in render_reaction("", "CC", "")
    assert "产物不能为空" in render_reaction("CC", "", "")
    assert "无法为" in render_reaction("XYZ_INVALID", "CC", "")


def test_arrow_basic():
    """ARROW：两个组件 + 箭头 + 类型标注（箭头上方，与 REACTION 一致）。"""
    out = render_arrow("c1ccccc1", "c1ccccc1N", "amination")
    assert out.count("\\begin{scope}[shift=") == 2
    assert "\\draw[->, thick]" in out
    assert "node[midway, above] {amination};" in out
    assert "\\itshape" not in out          # 不再斜体（与 REACTION 条件一致）
    assert "\\node[below]" not in out
    assert "无效" not in out


def test_arrow_error_paths():
    """ARROW 错误路径：无效反应物 / 无效产物。"""
    assert "无效反应物" in render_arrow("XYZ", "CC", "t")
    assert "无效产物" in render_arrow("CC", "XYZ", "t")


def test_retro_open_arrow():
    """RETRO：双线推导箭头 ⇒（±0.05 平行双线杆 + 开放折线尖），总长 1.3。"""
    out = render_retro("O=Cc1ccccc1", "c1ccccc1", "formylation")
    assert out.count("\\begin{scope}[shift=") == 2
    # 双线杆：两条 y=±0.05 的平行线（等距间距，普通粗细）
    shaft = re.findall(r"\\draw \([-\d.]+,(0\.05|-0\.05)\) -- \([-\d.]+,\1\);", out)
    assert sorted(s for s in shaft) == ["-0.05", "0.05"]
    # 开放式折线尖（无封口竖线、无填充、非实心箭头）
    assert re.search(
        r"\\draw \([-\d.]+,0.13\) -- \([-\d.]+,0\) -- \([-\d.]+,-0.13\);", out)
    assert "cycle" not in out
    assert "\\fill" not in out
    assert "\\draw[->" not in out
    # 总长：杆起点 → 尖端 = 1.3
    m1 = re.search(r"\\draw \(([-\d.]+),0.05\) -- \([-\d.]+,0.05\);", out)
    m2 = re.search(r"-- \(([-\d.]+),0\) -- \([-\d.]+,-0.13\);", out)
    assert abs((float(m2.group(1)) - float(m1.group(1))) - 1.3) < 0.01
    assert "\\itshape formylation" in out


def test_retro_error_paths():
    """RETRO 错误路径：无效目标 / 无效前体。"""
    assert "无效目标" in render_retro("XYZ", "CC")
    assert "无效前体" in render_retro("CC", "XYZ")


def test_composite_resonance_forms_differ():
    """row + [RESARROW] 共振式保留显式键级（两 Kekulé 式不同）。"""
    from core.tag_parser import parse_tags
    from renderers.composite import render_composite
    out = render_composite(*parse_tags(
        "[COMPOSITE:row]"
        "[STRUCT:C1=CC=CC=C1]"
        "[RESARROW]"
        "[STRUCT:C1C=CC=CC=1]"
        "[/COMPOSITE]"
    )[0].args)
    scopes = out.split("\\begin{scope}")[1:]
    assert len(scopes) == 2
    assert scopes[0] != scopes[1]


# ---------------------------------------------------------------------------
# 双轨制（20260814）：公式物种（KMnO4 等）渲染为下标文本节点。
# ---------------------------------------------------------------------------


def test_reaction_kmno4_oxidation_renders_text_species():
    """KMnO4 氧化乙醇：SMILES 物种画结构式、化学式物种渲染为下标文本。"""
    out = render_reaction("5CCO;4KMnO4;6H2SO4",
                          "5CH3COOH;4MnSO4;2K2SO4;11H2O", "Δ")
    assert out.startswith("\\begin{tikzpicture}")
    # 唯一结构式 scope = CCO（乙醇）；其余 6 物种均为文本节点
    assert out.count("\\begin{scope}[shift=") == 1
    assert "KMnO$_4$" in out
    assert "H$_2$SO$_4$" in out
    assert "MnSO$_4$" in out
    assert "K$_2$SO$_4$" in out
    assert "H$_2$O" in out
    assert "CH$_3$COOH" in out
    assert "无法为" not in out


def test_reaction_all_formula_species():
    """纯化学式轨：全部物种为公式文本节点，无结构式 scope。"""
    out = render_reaction("5CH3CH2OH;4KMnO4", "5CH3COOH;4MnSO4", "H+")
    assert out.count("\\begin{scope}[shift=") == 0
    assert "CH$_3$CH$_2$OH" in out
    assert "KMnO$_4$" in out
    assert "H$^{+}$" in out


def test_reaction_formula_species_coeff_nodes():
    """公式物种带系数：系数节点与文本节点并存（4KMnO4 → {4} + KMnO$_4$）。"""
    out = render_reaction("CCO;4KMnO4", "CH3COOH;4MnSO4", "")
    assert "KMnO$_4$" in out
    # 系数 4 节点（4KMnO4）
    assert re.search(r"\\node at \([-\d.]+,0\) \{4\};", out)


def test_reaction_formula_invalid_species_still_errors():
    """既非 SMILES 也非化学式仍报错（XYZ_INVALID 不受双轨制影响）。"""
    out = render_reaction("XYZ_INVALID", "CC", "")
    assert "无法为" in out


def test_arrow_formula_product_renders_text():
    """ARROW：化学式产物（CH3COOH）渲染为文本节点而非结构式。"""
    out = render_arrow("CCO", "CH3COOH", "KMnO4, H+")
    assert out.count("\\begin{scope}[shift=") == 1  # 仅反应物 CCO 为结构式
    assert "CH$_3$COOH" in out
    assert "KMnO$_4$, H$^{+}$" in out


def test_arrow_all_formula_species():
    """ARROW：两侧均为化学式 → 无结构式 scope，全部文本节点。"""
    out = render_arrow("KMnO4", "MnSO4", "")
    assert out.count("\\begin{scope}[shift=") == 0
    assert "KMnO$_4$" in out
    assert "MnSO$_4$" in out


def test_layout_text_item_no_overlap():
    """布局：文本节点与结构式节点按序排列不重叠（全局包围盒比较）。"""
    out = render_reaction("CCO;KMnO4", "CH3COOH;MnSO4", "")
    scopes = re.findall(r"\\begin\{scope\}\[shift=\{\(([-\d.]+),([-\d.]+)\)\}\]",
                        out)
    # 公式文本节点在 y=0（结构式 scope 内原子标签在分子局部坐标 y≠0）
    text_xs = [float(m.group(1)) for m in
               re.finditer(r"\\node\[fill=white, inner sep=1pt\] at "
                           r"\(([-\d.]+),0\.00\)", out)]
    assert len(scopes) == 1
    assert len(text_xs) == 3  # KMnO4 / CH3COOH / MnSO4
    all_x = sorted([float(sx) for sx, _ in scopes] + text_xs)
    assert len(all_x) == 4
    # 各组件 x 坐标严格递增（加号/箭头占位隔开，无重叠）
    assert all(b - a > 0.5 for a, b in zip(all_x, all_x[1:]))


# ---------- 可逆反应（双向箭头 ⇌，20260816） ----------


def _rev_bars_and_tips(out):
    """解析双向箭头四段：返回 (横线, 尖)（浮点线段列表）。

    只匹配顶层（行首 2 空格）裸 \draw——分子 scope 内骨架键为 4 空格缩进，
    不混入。横线 = 两端 y 相同；尖 = 端点 y 差 0.10（±0.10 偏移，45°）。
    """
    segs = [(float(a), float(b), float(c), float(d)) for a, b, c, d in
            re.findall(r"(?m)^  \\draw \(([-\d.]+),([-\d.]+)\) -- "
                       r"\(([-\d.]+),([-\d.]+)\)", out)]
    bars = [s for s in segs if abs(s[1] - s[3]) < 1e-9]
    tips = [s for s in segs if abs(s[1] - s[3]) > 1e-9]
    return bars, tips


def test_reaction_reversible_double_arrow():
    """REACTION 条件含 ⇌：双向箭头四段裸 \draw 拼成——两条等长横线
    （间距 0.10，y=±0.05）+ 两个 45° 尖（偏移 ±0.10：上尖右上、下尖左下）。"""
    out = render_reaction("CC(=O)O;CCO", "CC(=O)OCC", "浓H2SO4, Δ, ⇌, -H2O")
    bars, tips = _rev_bars_and_tips(out)
    assert len(bars) == 2 and len(tips) == 2
    # 横线 y = ±0.05（间距 0.10）；上尖右上、下尖左下（±0.10 偏移）
    up_bar = [s for s in bars if s[1] > 0][0]
    dn_bar = [s for s in bars if s[1] < 0][0]
    assert round(up_bar[1], 2) == 0.05 and round(dn_bar[1], 2) == -0.05
    up_tip = [s for s in tips if s[1] > 0][0]
    dn_tip = [s for s in tips if s[1] < 0][0]
    # 上尖起点 = 上横线右端，终点偏移 (+x?) → (x2-0.10, 0.15)：向右上回折
    assert (round(up_tip[0], 2), round(up_tip[1], 2)) == \
        (round(up_bar[2], 2), round(up_bar[3], 2))
    assert round(up_tip[2], 2) == round(up_tip[0], 2) - 0.10
    assert abs(up_tip[3] - (up_tip[1] + 0.10)) < 0.005      # 终点 y = 起点 + 0.10
    # 下尖起点 = 下横线左端，终点偏移 → (x1+0.10, -0.15)：向左下回折
    assert (round(dn_tip[0], 2), round(dn_tip[1], 2)) == \
        (round(dn_bar[0], 2), round(dn_bar[1], 2))
    assert round(dn_tip[2], 2) == round(dn_tip[0], 2) + 0.10
    assert abs(dn_tip[3] - (dn_tip[1] - 0.10)) < 0.005      # 终点 y = 起点 - 0.10
    # 不用 -> 箭头样式（四段裸 \draw 拼成）
    assert "\\draw[->" not in out
    # 条件分挂：正条件在上条 above、-H2O 在下条 below
    assert "node[midway, above] {浓H$_2$SO$_4$, $\\Delta$}" in out
    assert "node[midway, below] {-H$_2$O}" in out
    # ⇌ 令牌已剥离
    assert "⇌" not in out


def test_reaction_reversible_no_condition():
    """REACTION 仅 ⇌（无条件）：两条裸横线 + 两个尖，无节点。"""
    out = render_reaction("CCO", "CC=O", "⇌")
    bars, tips = _rev_bars_and_tips(out)
    assert len(bars) == 2 and len(tips) == 2
    assert "node[midway" not in out
    assert "⇌" not in out


def test_reaction_single_arrow_unchanged():
    """回归锚点：无 ⇌ 时单向输出与迁移前逐字符一致（单条箭头 + above 条件）。"""
    out = render_reaction("c1ccccc1;[O-][N+](=O)O",
                          "O=[N+]([O-])c1ccccc1;O", "H2SO4, Δ")
    assert out.count("\\draw[->, very thick]") == 1
    assert "node[midway, above] {H$_2$SO$_4$, $\\Delta$}" in out
    assert "node[midway, below]" not in out


def test_arrow_reversible():
    """ARROW 条件含 ⇌：双向四段（无 -> 样式），类型文本在上条。"""
    out = render_arrow("CCO", "CC=O", "Cu, ⇌")
    bars, tips = _rev_bars_and_tips(out)
    assert len(bars) == 2 and len(tips) == 2
    assert "\\draw[->" not in out
    assert "node[midway, above] {Cu}" in out
    assert "⇌" not in out


def test_parse_arrow_kind():
    """parse_arrow_kind：⇌ 识别并剥离，其余原样。"""
    from renderers.mol_primitives import parse_arrow_kind
    assert parse_arrow_kind("⇌") == ("reversible", "")
    assert parse_arrow_kind("H2SO4, Δ, ⇌") == ("reversible", "H2SO4, Δ")
    assert parse_arrow_kind("H2SO4, Δ") == ("single", "H2SO4, Δ")
    assert parse_arrow_kind("") == ("single", "")
