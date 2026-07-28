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
    out = render_reaction("c1ccccc1;[O-][N+](=O)[O-]",
                          "O=[N+]([O-])c1ccccc1;O", "H2SO4, 浓HNO3")
    assert out.startswith("\\begin{tikzpicture}")
    assert out.count("\\begin{scope}[shift=") == 4     # 4 个分子组件
    assert len(re.findall(r"\\node at \([-\d.]+,[-\d.]+\) \{\$\+\$\}", out)) == 2   # 两侧各一个加号
    assert out.count("\\draw[->, very thick]") == 1
    assert "H$_2$SO$_4$, 浓HNO$_3$" in out             # 条件自动下标
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
    """ARROW：两个组件 + 箭头 + 类型标注（下方斜体）。"""
    out = render_arrow("c1ccccc1", "c1ccccc1N", "amination")
    assert out.count("\\begin{scope}[shift=") == 2
    assert "\\draw[->, thick]" in out
    assert "\\itshape amination" in out
    assert "无效" not in out


def test_arrow_error_paths():
    """ARROW 错误路径：无效反应物 / 无效产物。"""
    assert "无效反应物" in render_arrow("XYZ", "CC", "t")
    assert "无效产物" in render_arrow("CC", "XYZ", "t")


def test_retro_open_arrow():
    """RETRO：双线推导箭头 ⇒（双线杆+实心尖，较粗），长度与普通反应箭头一致。"""
    out = render_retro("O=Cc1ccccc1", "c1ccccc1", "formylation")
    assert out.count("\\begin{scope}[shift=") == 2
    assert "double distance" in out                      # 双线杆
    assert re.search(r"\\fill \([-\d.]+,0\) -- \([-\d.]+,0.13\) -- \([-\d.]+,-0.13\) -- cycle;", out)
    assert "\\draw[->" not in out                        # 非实心箭头
    m = re.search(r"double distance=1.8pt, line width=0.9pt\] \(([-\d.]+),0\) -- \(([-\d.]+),0\)", out)
    assert m is not None
    assert abs((float(m.group(2)) + 0.22 - float(m.group(1))) - 1.3) < 0.01   # 含尖端总长 1.3
    assert "\\itshape formylation" in out


def test_retro_error_paths():
    """RETRO 错误路径：无效目标 / 无效前体。"""
    assert "无效目标" in render_retro("XYZ", "CC")
    assert "无效前体" in render_retro("CC", "XYZ")


def test_resonance_basic():
    """RESONANCE（布局引擎迁移）：极限式 scope 化 + ↔ + 孤对电子点 + 圆圈电荷。"""
    from renderers.resonance import render_resonance
    out = render_resonance("CC(=O)[O-]~CC([O-])=O")
    assert out.startswith("\\begin{tikzpicture}")
    assert out.count("\\begin{scope}[shift=") == 2
    assert out.count("$\\leftrightarrow$") == 1
    assert out.count("\\node[draw, circle") == 2       # 两个 O- 的圆圈电荷
    assert "\\fill" in out                            # 共振场景画出孤对电子


def test_resonance_error_paths():
    """RESONANCE 错误路径：空内容 / 少于 2 式 / 无效 SMILES。"""
    from renderers.resonance import render_resonance
    assert "内容为空" in render_resonance("")
    assert "至少需要 2 个" in render_resonance("CC")
    assert "无效 SMILES" in render_resonance("CC~XYZ_INVALID")


def test_resonance_kekule_forms_differ():
    """两个 Kekulé 式双键位置不同（跳过芳香化，不被统一成同一结构）。"""
    from renderers.mol_primitives import prepare_mol
    from renderers.resonance import render_resonance

    def doubles(m):
        return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                for b in m.GetBonds() if b.GetBondTypeAsDouble() >= 1.5}

    m1 = prepare_mol("C1=CC=CC=C1", allow_aromatic=False)
    m2 = prepare_mol("C1C=CC=CC=1", allow_aromatic=False)
    assert doubles(m1) != doubles(m2)

    out = render_resonance("C1=CC=CC=C1~C1C=CC=CC=1")
    scopes = out.split("\\begin{scope}")[1:]
    assert len(scopes) == 2
    assert scopes[0] != scopes[1]                # 两式绘制内容不同


def test_composite_resonance_forms_differ():
    """COMPOSITE resonance 布局同样保留显式键级（两式不同）。"""
    from core.tag_parser import parse_tags
    from renderers.composite import render_composite
    out = render_composite(*parse_tags(
        "[COMPOSITE:resonance]"
        "[STRUCT:C1=CC=CC=C1]"
        "[STRUCT:C1C=CC=CC=1]"
        "[/COMPOSITE]"
    )[0].args)
    scopes = out.split("\\begin{scope}")[1:]
    assert len(scopes) == 2
    assert scopes[0] != scopes[1]
