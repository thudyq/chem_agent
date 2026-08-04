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
