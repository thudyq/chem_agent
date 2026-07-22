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
    assert out.count("$+$") == 2                       # 两侧各一个加号
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
    """RETRO：空心三角箭头（fill=white 三角）+ transform 标注（上方）。"""
    out = render_retro("O=Cc1ccccc1", "c1ccccc1", "formylation")
    assert out.count("\\begin{scope}[shift=") == 2
    assert "fill=white" in out and "-- cycle;" in out   # 空心三角
    assert "\\itshape formylation" in out
    assert "\\draw[->" not in out                       # 非实心箭头


def test_retro_error_paths():
    """RETRO 错误路径：无效目标 / 无效前体。"""
    assert "无效目标" in render_retro("XYZ", "CC")
    assert "无效前体" in render_retro("CC", "XYZ")
