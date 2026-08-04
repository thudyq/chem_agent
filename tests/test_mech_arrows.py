# -*- coding: utf-8 -*-
"""renderers.composite.draw_mech_arrows 单元测试（改进 1 模块化）。

机理箭头几何逻辑（p0/p1 定位、端点吸附避让）提取为独立函数后，
可脱离完整 COMPOSITE 渲染独立测试。需要真实 RDKit。
"""

import pytest

from renderers.composite import _parse_mech_arrows, draw_mech_arrows
from renderers.mol_primitives import prepare_mol, scale_mol_coords

rdkit = pytest.importorskip("rdkit", reason="rdkit 未安装，跳过机理箭头测试")


def _mols(*smiles):
    """SMILES 列表 → {id: {"mol", "shift", ...}} 组件表（r0, r1, ...）。

    字段与 composite._collect_components 填充的组件表一致
    （_mech_labeler 依赖 explicit_hs / xh / bonds / hbonds）。
    """
    mols = {}
    for i, smi in enumerate(smiles):
        m = prepare_mol(smi)
        scale_mol_coords(m, 0.8)
        mols[f"r{i}"] = {
            "mol": m,
            "shift": (0.0, 0.0),
            "explicit_hs": {},
            "xh": [], "bonds": [], "hbonds": [], "charges": [],
        }
    return mols


def test_draw_mech_arrows_basic():
    """合法进攻箭头：输出含红色贝塞尔弯箭头。"""
    mols = _mols("CCl", "CO")
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:0>r1:0"]))
    assert lines, "应产出箭头 TikZ 行"
    assert any("red" in ln and "controls" in ln for ln in lines)


def test_draw_mech_arrows_fishhook():
    """鱼钩箭头（>>）同样绘制。"""
    mols = _mols("CCl", "CO")
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:0>>r1:0"]))
    assert any("red" in ln and "controls" in ln for ln in lines)


def test_draw_mech_arrows_bond_break():
    """断键箭头（起点为 a-b 键中点）绘制，向下弯曲。"""
    mols = _mols("CCl", "CO")
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:0-1>r0:1"]))
    assert any("red" in ln and "controls" in ln for ln in lines)


def test_draw_mech_arrows_skip_unknown_id():
    """引用未知组件 id 的箭头跳过，不崩溃。"""
    mols = _mols("CCl")
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["ghost:0>r0:0"]))
    assert lines == []


def test_draw_mech_arrows_skip_invalid_point():
    """端点超出原子数 / 空端点跳过，不崩溃。"""
    mols = _mols("CCl")  # 2 个原子
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:9>r0:0"]))
    assert lines == []
    lines = draw_mech_arrows(mols, _parse_mech_arrows(["r0:0>r0:9"]))
    assert lines == []


def test_draw_mech_arrows_empty():
    """空箭头列表返回空。"""
    assert draw_mech_arrows(_mols("CCl"), []) == []
