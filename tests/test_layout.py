# -*- coding: utf-8 -*-
"""tests/test_layout.py — 统一坐标布局引擎（renderers/layout.py）单元测试。

运行: python -m pytest tests/test_layout.py -v
"""

import pytest

from renderers.layout import layout_row
from renderers.mol_primitives import prepare_mol, scale_mol_coords

rdkit = pytest.importorskip("rdkit", reason="rdkit 未安装，跳过布局测试")


def _mol(smi: str):
    m = prepare_mol(smi)
    assert m is not None
    scale_mol_coords(m, 0.8)
    return m


def _global_bbox(placed):
    min_x, min_y, max_x, max_y = placed.bbox
    sx, sy = placed.shift
    return min_x + sx, min_y + sy, max_x + sx, max_y + sy


def test_basic_sequence_positions():
    """mol + plus + mol + arrow + mol：位置顺序与不重叠。"""
    items = [
        ("mol", "r0", _mol("CCl")),
        ("plus",),
        ("mol", "nu", _mol("[OH-]")),
        ("arrow", "SN2"),
        ("mol", "p0", _mol("CO")),
    ]
    result = layout_row(items)
    assert len(result.mols) == 3
    assert len(result.pluses) == 1
    assert len(result.arrows) == 1

    b0 = _global_bbox(result.mols[0])
    b1 = _global_bbox(result.mols[1])
    b2 = _global_bbox(result.mols[2])
    plus = result.pluses[0]
    arrow = result.arrows[0]

    assert b0[2] < plus < b1[0]        # 加号位于前两个分子之间
    assert b1[2] < arrow.x1            # 箭头在反应物右侧
    assert arrow.x2 < b2[0]            # 箭头在产物左侧
    assert b0[2] < b1[0] < b2[0]       # 分子互不重叠且有序


def test_y_centering():
    """每个分子的视觉包围盒中心对齐 y=0。"""
    items = [("mol", "a", _mol("CCl")), ("mol", "b", _mol("[OH-]"))]
    result = layout_row(items)
    for placed in result.mols:
        _, min_y, _, max_y = _global_bbox(placed)
        assert abs((min_y + max_y) / 2.0) < 1e-6


def test_mol_gap_without_connector():
    """相邻 mol 无连接符时保留 mol_gap 间距。"""
    items = [("mol", "a", _mol("CCl")), ("mol", "b", _mol("CO"))]
    result = layout_row(items, mol_gap=1.6)
    b0 = _global_bbox(result.mols[0])
    b1 = _global_bbox(result.mols[1])
    assert abs((b1[0] - b0[2]) - 1.6) < 1e-6


def test_arrow_span_and_width():
    """箭头跨度 = arrow_w - 2*arrow_pad；总宽为右端游标。"""
    items = [("mol", "a", _mol("CCl")), ("arrow", "x"), ("mol", "b", _mol("CO"))]
    result = layout_row(items, arrow_w=2.6, arrow_pad=0.65)
    arrow = result.arrows[0]
    assert abs((arrow.x2 - arrow.x1) - (2.6 - 2 * 0.65)) < 1e-6
    b1 = _global_bbox(result.mols[1])
    assert abs(result.width - b1[2]) < 1e-6


def test_keys_and_mol_map():
    """组件 key 保留，mol_map 可按 key 取位置。"""
    items = [
        ("mol", "sub", _mol("CCl")),
        ("mol", "nu", _mol("[OH-]")),
    ]
    result = layout_row(items)
    m = result.mol_map()
    assert set(m.keys()) == {"sub", "nu"}
    assert m["sub"].shift != m["nu"].shift


def test_multi_arrows_sequence():
    """多步序列（A → B → C）：多个箭头按序排布。"""
    items = [
        ("mol", "a", _mol("C=C")),
        ("arrow", "H2O"),
        ("mol", "b", _mol("CCO")),
        ("arrow", "CuO"),
        ("mol", "c", _mol("CC=O")),
    ]
    result = layout_row(items)
    assert len(result.arrows) == 2
    assert result.arrows[0].condition == "H2O"
    assert result.arrows[1].condition == "CuO"
    assert result.arrows[0].x2 < result.arrows[1].x1
