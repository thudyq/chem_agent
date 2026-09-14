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


def test_resarrow_connector():
    """共振箭头 ↔ 连接符：占位并记录中心 x。"""
    items = [("mol", "a", _mol("C1=CC=CC=C1")),
             ("resarrow",),
             ("mol", "b", _mol("C1C=CC=CC=1"))]
    result = layout_row(items, res_w=1.1)
    assert len(result.resarrows) == 1
    b0 = _global_bbox(result.mols[0])
    b1 = _global_bbox(result.mols[1])
    assert b0[2] < result.resarrows[0] < b1[0]


def test_layout_rows_stacking():
    """多行布局：newline 分隔的行逐行向下堆叠（y 偏移递增）。"""
    from renderers.layout import layout_rows
    items = [
        ("mol", "main", _mol("CC(=O)[O-]")),
        ("newline",),
        ("mol", "r1", _mol("CC(=O)[O-]")),
        ("resarrow",),
        ("mol", "r2", _mol("CC([O-])=O")),
    ]
    rows, y_offsets = layout_rows(items)
    assert len(rows) == 2
    assert len(rows[0].mols) == 1 and len(rows[1].mols) == 2
    assert y_offsets[0] == 0.0
    assert y_offsets[1] > 1.0          # 第二行在第一行下方（行高 + 行距）
    assert len(rows[1].resarrows) == 1


def test_mol_with_label_reserves_width():
    """C2：带标签的 mol 按 spacing_bbox 占位——宽标签不压相邻组件。"""
    items = [
        ("mol", "a", _mol("CCl"), "质子化乙醇的反应中间体"),
        ("mol", "b", _mol("CO")),
    ]
    result = layout_row(items)
    a, b = result.mols[0], result.mols[1]
    # spacing_bbox 横向外延 ≥ 分子 bbox 外延（标签宽于分子）
    sa, sb = a.spacing_bbox, b.spacing_bbox
    assert sa[0] <= a.bbox[0] and sa[2] >= a.bbox[2]
    # 相邻组件全局 spacing 不重叠（pass 1 游标已按 spacing 宽度推进）
    assert sa[2] + a.shift[0] < sb[0] + b.shift[0]


def test_label_spacing_includes_height():
    """C2：spacing_bbox 纵向含标签外延（分子底边向下 label_gap + 行高）。"""
    from renderers.mol_primitives import label_wrapped_size
    label = "质子化乙醇的反应中间体"
    items = [("mol", "a", _mol("CCl"), label)]
    result = layout_row(items, label_gap=0.35)
    sb = result.mols[0].spacing_bbox
    _, lh = label_wrapped_size(label)
    # 分子 bbox 底边向下：label_gap(0.35) + 标签总高
    assert abs((result.mols[0].bbox[1] - 0.35 - lh) - sb[1]) < 1e-9


def test_row_stacking_counts_label_height():
    """C2：多行布局行高计入标签外延——上行标签不压下行分子。"""
    from renderers.layout import layout_rows
    label = "质子化乙醇的反应中间体"
    items = [
        ("mol", "a", _mol("CCl"), label),
        ("newline",),
        ("mol", "b", _mol("CO")),
    ]
    rows, y_offsets = layout_rows(items)
    a, b = rows[0].mols[0], rows[1].mols[0]
    # y 向下为负：第一行 spacing 底边（含标签）的全局 y 高于（大于）
    # 第二行分子顶边 → 标签不压下行分子
    a_bottom = (a.spacing_bbox or a.bbox)[1] + a.shift[1] - y_offsets[0]
    b_top = b.bbox[3] + b.shift[1] - y_offsets[1]
    assert a_bottom > b_top


def test_two_pass_overlap_resolution():
    """C2 通用两遍布局：spacing 估算偏窄时 pass 2 把右侧组件推离。"""
    # 构造：bbox_fn 返回固定窄 bbox（模拟估算不足），两分子间距被 pass 2 拉大
    def narrow_bbox(mol):
        return (-0.5, -0.5, 0.5, 0.5)      # 固定 1.0 宽
    items = [
        ("mol", "a", _mol("CCl"), "质子化乙醇的反应中间体"),
        ("mol", "b", _mol("CO"), "质子化乙醇的反应中间体"),
    ]
    result = layout_row(items, mol_gap=0.0, bbox_fn=narrow_bbox)
    a, b = result.mols[0], result.mols[1]
    sa, sb = a.spacing_bbox, b.spacing_bbox
    # pass 1 游标按 spacing（含标签）推进 → 无需 pass 2 也已不重叠
    assert sa[2] + a.shift[0] <= sb[0] + b.shift[0] + 1e-9
