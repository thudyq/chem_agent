# -*- coding: utf-8 -*-
"""renderers/layout.py — 统一坐标布局引擎（R-7）。

把"分子 + 连接符 + 反应箭头"组成的组件序列排布到统一水平坐标系：
按视觉包围盒计算每个组件的位置、避免重叠、对齐箭头。

composite / reaction_mech 及后续组合式渲染器（多步序列、共振组合、
势能面驻点结构叠加）共用此引擎；调用方只负责提供组件与顺序，
位置计算全部交给 layout_row。
"""

from dataclasses import dataclass, field
from typing import Any, Hashable, List, Tuple

from .mol_primitives import (
    atom_pos, bond_segments, condensed_atom_label, label_bond_margin,
    lone_pair_tikz, mol_visual_bbox,
)


@dataclass
class PlacedMol:
    """已定位的分子组件。"""
    key: Hashable                                  # 调用方标识（如组件 id / 序号）
    mol: Any                                       # RDKit Mol（局部 2D 坐标）
    shift: Tuple[float, float]                     # scope 平移量（全局 = 局部 + shift）
    bbox: Tuple[float, float, float, float]        # 局部视觉包围盒


@dataclass
class PlacedArrow:
    """已定位的主反应箭头。"""
    x1: float
    x2: float
    condition: str = ""


@dataclass
class RowLayout:
    """一行组件的布局结果。"""
    mols: List[PlacedMol]
    pluses: List[float]          # + 号中心 x 坐标
    arrows: List[PlacedArrow]
    width: float                 # 总宽（右端游标）
    resarrows: List[float] = field(default_factory=list)   # ↔ 中心 x 坐标

    def mol_map(self) -> dict:
        """key → PlacedMol 映射，便于按组件 id 取位置。"""
        return {p.key: p for p in self.mols}


def layout_row(items: list, *, mol_gap: float = 1.6, plus_w: float = 1.1,
               arrow_w: float = 2.6, arrow_pad: float = 0.65,
               res_w: float = 1.1,
               bbox_fn=mol_visual_bbox) -> RowLayout:
    """把组件序列排成一行。

    参数:
        items: 组件序列，元素为
            ("mol", key, mol)  分子组件（RDKit Mol，已有 2D 坐标）；
            ("plus",)          加号连接符；
            ("arrow", cond)    主反应箭头（cond 为条件文本，可空）；
            ("resarrow",)      共振箭头 ↔ 连接符。
        mol_gap: 相邻两个 mol 之间无连接符时的水平间距。
        plus_w / res_w: 加号 / 共振箭头占位宽度。
        arrow_w: 反应箭头占位宽度。
        arrow_pad: 箭头实际线段两端内缩量。
        bbox_fn: 视觉包围盒函数（默认含标签与孤对电子外延）。

    规则:
        - 每个 mol 按视觉包围盒宽度占位，包围盒中心垂直对齐 y=0；
        - 组件按给定顺序从左到右排布，互不重叠。
    """
    cursor = 0.0
    mols: List[PlacedMol] = []
    pluses: List[float] = []
    arrows: List[PlacedArrow] = []
    resarrows: List[float] = []
    prev_kind = None
    for item in items:
        kind = item[0]
        if kind == "mol":
            if prev_kind == "mol":
                cursor += mol_gap
            _, key, mol = item
            bbox = bbox_fn(mol)
            min_x, min_y, max_x, max_y = bbox
            w = max_x - min_x
            local_cx = (min_x + max_x) / 2.0
            local_cy = (min_y + max_y) / 2.0
            shift = (cursor + w / 2.0 - local_cx, -local_cy)
            mols.append(PlacedMol(key=key, mol=mol, shift=shift, bbox=bbox))
            cursor += w
        elif kind == "plus":
            pluses.append(cursor + plus_w / 2.0)
            cursor += plus_w
        elif kind == "resarrow":
            resarrows.append(cursor + res_w / 2.0)
            cursor += res_w
        elif kind == "arrow":
            cond = item[1] if len(item) > 1 else ""
            arrows.append(PlacedArrow(x1=cursor + arrow_pad,
                                      x2=cursor + arrow_w - arrow_pad,
                                      condition=cond))
            cursor += arrow_w
        prev_kind = kind
    return RowLayout(mols=mols, pluses=pluses, arrows=arrows, width=cursor,
                     resarrows=resarrows)


def layout_rows(items: list, *, row_gap: float = 1.2, **kwargs):
    """多行布局：items 中的 ("newline",) 分隔各行（R-6 上下排列）。

    每行用 layout_row 排布（kwargs 透传），再按行内最高组件的高度 + row_gap
    逐行向下堆叠（y 向下为负方向）。

    返回:
        (rows, y_offsets)：rows 为每行的 RowLayout，y_offsets 为每行相对
        第一行的向下偏移量（渲染时从组件 y 坐标中减去）。
    """
    rows_items, current = [], []
    for item in items:
        if item[0] == "newline":
            rows_items.append(current)
            current = []
        else:
            current.append(item)
    rows_items.append(current)

    rows, y_offsets = [], []
    y = 0.0
    for row_items in rows_items:
        layout = layout_row(row_items, **kwargs)
        rows.append(layout)
        y_offsets.append(y)
        height = max((p.bbox[3] - p.bbox[1] for p in layout.mols), default=1.0)
        y += height + row_gap
    return rows, y_offsets


def molecule_scope_lines(mol, shift: Tuple[float, float], *,
                         show_numbers: bool = False,
                         show_lone_pairs: bool = True) -> List[str]:
    r"""分子组件的 scope 绘制行（内部全部局部坐标，位置由 shift 决定）。

    普通方程式（show_lone_pairs=False）不画孤对电子点；
    机理场景（show_lone_pairs=True）画出电子点；show_numbers 显示原子序号。
    """
    lines = [
        f"  \\begin{{scope}}[shift={{({shift[0]:.2f},{shift[1]:.2f})}}]"
    ]
    for segs in bond_segments(mol, labeler=condensed_atom_label,
                              margin_fn=label_bond_margin):
        for x1, y1, x2, y2 in segs:
            lines.append(f"    \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")
    for atom in mol.GetAtoms():
        x, y = atom_pos(mol, atom.GetIdx())
        lab = condensed_atom_label(atom)
        if lab:
            lines.append(
                f"    \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};"
            )
        if show_numbers:
            lines.append(
                f"    \\node[font=\\tiny, gray, below right] at ({x:.2f},{y:.2f}) "
                f"{{{atom.GetIdx()}}};"
            )
    if show_lone_pairs:
        for atom in mol.GetAtoms():
            for dot_line in lone_pair_tikz(mol, atom.GetIdx()):
                lines.append(f"    {dot_line}")
    lines.append("  \\end{scope}")
    return lines


def energy_point_coords(values: list, *, x0: float = 1.0, xstep: float = 1.5,
                        height: float = 3.0) -> dict:
    """能量点序列 → 势能面画布坐标（R-3 驻点布局）。

    返回 dict：
        points: [(i, value, x, y), ...]  每个驻点的序号、能量值、画布坐标；
        max_idx: 能量最高点下标（过渡态候选）；
        x_last: 最后一个点的 x 坐标（坐标轴/标注框定位用）。
    """
    n = len(values)
    emax, emin = max(values), min(values)
    erange = (emax - emin) or 1.0
    points = []
    for i, v in enumerate(values):
        x = x0 + i * xstep
        y = (v - emin) / erange * height
        points.append((i, v, x, y))
    return {
        "points": points,
        "max_idx": values.index(emax),
        "x_last": x0 + (n - 1) * xstep,
    }


def energy_point_roles(values: list) -> dict:
    """能量点角色判定（两步/多步反应标注）。

    首点 → 反应物，尾点 → 产物，局部极大 → 过渡态，局部极小 → 反应中间体；
    平台段（与相邻等值）不标注。
    """
    n = len(values)
    roles = {}
    if n < 2:
        return roles
    roles[0] = "反应物"
    roles[n - 1] = "产物"
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            roles[i] = "过渡态"
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            roles[i] = "反应中间体"
    return roles


def place_bbox(bbox: Tuple[float, float, float, float], x: float, y: float,
               side: str = "above", margin: float = 0.5) -> Tuple[float, float]:
    """把组件包围盒放到 (x, y) 的指定方位，返回 scope shift。

    side="above"：包围盒底边距 (x,y) 上方 margin；
    side="below"：包围盒顶边距 (x,y) 下方 margin。
    """
    min_x, min_y, max_x, max_y = bbox
    sx = x - (min_x + max_x) / 2.0
    sy = (y + margin - min_y) if side == "above" else (y - margin - max_y)
    return (sx, sy)


def energy_annotation_placement(occupied: list, x_last: float, *,
                                box_h: float = 0.8, gap: float = 0.3,
                                min_top: float = 3.5):
    """势能面 Ea/ΔH 标注框与纵轴高度的无遮挡布局。

    occupied: 已占区域 [(min_x, min_y, max_x, max_y), ...]（全局坐标，
    含驻点结构、驻点标签）。标注框放在最高组件上方的净空带（box 底边
    高于一切组件），水平方向置于离最高组件较远的一侧；纵轴随之加高。

    返回 (x, y, anchor, axis_top)：标注框节点坐标与 anchor（north east/
    north west），以及纵轴箭头顶端高度。
    """
    max_top = max([min_top] + [r[3] for r in occupied])
    box_top = max_top + gap + box_h
    axis_top = box_top + 0.2
    highest = max(occupied, key=lambda r: r[3]) if occupied else None
    mid = (x_last + 0.8) / 2.0
    if highest is not None and (highest[0] + highest[2]) / 2.0 < mid:
        return x_last + 0.7, box_top, "north east", axis_top
    return 0.15, box_top, "north west", axis_top


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from renderers.mol_primitives import prepare_mol, scale_mol_coords

    def _mol(smi):
        m = prepare_mol(smi)
        scale_mol_coords(m, 0.8)
        return m

    items = [
        ("mol", "r0", _mol("CCl")),
        ("plus",),
        ("mol", "nu", _mol("[OH-]")),
        ("arrow", "SN2"),
        ("mol", "p0", _mol("CO")),
        ("plus",),
        ("mol", "p1", _mol("[Cl-]")),
    ]
    result = layout_row(items)
    print(f"总宽: {result.width:.2f}")
    for p in result.mols:
        print(f"  mol {p.key}: shift=({p.shift[0]:.2f},{p.shift[1]:.2f}), "
              f"bbox w={p.bbox[2] - p.bbox[0]:.2f}")
    print(f"  pluses: {[f'{x:.2f}' for x in result.pluses]}")
    for a in result.arrows:
        print(f"  arrow: ({a.x1:.2f},{a.x2:.2f}) cond={a.condition}")
