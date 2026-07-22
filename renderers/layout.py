# -*- coding: utf-8 -*-
"""renderers/layout.py — 统一坐标布局引擎（R-7）。

把"分子 + 连接符 + 反应箭头"组成的组件序列排布到统一水平坐标系：
按视觉包围盒计算每个组件的位置、避免重叠、对齐箭头。

composite / reaction_mech 及后续组合式渲染器（多步序列、共振组合、
势能面驻点结构叠加）共用此引擎；调用方只负责提供组件与顺序，
位置计算全部交给 layout_row。
"""

from dataclasses import dataclass
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

    def mol_map(self) -> dict:
        """key → PlacedMol 映射，便于按组件 id 取位置。"""
        return {p.key: p for p in self.mols}


def layout_row(items: list, *, mol_gap: float = 1.6, plus_w: float = 1.1,
               arrow_w: float = 2.6, arrow_pad: float = 0.65,
               bbox_fn=mol_visual_bbox) -> RowLayout:
    """把组件序列排成一行。

    参数:
        items: 组件序列，元素为
            ("mol", key, mol)  分子组件（RDKit Mol，已有 2D 坐标）；
            ("plus",)          加号连接符；
            ("arrow", cond)    主反应箭头（cond 为条件文本，可空）。
        mol_gap: 相邻两个 mol 之间无连接符时的水平间距。
        plus_w: 加号占位宽度。
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
        elif kind == "arrow":
            cond = item[1] if len(item) > 1 else ""
            arrows.append(PlacedArrow(x1=cursor + arrow_pad,
                                      x2=cursor + arrow_w - arrow_pad,
                                      condition=cond))
            cursor += arrow_w
        prev_kind = kind
    return RowLayout(mols=mols, pluses=pluses, arrows=arrows, width=cursor)


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
