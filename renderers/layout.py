# -*- coding: utf-8 -*-
"""renderers/layout.py — 统一坐标布局引擎（R-7）。

把"分子 + 连接符 + 反应箭头"组成的组件序列排布到统一水平坐标系：
按视觉包围盒计算每个组件的位置、避免重叠、对齐箭头。

composite 及组合式渲染器（多步序列、共振组合、
势能面驻点结构叠加）共用此引擎；调用方只负责提供组件与顺序，
位置计算全部交给 layout_row。
"""

from dataclasses import dataclass, field
from typing import Any, Hashable, List, Tuple

from .collide import DOT_R, Occupancy
from .mol_primitives import (
    _label_flip_for, aromatic_ring_info, atom_label, atom_main_label, atom_pos,
    bond_segments, charge_tikz, heavy_atom_count, label_bond_margin,
    label_visual_width,
    label_wrapped_size, lone_pair_dot_groups, lone_pair_tikz, mol_visual_bbox,
)


@dataclass
class PlacedMol:
    """已定位的分子组件。"""
    key: Hashable                                  # 调用方标识（如组件 id / 序号）
    mol: Any                                       # RDKit Mol（局部 2D 坐标）
    shift: Tuple[float, float]                     # scope 平移量（全局 = 局部 + shift）
    bbox: Tuple[float, float, float, float]        # 局部视觉包围盒
    label: str = ""                                # 组件级标签（宽于分子时计入占位）
    coeff: float = 1.0                             # 化学计量系数（渲染系数节点）
    spacing_bbox: Tuple[float, float, float, float] = None  # 含标签外延的间距包围盒


@dataclass
class PlacedArrow:
    """已定位的主反应箭头。"""
    x1: float
    x2: float
    condition: str = ""
    kind: str = "single"     # 大一统架构箭头类型（single/reversible/resonance/retro）
    sup: list = field(default_factory=list)  # 附件引用（+E 副反应物 / -F 副产物）


@dataclass
class PlacedText:
    """已定位的纯文本组件（化学式文本节点，如 KMnO4，无分子结构）。"""
    key: Hashable                                 # 调用方标识（如组件 id / 序号）
    text: str                                     # 原始文本（渲染端 wrap_format_text 排版）
    shift: Tuple[float, float]                    # 节点平移量（全局 = 局部 + shift）
    bbox: Tuple[float, float, float, float]       # 局部包围盒（中心在 y=0）
    coeff: float = 1.0                            # 化学计量系数（渲染系数节点）
    spacing_bbox: Tuple[float, float, float, float] = None  # 间距包围盒（同 bbox）


@dataclass
class PlacedBlock:
    """已定位的复合块组件（[BLOCK] 共振块，20260819 大一统架构）。

    lines 为块内部预渲染的 TikZ 行（局部坐标，含分子 scope），
    整体随 shift 平移；bbox 为块内部布局的局部包围盒（占位宽度）。
    """
    key: Hashable
    lines: List[str]
    shift: Tuple[float, float]
    bbox: Tuple[float, float, float, float]


@dataclass
class RowLayout:
    """一行组件的布局结果。"""
    mols: List[PlacedMol]
    pluses: List[float]          # + 号中心 x 坐标
    arrows: List[PlacedArrow]
    width: float                 # 总宽（右端游标）
    resarrows: List[float] = field(default_factory=list)   # ↔ 中心 x 坐标
    texts: List[PlacedText] = field(default_factory=list)  # 纯文本组件（化学式）
    blocks: List[PlacedBlock] = field(default_factory=list)  # 复合块组件

    def mol_map(self) -> dict:
        """key → PlacedMol 映射，便于按组件 id 取位置。"""
        return {p.key: p for p in self.mols}


def layout_row(items: list, *, mol_gap: float = 1.6, plus_w: float = 1.1,
               arrow_w: float = 2.6, arrow_pad: float = 0.65,
               res_w: float = 1.1,
               label_gap: float = 0.35,
               bbox_fn=mol_visual_bbox) -> RowLayout:
    """把组件序列排成一行。

    参数:
        items: 组件序列，元素为
            ("mol", key, mol)             分子组件（RDKit Mol，已有 2D 坐标）；
            ("mol", key, mol, label)      同上，附组件级标签（字符串，宽于分子时
                                          spacing_bbox 计入标签外延，防重叠）；
            ("mol", key, mol, coeff)      同上，附化学计量系数（数字，如 2 或
                                          0.5；渲染端负责画系数节点，布局仅预留
                                          左侧宽度）；
            ("text", key, text, coeff)    纯文本组件（化学式如 KMnO4，无分子结构，
                                          渲染端画文本节点；coeff 语义同上）；
            ("plus",)          加号连接符；
            ("arrow", cond)    主反应箭头（cond 为条件文本，可空）；
            ("resarrow",)      共振箭头 ↔ 连接符。
        mol_gap: 相邻两个 mol 之间无连接符时的水平间距。
        plus_w / res_w: 加号 / 共振箭头占位宽度。
        arrow_w: 反应箭头占位宽度。
        arrow_pad: 箭头实际线段两端内缩量。
        label_gap: 标签顶边到分子 bbox 底边的间距（与绘制端一致）。
        bbox_fn: 分子视觉包围盒函数（默认含原子标签与孤对电子外延）。

    规则:
        - 每个 mol 按 spacing_bbox 宽度占位（含标签外延时宽于分子），
          包围盒中心垂直对齐 y=0；带标签的分子垂直中心仍按分子 bbox
          （标签在分子下方，不抬升分子）；
        - 组件按给定顺序从左到右排布；两遍布局（pass 2）：排布后检测
          相邻组件全局 spacing_bbox 重叠，把右侧组件向右推离（最多 5 轮）。
    """
    cursor = 0.0
    mols: List[PlacedMol] = []
    texts: List[PlacedText] = []
    pluses: List[float] = []
    arrows: List[PlacedArrow] = []
    resarrows: List[float] = []
    blocks: List[PlacedBlock] = []
    prev_kind = None
    for item in items:
        kind = item[0]
        if kind == "mol":
            if prev_kind == "mol":
                cursor += mol_gap
            _, key, mol = item[:3]
            extra = item[3] if len(item) > 3 else ""
            label = extra if isinstance(extra, str) else ""
            coeff = extra if isinstance(extra, (int, float)) else 0.0
            bbox = bbox_fn(mol)
            min_x, min_y, max_x, max_y = bbox
            local_cy = (min_y + max_y) / 2.0
            # 系数节点左侧留白（"2" / "1/2" 约 0.5 宽，乘位数缩放）
            coeff_pad = 0.6 if coeff and coeff != 1 else 0.0
            if label:
                # 标签外延：以分子 bbox 中心 x 为轴，左右各扩 label 半宽；
                # 纵向从分子底边向下 label_gap + 行高×行数
                cx = (min_x + max_x) / 2.0
                lw, lh = label_wrapped_size(label)
                label_half = lw / 2.0
                spacing = (min(min_x, cx - label_half),
                           min_y - label_gap - lh,
                           max(max_x, cx + label_half),
                           max_y)
            else:
                spacing = bbox
            smin_x, smin_y, smax_x, smax_y = spacing
            w = smax_x - smin_x + coeff_pad
            local_cx = (smin_x + smax_x) / 2.0 + coeff_pad / 2.0
            shift = (cursor + w / 2.0 - local_cx, -local_cy)
            mols.append(PlacedMol(key=key, mol=mol, shift=shift, bbox=bbox,
                                  label=label, coeff=coeff or 1.0,
                                  spacing_bbox=spacing))
            cursor += w
        elif kind == "text":
            if prev_kind in ("mol", "text"):
                cursor += mol_gap
            _, key, text = item[:3]
            coeff = item[3] if len(item) > 3 else 0.0
            w_t, h_t = label_wrapped_size(text)
            bbox = (-w_t / 2.0, -h_t / 2.0, w_t / 2.0, h_t / 2.0)
            # 系数节点左侧留白（与 mol 分支一致）
            coeff_pad = 0.6 if coeff and coeff != 1 else 0.0
            w = w_t + coeff_pad
            local_cx = coeff_pad / 2.0
            shift = (cursor + w / 2.0 - local_cx, 0.0)
            texts.append(PlacedText(key=key, text=text, shift=shift, bbox=bbox,
                                    coeff=coeff or 1.0, spacing_bbox=bbox))
            cursor += w
        elif kind == "plus":
            pluses.append(cursor + plus_w / 2.0)
            cursor += plus_w
        elif kind == "resarrow":
            resarrows.append(cursor + res_w / 2.0)
            cursor += res_w
        elif kind == "arrow":
            cond = item[1] if len(item) > 1 else ""
            a_kind = item[2] if len(item) > 2 else "single"
            sup = item[3] if len(item) > 3 else []
            arrows.append(PlacedArrow(x1=cursor + arrow_pad,
                                      x2=cursor + arrow_w - arrow_pad,
                                      condition=cond, kind=a_kind, sup=sup))
            cursor += arrow_w
        elif kind == "block":
            # 复合块组件（[BLOCK] 共振块）：按块 bbox 占位，整体平移
            if prev_kind in ("mol", "text", "block"):
                cursor += mol_gap
            _, key, blines, bbox = item[:4]
            min_x, min_y, max_x, max_y = bbox
            w = max_x - min_x
            local_cy = (min_y + max_y) / 2.0
            local_cx = (min_x + max_x) / 2.0
            shift = (cursor + w / 2.0 - local_cx, -local_cy)
            blocks.append(PlacedBlock(key=key, lines=blines, shift=shift,
                                      bbox=bbox))
            cursor += w
        prev_kind = kind
    return RowLayout(mols=mols, pluses=pluses, arrows=arrows, width=cursor,
                     resarrows=resarrows, texts=texts, blocks=blocks)


def _resolve_row_overlaps(layout: RowLayout, pad: float = 0.05) -> float:
    """两遍布局 pass 2：相邻组件全局 spacing_bbox 重叠时右推（原地修改 shift）。

    游标式排布（按 spacing 宽度）本身不重叠，本遍是兜底——当 spacing
    估算与实际渲染仍有偏差（如标签实际更宽）时把右侧组件推离。
    返回修正后的总宽（右端最大坐标）。
    """
    for _ in range(5):
        moved = False
        placed = sorted(list(layout.mols) + list(layout.texts),
                        key=lambda p: p.shift[0])
        for i in range(1, len(placed)):
            a, b = placed[i - 1], placed[i]
            sba = a.spacing_bbox or a.bbox
            sbb = b.spacing_bbox or b.bbox
            ax1 = sba[0] + a.shift[0]
            bx1 = sbb[0] + b.shift[0]
            overlap = ax1 + pad - bx1
            if overlap > 0:
                b.shift = (b.shift[0] + overlap, b.shift[1])
                moved = True
        if not moved:
            break
    right = 0.0
    for p in list(layout.mols) + list(layout.texts):
        sbb = p.spacing_bbox or p.bbox
        right = max(right, sbb[2] + p.shift[0])
    return right


def layout_rows(items: list, *, row_gap: float = 1.2, **kwargs):
    """多行布局：items 中的 ("newline",) 分隔各行（R-6 上下排列）。

    每行用 layout_row 排布（kwargs 透传），再按行内最高组件的间距包围盒
    高度（含标签向下外延）+ row_gap 逐行向下堆叠（y 向下为负方向）。

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
        # pass 2：相邻组件间距包围盒重叠时右推，总宽随之修正
        _resolve_row_overlaps(layout)
        rows.append(layout)
        y_offsets.append(y)
        height = max(
            ((p.spacing_bbox or p.bbox)[3] - (p.spacing_bbox or p.bbox)[1]
             for p in list(layout.mols) + list(layout.texts)),
            default=1.0,
        )
        y += height + row_gap
    return rows, y_offsets


def molecule_scope_lines(mol, shift: Tuple[float, float], *,
                         show_numbers: bool = False,
                         show_lone_pairs: bool = True,
                         explicit_hs: dict | None = None,
                         aromatic_rings: list | None = None,
                         occupancy=None,
                         bond_margin_scale: float = 1.0) -> List[str]:
    r"""分子组件的 scope 绘制行（内部全部局部坐标，位置由 shift 决定）。

    普通方程式（show_lone_pairs=False）不画孤对电子点；
    机理场景（show_lone_pairs=True）画出电子点；show_numbers 显示原子序号。
    explicit_hs：{原子序号: 已显式画出 H 数}，标签 H 计数自动扣减
    （[XH] 显式氢 / 氢键给体，保证"标签 H + 画出 H"总数正确）。
    标签风格统一规则（heavy_atom_count）：重原子数 ≤ 2 的小分子
    （CH₃Cl/CH₂=CH₂/CH₄…）用结构简式（非环碳写 CHn，教科书式 H₃C—Cl）；
    其余用键线式（碳原子不标 CHn，仅杂原子带 H 标签）。
    bond_margin_scale: 键线留白缩放系数——容器内分子坐标按 _MOL_SCALE
    缩放后，标签字号不变、键线修剪量（label_bond_margin）若仍按原标签
    宽度计算，键线两端会深入标签背景被 fill=white 盖住（如 CH₃Cl 只剩
    "—Cl"）；调用方应传分子坐标缩放系数使修剪量同步缩放（20260821 修复）。
    aromatic_rings: aromatic_ring_info() 的输出——全芳香单环跳过环内键、
        在质心画圆（芳香小写 c1ccccc1 风格）；为 None 时不画圈（凯库勒交替键）。
    occupancy: 可选的 collide.Occupancy（局部坐标）——键/环/标签/电子点
        逐笔登记，电荷圈据此选零冲突候选位；传入时供调用方后续注解
        （XH 显式 H 等）继续避让（R-8）。
    """
    hs = explicit_hs or {}
    # 统一标签规则：≤2 重原子小分子结构简式，其余键线式
    if heavy_atom_count(mol) <= 2:
        labeler = (lambda a, flip=False:
                   atom_main_label(a, hs.get(a.GetIdx(), 0), flip))
    else:
        labeler = (lambda a, flip=False:
                   atom_label(a, hs.get(a.GetIdx(), 0), flip))
    # 键线留白随分子缩放补偿（见 bond_margin_scale 说明）
    margin_fn = (lambda lab: label_bond_margin(lab) * bond_margin_scale)
    # 芳香画圈：调用方显式传入（structure.py）或从 mol property 自动读取
    # （prepare_mol 已存 _aromatic_lowercase，ARROW/REACTION/COMPOSITE 共用）
    if aromatic_rings is None:
        try:
            if mol.HasProp("_aromatic_lowercase") \
                    and mol.GetProp("_aromatic_lowercase") == "1":
                aromatic_rings = aromatic_ring_info(mol)
        except Exception:
            aromatic_rings = None
    lines = [
        f"  \\begin{{scope}}[shift={{({shift[0]:.2f},{shift[1]:.2f})}}]"
    ]
    # 占据注册表（R-8，局部坐标）：键线段/芳香环圆/标签矩形逐笔登记，
    # 电荷圈等注解元素据此选零冲突候选位（不撞键/标签/彼此）
    occ = occupancy if occupancy is not None else Occupancy()
    skip_rings = [aromatic_rings[k][0] for k in range(len(aromatic_rings or []))]
    for segs in bond_segments(mol, labeler=labeler, margin_fn=margin_fn,
                              skip_aromatic_rings=skip_rings):
        for x1, y1, x2, y2 in segs:
            lines.append(f"    \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")
            occ.add_segment(x1, y1, x2, y2)
    # 芳香环画圈：在质心画圆（半径取环原子平均距离），替代环内键
    for _, cx, cy, radius in (aromatic_rings or []):
        lines.append(
            f"    \\draw ({cx:.2f},{cy:.2f}) circle ({radius:.2f});"
        )
        occ.add_circle(cx, cy, radius)
    for atom in mol.GetAtoms():
        x, y = atom_pos(mol, atom.GetIdx())
        idx = atom.GetIdx()
        # 键端在标签右侧时翻转标签（OH→HO），使键连的元素符号靠近键端
        lab = labeler(atom, flip=_label_flip_for(mol, idx))
        if lab:
            lines.append(
                f"    \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};"
            )
            # 标签占据按字形估算（label_visual_width 的 0.6 折减——该宽度
            # 是为组件间距设计的保守上限，字形实际约占六成）
            hw = max(0.11, label_visual_width(lab) * 0.3)
            occ.add_rect(x - hw, y - 0.12, x + hw, y + 0.12)
        charge = charge_tikz(mol, idx, explicit_hs=hs.get(idx, 0),
                             occupancy=occ)
        if charge:
            lines.append(f"    {charge}")
        if show_numbers:
            lines.append(
                f"    \\node[font=\\tiny, gray, below right] at ({x:.2f},{y:.2f}) "
                f"{{{idx}}};"
            )
    if show_lone_pairs:
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            # 与 lone_pair_tikz 同格式输出，同时把电子点登记进占据表
            groups, singles = lone_pair_dot_groups(
                mol, idx, explicit_hs=hs.get(idx, 0))
            for (x1, y1), (x2, y2) in groups:
                for dx, dy in ((x1, y1), (x2, y2)):
                    lines.append(f"    \\fill ({dx:.2f},{dy:.2f}) circle (0.028);")
                    occ.add_circle(dx, dy, DOT_R)
            for dx, dy in singles:
                lines.append(f"    \\fill ({dx:.2f},{dy:.2f}) circle (0.028);")
                occ.add_circle(dx, dy, DOT_R)
    else:
        # 自由基单电子不受孤对电子开关影响：不画点会被误读为离子
        # （如 ·CH3 变成 CH3±）——show_lone_pairs=False 只抑制孤对电子对
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            _, singles = lone_pair_dot_groups(
                mol, idx, explicit_hs=hs.get(idx, 0))
            for dx, dy in singles:
                lines.append(f"    \\fill ({dx:.2f},{dy:.2f}) circle (0.028);")
                occ.add_circle(dx, dy, DOT_R)
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
