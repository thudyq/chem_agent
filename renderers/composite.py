# -*- coding: utf-8 -*-
r"""renderers/composite.py — [COMPOSITE] 容器式复合标记渲染器。

LLM 在容器内显式列出结构组件、连接符与机理箭头，渲染器只负责把这些组件
放进统一 TikZ 坐标系组合绘制，实现“LLM 组装组件、渲染器组合绘制”的架构。

容器语法：
    [COMPOSITE:reaction_mech]
    [STRUCT:CCl,label=CH3Cl]          （id 省略，自动编号 r0）
    [PLUS]
    [STRUCT:[OH-],id=nu,label=OH-]    （显式 id）
    [RXNARROW]                         （主反应箭头，兼作反应物/产物分界）
    [STRUCT:CO,label=CH3OH]
    [PLUS]
    [STRUCT:[Cl-],label=Cl-]
    [MECHARROW:nu:0>r0:0]              （孤对电子进攻箭头，双电子；>> 为鱼钩）
    [MECHARROW:r0:0-1>r0:1]            （σ 键断裂箭头：从 C(0)-Cl(1) 键中点指向 Cl(1)）
    [CONDITION:SN2]                    （主箭头上方的条件文本）
    [/COMPOSITE]

布局种类：
    reaction_mech: 反应式 + 机理场景，必须包含至少一个 [RXNARROW]；
    row: 纯横向组件排列（共振式、多步序列等），[RXNARROW] 可选。
    energy: 势能面 + 驻点结构（R-3）：容器内需一个 [ENERGY:点序列]，
        每个 STRUCT 用 at=点序号 挂到驻点上（pos=above/below 可选，默认 above）。
    resonance: 共振式组合（R-6）：连续 STRUCT 之间自动插入共振箭头 ↔，
        无需手写连接符：[COMPOSITE:resonance][STRUCT:式1][STRUCT:式2][/COMPOSITE]。
    通用连接符（row / resonance 均可显式使用）：
        [RESARROW] 共振箭头 ↔；[NEWLINE] 换行（组件在多行中上下排列）。
    头部可追加标志：[COMPOSITE:reaction_mech,numbering] 打开原子序号标注
    （默认不显示；仅在碳原子较多、需要指明参与反应的原子时使用）。

连接符规则：
    - 相邻 [STRUCT] 之间默认只留间距；需要“+”必须显式写 [PLUS]；
    - [RXNARROW:条件] 可内联条件文本；[CONDITION:x] 填充第一个无内联条件的
      主箭头；
    - 多个 [RXNARROW] 可形成 A → B → C 多步序列。

机理箭头引用：组件 id（显式 id= 或自动 r0/r1/...）+ 端点引用。
端点可以是原子序号（SMILES 顺序，0 起），也可以是 "a-b" 形式的键中点
（σ 键断裂箭头从键发出，如 r0:0-1>r0:1）。杂原子起点自动上移到孤对电子
区域；键中点出发的箭头向下弯，其余向上弯。引用未知 id 或越界原子的箭头
会被跳过，不影响整体渲染。

组件级标注（R-2，随分子 scope 一起移动）：
    [CHARGE:ref|idx:δ+,idx:δ-,...]   组件 ref 上的部分电荷（红色）
    [HBOND:ref|from-to,...]          组件 ref 内的氢键虚线（teal dashed）

绘制风格：分子按结构简式绘制（非环碳原子写出 CH₃/CH₂/CH，环上碳保持
键线式）。label 中的纯化学式（如 CH3Cl）不会重复显示——分子本身已是简式；
中文名称/角色标注（如 底物、亲核试剂）仍显示在分子下方。
label 与条件写普通文本即可（CH3Cl、OH-、H2SO4），渲染器自动把数字转为
下标、尾部电荷转为上标。
"""

import math
import re

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from renderers.mol_primitives import (
        _ARROW_LABEL_GAP, _ARROW_POINT_GAP, _LABEL_TEXT_HALF_H, bond_order_of, format_chem_text,
        format_partial_charge,
        hbond_dots_tikz, mech_arrow_between, mech_arrow_origin,
        mol_visual_bbox, parse_charge_pairs, parse_hbond_pairs, atom_label,
        atom_main_label, bond_segments_for, label_bond_margin,
        label_edge_point, prepare_mol, scale_mol_coords, symbol_center,
        atom_pos, place_donor_h, place_explicit_hs, adjust_hbond_conformation,
    )
    from renderers.layout import (
        energy_annotation_placement, energy_point_coords, energy_point_roles,
        layout_row, layout_rows, molecule_scope_lines, place_bbox,
    )
else:
    from .mol_primitives import (
        _ARROW_LABEL_GAP, _ARROW_POINT_GAP, _LABEL_TEXT_HALF_H, bond_order_of, format_chem_text,
        format_partial_charge,
        hbond_dots_tikz, mech_arrow_between, mech_arrow_origin,
        mol_visual_bbox, parse_charge_pairs, parse_hbond_pairs, atom_label,
        atom_main_label, bond_segments_for, label_bond_margin,
        label_edge_point, prepare_mol, scale_mol_coords, symbol_center,
        atom_pos, place_donor_h, place_explicit_hs, adjust_hbond_conformation,
    )
    from .layout import (
        energy_annotation_placement, energy_point_coords, energy_point_roles,
        layout_row, layout_rows, molecule_scope_lines, place_bbox,
    )


_MOL_GAP = 1.6    # 无连接符时相邻分子的水平间距
_PLUS_W = 1.1     # [PLUS] 连接符占宽
_ARR_W = 2.6      # [RXNARROW] 占宽
_ARR_PAD = 0.65   # 主箭头两端内缩余量（箭头实际长度 1.3）
_MOL_SCALE = 0.8  # 分子坐标缩放因子（紧凑化，不影响字号）

_MECH_ARROW_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*:\s*(\d+(?:-\d+)?)\s*(>>|>)\s*"
    r"([A-Za-z0-9_]+)\s*:\s*(\d+(?:-\d+)?)\s*$"
)

_SUPPORTED_LAYOUTS = ("reaction_mech", "row", "energy", "resonance")


def _rects_intersect(a: tuple, b: tuple, pad: float = 0.15) -> bool:
    """两个 (min_x, min_y, max_x, max_y) 矩形是否相交（含 pad 间距）。"""
    return not (a[2] + pad < b[0] or b[2] + pad < a[0]
                or a[3] + pad < b[1] or b[3] + pad < a[1])


def _parse_mech_arrows(specs):
    arrows = []
    for spec in specs:
        m = _MECH_ARROW_RE.match(spec)
        if not m:
            continue
        src_id, src_pt, sep, dst_id, dst_pt = m.groups()
        kind = "fishhook" if sep == ">>" else "standard"
        arrows.append((src_id, src_pt, dst_id, dst_pt, kind))
    return arrows


def _mech_labeler(info):
    """组件级标签器（与分子绘制同款）：带 XH/BOND/HBOND 标注的分子按键线式
    （碳原子不标 CHn），其余保持结构简式。供机理箭头端点吸附到标签边缘用。"""
    hs = info["explicit_hs"]
    if info["xh"] or info["bonds"] or info["hbonds"]:
        return lambda a: atom_label(a, hs.get(a.GetIdx(), 0))
    return lambda a: atom_main_label(a, hs.get(a.GetIdx(), 0))


def draw_mech_arrows(mols: dict, arrows: list) -> list:
    """绘制机理弯箭头（p0/p1 定位、端点吸附避让），返回 TikZ 行列表。

    mols: {组件 id: {"mol": RDKit Mol, "shift": (x, y), ...}} 组件表。
    arrows: [(src_id, src_pt, dst_id, dst_pt, kind), ...]——_parse_mech_arrows
        输出；src_pt/dst_pt 为原子序号或 "a-b" 键中点。
    未知组件 id / 无效端点的箭头跳过，不影响整体渲染。

    逻辑：目标端先按原子中心定位（供源端选孤对槽位）；源端确定后，
    再按实际源端把目标端吸附到标签边缘空隙（箭头尖不压标签）。
    """
    lines = []
    for src_id, src_pt, dst_id, dst_pt, kind in arrows:
        if src_id not in mols or dst_id not in mols:
            continue
        sm = mols[src_id]
        dm = mols[dst_id]
        slab = _mech_labeler(sm)
        dlab = _mech_labeler(dm)
        p1 = mech_arrow_origin(dm["mol"], dst_pt, dm["shift"],
                               lone_pair_offset=False)
        if p1 is None:
            continue
        p0 = mech_arrow_origin(sm["mol"], src_pt, sm["shift"],
                               toward=(p1[0], p1[1]),
                               prefer_single=(kind == "fishhook"),
                               labeler=slab)
        if p0 is None:
            continue
        p1 = mech_arrow_origin(dm["mol"], dst_pt, dm["shift"],
                               lone_pair_offset=False,
                               toward=(p0[0], p0[1]), labeler=dlab,
                               bend_side=-1.0 if p0[2] else 1.0)
        if p1 is None:
            continue
        bond_break = ("-" in src_pt
                      and bond_order_of(sm["mol"], src_pt) == 1)
        inset_start = (_ARROW_POINT_GAP if bond_break
                       else (0.0 if (p0[2] or p0[3] or p0[4]) else 0.15))
        aim_end = ("-" not in dst_pt and p1[4]
                   and dm["mol"].GetAtomWithIdx(int(dst_pt)).GetAtomicNum() == 6)
        tb = None
        if aim_end:
            da = dm["mol"].GetAtomWithIdx(int(dst_pt))
            ax, ay = symbol_center(dm["mol"], int(dst_pt))
            tb = (ax + dm["shift"][0], ay + dm["shift"][1],
                  label_bond_margin(dlab(da)), _LABEL_TEXT_HALF_H)
        inset_end = (_ARROW_LABEL_GAP if aim_end
                     else (0.0 if p1[4] else 0.10))
        lines.extend(
            mech_arrow_between(p0[0], p0[1], p1[0], p1[1], kind,
                               from_bond=p0[2], inset_start=inset_start,
                               inset_end=inset_end, bond_break=bond_break,
                               aim_end=aim_end, text_box=tb)
        )
    return lines


def _collect_components(children):
    structs = []
    sequence = []
    mech_specs = []
    global_cond = ""
    annotations = {}
    for child in children:
        if child.type == "STRUCT":
            cid = child.attrs.get("id") or f"r{len(structs)}"
            label = child.args[1] if len(child.args) > 1 else None
            structs.append({
                "id": cid,
                "smiles": child.args[0].strip(),
                "label": label,
                "at": child.attrs.get("at"),
                "pos": child.attrs.get("pos", "above"),
            })
            sequence.append(("mol", len(structs) - 1))
        elif child.type == "PLUS":
            sequence.append(("plus",))
        elif child.type == "RESARROW":
            sequence.append(("resarrow",))
        elif child.type == "NEWLINE":
            sequence.append(("newline",))
        elif child.type == "RXNARROW":
            cond = child.args[0].strip() if child.args else ""
            sequence.append(("arrow", cond))
        elif child.type == "CONDITION":
            if child.args and not global_cond:
                global_cond = child.args[0].strip()
        elif child.type == "MECHARROW":
            if child.args:
                mech_specs.extend(child.args[0].split(","))
        elif child.type in ("CHARGE", "HBOND") and len(child.args) >= 2:
            ref = child.args[0].strip()
            annotations.setdefault(ref, {})[child.type.lower()] = child.args[1]
        elif child.type == "XH" and len(child.args) >= 2:
            ref = child.args[0].strip()
            try:
                a = int(child.args[1])
                annotations.setdefault(ref, {}).setdefault("xh", []).append(a)
            except ValueError:
                pass
        elif child.type == "BOND" and len(child.args) >= 2:
            ref = child.args[0].strip()
            annotations.setdefault(ref, {}).setdefault("bonds", []).append(
                child.args[1].strip())
    return structs, sequence, mech_specs, global_cond, annotations


def _render_energy_layout(points_str: str, structs: list, mols: dict,
                          show_numbers: bool) -> str:
    """energy 布局：势能面曲线 + 驻点结构组件（R-3）。

    每个 STRUCT 通过 at= 挂到能量点上（pos=above/below，默认 above），
    分子按视觉包围盒置于驻点正上方/下方；驻点标签优先用 STRUCT 的 label。
    """
    try:
        values = [float(v.strip()) for v in points_str.split(",") if v.strip()]
    except ValueError:
        return f"（COMPOSITE 渲染失败：能量点序列格式错误「{points_str}」）"
    if len(values) < 2:
        return "（COMPOSITE 渲染失败：能量点至少需要 2 个）"

    n = len(values)
    info = energy_point_coords(values)
    x_last = info["x_last"]
    roles = energy_point_roles(values)

    at_map = {}
    for comp in structs:
        if comp["at"] is None:
            return (f"（COMPOSITE 渲染失败：energy 布局中 STRUCT 组件 "
                    f"{comp['id']} 需要 at=点序号）")
        if comp["at"] >= n:
            return (f"（COMPOSITE 渲染失败：组件 {comp['id']} 的 "
                    f"at={comp['at']} 超出能量点范围 0~{n - 1}）")
        at_map[comp["at"]] = comp

    # 先计算全部组件的已占区域，再决定标注框与纵轴高度（避免遮挡）
    # P3 冲突消解：先预置驻点标签区域；结构按驻点序号放置（相邻先检测），
    # 与已占区域重叠时向右错开（最多 8.0），避免相邻驻点结构互相压叠。
    occupied = []
    for i, v, x, y in info["points"]:
        label = (at_map[i]["label"]
                 if (i in at_map and at_map[i]["label"]) else roles.get(i))
        if not label:
            continue
        yoff = 0.35 if roles.get(i) == "过渡态" else -0.3
        occupied.append((x - 0.85, y + yoff - 0.22, x + 0.85, y + yoff + 0.22))

    mol_placements = []
    for comp in sorted(structs, key=lambda c: c["at"]):
        _, _, x, y = info["points"][comp["at"]]
        mol = mols[comp["id"]]["mol"]
        bbox = mol_visual_bbox(mol, include_lone_pairs=False)
        shift = place_bbox(bbox, x, y, comp["pos"], margin=0.6)
        for _ in range(20):  # 最多右移 20×0.4 = 8.0
            rect = (bbox[0] + shift[0], bbox[1] + shift[1],
                    bbox[2] + shift[0], bbox[3] + shift[1])
            if not any(_rects_intersect(rect, o) for o in occupied):
                break
            shift = (shift[0] + 0.4, shift[1])
        mol_placements.append((mol, shift))
        occupied.append(rect)

    box_x, box_y, box_anchor, axis_top = energy_annotation_placement(
        occupied, x_last)

    lines = [r"\begin{tikzpicture}"]
    lines.append(f"  \\draw[->] (0,0) -- ({x_last + 0.8:.1f},0);")
    lines.append(f"  \\draw[->] (0,0) -- (0,{axis_top:.1f});")
    lines.append(f"  \\node[font=\\small] at ({(x_last + 0.8) / 2:.1f},-0.30) {{反应进程}};")
    lines.append(
        f"  \\node[font=\\small, rotate=90, anchor=south] at (-0.10,{axis_top - 0.5:.1f}) "
        "{能量 (kJ/mol)};"
    )
    y0 = info["points"][0][3]
    lines.append(f"  \\draw[gray, dashed] (0,{y0:.2f}) -- ({x_last:.1f},{y0:.2f});")
    coords = " ".join(f"({x:.1f},{y:.2f})" for _, _, x, y in info["points"])
    lines.append(f"  \\draw[thick, blue, smooth] plot coordinates {{{coords}}};")

    for i, v, x, y in info["points"]:
        comp = at_map.get(i)
        label = comp["label"] if (comp and comp["label"]) else roles.get(i)
        lines.append(f"  \\begin{{scope}}[shift={{({x:.1f},{y:.2f})}}]")
        lines.append("    \\fill[blue] (0,0) circle (0.06);")
        if label:
            yoff = 0.35 if roles.get(i) == "过渡态" else -0.3
            lines.append(f"    \\node[font=\\small] at (0,{yoff:.2f}) "
                         f"{{{format_chem_text(label)} ({v:+.0f})}};")
        lines.append("  \\end{scope}")

    for mol, shift in mol_placements:
        lines.extend(molecule_scope_lines(mol, shift, show_numbers=show_numbers,
                                          show_lone_pairs=False))

    ea = max(values) - values[0]
    dh = values[-1] - values[0]
    node_text = f"Ea $\\approx$ {ea:.0f} kJ/mol\\\\$\\Delta$H $\\approx$ {dh:+.0f} kJ/mol"
    lines.append(
        "    \\node[draw, rounded corners, fill=yellow!10, font=\\small, align=left, "
        f"anchor={box_anchor}] at ({box_x:.2f},{box_y:.2f}) {{{node_text}}};"
    )
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def render_composite(layout: str, children: list) -> str:
    r"""[COMPOSITE] 渲染：容器内组件 → 统一坐标系单张 TikZ。

    参数:
        layout: 布局名（reaction_mech / row），可追加逗号分隔的标志
                （如 "reaction_mech,numbering" 打开原子序号标注）。
        children: 容器内子标记 RenderTag 列表（core.tag_parser 解析结果）。

    返回:
        可编译的 TikZ 代码；失败返回可读错误提示。
    """
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（COMPOSITE 渲染失败：rdkit 未安装）"

    header = [p.strip() for p in (layout or "").split(",")]
    layout_name = header[0]
    flags = set(header[1:])
    if layout_name not in _SUPPORTED_LAYOUTS:
        return (
            f"（COMPOSITE 渲染失败：未知布局「{layout_name}」，"
            f"支持 {' / '.join(_SUPPORTED_LAYOUTS)}）"
        )
    show_numbers = "numbering" in flags

    structs, sequence, mech_specs, global_cond, annotations = _collect_components(children)

    if not structs:
        return "（COMPOSITE 渲染失败：容器内缺少 [STRUCT] 组件）"
    if layout_name == "reaction_mech" and not any(el[0] == "arrow" for el in sequence):
        return "（COMPOSITE 渲染失败：reaction_mech 布局需要 [RXNARROW] 标记主反应箭头位置）"

    # resonance 布局/含共振箭头时，极限式必须保留显式键级（跳过芳香化），
    # 否则不同 Kekulé 式会被统一芳香化成同一结构
    allow_aromatic = not (
        layout_name == "resonance"
        or any(el[0] == "resarrow" for el in sequence)
    )

    mols = {}
    for comp in structs:
        mol = prepare_mol(comp["smiles"], allow_aromatic=allow_aromatic)
        if mol is None:
            return f"（COMPOSITE 渲染失败：无效 SMILES「{comp['smiles']}」（组件 {comp['id']}）"
        scale_mol_coords(mol, _MOL_SCALE)
        anno = annotations.get(comp["id"], {})
        # 显式 H 对账：XH 子标记 + 氢键给体/受体，标签 H 计数自动扣减
        explicit_hs = {}
        for a in anno.get("xh", []):
            explicit_hs[a] = explicit_hs.get(a, 0) + 1
        for fi, _ in parse_hbond_pairs(anno.get("hbond", "")):
            explicit_hs[fi] = explicit_hs.get(fi, 0) + 1
        for _, ti in parse_hbond_pairs(anno.get("hbond", "")):
            explicit_hs[ti] = explicit_hs.get(ti, 0) + 1
        mols[comp["id"]] = {
            "mol": mol,
            "label": comp["label"],
            "shift": (0.0, 0.0),
            "charges": parse_charge_pairs(anno.get("charge", "")),
            "hbonds": parse_hbond_pairs(anno.get("hbond", "")),
            "xh": anno.get("xh", []),
            "bonds": anno.get("bonds", []),
            "explicit_hs": explicit_hs,
        }

    # 氢键场景构象调整（布局前）：给体与受体折到主链同一侧
    for comp in structs:
        mol = mols[comp["id"]]["mol"]
        for fi, ti in mols[comp["id"]]["hbonds"]:
            if fi < mol.GetNumAtoms() and ti < mol.GetNumAtoms():
                adjust_hbond_conformation(mol, fi, ti)

    if layout_name == "energy":
        energy_child = next((c for c in children if c.type == "ENERGY"), None)
        if energy_child is None or not energy_child.args:
            return "（COMPOSITE 渲染失败：energy 布局需要 [ENERGY:点序列] 组件）"
        return _render_energy_layout(energy_child.args[0], structs, mols,
                                     show_numbers)

    # resonance 布局：连续 mol 之间自动插入共振箭头 ↔
    if layout_name == "resonance":
        new_seq = []
        for el in sequence:
            if el[0] == "mol" and new_seq and new_seq[-1][0] == "mol":
                new_seq.append(("resarrow",))
            new_seq.append(el)
        sequence = new_seq

    # 统一布局引擎：组件序列 → 位置/加号/共振箭头/反应箭头（视觉包围盒防重叠）
    items = []
    for el in sequence:
        if el[0] == "mol":
            cid = structs[el[1]]["id"]
            items.append(("mol", cid, mols[cid]["mol"]))
        elif el[0] in ("plus", "resarrow", "newline"):
            items.append((el[0],))
        elif el[0] == "arrow":
            items.append(("arrow", el[1]))
    rows, y_offsets = layout_rows(items, mol_gap=_MOL_GAP, plus_w=_PLUS_W,
                                  arrow_w=_ARR_W, arrow_pad=_ARR_PAD)
    plus_positions = []      # (x, yoff)
    res_positions = []       # (x, yoff)
    main_arrows = []         # [x1, x2, cond, yoff]
    for row_layout, yoff in zip(rows, y_offsets):
        for placed in row_layout.mols:
            mols[placed.key]["shift"] = (placed.shift[0], placed.shift[1] - yoff)
            mols[placed.key]["bbox"] = placed.bbox
        plus_positions.extend((px, yoff) for px in row_layout.pluses)
        res_positions.extend((rx, yoff) for rx in row_layout.resarrows)
        main_arrows.extend([a.x1, a.x2, a.condition, yoff]
                           for a in row_layout.arrows)

    if global_cond:
        for arr in main_arrows:
            if not arr[2]:
                arr[2] = global_cond
                break

    lines = [r"\begin{tikzpicture}"]

    # 键线式默认不标孤对电子（规范第 3 条）；仅机理场景（弯箭头起点）、
    # 共振场景（孤对电子参与共轭）自动画出
    show_lone_pairs = (
        bool(mech_specs)
        or layout_name == "resonance"
        or any(el[0] == "resarrow" for el in sequence)
    )

    # 每个分子一个 scope（布局引擎积木），组件级标注（电荷/氢键）随分子移动
    for comp in structs:
        info = mols[comp["id"]]
        mol = info["mol"]
        hs = info["explicit_hs"]
        # 带 [XH]/[BOND]/[HBOND] 标注的分子按键线式绘制（碳原子不标 CHn，
        # 只在反应位点画出显式键），普通分子保持结构简式（原有逻辑不变）
        bond_line = bool(info["xh"] or info["bonds"] or info["hbonds"])
        labeler = _mech_labeler(info)
        lines.extend(molecule_scope_lines(mol, info["shift"],
                                          show_numbers=show_numbers,
                                          show_lone_pairs=show_lone_pairs,
                                          explicit_hs=hs,
                                          bond_line=bond_line))
        for idx, raw_label in info["charges"].items():
            if idx >= mol.GetNumAtoms():
                continue
            # 部分电荷以元素符号中心为基准（与孤对电子同一基准）
            x, y = symbol_center(mol, idx, hs.get(idx, 0))
            x += info["shift"][0]
            y += info["shift"][1]
            lines.append(
                f"  \\node[font=\\small, red] at ({x + 0.30:.2f},{y + 0.30:.2f}) "
                f"{{{format_partial_charge(raw_label)}}};"
            )
        for fi, ti in info["hbonds"]:
            if fi >= mol.GetNumAtoms() or ti >= mol.GetNumAtoms():
                continue
            # X—H 实线（从标签边缘起笔，不压标签）+ H···Y 点状虚线
            tx, ty = symbol_center(mol, ti, hs.get(ti, 0))
            hx, hy = place_donor_h(mol, fi, (tx, ty))
            sx, sy = label_edge_point(mol, fi, (hx, hy), labeler=labeler)
            tx += info["shift"][0]
            ty += info["shift"][1]
            hx += info["shift"][0]
            hy += info["shift"][1]
            sx += info["shift"][0]
            sy += info["shift"][1]
            lines.append(f"  \\draw ({sx:.2f},{sy:.2f}) -- ({hx:.2f},{hy:.2f});")
            lines.append(
                f"  \\node[fill=white, inner sep=1pt] at ({hx:.2f},{hy:.2f}) {{H}};"
            )
            for dot_line in hbond_dots_tikz(hx, hy, tx, ty):
                lines.append(f"  {dot_line}")
            # 受体显式 H：朝向远离给体方向（避开氢键点线），标签已扣减
            ax, ay = atom_pos(mol, ti)
            ahx, ahy = place_explicit_hs(mol, ti, 1,
                                         toward=(2 * ax - tx, 2 * ay - ty))[0]
            asx, asy = label_edge_point(mol, ti, (ahx, ahy), labeler=labeler)
            ahx += info["shift"][0]
            ahy += info["shift"][1]
            asx += info["shift"][0]
            asy += info["shift"][1]
            lines.append(f"  \\draw ({asx:.2f},{asy:.2f}) -- ({ahx:.2f},{ahy:.2f});")
            lines.append(
                f"  \\node[fill=white, inner sep=1pt] at ({ahx:.2f},{ahy:.2f}) {{H}};"
            )
        # [XH] 显式氢：标签已按 explicit_hs 扣减，此处画出 X—H 实线 + H 节点
        xh_counts = {}
        for a in info["xh"]:
            xh_counts[a] = xh_counts.get(a, 0) + 1
        for a, count in xh_counts.items():
            if a >= mol.GetNumAtoms():
                continue
            for hx, hy in place_explicit_hs(mol, a, count):
                sx, sy = label_edge_point(mol, a, (hx, hy), labeler=labeler)
                hx += info["shift"][0]
                hy += info["shift"][1]
                sx += info["shift"][0]
                sy += info["shift"][1]
                lines.append(f"  \\draw ({sx:.2f},{sy:.2f}) -- ({hx:.2f},{hy:.2f});")
                lines.append(
                    f"  \\node[fill=white, inner sep=1pt] at ({hx:.2f},{hy:.2f}) {{H}};"
                )
        # [BOND] 反应位点键突出：复用骨架修剪段，红色粗线与原键完全对齐
        for spec in info["bonds"]:
            if "-" not in spec:
                continue
            sa, _, sb = spec.partition("-")
            try:
                a, b = int(sa), int(sb)
            except ValueError:
                continue
            if a >= mol.GetNumAtoms() or b >= mol.GetNumAtoms():
                continue
            segs = bond_segments_for(mol, a, b, labeler=labeler,
                                     margin_fn=label_bond_margin)
            if segs is None:
                continue
            for x1, y1, x2, y2 in segs:
                lines.append(
                    f"  \\draw[very thick, red] "
                    f"({x1 + info['shift'][0]:.2f},{y1 + info['shift'][1]:.2f}) -- "
                    f"({x2 + info['shift'][0]:.2f},{y2 + info['shift'][1]:.2f});"
                )

    for comp in structs:
        info = mols[comp["id"]]
        label = info["label"]
        if label and not label.isascii():
            min_x, min_y, max_x, _ = info["bbox"]
            shift = info["shift"]
            cx = (min_x + max_x) / 2.0 + shift[0]
            ly = min_y + shift[1] - 0.35
            lines.append(
                f"  \\node[below] at ({cx:.2f},{ly:.2f}) "
                f"{{{format_chem_text(label)}}};"
            )

    for px, yoff in plus_positions:
        lines.append(f"  \\node at ({px:.2f},{-yoff:.2f}) {{$+$}};")

    for rx, yoff in res_positions:
        lines.append(f"  \\node[font=\\large] at ({rx:.2f},{-yoff:.2f}) {{$\\leftrightarrow$}};")

    for x1, x2, cond, yoff in main_arrows:
        cond_text = format_chem_text(cond)
        if cond_text:
            lines.append(
                f"  \\draw[->, very thick] ({x1:.2f},{-yoff:.2f}) -- ({x2:.2f},{-yoff:.2f}) "
                f"node[midway, above] {{{cond_text}}};"
            )
        else:
            lines.append(f"  \\draw[->, very thick] ({x1:.2f},{-yoff:.2f}) -- ({x2:.2f},{-yoff:.2f});")

    lines.extend(draw_mech_arrows(mols, _parse_mech_arrows(mech_specs)))

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from core.tag_parser import parse_tags

    demos = [
        (
            "SN2 机理（标准范本风格：孤对电子起点 + 键中点断键箭头）",
            "[COMPOSITE:reaction_mech]"
            "[STRUCT:CCl,label=CH3Cl][PLUS][STRUCT:[OH-],id=nu,label=OH-]"
            "[RXNARROW]"
            "[STRUCT:CO,label=CH3OH][PLUS][STRUCT:[Cl-],label=Cl-]"
            "[MECHARROW:nu:0>r0:0][MECHARROW:r0:0-1>r0:1]"
            "[CONDITION:SN2]"
            "[/COMPOSITE]",
        ),
        (
            "多步序列（row 布局，多个 RXNARROW，内联条件）",
            "[COMPOSITE:row]"
            "[STRUCT:C=C,label=乙烯][RXNARROW:H2O / H+]"
            "[STRUCT:CCO,label=乙醇][RXNARROW:CuO, Δ]"
            "[STRUCT:CC=O,label=乙醛]"
            "[/COMPOSITE]",
        ),
        (
            "鱼钩箭头（自由基加成）",
            "[COMPOSITE:reaction_mech]"
            "[STRUCT:C=C][PLUS][STRUCT:[Br],id=br]"
            "[RXNARROW:hv]"
            "[STRUCT:[CH2]CBr]"
            "[MECHARROW:br:0>>r0:1,r0:0-1>>r0:1]"
            "[/COMPOSITE]",
        ),
        (
            "错误：reaction_mech 缺少 RXNARROW",
            "[COMPOSITE:reaction_mech][STRUCT:CCl][/COMPOSITE]",
        ),
        (
            "错误：未知布局",
            "[COMPOSITE:grid][STRUCT:CCl][/COMPOSITE]",
        ),
        (
            "容错：机理箭头引用未知 id（应跳过，不崩溃）",
            "[COMPOSITE:reaction_mech]"
            "[STRUCT:CCl][RXNARROW][STRUCT:CO]"
            "[MECHARROW:r9:0>r0:0]"
            "[/COMPOSITE]",
        ),
        (
            "numbering 标志：打开原子序号标注",
            "[COMPOSITE:reaction_mech,numbering]"
            "[STRUCT:CCl,label=CH3Cl][PLUS][STRUCT:[OH-],label=OH-]"
            "[RXNARROW:SN2]"
            "[STRUCT:CO,label=CH3OH][PLUS][STRUCT:[Cl-],label=Cl-]"
            "[MECHARROW:r1:0>r0:0,r0:0-1>r0:1]"
            "[/COMPOSITE]",
        ),
        (
            "energy 布局：SN2 势能面 + 三个驻点结构",
            "[COMPOSITE:energy]"
            "[ENERGY:0,108,-20]"
            "[STRUCT:CCl.[OH-],label=反应物,at=0]"
            "[STRUCT:CCl.[OH-],label=过渡态,at=1]"
            "[STRUCT:CO.[Cl-],label=产物,at=2]"
            "[/COMPOSITE]",
        ),
        (
            "resonance 布局：苯的两个 Kekulé 式（自动 ↔）",
            "[COMPOSITE:resonance]"
            "[STRUCT:C1=CC=CC=C1,label=Kekulé 式 I]"
            "[STRUCT:C1C=CC=CC=1,label=Kekulé 式 II]"
            "[/COMPOSITE]",
        ),
        (
            "NEWLINE 多行：主结构在上、共振式在下",
            "[COMPOSITE:row]"
            "[STRUCT:CC(=O)[O-],label=羧酸根]"
            "[NEWLINE]"
            "[STRUCT:CC(=O)[O-]][RESARROW][STRUCT:CC([O-])=O]"
            "[/COMPOSITE]",
        ),
    ]

    for title, text in demos:
        print("=" * 60)
        print(title)
        print("=" * 60)
        tags = parse_tags(text)
        composite = next((t for t in tags if t.type == "COMPOSITE"), None)
        if composite is None:
            print("（未解析到 COMPOSITE 标记）")
            continue
        print(render_composite(composite.args[0], composite.args[1]))
        print()