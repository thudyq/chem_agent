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
        format_chem_text, format_partial_charge, hbond_line_tikz,
        mech_arrow_between, mech_arrow_origin, mol_visual_bbox,
        parse_charge_pairs, parse_hbond_pairs, atom_pos, prepare_mol,
        scale_mol_coords,
    )
    from renderers.layout import (
        energy_annotation_placement, energy_point_coords, layout_row,
        layout_rows, molecule_scope_lines, place_bbox,
    )
else:
    from .mol_primitives import (
        format_chem_text, format_partial_charge, hbond_line_tikz,
        mech_arrow_between, mech_arrow_origin, mol_visual_bbox,
        parse_charge_pairs, parse_hbond_pairs, atom_pos, prepare_mol,
        scale_mol_coords,
    )
    from .layout import (
        energy_annotation_placement, energy_point_coords, layout_row,
        layout_rows, molecule_scope_lines, place_bbox,
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
_STRUCT_ID_RE = re.compile(r",id=([A-Za-z0-9_]+)")
_STRUCT_AT_RE = re.compile(r",at=(\d+)")
_STRUCT_POS_RE = re.compile(r",pos=(above|below)")

_SUPPORTED_LAYOUTS = ("reaction_mech", "row", "energy", "resonance")


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


def _collect_components(children):
    structs = []
    sequence = []
    mech_specs = []
    global_cond = ""
    annotations = {}
    for child in children:
        if child.type == "STRUCT":
            m = _STRUCT_ID_RE.search(child.raw)
            cid = m.group(1) if m else f"r{len(structs)}"
            label = child.args[1] if len(child.args) > 1 else None
            at_m = _STRUCT_AT_RE.search(child.raw)
            pos_m = _STRUCT_POS_RE.search(child.raw)
            structs.append({
                "id": cid,
                "smiles": child.args[0].strip(),
                "label": label,
                "at": int(at_m.group(1)) if at_m else None,
                "pos": pos_m.group(1) if pos_m else "above",
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
    max_idx = info["max_idx"]
    x_last = info["x_last"]
    role_map = {0: "反应物", n - 1: "产物"}
    if max_idx not in role_map:
        role_map[max_idx] = "过渡态"

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
    mol_placements = []
    occupied = []
    for comp in structs:
        _, _, x, y = info["points"][comp["at"]]
        mol = mols[comp["id"]]["mol"]
        bbox = mol_visual_bbox(mol, include_lone_pairs=False)
        shift = place_bbox(bbox, x, y, comp["pos"], margin=0.6)
        mol_placements.append((mol, shift))
        occupied.append((bbox[0] + shift[0], bbox[1] + shift[1],
                         bbox[2] + shift[0], bbox[3] + shift[1]))
    for i, v, x, y in info["points"]:
        yoff = 0.35 if i == max_idx else -0.3
        occupied.append((x - 0.85, y + yoff - 0.22, x + 0.85, y + yoff + 0.22))

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
        label = comp["label"] if (comp and comp["label"]) else role_map.get(i)
        lines.append(f"  \\begin{{scope}}[shift={{({x:.1f},{y:.2f})}}]")
        lines.append("    \\fill[blue] (0,0) circle (0.06);")
        if label:
            yoff = 0.35 if i == max_idx else -0.3
            lines.append(f"    \\node[font=\\small] at (0,{yoff:.2f}) {{{label} ({v:+.0f})}};")
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
        mols[comp["id"]] = {
            "mol": mol,
            "label": comp["label"],
            "shift": (0.0, 0.0),
            "charges": parse_charge_pairs(anno.get("charge", "")),
            "hbonds": parse_hbond_pairs(anno.get("hbond", "")),
        }

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

    # 每个分子一个 scope（布局引擎积木），组件级标注（电荷/氢键）随分子移动
    for comp in structs:
        info = mols[comp["id"]]
        mol = info["mol"]
        lines.extend(molecule_scope_lines(mol, info["shift"],
                                          show_numbers=show_numbers))
        for idx, raw_label in info["charges"].items():
            if idx >= mol.GetNumAtoms():
                continue
            x, y = atom_pos(mol, idx)
            x += info["shift"][0]
            y += info["shift"][1]
            lines.append(
                f"  \\node[font=\\small, red] at ({x + 0.30:.2f},{y + 0.25:.2f}) "
                f"{{{format_partial_charge(raw_label)}}};"
            )
        for fi, ti in info["hbonds"]:
            if fi >= mol.GetNumAtoms() or ti >= mol.GetNumAtoms():
                continue
            fx, fy = atom_pos(mol, fi)
            tx, ty = atom_pos(mol, ti)
            lines.append(
                "  " + hbond_line_tikz(fx + info["shift"][0], fy + info["shift"][1],
                                       tx + info["shift"][0], ty + info["shift"][1])
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

    for src_id, src_pt, dst_id, dst_pt, kind in _parse_mech_arrows(mech_specs):
        if src_id not in mols or dst_id not in mols:
            continue
        sm = mols[src_id]
        dm = mols[dst_id]
        p1 = mech_arrow_origin(dm["mol"], dst_pt, dm["shift"],
                               lone_pair_offset=False)
        if p1 is None:
            continue
        p0 = mech_arrow_origin(sm["mol"], src_pt, sm["shift"],
                               toward=(p1[0], p1[1]),
                               prefer_single=(kind == "fishhook"))
        if p0 is None:
            continue
        inset_start = 0.0 if p0[3] else (0.05 if p0[2] else 0.15)
        lines.extend(
            mech_arrow_between(p0[0], p0[1], p1[0], p1[1], kind,
                               from_bond=p0[2], inset_start=inset_start)
        )

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
            "[MECHARROW:br:0>>r0:0,r0:0>>r0:1]"
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
