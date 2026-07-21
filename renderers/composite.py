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
        bond_segments, condensed_atom_label, format_chem_text, label_plain_len,
        lone_pair_tikz, mech_arrow_between, mech_arrow_origin,
        atom_pos, prepare_mol, scale_mol_coords,
    )
    from renderers.layout import layout_row
else:
    from .mol_primitives import (
        bond_segments, condensed_atom_label, format_chem_text, label_plain_len,
        lone_pair_tikz, mech_arrow_between, mech_arrow_origin,
        atom_pos, prepare_mol, scale_mol_coords,
    )
    from .layout import layout_row


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

_SUPPORTED_LAYOUTS = ("reaction_mech", "row")


def _bond_margin(label: str) -> float:
    n = label_plain_len(label)
    if n <= 2:
        return 0.30
    if n == 3:
        return 0.45
    return 0.58


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
    for child in children:
        if child.type == "STRUCT":
            m = _STRUCT_ID_RE.search(child.raw)
            cid = m.group(1) if m else f"r{len(structs)}"
            label = child.args[1] if len(child.args) > 1 else None
            structs.append({"id": cid, "smiles": child.args[0].strip(), "label": label})
            sequence.append(("mol", len(structs) - 1))
        elif child.type == "PLUS":
            sequence.append(("plus",))
        elif child.type == "RXNARROW":
            cond = child.args[0].strip() if child.args else ""
            sequence.append(("arrow", cond))
        elif child.type == "CONDITION":
            if child.args and not global_cond:
                global_cond = child.args[0].strip()
        elif child.type == "MECHARROW":
            if child.args:
                mech_specs.extend(child.args[0].split(","))
    return structs, sequence, mech_specs, global_cond


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

    structs, sequence, mech_specs, global_cond = _collect_components(children)

    if not structs:
        return "（COMPOSITE 渲染失败：容器内缺少 [STRUCT] 组件）"
    if layout_name == "reaction_mech" and not any(el[0] == "arrow" for el in sequence):
        return "（COMPOSITE 渲染失败：reaction_mech 布局需要 [RXNARROW] 标记主反应箭头位置）"

    mols = {}
    for comp in structs:
        mol = prepare_mol(comp["smiles"])
        if mol is None:
            return f"（COMPOSITE 渲染失败：无效 SMILES「{comp['smiles']}」（组件 {comp['id']}）"
        scale_mol_coords(mol, _MOL_SCALE)
        mols[comp["id"]] = {
            "mol": mol,
            "label": comp["label"],
            "shift": (0.0, 0.0),
        }

    # 统一布局引擎：组件序列 → 位置/加号/箭头（视觉包围盒防重叠）
    items = []
    for el in sequence:
        if el[0] == "mol":
            cid = structs[el[1]]["id"]
            items.append(("mol", cid, mols[cid]["mol"]))
        elif el[0] == "plus":
            items.append(("plus",))
        elif el[0] == "arrow":
            items.append(("arrow", el[1]))
    layout = layout_row(items, mol_gap=_MOL_GAP, plus_w=_PLUS_W,
                        arrow_w=_ARR_W, arrow_pad=_ARR_PAD)
    for placed in layout.mols:
        mols[placed.key]["shift"] = placed.shift
        mols[placed.key]["bbox"] = placed.bbox
    plus_positions = layout.pluses
    main_arrows = [[a.x1, a.x2, a.condition] for a in layout.arrows]

    if global_cond:
        for arr in main_arrows:
            if not arr[2]:
                arr[2] = global_cond
                break

    lines = [r"\begin{tikzpicture}"]

    # 每个分子一个 scope：内部全部局部坐标，位置由 shift 决定，
    # 整体移动分子不破坏键/标签/电子点的内部比例
    for comp in structs:
        info = mols[comp["id"]]
        mol = info["mol"]
        shift = info["shift"]
        lines.append(
            f"  \\begin{{scope}}[shift={{({shift[0]:.2f},{shift[1]:.2f})}}]"
        )
        for segs in bond_segments(mol, labeler=condensed_atom_label,
                                  margin_fn=_bond_margin):
            for x1, y1, x2, y2 in segs:
                lines.append(
                    f"    \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});"
                )
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
        for atom in mol.GetAtoms():
            for dot_line in lone_pair_tikz(mol, atom.GetIdx()):
                lines.append(f"    {dot_line}")
        lines.append("  \\end{scope}")

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

    for px in plus_positions:
        lines.append(f"  \\node at ({px:.2f},0) {{$+$}};")

    for x1, x2, cond in main_arrows:
        cond_text = format_chem_text(cond)
        if cond_text:
            lines.append(
                f"  \\draw[->, very thick] ({x1:.2f},0) -- ({x2:.2f},0) "
                f"node[midway, above] {{{cond_text}}};"
            )
        else:
            lines.append(f"  \\draw[->, very thick] ({x1:.2f},0) -- ({x2:.2f},0);")

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
