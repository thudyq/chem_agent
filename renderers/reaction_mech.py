# -*- coding: utf-8 -*-
r"""renderers/reaction_mech.py — [REACTIONMECH] 机理图与反应式组合渲染器。

把多个反应物、多个产物横向排布成一条反应方程式，并在上方叠加电子推进
弯箭头，形成“反应式 + 机理”的复合图。所有分子绘制在同一个 TikZ 坐标系
内，保证机理箭头与分子骨架对齐。

标记格式：
    [REACTIONMECH:反应物1;反应物2;...|产物1;产物2;...|反应条件|机理箭头]
    [REACTIONMECH:反应物|产物|条件|机理箭头|numbering]   （显示原子序号）

机理箭头格式：
    src_mol:src_atom>dst_mol:dst_atom        （双头弯箭头，电子对转移）
    src_mol:src_atom>>dst_mol:dst_atom       （鱼钩箭头，单电子转移）
    端点也可以是键中点 "a-b"：src_mol:a-b>dst_mol:dst_atom
    （σ 键断裂箭头从键发出，如 0:0-1>0:1）。杂原子起点自动上移到孤对电子
    区域；键中点出发的箭头向下弯，其余向上弯。

分子编号：先反应物后产物，从 0 开始。例如 2 个反应物 + 2 个产物时，产物
Cl- 的编号为 2，CH3OH 的编号为 3。
默认不显示原子序号；仅当需要核对编号时在第五段写 numbering。

示例（SN2）：
    [REACTIONMECH:CCl;[OH-]|[Cl-];CO|SN2|1:0>0:0,0:1>2:0]
含义：
    - 反应物 0：CH3Cl；反应物 1：OH-
    - 产物 2：Cl-；产物 3：CH3OH
    - 箭头 1:0>0:0：OH- 的 O 原子进攻 CH3Cl 的 C 原子
    - 箭头 0:1>2:0：CH3Cl 的 Cl 原子上的 C-Cl 键电子流向产物 Cl-

设计选择：
- 使用 RDKit 2D 坐标绘制所有分子骨架，和 [MECH] 保持一致；
- 分子之间按水平方向排列，反应条件标于主箭头上方；
- 机理箭头为红色 Bezier 曲线，与 [MECH] 风格一致；
- 产物、试剂也参与编号，LLM 可以画出“试剂 → 底物”或“键电子 → 离去基团”的箭头。
"""

from typing import List, Tuple

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from renderers.mol_primitives import (
        _ARROW_LABEL_GAP, _ARROW_POINT_GAP, _LABEL_TEXT_HALF_H,
        atom_main_label, atom_pos, bond_order_of, format_chem_text,
        label_bond_margin, mech_arrow_between, mech_arrow_origin, prepare_mol,
        scale_mol_coords, symbol_center,
    )
    from renderers.layout import layout_row, molecule_scope_lines
else:
    from .mol_primitives import (
        _ARROW_LABEL_GAP, _ARROW_POINT_GAP, _LABEL_TEXT_HALF_H,
        atom_main_label, atom_pos, bond_order_of, format_chem_text,
        label_bond_margin, mech_arrow_between, mech_arrow_origin, prepare_mol,
        scale_mol_coords,
    )
    from .layout import layout_row, molecule_scope_lines


_MOL_GAP = 1.8        # 同一侧分子之间的水平间距
_ARR_MARGIN = 1.9     # 分子与反应箭头之间的余量；反应箭头实际宽度等于此值
_MOL_SCALE = 0.8      # 分子坐标缩放因子（紧凑化，不影响字号）


def _parse_molecules(s: str) -> List[str]:
    """分号分隔的 SMILES 列表，去空。"""
    return [x.strip() for x in s.split(";") if x.strip()]


def _parse_arrows(arrows_str: str) -> List[Tuple[int, str, int, str, str]]:
    """把机理箭头字符串解析为 (src_mol, src_pt, dst_mol, dst_pt, kind) 列表。

    src_pt / dst_pt 为原子序号字符串或 "a-b" 键中点字符串；
    kind 为 'standard' 或 'fishhook'。
    """
    arrows = []
    if not arrows_str:
        return arrows
    for part in arrows_str.split(","):
        part = part.strip()
        if not part:
            continue
        kind = "standard"
        sep = ">"
        if ">>" in part:
            kind = "fishhook"
            sep = ">>"
        elif ">" not in part:
            continue
        src, dst = part.split(sep, 1)
        src = src.strip()
        dst = dst.strip()
        try:
            src_mol_s, src_pt = src.split(":", 1)
            dst_mol_s, dst_pt = dst.split(":", 1)
            src_mol, dst_mol = int(src_mol_s), int(dst_mol_s)
        except ValueError:
            continue
        arrows.append((src_mol, src_pt.strip(), dst_mol, dst_pt.strip(), kind))
    return arrows


def render_reaction_mech(reactants_str: str, products_str: str,
                         conditions: str = "", arrows_str: str = "",
                         flags: str = "") -> str:
    r"""[REACTIONMECH] 渲染：反应式 + 机理弯箭头 → 单张 TikZ。

    flags 含 "numbering" 时显示原子序号（默认隐藏）。
    """
    try:
        from rdkit import Chem
    except ImportError:
        return "（机理反应图渲染失败：rdkit 未安装）"

    show_numbers = "numbering" in (flags or "")

    reactants = _parse_molecules(reactants_str)
    products = _parse_molecules(products_str)

    if not reactants:
        return "（机理反应图渲染失败：反应物不能为空）"
    if not products:
        return "（机理反应图渲染失败：产物不能为空）"

    all_smiles = reactants + products
    molecules = []
    for smi in all_smiles:
        mol = prepare_mol(smi)
        if mol is None:
            return f"（机理反应图渲染失败：无效 SMILES「{smi}」）"
        scale_mol_coords(mol, _MOL_SCALE)
        molecules.append(mol)

    arrows = _parse_arrows(arrows_str)

    # 统一布局引擎：反应物 → 主箭头 → 产物（视觉包围盒防重叠）
    items = []
    for i, mol in enumerate(molecules):
        if i == len(reactants):
            items.append(("arrow", conditions.strip()))
        items.append(("mol", i, mol))
    layout = layout_row(items, mol_gap=_MOL_GAP,
                        arrow_w=2 * _ARR_MARGIN, arrow_pad=_ARR_MARGIN / 2)
    shifts = [p.shift for p in layout.mols]
    main_arrow = layout.arrows[0]

    lines = [r"\begin{tikzpicture}"]

    # 每个分子一个 scope（布局引擎积木）
    for mol, shift in zip(molecules, shifts):
        lines.extend(molecule_scope_lines(mol, shift, show_numbers=show_numbers))

    cond_text = format_chem_text(main_arrow.condition)
    if cond_text:
        lines.append(
            f"  \\draw[->, very thick] ({main_arrow.x1:.2f},0) -- ({main_arrow.x2:.2f},0) "
            f"node[midway, above] {{{cond_text}}};"
        )
    else:
        lines.append(
            f"  \\draw[->, very thick] ({main_arrow.x1:.2f},0) -- ({main_arrow.x2:.2f},0);"
        )

    for src_mol, src_pt, dst_mol, dst_pt, kind in arrows:
        if src_mol >= len(molecules) or dst_mol >= len(molecules):
            continue
        sm = molecules[src_mol]
        dm = molecules[dst_mol]
        p1 = mech_arrow_origin(dm, dst_pt, shifts[dst_mol],
                               lone_pair_offset=False)
        if p1 is None:
            continue
        p0 = mech_arrow_origin(sm, src_pt, shifts[src_mol],
                               toward=(p1[0], p1[1]),
                               prefer_single=(kind == "fishhook"),
                               labeler=atom_main_label)
        if p0 is None:
            continue
        p1 = mech_arrow_origin(dm, dst_pt, shifts[dst_mol],
                               lone_pair_offset=False,
                               toward=(p0[0], p0[1]),
                               labeler=atom_main_label,
                               bend_side=-1.0 if p0[2] else 1.0)
        if p1 is None:
            continue
        bond_break = "-" in src_pt and bond_order_of(sm, src_pt) == 1
        inset_start = (_ARROW_POINT_GAP if bond_break
                       else (0.0 if (p0[2] or p0[3] or p0[4]) else 0.15))
        aim_end = ("-" not in dst_pt and p1[4]
                   and dm.GetAtomWithIdx(int(dst_pt)).GetAtomicNum() == 6)
        tb = None
        if aim_end:
            da = dm.GetAtomWithIdx(int(dst_pt))
            ax, ay = symbol_center(dm, int(dst_pt))
            tb = (ax + shifts[dst_mol][0], ay + shifts[dst_mol][1],
                  label_bond_margin(atom_main_label(da)), _LABEL_TEXT_HALF_H)
        inset_end = (_ARROW_LABEL_GAP if aim_end
                     else (0.0 if p1[4] else 0.10))
        lines.extend(
            mech_arrow_between(p0[0], p0[1], p1[0], p1[1], kind,
                               from_bond=p0[2], inset_start=inset_start,
                               inset_end=inset_end, bond_break=bond_break,
                               aim_end=aim_end, text_box=tb)
        )

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("=" * 60)
    print("[REACTIONMECH] 机理反应图测试")
    print("=" * 60)

    print("\n[1] SN2：OH- 孤对电子进攻 CH3Cl，C-Cl 键断裂")
    print(render_reaction_mech(
        "CCl;[OH-]",
        "CO;[Cl-]",
        "SN2",
        "1:0>0:0,0:0-1>0:1",
    ))

    print("\n[2] 苯的硝化：NO2+ 进攻苯环")
    print(render_reaction_mech(
        "c1ccccc1;[O-][N+](=O)[O-]",
        "O=[N+]([O-])c1ccccc1;O",
        "H2SO4, 浓HNO3",
        "1:1>0:0",
    ))

    print("\n[3] 自由基加成：Br· 加到乙烯（鱼钩箭头）")
    print(render_reaction_mech(
        "C=C;[Br]",
        "[CH2]CBr",
        "hv 或 ROOR",
        "1:0>>0:1,0:0-1>>0:1",
    ))

    print("\n[4] 空反应物错误")
    print(render_reaction_mech(
        "",
        "[Cl-];CO",
        "SN2",
        "1:0>0:0",
    ))

    print("\n[5] 无效箭头索引（应被忽略，不崩溃）")
    print(render_reaction_mech(
        "CCl;[OH-]",
        "[Cl-];CO",
        "SN2",
        "1:0>0:99",
    ))

    print("\n[6] numbering 标志：显示原子序号")
    print(render_reaction_mech(
        "CCl;[OH-]",
        "CO;[Cl-]",
        "SN2",
        "1:0>0:0,0:0-1>0:1",
        "numbering",
    ))
