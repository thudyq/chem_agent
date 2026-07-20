# -*- coding: utf-8 -*-
r"""renderers/reaction_mech.py — [REACTIONMECH] 机理图与反应式组合渲染器。

把多个反应物、多个产物横向排布成一条反应方程式，并在上方叠加电子推进
弯箭头，形成“反应式 + 机理”的复合图。所有分子绘制在同一个 TikZ 坐标系
内，保证机理箭头与分子骨架对齐。

标记格式：
    [REACTIONMECH:反应物1;反应物2;...|产物1;产物2;...|反应条件|机理箭头]

机理箭头格式：
    src_mol:src_atom>dst_mol:dst_atom        （双头弯箭头，电子对转移）
    src_mol:src_atom>>dst_mol:dst_atom       （鱼钩箭头，单电子转移）

分子编号：先反应物后产物，从 0 开始。例如 2 个反应物 + 2 个产物时，产物
Cl- 的编号为 2，CH3OH 的编号为 3。

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

import math
from typing import List, Tuple

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from renderers.mol_primitives import atom_label, atom_pos, bond_segments, prepare_mol
else:
    from .mol_primitives import atom_label, atom_pos, bond_segments, prepare_mol


_MOL_GAP = 1.5        # 同一侧分子之间的水平间距
_ARR_MARGIN = 1.5     # 分子与反应箭头之间的余量；反应箭头实际宽度等于此值


def _parse_molecules(s: str) -> List[str]:
    """分号分隔的 SMILES 列表，去空。"""
    return [x.strip() for x in s.split(";") if x.strip()]


def _parse_arrows(arrows_str: str) -> List[Tuple[int, int, int, int, str]]:
    """把机理箭头字符串解析为 (src_mol, src_atom, dst_mol, dst_atom, kind) 列表。

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
            src_mol, src_atom = map(int, src.split(":", 1))
            dst_mol, dst_atom = map(int, dst.split(":", 1))
        except ValueError:
            continue
        arrows.append((src_mol, src_atom, dst_mol, dst_atom, kind))
    return arrows


def _mol_bbox(mol) -> Tuple[float, float, float, float]:
    """返回分子 2D 坐标包围盒 (min_x, min_y, max_x, max_y)。"""
    xs = []
    ys = []
    for atom in mol.GetAtoms():
        x, y = atom_pos(mol, atom.GetIdx())
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def _transformed_pos(mol, shift: Tuple[float, float], idx: int) -> Tuple[float, float]:
    x, y = atom_pos(mol, idx)
    return x + shift[0], y + shift[1]


def _format_conditions(text: str) -> str:
    """把化学式中的数字自动转为下标（如 H2SO4 → H$_2$SO$_4$）。"""
    if not text or "$" in text:
        return text
    import re
    return re.sub(r"([A-Za-z])(\d+)", r"\1$_\2$", text)


def render_reaction_mech(reactants_str: str, products_str: str,
                         conditions: str = "", arrows_str: str = "") -> str:
    r"""[REACTIONMECH] 渲染：反应式 + 机理弯箭头 → 单张 TikZ。"""
    try:
        from rdkit import Chem
    except ImportError:
        return "（机理反应图渲染失败：rdkit 未安装）"

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
        molecules.append(mol)

    arrows = _parse_arrows(arrows_str)

    # 计算布局：反应物 → 箭头空档 → 产物
    bboxes = [_mol_bbox(mol) for mol in molecules]
    widths = [bbox[2] - bbox[0] for bbox in bboxes]
    n_react = len(reactants)
    n_prod = len(products)

    left_width = sum(widths[:n_react]) + _MOL_GAP * (n_react - 1)
    right_width = sum(widths[n_react:]) + _MOL_GAP * (n_prod - 1)

    total_left = left_width + _ARR_MARGIN
    total_right = right_width + _ARR_MARGIN
    max_side = max(total_left, total_right)

    shifts = []
    cursor = -max_side
    for i in range(n_react):
        w = widths[i]
        min_x, min_y, max_x, max_y = bboxes[i]
        local_cx = (min_x + max_x) / 2.0
        local_cy = (min_y + max_y) / 2.0
        target_x = cursor + w / 2.0
        shifts.append((target_x - local_cx, -local_cy))
        cursor += w + _MOL_GAP

    cursor = _ARR_MARGIN
    for i in range(n_react, len(molecules)):
        w = widths[i]
        min_x, min_y, max_x, max_y = bboxes[i]
        local_cx = (min_x + max_x) / 2.0
        local_cy = (min_y + max_y) / 2.0
        target_x = cursor + w / 2.0
        shifts.append((target_x - local_cx, -local_cy))
        cursor += w + _MOL_GAP

    lines = [r"\begin{tikzpicture}"]

    for mol, shift in zip(molecules, shifts):
        for segs in bond_segments(mol, label_margin=0.25, bond_gap=0.08):
            for x1, y1, x2, y2 in segs:
                lines.append(
                    f"  \\draw ({x1 + shift[0]:.2f},{y1 + shift[1]:.2f}) "
                    f"-- ({x2 + shift[0]:.2f},{y2 + shift[1]:.2f});"
                )

    for mol, shift in zip(molecules, shifts):
        for atom in mol.GetAtoms():
            x, y = _transformed_pos(mol, shift, atom.GetIdx())
            lab = atom_label(atom)
            if lab:
                lines.append(
                    f"  \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};"
                )
            lines.append(
                f"  \\node[font=\\tiny, gray, below right] at ({x:.2f},{y:.2f}) "
                f"{{{atom.GetIdx()}}};"
            )

    arrow_left = -_ARR_MARGIN / 2.0
    arrow_right = _ARR_MARGIN / 2.0
    cond_text = _format_conditions(conditions.strip())
    if cond_text:
        lines.append(
            f"  \\draw[->, very thick] ({arrow_left:.2f},0) -- ({arrow_right:.2f},0) "
            f"node[midway, above] {{{cond_text}}};"
        )
    else:
        lines.append(
            f"  \\draw[->, very thick] ({arrow_left:.2f},0) -- ({arrow_right:.2f},0);"
        )

    for src_mol, src_atom, dst_mol, dst_atom, kind in arrows:
        if src_mol >= len(molecules) or dst_mol >= len(molecules):
            continue
        sm = molecules[src_mol]
        dm = molecules[dst_mol]
        if src_atom >= sm.GetNumAtoms() or dst_atom >= dm.GetNumAtoms():
            continue
        fx, fy = _transformed_pos(sm, shifts[src_mol], src_atom)
        tx, ty = _transformed_pos(dm, shifts[dst_mol], dst_atom)
        dx, dy = tx - fx, ty - fy
        L = math.hypot(dx, dy) or 1.0
        off = 0.5
        sign = -1 if (src_mol + dst_mol) % 2 == 0 else 1
        mx = (fx + tx) / 2.0 + (-dy / L) * off * sign
        my = (fy + ty) / 2.0 + (dx / L) * off * sign
        if kind == "fishhook":
            lines.append(
                f"  \\draw[thick, red] ({fx:.2f},{fy:.2f}) "
                f".. controls ({mx:.2f},{my:.2f}) .. ({tx:.2f},{ty:.2f});"
            )
            incoming = math.atan2(ty - my, tx - mx)
            barb_ang = incoming + math.pi + math.radians(25)
            blen = 0.18
            bx = tx + blen * math.cos(barb_ang)
            by = ty + blen * math.sin(barb_ang)
            lines.append(
                f"  \\draw[thick, red] ({tx:.2f},{ty:.2f}) -- ({bx:.2f},{by:.2f});"
            )
        else:
            lines.append(
                f"  \\draw[->, thick, red] ({fx:.2f},{fy:.2f}) "
                f".. controls ({mx:.2f},{my:.2f}) .. ({tx:.2f},{ty:.2f});"
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

    print("\n[1] SN2：OH- 进攻 CH3Cl，Cl- 离去")
    print(render_reaction_mech(
        "CCl;[OH-]",
        "[Cl-];CO",
        "SN2",
        "1:0>0:0,0:1>2:0",
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
        "1:0>>0:0,0:0>>0:1",
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
