# -*- coding: utf-8 -*-
"""renderers/stereo.py — [STEREO] 标记渲染器：楔形式（wedge/dash）立体结构。

RDKit 解析含 @/@@ 的 SMILES → PrepareMolForDrawing（含坐标+楔形方向）→
对 BEGINWEDGE 画实心三角（前向）、BEGINDASH 画渐宽短横（后向）、
其余画普通线。不使用 mol2chemfig，完全 RDKit 2D 坐标自绘（与 Lewis 同模式）。
"""

import math

from .mol_primitives import _label_flip_for, atom_label, atom_pos, charge_tikz, \
    label_bond_margin, label_node_pos, mol_visual_bbox, prepare_mol, wrap_format_text


def stereo_scope_lines(mol) -> list:
    """楔形式 scope 绘制行（不含 tikzpicture 包装与 label）。

    供顶层 render_stereo 与 COMPOSITE 容器内 mode=stereo 组件复用——
    容器负责布局平移（scope shift），此处按 mol 当前坐标原样输出。
    """
    from rdkit.Chem import BondDir

    lines = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        xi, yi = atom_pos(mol, i)
        xj, yj = atom_pos(mol, j)

        order = b.GetBondTypeAsDouble()
        order = 3 if order >= 2.5 else (2 if order >= 1.5 else 1)
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        px, py = -uy, ux
        # 标签留白按 label_bond_margin 分档（OH/NH₂=0.30、CH₃=0.45…）——
        # 原固定 0.25 对双字符标签（半宽≈0.26）留白不足、端点落入标签字符区
        lab_i = atom_label(mol.GetAtomWithIdx(i))
        lab_j = atom_label(mol.GetAtomWithIdx(j))
        si = label_bond_margin(lab_i) if lab_i else 0.0
        sj = label_bond_margin(lab_j) if lab_j else 0.0
        x1, y1 = xi + ux * si, yi + uy * si
        x2, y2 = xj - ux * sj, yj - uy * sj
        bdir = b.GetBondDir()

        if bdir == BondDir.BEGINWEDGE:
            # 实心楔形：窄端在 i，宽端在 j
            w = 0.09
            lines.append(
                f"  \\fill ({x1:.2f},{y1:.2f}) -- "
                f"({x2+px*w:.2f},{y2+py*w:.2f}) -- "
                f"({x2-px*w:.2f},{y2-py*w:.2f}) -- cycle;"
            )
        elif bdir == BondDir.BEGINDASH:
            # 虚楔形：5 条渐宽短横，从窄（i）到宽（j）
            for k in range(1, 6):
                t = k / 5
                w = 0.02 + 0.07 * t
                cx = x1 + (x2 - x1) * t
                cy = y1 + (y2 - y1) * t
                lines.append(
                    f"  \\draw[thick] ({cx+px*w:.2f},{cy+py*w:.2f}) -- "
                    f"({cx-px*w:.2f},{cy-py*w:.2f});"
                )
        else:
            lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")
            if order >= 2:
                d = 0.08
                lines.append(f"  \\draw ({x1+px*d:.2f},{y1+py*d:.2f}) -- ({x2+px*d:.2f},{y2+py*d:.2f});")
            if order >= 3:
                d = 0.08
                lines.append(f"  \\draw ({x1-px*d:.2f},{y1-py*d:.2f}) -- ({x2-px*d:.2f},{y2-py*d:.2f});")

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        lab = atom_label(atom, flip=_label_flip_for(mol, idx))
        if lab:
            nx, ny = label_node_pos(mol, idx)
            lines.append(f"  \\node[fill=white, inner sep=1pt] at ({nx:.2f},{ny:.2f}) {{{lab}}};")
        # 形式电荷统一用圆圈电荷显示，不内嵌上标
        charge = charge_tikz(mol, idx)
        if charge:
            lines.append(f"  {charge}")
    return lines


def render_stereo(smiles: str, label: str = None) -> str:
    """[STEREO] 渲染：含立体信息的 SMILES → 楔形式 TikZ。失败返回错误提示。

    label 可选：置于结构下方（如 (R)-乳酸）。
    """
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（立体结构渲染失败：rdkit 未安装）"

    mol = prepare_mol(smiles)
    if mol is None:
        return f"（立体结构渲染失败：无效 SMILES「{smiles}」）"

    lines = ["\\begin{tikzpicture}"]
    lines.extend(stereo_scope_lines(mol))
    if label:
        min_x, min_y, max_x, _ = mol_visual_bbox(mol, include_lone_pairs=False)
        text = wrap_format_text(label)
        align = "align=center, " if "\\\\" in text else ""
        lines.append(
            f"  \\node[{align}below] at ({(min_x + max_x) / 2.0:.2f},{min_y - 0.15:.2f}) "
            f"{{{text}}};"
        )

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] (R)-乳酸 C[C@@H](O)C(=O)O:")
    print(render_stereo("C[C@@H](O)C(=O)O"))
    print("\n[2] (S)-2-氨基丁酸 CC[C@@H](N)C(=O)O:")
    print(render_stereo("CC[C@@H](N)C(=O)O"))
    print("\n[3] 无手性中心（应提示）:")
    print(render_stereo("CCO"))
    print("\n[4] 带 label：(R)-乳酸 C[C@@H](O)C(=O)O, label=(R)-乳酸:")
    print(render_stereo("C[C@@H](O)C(=O)O", label="(R)-乳酸"))
