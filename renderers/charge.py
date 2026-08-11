# -*- coding: utf-8 -*-
"""renderers/charge.py — [CHARGE] 标记渲染器：结构式 + 部分电荷标注。

RDKit 2D 坐标自绘分子骨架，在指定原子旁标注 δ+/δ-（红色）。
标注以元素符号中心为基准（与孤对电子同一基准，而非整个基团标签中心）。
标记格式：[CHARGE:SMILES|0:δ+,1:δ-,...]
"""

from .mol_primitives import (
    atom_label, atom_pos, charge_tikz, format_partial_charge,
    parse_charge_pairs, partial_charge_pos, prepare_mol, bond_segments,
)


def render_charge(smiles: str, charges_str: str = "") -> str:
    """[CHARGE] 渲染：SMILES + 部分电荷标注 → TikZ。"""
    try:
        from rdkit import Chem
    except ImportError:
        return "（电荷标注渲染失败：rdkit 未安装）"

    mol = prepare_mol(smiles)
    if mol is None:
        return f"（电荷标注渲染失败：无效 SMILES「{smiles}」）"

    charges = parse_charge_pairs(charges_str)

    lines = ["\\begin{tikzpicture}"]

    # 骨架键
    for segs in bond_segments(mol, label_margin=0.25, bond_gap=0.08):
        for x1, y1, x2, y2 in segs:
            lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")

    # 原子标签（形式电荷统一用圆圈电荷显示，不内嵌上标）
    for atom in mol.GetAtoms():
        lab = atom_label(atom)
        if lab:
            x, y = atom_pos(mol, atom.GetIdx())
            lines.append(f"  \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")
        charge = charge_tikz(mol, atom.GetIdx())
        if charge:
            lines.append(f"  {charge}")

    # 部分电荷标注（红色，以元素符号中心为基准，方向避让键与标签氢）
    for idx, raw_label in charges.items():
        if idx >= mol.GetNumAtoms():
            continue
        x, y = partial_charge_pos(mol, idx)
        label = format_partial_charge(raw_label)
        lines.append(f"  \\node[font=\\small, red] at ({x:.2f},{y:.2f}) {{{label}}};")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 乙醇部分电荷: [CHARGE:OCC|0:δ-,1:δ+]")
    print(render_charge("OCC", "0:δ-,1:δ+"))
    print("\n[2] 无效 SMILES（应降级提示）:")
    print(render_charge("XYZ", "0:δ-"))
