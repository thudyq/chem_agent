# -*- coding: utf-8 -*-
"""renderers/hbond.py — [HBOND] 标记渲染器：氢键虚线连接。

RDKit 2D 坐标自绘分子骨架，在指定原子对之间画虚线表示氢键。
标记格式：[HBOND:SMILES|from-to,from-to,...]
示例：[HBOND:O|0-0]（两个水分子间 O···H-O 氢键，需两个片段 SMILES）
      [HBOND:OCO|0-2]（分子内 O(0)···O(2) 氢键）
"""

from .mol_primitives import (
    atom_label, atom_pos, hbond_line_tikz, parse_hbond_pairs,
    prepare_mol, bond_segments,
)


def render_hbond(smiles: str, pairs_str: str = "") -> str:
    """[HBOND] 渲染：SMILES + 氢键原子对 → TikZ（骨架 + 虚线）。"""
    try:
        from rdkit import Chem
    except ImportError:
        return "（氢键渲染失败：rdkit 未安装）"

    mol = prepare_mol(smiles)
    if mol is None:
        return f"（氢键渲染失败：无效 SMILES「{smiles}」）"

    hbonds = parse_hbond_pairs(pairs_str)

    lines = ["\\begin{tikzpicture}"]

    # 骨架键
    for segs in bond_segments(mol, label_margin=0.25, bond_gap=0.08):
        for x1, y1, x2, y2 in segs:
            lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")

    # 原子标签
    for atom in mol.GetAtoms():
        lab = atom_label(atom)
        if lab:
            x, y = atom_pos(mol, atom.GetIdx())
            lines.append(f"  \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")

    # 氢键虚线
    for fi, ti in hbonds:
        if fi >= mol.GetNumAtoms() or ti >= mol.GetNumAtoms():
            continue
        fx, fy = atom_pos(mol, fi)
        tx, ty = atom_pos(mol, ti)
        lines.append("  " + hbond_line_tikz(fx, fy, tx, ty))

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 分子内氢键: OCO | 0-2")
    print(render_hbond("OCO", "0-2"))
