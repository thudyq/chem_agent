# -*- coding: utf-8 -*-
"""renderers/hbond.py — [HBOND] 标记渲染器：氢键点状虚线。

RDKit 2D 坐标自绘分子骨架（键线式：碳原子不标 CHn，杂原子带 H 标签），
在给体 H 与受体 Y 之间布点表示氢键（规范：X—H 共价键为实线、H···Y 为
点状虚线 ≥3 点、全图点径间距一致）。给体与受体的标签 H 计数均 -1，
显式画出的 H 不在标签中重复（如给体 -O-H 而非 -OH-H；受体同样画成
-O-H，且其 H 朝向远离给体方向，避开氢键点线）。
标记格式：[HBOND:SMILES|from-to,from-to,...]（from 为给体原子，to 为受体原子）
示例：[HBOND:OCCO|0-3]（乙二醇分子内 O—H···O 氢键）
"""

from .mol_primitives import (
    adjust_hbond_conformation, atom_label, atom_pos, charge_tikz,
    hbond_dots_tikz, label_bond_margin, label_edge_point, parse_hbond_pairs,
    place_donor_h, place_explicit_hs, prepare_mol, bond_segments,
)


def render_hbond(smiles: str, pairs_str: str = "") -> str:
    """[HBOND] 渲染：SMILES + 氢键原子对 → TikZ（骨架 + 点状虚线）。"""
    try:
        from rdkit import Chem
    except ImportError:
        return "（氢键渲染失败：rdkit 未安装）"

    mol = prepare_mol(smiles)
    if mol is None:
        return f"（氢键渲染失败：无效 SMILES「{smiles}」）"

    hbonds = parse_hbond_pairs(pairs_str)
    # 构象调整：把给体与受体折到主链同一侧（规范第 4 条）
    for fi, ti in hbonds:
        if fi < mol.GetNumAtoms() and ti < mol.GetNumAtoms():
            adjust_hbond_conformation(mol, fi, ti)

    lines = ["\\begin{tikzpicture}"]

    # 键线式骨架：碳原子不标 CHn（atom_label 返回 None），
    # 给体与受体标签 H 均已扣减：显式画出的 H 不在标签中重复
    hbond_atoms = {fi for fi, _ in hbonds} | {ti for _, ti in hbonds}
    hbond_lab = lambda a: atom_label(a, 1 if a.GetIdx() in hbond_atoms else 0)
    for segs in bond_segments(mol, labeler=hbond_lab,
                              margin_fn=label_bond_margin):
        for x1, y1, x2, y2 in segs:
            lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")

    # 原子标签（键线式：仅杂原子；给体/受体标签 H 计数 -1）
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        lab = hbond_lab(atom)
        if lab:
            x, y = atom_pos(mol, idx)
            lines.append(f"  \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")
        charge = charge_tikz(mol, idx,
                             explicit_hs=1 if idx in hbond_atoms else 0)
        if charge:
            lines.append(f"  {charge}")

    # 氢键：X—H 实线（从标签边缘起笔，不压标签）+ H···Y 点状虚线（从 H 出发）
    for fi, ti in hbonds:
        if fi >= mol.GetNumAtoms() or ti >= mol.GetNumAtoms():
            continue
        tx, ty = atom_pos(mol, ti)
        hx, hy = place_donor_h(mol, fi, (tx, ty))
        sx, sy = label_edge_point(mol, fi, (hx, hy), labeler=hbond_lab)
        lines.append(f"  \\draw ({sx:.2f},{sy:.2f}) -- ({hx:.2f},{hy:.2f});")
        lines.append(f"  \\node[fill=white, inner sep=1pt] at ({hx:.2f},{hy:.2f}) {{H}};")
        for dot_line in hbond_dots_tikz(hx, hy, tx, ty):
            lines.append(f"  {dot_line}")
        # 受体显式 H：朝向远离给体方向（避开氢键点线），标签已扣减
        ax, ay = atom_pos(mol, ti)
        ahx, ahy = place_explicit_hs(mol, ti, 1,
                                     toward=(2 * ax - tx, 2 * ay - ty))[0]
        asx, asy = label_edge_point(mol, ti, (ahx, ahy), labeler=hbond_lab)
        lines.append(f"  \\draw ({asx:.2f},{asy:.2f}) -- ({ahx:.2f},{ahy:.2f});")
        lines.append(f"  \\node[fill=white, inner sep=1pt] at ({ahx:.2f},{ahy:.2f}) {{H}};")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 乙二醇分子内氢键: OCCO | 0-3")
    print(render_hbond("OCCO", "0-3"))
    print("\n[2] 无效 SMILES（应降级提示）:")
    print(render_hbond("XYZ", "0-1"))
