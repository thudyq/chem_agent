# -*- coding: utf-8 -*-
"""renderers/charge.py — [CHARGE] 标记渲染器：结构式 + 部分电荷标注。

RDKit 2D 坐标自绘分子骨架，在指定原子旁标注 δ+/δ-（红色）。
标记格式：[CHARGE:SMILES|0:δ+,1:δ-,...]
"""

from ._mol_base import atom_label, atom_pos, prepare_mol, bond_segments


def _fmt_charge(raw: str) -> str:
    """δ+ → $\\delta^+$, δ- → $\\delta^-$"""
    raw = raw.strip()
    if "δ" in raw:
        s = raw.replace("δ", "\\delta")
        if s.endswith("+"):
            s = s[:-1] + "^+"
        elif s.endswith("-"):
            s = s[:-1] + "^-"
        return f"${s}$"
    return raw


def _parse_charges(charges_str: str) -> dict:
    """'0:δ+,1:δ-' → {0: 'δ+', 1: 'δ-'}"""
    result = {}
    for pair in charges_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            idx_str, _, label = pair.partition(":")
            try:
                result[int(idx_str.strip())] = label.strip()
            except ValueError:
                pass
    return result


def render_charge(smiles: str, charges_str: str = "") -> str:
    """[CHARGE] 渲染：SMILES + 部分电荷标注 → TikZ。"""
    try:
        from rdkit import Chem
    except ImportError:
        return "（电荷标注渲染失败：rdkit 未安装）"

    mol = prepare_mol(smiles)
    if mol is None:
        return f"（电荷标注渲染失败：无效 SMILES「{smiles}」）"

    charges = _parse_charges(charges_str)

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

    # 部分电荷标注（红色，置于原子右上方）
    for idx, raw_label in charges.items():
        if idx >= mol.GetNumAtoms():
            continue
        x, y = atom_pos(mol, idx)
        label = _fmt_charge(raw_label)
        lines.append(f"  \\node[font=\\small, red] at ({x+0.30:.2f},{y+0.25:.2f}) {{{label}}};")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] HCl 的极性: [CHARGE:Cl|0:δ-,1:δ+]")
    print(render_charge("Cl", "0:δ-"))
