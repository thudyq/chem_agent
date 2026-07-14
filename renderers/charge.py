# -*- coding: utf-8 -*-
"""renderers/charge.py — [CHARGE] 标记渲染器：结构式 + 部分电荷标注。

RDKit 2D 坐标自绘分子骨架，在指定原子旁标注 δ+/δ-（红色）。
标记格式：[CHARGE:SMILES|0:δ+,1:δ-,...]
"""

import math


def _atom_label(atom):
    z = atom.GetAtomicNum()
    if z == 6 and atom.GetFormalCharge() == 0:
        return None
    sym = atom.GetSymbol()
    sym = sym[0].upper() + sym[1:]
    h = atom.GetTotalNumHs()
    parts = sym
    if h == 1:
        parts += "H"
    elif h > 1:
        parts += f"H$_{{{h}}}$"
    fc = atom.GetFormalCharge()
    if fc:
        num = str(abs(fc)) if abs(fc) > 1 else ""
        sign = "+" if fc > 0 else "-"
        parts += f"$^{{{num}{sign}}}$"
    return parts


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
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return "（电荷标注渲染失败：rdkit 未安装）"

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return f"（电荷标注渲染失败：无效 SMILES「{smiles}」）"
    try:
        prepared = rdMolDraw2D.PrepareMolForDrawing(mol)
        if prepared is not None:
            mol = prepared
    except Exception:
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()

    def pos(i):
        p = conf.GetAtomPosition(i)
        return p.x, p.y

    charges = _parse_charges(charges_str)

    lines = ["\\begin{tikzpicture}"]

    # 骨架键
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        xi, yi = pos(i)
        xj, yj = pos(j)
        order = b.GetBondTypeAsDouble()
        order = 3 if order >= 2.5 else (2 if order >= 1.5 else 1)
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        px, py = -uy, ux
        si = 0.25 if _atom_label(mol.GetAtomWithIdx(i)) else 0.0
        sj = 0.25 if _atom_label(mol.GetAtomWithIdx(j)) else 0.0
        x1, y1 = xi + ux * si, yi + uy * si
        x2, y2 = xj - ux * sj, yj - uy * sj
        lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")
        if order >= 2:
            d = 0.08
            lines.append(f"  \\draw ({x1+px*d:.2f},{y1+py*d:.2f}) -- ({x2+px*d:.2f},{y2+py*d:.2f});")

    # 原子标签
    for atom in mol.GetAtoms():
        lab = _atom_label(atom)
        if lab:
            x, y = pos(atom.GetIdx())
            lines.append(f"  \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")

    # 部分电荷标注（红色，置于原子右上方）
    for idx, raw_label in charges.items():
        if idx >= mol.GetNumAtoms():
            continue
        x, y = pos(idx)
        label = _fmt_charge(raw_label)
        lines.append(f"  \\node[font=\\small, red] at ({x+0.30:.2f},{y+0.25:.2f}) {{{label}}};")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] HCl 的极性: [CHARGE:Cl|0:δ-,1:δ+]")
    print(render_charge("Cl", "0:δ-"))
