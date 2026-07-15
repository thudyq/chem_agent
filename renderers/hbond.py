# -*- coding: utf-8 -*-
"""renderers/hbond.py — [HBOND] 标记渲染器：氢键虚线连接。

RDKit 2D 坐标自绘分子骨架，在指定原子对之间画虚线表示氢键。
标记格式：[HBOND:SMILES|from-to,from-to,...]
示例：[HBOND:O|0-0]（两个水分子间 O···H-O 氢键，需两个片段 SMILES）
      [HBOND:OCO|0-2]（分子内 O(0)···O(2) 氢键）
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


def _parse_hbonds(pairs_str: str):
    """'0-2,3-5' → [(0,2),(3,5)]"""
    pairs = []
    for s in pairs_str.split(","):
        s = s.strip()
        if "-" in s:
            try:
                f, t = s.split("-", 1)
                pairs.append((int(f.strip()), int(t.strip())))
            except ValueError:
                pass
    return pairs


def render_hbond(smiles: str, pairs_str: str = "") -> str:
    """[HBOND] 渲染：SMILES + 氢键原子对 → TikZ（骨架 + 虚线）。"""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return "（氢键渲染失败：rdkit 未安装）"

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return f"（氢键渲染失败：无效 SMILES「{smiles}」）"
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

    hbonds = _parse_hbonds(pairs_str)

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

    # 氢键虚线（蓝绿色，dashed，缩短两端避免压住原子）
    for fi, ti in hbonds:
        if fi >= mol.GetNumAtoms() or ti >= mol.GetNumAtoms():
            continue
        fx, fy = pos(fi)
        tx, ty = pos(ti)
        dx, dy = tx - fx, ty - fy
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        margin = 0.25
        x1, y1 = fx + ux * margin, fy + uy * margin
        x2, y2 = tx - ux * margin, ty - uy * margin
        lines.append(
            f"  \\draw[dashed, teal, thick] ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});"
        )

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 分子内氢键: OCO | 0-2")
    print(render_hbond("OCO", "0-2"))
