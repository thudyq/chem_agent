# -*- coding: utf-8 -*-
"""renderers/lewis.py — [LEWIS] 标记渲染器：Lewis 结构式（显示孤对电子）。

与 STRUCT（mol2chemfig 骨架式）不同，Lewis 需显示孤对电子点，而 mol2chemfig 的
布局坐标与 RDKit 不一致——故 Lewis 完全用 RDKit 2D 坐标自绘：键、原子标签、
孤对电子点统一坐标系，保证对齐。

孤对电子数 = (价电子 − 键级和 − 形式电荷) / 2。
"""

import math

# 原子序 → 价电子数
_VALENCE_E = {1: 1, 5: 5, 6: 4, 7: 5, 8: 6, 9: 7, 14: 4, 15: 5, 16: 6, 17: 7, 35: 7, 53: 7}


def _num_lone_pairs(atom) -> int:
    z = atom.GetAtomicNum()
    ve = _VALENCE_E.get(z)
    if ve is None:
        return 0
    bond_sum = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()) + atom.GetTotalNumHs()
    fc = atom.GetFormalCharge()
    lp = (ve - bond_sum - fc) / 2
    return int(lp) if lp >= 0 else 0


def _atom_label(atom):
    """杂原子/带电荷原子 → 标签（隐式 C 返回 None）。"""
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


def render_lewis(smiles: str) -> str:
    """[LEWIS] 渲染：SMILES → 含孤对电子的 Lewis 结构式 TikZ。失败返回错误提示。"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return "（Lewis 渲染失败：rdkit 未安装）"

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return f"（Lewis 渲染失败：无效 SMILES「{smiles}」）"
    mol = Chem.AddHs(mol)  # 显式化 H，Lewis 需显示所有原子与键
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)  # 芳香键 → 明确单/双键
    except Exception:
        pass
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()

    def pos(i):
        p = conf.GetAtomPosition(i)
        return p.x, p.y

    lines = ["\\begin{tikzpicture}"]

    # 键
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        xi, yi = pos(i)
        xj, yj = pos(j)
        order = b.GetBondTypeAsDouble()
        order = 3 if order >= 2.5 else (2 if order >= 1.5 else 1)
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        px, py = -uy, ux  # 垂直方向
        si = 0.30 if _atom_label(mol.GetAtomWithIdx(i)) else 0.0
        sj = 0.30 if _atom_label(mol.GetAtomWithIdx(j)) else 0.0
        x1, y1 = xi + ux * si, yi + uy * si
        x2, y2 = xj - ux * sj, yj - uy * sj
        seg = lambda ox, oy: lines.append(f"  \\draw ({x1+ox:.2f},{y1+oy:.2f}) -- ({x2+ox:.2f},{y2+oy:.2f});")
        seg(0, 0)
        if order >= 2:
            d = 0.09
            seg(px * d, py * d)
        if order >= 3:
            d = 0.09
            seg(-px * d, -py * d)

    # 原子标签
    for atom in mol.GetAtoms():
        lab = _atom_label(atom)
        if lab:
            x, y = pos(atom.GetIdx())
            lines.append(f"  \\node[fill=white,inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")

    # 孤对电子点
    DOT, R = 0.03, 0.28
    for atom in mol.GetAtoms():
        lp = _num_lone_pairs(atom)
        if lp <= 0:
            continue
        xi, yi = pos(atom.GetIdx())
        dirs = []
        for n in atom.GetNeighbors():
            xn, yn = pos(n.GetIdx())
            dL = math.hypot(xn - xi, yn - yi) or 1.0
            dirs.append(((xn - xi) / dL, (yn - yi) / dL))
        avg_ang = math.atan2(sum(d[1] for d in dirs), sum(d[0] for d in dirs)) if dirs else math.pi / 2
        opp_ang = avg_ang + math.pi  # 孤对电子置于键方向的反侧
        spread = 0.90
        for k in range(lp):
            off = (k - (lp - 1) / 2) * spread
            ang = opp_ang + off
            cx, cy = xi + R * math.cos(ang), yi + R * math.sin(ang)
            perp = ang + math.pi / 2
            dd = 0.07
            for s in (-1, 1):
                dx, dy = cx + s * dd * math.cos(perp), cy + s * dd * math.sin(perp)
                lines.append(f"  \\fill ({dx:.2f},{dy:.2f}) circle ({DOT});")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 水 O:")
    print(render_lewis("O"))
    print("\n[2] 氨 N:")
    print(render_lewis("N"))
    print("\n[3] 甲醇 CO:")
    print(render_lewis("CO"))
