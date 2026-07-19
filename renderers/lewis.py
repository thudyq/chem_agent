# -*- coding: utf-8 -*-
"""renderers/lewis.py — [LEWIS] 标记渲染器：Lewis 结构式（显示孤对电子）。

与 STRUCT（mol2chemfig 骨架式）不同，Lewis 需显示孤对电子点，而 mol2chemfig 的
布局坐标与 RDKit 不一致——故 Lewis 完全用 RDKit 2D 坐标自绘：键、原子标签、
孤对电子点统一坐标系，保证对齐。

孤对电子数 = (价电子 − 键级和 − 形式电荷) / 2。
"""

import math

from .mol_primitives import atom_label, atom_pos, prepare_mol, bond_segments


# 原子序 → 价电子数
_VALENCE_E = {1: 1, 5: 5, 6: 4, 7: 5, 8: 6, 9: 7, 14: 4, 15: 5, 16: 6, 17: 7, 35: 7, 53: 7}


def _num_lone_pairs(atom) -> int:
    """计算原子上的孤对电子数。"""
    z = atom.GetAtomicNum()
    ve = _VALENCE_E.get(z)
    if ve is None:
        return 0
    bond_sum = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()) + atom.GetTotalNumHs()
    fc = atom.GetFormalCharge()
    lp = (ve - bond_sum - fc) / 2
    return int(lp) if lp >= 0 else 0


def render_lewis(smiles: str) -> str:
    """[LEWIS] 渲染：SMILES → 含孤对电子的 Lewis 结构式 TikZ。失败返回错误提示。"""
    try:
        from rdkit import Chem
    except ImportError:
        return "（Lewis 渲染失败：rdkit 未安装）"

    mol = prepare_mol(smiles, add_hs=True, kekulize=True, use_prepare=False)
    if mol is None:
        return f"（Lewis 渲染失败：无效 SMILES「{smiles}」）"

    lines = ["\\begin{tikzpicture}"]

    # 键（Lewis 使用更大的标签边距和键间距）
    for segs in bond_segments(mol, label_margin=0.30, bond_gap=0.09):
        for x1, y1, x2, y2 in segs:
            lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")

    # 原子标签
    for atom in mol.GetAtoms():
        lab = atom_label(atom)
        if lab:
            x, y = atom_pos(mol, atom.GetIdx())
            lines.append(f"  \\node[fill=white,inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")

    # 孤对电子点
    DOT, R = 0.03, 0.28
    for atom in mol.GetAtoms():
        lp = _num_lone_pairs(atom)
        if lp <= 0:
            continue
        xi, yi = atom_pos(mol, atom.GetIdx())
        dirs = []
        for n in atom.GetNeighbors():
            xn, yn = atom_pos(mol, n.GetIdx())
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
