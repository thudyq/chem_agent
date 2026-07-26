# -*- coding: utf-8 -*-
"""renderers/newman.py — [NEWMAN] 标记渲染器：沿 C-C 键的纽曼投影式。

输入 SMILES + 二面角，用 tikzpicture 画纽曼投影：前碳为实心圆、3 键 120° 分布，
后碳 3 键旋转 θ°（灰色、先画以置于后）。取代基标签由 RDKit 解析（C→CH₃、O→OH、隐式 H）。
"""

import math


def _atom_label(atom):
    """取代基原子 → 简化标签（按原子符号+隐式 H，如 CH₃/OH/Cl）。"""
    sym = atom.GetSymbol()
    h = atom.GetTotalNumHs()
    if sym == "C":
        return f"CH$_{{{h}}}$" if h else "C"
    if h == 1:
        return f"{sym}H"
    if h > 1:
        return f"{sym}H$_{{{h}}}$"
    return sym


def render_newman(smiles: str, angle="60") -> str:
    """[NEWMAN] 渲染：SMILES + 二面角 → 纽曼投影 TikZ 代码。

    失败（无效 SMILES/无 C-C 单键/rdkit 未装）返回可读错误提示。
    """
    try:
        from rdkit import Chem
    except ImportError:
        return "（纽曼投影渲染失败：rdkit 未安装）"

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return f"（纽曼投影渲染失败：无效 SMILES「{smiles}」）"

    # 找首个 C-C 单键作为观察键（复杂分子的特定键选择留作后续优化）
    bond = None
    for b in mol.GetBonds():
        a, b2 = b.GetBeginAtom(), b.GetEndAtom()
        if a.GetSymbol() == "C" and b2.GetSymbol() == "C" and b.GetBondTypeAsDouble() == 1:
            bond = b
            break
    if bond is None:
        return f"（纽曼投影渲染失败：「{smiles}」无 C-C 单键）"

    front_atom, back_atom = bond.GetBeginAtom(), bond.GetEndAtom()

    def get_subs(atom, exclude):
        subs = [_atom_label(n) for n in atom.GetNeighbors() if n.GetIdx() != exclude.GetIdx()]
        subs.extend(["H"] * atom.GetTotalNumHs())
        return subs

    front_subs = get_subs(front_atom, back_atom)
    back_subs = get_subs(back_atom, front_atom)

    try:
        theta = float(angle)
    except (TypeError, ValueError):
        theta = 60.0

    front_angles = [90.0, 210.0, 330.0]
    back_angles = [a + theta for a in front_angles]
    R = 0.5              # 前碳空心圆半径
    D = R * (1 + 2 / 3)  # 取代基距圆心：圆外键长统一为圆半径的 2/3
    LBL = D * 1.25       # 标签距圆心（比键末端稍远）
    _COLLIDE = 5.0       # 前后键角差小于该值视为重叠（重叠式）
    _LABEL_SHIFT = 15.0  # 重叠式后标签顺时针偏移角（露出）

    def pt(length, deg):
        r = math.radians(deg)
        return length * math.cos(r), length * math.sin(r)

    def _collides(deg):
        return any(min(abs(deg - f) % 360.0,
                       360.0 - abs(deg - f) % 360.0) < _COLLIDE
                   for f in front_angles)

    lines = ["\\begin{tikzpicture}[scale=1.1]"]
    # 后键：从圆周(R)向外到 D（圆外部分 = 2R/3），先画（灰色、粗线与圆一致，在后）；
    # 重叠式中后键与后标签同步顺时针偏移 15°，避免被前键完全遮挡
    for a, sub in zip(back_angles, back_subs):
        ea = a - _LABEL_SHIFT if _collides(a) else a
        x0, y0 = pt(R, ea)
        x1, y1 = pt(D, ea)
        lx, ly = pt(LBL, ea)
        lines.append(f"  \\draw[thick, gray] ({x0:.2f},{y0:.2f}) -- ({x1:.2f},{y1:.2f});")
        lines.append(f"  \\node[gray] at ({lx:.2f},{ly:.2f}) {{\\small {sub}}};")
    # 前键：从圆心(0,0)到 D（圆外部分 = 2R/3，粗线与圆一致）
    for a, sub in zip(front_angles, front_subs):
        x1, y1 = pt(D, a)
        lx, ly = pt(LBL, a)
        lines.append(f"  \\draw[thick] (0,0) -- ({x1:.2f},{y1:.2f});")
        lines.append(f"  \\node at ({lx:.2f},{ly:.2f}) {{\\small {sub}}};")
    # 前碳：空心大圆（最后画，圆环压在键交叉之上，保持清晰）
    lines.append(f"  \\draw[thick] (0,0) circle ({R:.2f});")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 乙烷 CC, 60°（交叉式）:")
    print(render_newman("CC", "60"))
    print("\n[2] 乙烷 CC, 0°（重叠式）:")
    print(render_newman("CC", "0"))
