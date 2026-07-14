# -*- coding: utf-8 -*-
"""renderers/mechanism.py — [MECH] 标记渲染器：在分子上画电子推进弯箭头。

MVP 方案：LLM 指定 SMILES + 原子索引间的电子流向（from>to），
renderer 画分子骨架 + 红色 Bezier 弯箭头。原子索引按 SMILES 出现顺序（0 起）。
renderer 额外标注索引号，便于核对箭头指向。

标记格式：[MECH:SMILES|from>to,from>to,...]
示例：[MECH:CCl.[OH-]|2>0,0>1]  （O(2)进攻C(0)，C(0)-Cl(1)键断裂）
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


def _parse_arrows(arrows_str):
    """'2>0,0>1' → [(2,0),(0,1)]"""
    pairs = []
    for s in arrows_str.split(","):
        s = s.strip()
        if ">" in s:
            try:
                f, t = s.split(">", 1)
                pairs.append((int(f.strip()), int(t.strip())))
            except ValueError:
                pass
    return pairs


def render_mechanism(smiles: str, arrows_str: str = "") -> str:
    """[MECH] 渲染：SMILES + 电子流向 → 分子骨架 + 弯箭头 TikZ。"""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return "（机理渲染失败：rdkit 未安装）"

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return f"（机理渲染失败：无效 SMILES「{smiles}」）"
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

    pairs = _parse_arrows(arrows_str)

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

    # 原子标签 + 索引号
    for atom in mol.GetAtoms():
        x, y = pos(atom.GetIdx())
        lab = _atom_label(atom)
        if lab:
            lines.append(f"  \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")
        lines.append(f"  \\node[font=\\tiny, gray, below right] at ({x:.2f},{y:.2f}) {{{atom.GetIdx()}}};")

    # 弯箭头（红色 Bezier）
    for idx, (fi, ti) in enumerate(pairs):
        if fi >= mol.GetNumAtoms() or ti >= mol.GetNumAtoms():
            continue
        fx, fy = pos(fi)
        tx, ty = pos(ti)
        dx, dy = tx - fx, ty - fy
        L = math.hypot(dx, dy) or 1.0
        sign = 1 if idx % 2 == 0 else -1
        off = 0.6 * sign
        mx = (fx + tx) / 2 + (-dy / L) * off
        my = (fy + ty) / 2 + (dx / L) * off
        lines.append(
            f"  \\draw[->, thick, red] ({fx:.2f},{fy:.2f}) "
            f".. controls ({mx:.2f},{my:.2f}) .. ({tx:.2f},{ty:.2f});"
        )

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] SN2: CCl.[OH-] | 2>0,0>1")
    print(render_mechanism("CCl.[OH-]", "2>0,0>1"))
