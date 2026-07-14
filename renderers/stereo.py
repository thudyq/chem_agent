# -*- coding: utf-8 -*-
"""renderers/stereo.py — [STEREO] 标记渲染器：楔形式（wedge/dash）立体结构。

RDKit 解析含 @/@@ 的 SMILES → PrepareMolForDrawing（含坐标+楔形方向）→
对 BEGINWEDGE 画实心三角（前向）、BEGINDASH 画渐宽短横（后向）、
其余画普通线。不使用 mol2chemfig，完全 RDKit 2D 坐标自绘（与 Lewis 同模式）。
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


def render_stereo(smiles: str) -> str:
    """[STEREO] 渲染：含立体信息的 SMILES → 楔形式 TikZ。失败返回错误提示。"""
    try:
        from rdkit import Chem
        from rdkit.Chem import BondDir
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return "（立体结构渲染失败：rdkit 未安装）"

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return f"（立体结构渲染失败：无效 SMILES「{smiles}」）"
    # PrepareMolForDrawing 内含 Kekulize + Compute2DCoords + 设置楔形键方向（BondDir）
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

    lines = ["\\begin{tikzpicture}"]

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
        bdir = b.GetBondDir()

        if bdir == BondDir.BEGINWEDGE:
            # 实心楔形：窄端在 i，宽端在 j
            w = 0.09
            lines.append(
                f"  \\fill ({x1:.2f},{y1:.2f}) -- "
                f"({x2+px*w:.2f},{y2+py*w:.2f}) -- "
                f"({x2-px*w:.2f},{y2-py*w:.2f}) -- cycle;"
            )
        elif bdir == BondDir.BEGINDASH:
            # 虚楔形：5 条渐宽短横，从窄（i）到宽（j）
            for k in range(1, 6):
                t = k / 5
                w = 0.02 + 0.07 * t
                cx = x1 + (x2 - x1) * t
                cy = y1 + (y2 - y1) * t
                lines.append(
                    f"  \\draw[thick] ({cx+px*w:.2f},{cy+py*w:.2f}) -- "
                    f"({cx-px*w:.2f},{cy-py*w:.2f});"
                )
        else:
            lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")
            if order >= 2:
                d = 0.08
                lines.append(f"  \\draw ({x1+px*d:.2f},{y1+py*d:.2f}) -- ({x2+px*d:.2f},{y2+py*d:.2f});")
            if order >= 3:
                d = 0.08
                lines.append(f"  \\draw ({x1-px*d:.2f},{y1-py*d:.2f}) -- ({x2-px*d:.2f},{y2-py*d:.2f});")

    for atom in mol.GetAtoms():
        lab = _atom_label(atom)
        if lab:
            x, y = pos(atom.GetIdx())
            lines.append(f"  \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] (R)-乳酸 C[C@H](O)C(=O)O:")
    print(render_stereo("C[C@H](O)C(=O)O"))
    print("\n[2] (S)-2-氨基丁酸 CC[C@@H](N)C(=O)O:")
    print(render_stereo("CC[C@@H](N)C(=O)O"))
