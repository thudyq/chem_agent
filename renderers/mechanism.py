# -*- coding: utf-8 -*-
"""renderers/mechanism.py — [MECH] 标记渲染器：在分子上画电子推进弯箭头。

MVP 方案：LLM 指定 SMILES + 原子索引间的电子流向（from>to），
renderer 画分子骨架 + 红色 Bezier 弯箭头。原子索引按 SMILES 出现顺序（0 起）。
renderer 额外标注索引号，便于核对箭头指向。

标记格式：[MECH:SMILES|from>to,from>to,...]
示例：[MECH:CCl.[OH-]|2>0,0>1]  （O(2)进攻C(0)，C(0)-Cl(1)键断裂）
"""

import math

try:
    from renderers._mol_base import atom_label, atom_pos, prepare_mol, bond_segments
except ImportError:  # noqa: E722
    from _mol_base import atom_label, atom_pos, prepare_mol, bond_segments


def _parse_arrows(arrows_str):
    """'2>0,0>1' → [(2,0,'standard'),(0,1,'standard')]
    '2>>0' → [(2,0,'fishhook')]（鱼钩/单电子箭头）"""
    pairs = []
    for s in arrows_str.split(","):
        s = s.strip()
        if ">>" in s:
            try:
                f, t = s.split(">>", 1)
                pairs.append((int(f.strip()), int(t.strip()), "fishhook"))
            except ValueError:
                pass
        elif ">" in s:
            try:
                f, t = s.split(">", 1)
                pairs.append((int(f.strip()), int(t.strip()), "standard"))
            except ValueError:
                pass
    return pairs


def render_mechanism(smiles: str, arrows_str: str = "") -> str:
    """[MECH] 渲染：SMILES + 电子流向 → 分子骨架 + 弯箭头 TikZ。"""
    try:
        from rdkit import Chem
    except ImportError:
        return "（机理渲染失败：rdkit 未安装）"

    mol = prepare_mol(smiles)
    if mol is None:
        return f"（机理渲染失败：无效 SMILES「{smiles}」）"

    pairs = _parse_arrows(arrows_str)

    lines = ["\\begin{tikzpicture}"]

    # 骨架键
    for segs in bond_segments(mol, label_margin=0.25, bond_gap=0.08):
        for x1, y1, x2, y2 in segs:
            lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")

    # 原子标签 + 索引号
    for atom in mol.GetAtoms():
        x, y = atom_pos(mol, atom.GetIdx())
        lab = atom_label(atom)
        if lab:
            lines.append(f"  \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")
        lines.append(f"  \\node[font=\\tiny, gray, below right] at ({x:.2f},{y:.2f}) {{{atom.GetIdx()}}};")

    # 弯箭头（红色 Bezier）
    for idx, (fi, ti, atype) in enumerate(pairs):
        if fi >= mol.GetNumAtoms() or ti >= mol.GetNumAtoms():
            continue
        fx, fy = atom_pos(mol, fi)
        tx, ty = atom_pos(mol, ti)
        dx, dy = tx - fx, ty - fy
        L = math.hypot(dx, dy) or 1.0
        sign = 1 if idx % 2 == 0 else -1
        off = 0.6 * sign
        mx = (fx + tx) / 2 + (-dy / L) * off
        my = (fy + ty) / 2 + (dx / L) * off
        if atype == "fishhook":
            # 鱼钩箭头：曲线（无 -> 全箭头）+ 单边半 barb
            lines.append(
                f"  \\draw[thick, red] ({fx:.2f},{fy:.2f}) "
                f".. controls ({mx:.2f},{my:.2f}) .. ({tx:.2f},{ty:.2f});"
            )
            incoming = math.atan2(ty - my, tx - mx)
            barb_ang = incoming + math.pi + math.radians(25)
            blen = 0.18
            bx = tx + blen * math.cos(barb_ang)
            by = ty + blen * math.sin(barb_ang)
            lines.append(f"  \\draw[thick, red] ({tx:.2f},{ty:.2f}) -- ({bx:.2f},{by:.2f});")
        else:
            lines.append(
                f"  \\draw[->, thick, red] ({fx:.2f},{fy:.2f}) "
                f".. controls ({mx:.2f},{my:.2f}) .. ({tx:.2f},{ty:.2f});"
            )

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] SN2: CCl.[OH-] | 2>0,0>1")
    print(render_mechanism("CCl.[OH-]", "2>0,0>1"))
