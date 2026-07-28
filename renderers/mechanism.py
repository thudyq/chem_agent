# -*- coding: utf-8 -*-
"""renderers/mechanism.py — [MECH] 标记渲染器：在分子上画电子推进弯箭头。

LLM 指定 SMILES + 原子/键间的电子流向，renderer 画结构简式骨架 + 红色
Bezier 弯箭头。原子索引按 SMILES 出现顺序（0 起）。

标记格式：[MECH:SMILES|from>to,from>to,...] 或 [MECH:SMILES|箭头|numbering]
端点引用：from/to 可以是原子序号（如 2>0），也可以是键中点 "a-b"（σ 键
断裂箭头从键发出，如 0-1>1）。
示例：[MECH:CCl.[OH-]|2>0,0-1>1]  （O(2)孤对电子进攻C(0)，C(0)-Cl(1)键断裂）

杂原子起点自动上移到孤对电子区域；键中点出发的箭头向下弯，其余向上弯。
默认不显示原子序号；仅当碳原子较多、需要指明参与反应的原子时，
在第三段写 numbering 打开序号标注（调试/核对用途）。
"""

from .mol_primitives import (
    atom_main_label, atom_pos, bond_segments, charge_tikz, label_bond_margin,
    lone_pair_tikz, mech_arrow_between, mech_arrow_origin, prepare_mol,
    scale_mol_coords,
)

_MOL_SCALE = 0.8    # 分子坐标缩放因子（与其他机理渲染器一致）


def _parse_arrows(arrows_str):
    """'2>0,0-1>1' → [('2','0','standard'),('0-1','1','standard')]
    '2>>0' → [('2','0','fishhook')]（鱼钩/单电子箭头）"""
    pairs = []
    for s in arrows_str.split(","):
        s = s.strip()
        if ">>" in s:
            f, _, t = s.partition(">>")
            pairs.append((f.strip(), t.strip(), "fishhook"))
        elif ">" in s:
            f, _, t = s.partition(">")
            pairs.append((f.strip(), t.strip(), "standard"))
    return [(f, t, k) for f, t, k in pairs if f and t]


def render_mechanism(smiles: str, arrows_str: str = "", flags: str = "") -> str:
    """[MECH] 渲染：SMILES + 电子流向 → 结构简式骨架 + 弯箭头 TikZ。

    参数:
        smiles: 分子 SMILES（可含 . 分隔的多组分）。
        arrows_str: 电子流向，如 "2>0,0-1>1"（>> 为鱼钩箭头）。
        flags: 可选标志，含 "numbering" 时显示原子序号。
    """
    try:
        from rdkit import Chem
    except ImportError:
        return "（机理渲染失败：rdkit 未安装）"

    mol = prepare_mol(smiles)
    if mol is None:
        return f"（机理渲染失败：无效 SMILES「{smiles}」）"
    scale_mol_coords(mol, _MOL_SCALE)

    pairs = _parse_arrows(arrows_str)
    show_numbers = "numbering" in (flags or "")

    lines = ["\\begin{tikzpicture}"]

    for segs in bond_segments(mol, labeler=atom_main_label,
                              margin_fn=label_bond_margin):
        for x1, y1, x2, y2 in segs:
            lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")

    for atom in mol.GetAtoms():
        x, y = atom_pos(mol, atom.GetIdx())
        lab = atom_main_label(atom)
        if lab:
            lines.append(f"  \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")
        charge = charge_tikz(mol, atom.GetIdx())
        if charge:
            lines.append(f"  {charge}")
        if show_numbers:
            lines.append(f"  \\node[font=\\tiny, gray, below right] at ({x:.2f},{y:.2f}) {{{atom.GetIdx()}}};")

    for atom in mol.GetAtoms():
        for dot_line in lone_pair_tikz(mol, atom.GetIdx()):
            lines.append(f"  {dot_line}")

    for fs, ts, atype in pairs:
        p1 = mech_arrow_origin(mol, ts, lone_pair_offset=False)
        if p1 is None:
            continue
        p0 = mech_arrow_origin(mol, fs, toward=(p1[0], p1[1]),
                               prefer_single=(atype == "fishhook"))
        if p0 is None:
            continue
        inset_start = 0.0 if p0[3] else (0.05 if p0[2] else 0.15)
        lines.extend(
            mech_arrow_between(p0[0], p0[1], p1[0], p1[1], atype,
                               from_bond=p0[2], inset_start=inset_start)
        )

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] SN2: CCl.[OH-] | 2>0,0-1>1（默认无序号）")
    print(render_mechanism("CCl.[OH-]", "2>0,0-1>1"))
    print()
    print("[2] 同上，numbering 打开序号")
    print(render_mechanism("CCl.[OH-]", "2>0,0-1>1", "numbering"))
