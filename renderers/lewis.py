# -*- coding: utf-8 -*-
"""renderers/lewis.py — [LEWIS] 标记渲染器：Lewis 结构式（显示孤对电子）。

与 STRUCT（mol2chemfig 骨架式）不同，Lewis 需显示孤对电子点，而 mol2chemfig 的
布局坐标与 RDKit 不一致——故 Lewis 完全用 RDKit 2D 坐标自绘：键、原子标签、
孤对电子点统一坐标系，保证对齐。

孤对电子的计数、正交优先摆放、固定点距与符号中心修正与其他机理渲染器
共享同一套逻辑（renderers/mol_primitives.py）。
"""

from .mol_primitives import (
    atom_main_label, atom_pos, bond_segments, charge_tikz, lone_pair_tikz,
    prepare_mol,
)


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

    # 原子标签（主标签不含电荷）+ 圆圈电荷（右上角）
    for atom in mol.GetAtoms():
        lab = atom_main_label(atom)
        if lab:
            x, y = atom_pos(mol, atom.GetIdx())
            lines.append(f"  \\node[fill=white,inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")
        charge = charge_tikz(mol, atom.GetIdx())
        if charge:
            lines.append(f"  {charge}")

    # 孤对电子点（共享逻辑：正交优先、点距 0.30、绕元素符号中心）
    for atom in mol.GetAtoms():
        for dot_line in lone_pair_tikz(mol, atom.GetIdx()):
            lines.append(f"  {dot_line}")

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 水 O:")
    print(render_lewis("O"))
    print("\n[2] 氨 N:")
    print(render_lewis("N"))
    print("\n[3] 甲醇 CO:")
    print(render_lewis("CO"))
    print("\n[4] 无效 SMILES（应降级提示）:")
    print(render_lewis("XYZ"))
