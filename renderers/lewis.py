# -*- coding: utf-8 -*-
"""renderers/lewis.py — [LEWIS] 标记渲染器：Lewis 结构式（显示孤对电子）。

与 STRUCT（mol2chemfig 骨架式）不同，Lewis 需显示孤对电子点，而 mol2chemfig 的
布局坐标与 RDKit 不一致——故 Lewis 完全用 RDKit 2D 坐标自绘：键、原子标签、
孤对电子点统一坐标系，保证对齐。

孤对电子的计数、正交优先摆放、固定点距与符号中心修正与其他机理渲染器
共享同一套逻辑（renderers/mol_primitives.py）。
"""

from .mol_primitives import (
    _label_flip_for, atom_main_label, bond_segments, charge_tikz,
    label_bond_margin, label_node_pos, lone_pair_tikz, mol_visual_bbox,
    prepare_mol, wrap_format_text,
)


def render_lewis(smiles: str, label: str = None) -> str:
    """[LEWIS] 渲染：SMILES → 含孤对电子的 Lewis 结构式 TikZ。失败返回错误提示。

    label 可选：置于结构下方（如 水、H₂O——中文/化学式名称）。
    """
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（Lewis 渲染失败：rdkit 未安装）"

    mol = prepare_mol(smiles, add_hs=True, kekulize=True, use_prepare=False)
    if mol is None:
        return f"（Lewis 渲染失败：无效 SMILES「{smiles}」）"

    lines = ["\\begin{tikzpicture}"]

    # 键：与标签绘制端同一 labeler（atom_main_label）计算留白——否则键线式
    # 口径（atom_label）对非环碳返回 None → 碳标签 "C" 无留白、键线从标签
    # 中心穿出；留白按 label_bond_margin 分档（单字符 0.30），Lewis 标签
    # 基本为单/双字符，数值与原固定 0.30 一致（碳标签是新获得留白者）。
    for segs in bond_segments(mol, labeler=atom_main_label,
                              margin_fn=label_bond_margin, bond_gap=0.09):
        for x1, y1, x2, y2 in segs:
            lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")

    # 原子标签（主标签不含电荷）+ 圆圈电荷（右上角）
    for atom in mol.GetAtoms():
        lab = atom_main_label(atom, flip=_label_flip_for(mol, atom.GetIdx()))
        if lab:
            nx, ny = label_node_pos(mol, atom.GetIdx())
            lines.append(f"  \\node[fill=white,inner sep=1pt] at ({nx:.2f},{ny:.2f}) {{{lab}}};")
        charge = charge_tikz(mol, atom.GetIdx())
        if charge:
            lines.append(f"  {charge}")

    # 孤对电子点（共享逻辑：正交优先、点距 0.24、绕元素符号中心）
    for atom in mol.GetAtoms():
        for dot_line in lone_pair_tikz(mol, atom.GetIdx()):
            lines.append(f"  {dot_line}")

    if label:
        # label 置于结构正下方（含孤对电子点外延——底部有电子点时 label 不压点）
        min_x, min_y, max_x, _ = mol_visual_bbox(mol, include_lone_pairs=True)
        text = wrap_format_text(label)
        align = "align=center, " if "\\\\" in text else ""
        lines.append(
            f"  \\node[{align}below] at ({(min_x + max_x) / 2.0:.2f},{min_y - 0.15:.2f}) "
            f"{{{text}}};"
        )

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
