# -*- coding: utf-8 -*-
"""renderers/resonance.py — [RESONANCE] 标记渲染器：共振结构式。

LLM 提供多个共振极限式的 SMILES（~ 分隔），renderer 用统一布局引擎
（R-7，renderers/layout.py）把各极限式排成一行，用 ↔（共振符号，
非平衡箭头）连接。共振涉及电子离域，画出孤对电子点。

标记格式：[RESONANCE:SMILES1~SMILES2~SMILES3]
示例：[RESONANCE:C1=CC=CC=C1~C1=CC=CC=C1]（苯的两个 Kekulé 式）
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from renderers.mol_primitives import prepare_mol, scale_mol_coords
    from renderers.layout import layout_row, molecule_scope_lines
else:
    from .mol_primitives import prepare_mol, scale_mol_coords
    from .layout import layout_row, molecule_scope_lines


_RES_W = 1.1
_MOL_SCALE = 0.8


def render_resonance(content: str) -> str:
    """[RESONANCE] 渲染：多个共振式 SMILES → TikZ 横向排列 + ↔。"""
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（共振式渲染失败：rdkit 未安装）"

    if not content or not isinstance(content, str):
        return "（共振式渲染失败：内容为空）"

    smiles_list = [s.strip() for s in content.split("~") if s.strip()]
    if len(smiles_list) < 2:
        return "（共振式渲染失败：至少需要 2 个共振极限式，用 ~ 分隔）"

    mols = []
    for smi in smiles_list:
        mol = prepare_mol(smi)
        if mol is None:
            return f"（共振式渲染失败：无效 SMILES「{smi}」）"
        scale_mol_coords(mol, _MOL_SCALE)
        mols.append(mol)

    items = []
    for i, mol in enumerate(mols):
        if i > 0:
            items.append(("resarrow",))
        items.append(("mol", i, mol))
    layout = layout_row(items, res_w=_RES_W)

    lines = ["\\begin{tikzpicture}"]
    for placed in layout.mols:
        lines.extend(molecule_scope_lines(placed.mol, placed.shift))
    for rx in layout.resarrows:
        lines.append(f"  \\node[font=\\large] at ({rx:.2f},0) {{$\\leftrightarrow$}};")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 苯 Kekulé:")
    print(render_resonance("C1=CC=CC=C1~C1C=CC=CC=1"))
    print("\n[2] 羧酸根:")
    print(render_resonance("CC(=O)[O-]~CC([O-])=O"))
