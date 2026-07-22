# -*- coding: utf-8 -*-
r"""renderers/arrow.py — [ARROW] 标记渲染器：反应物 → 产物 的带箭头反应式。

由统一布局引擎（R-7，renderers/layout.py）排布：反应物 → 产物，
反应类型标注置于箭头下方中点。每个分子封装为独立 TikZ scope（局部坐标）。

标记格式：[ARROW:反应物SMILES,产物SMILES,反应类型]
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from renderers.mol_primitives import format_chem_text, prepare_mol, scale_mol_coords
    from renderers.layout import layout_row, molecule_scope_lines
else:
    from .mol_primitives import format_chem_text, prepare_mol, scale_mol_coords
    from .layout import layout_row, molecule_scope_lines


_ARR_W = 3.0
_ARR_PAD = 0.5
_MOL_SCALE = 0.8


def render_arrow(reactant_smi: str, product_smi: str, reaction_type: str = None) -> str:
    """[ARROW] 渲染：反应物 + 产物 + 反应类型 → TikZ 反应式。

    任一 SMILES 无效/渲染失败时返回可读的错误提示字符串。
    """
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（反应箭头渲染失败：rdkit 未安装）"

    reactant = prepare_mol(reactant_smi)
    if reactant is None:
        return f"（反应箭头渲染失败：无效反应物 SMILES「{reactant_smi}」）"
    product = prepare_mol(product_smi)
    if product is None:
        return f"（反应箭头渲染失败：无效产物 SMILES「{product_smi}」）"
    scale_mol_coords(reactant, _MOL_SCALE)
    scale_mol_coords(product, _MOL_SCALE)

    layout = layout_row(
        [("mol", "r", reactant), ("arrow", ""), ("mol", "p", product)],
        arrow_w=_ARR_W, arrow_pad=_ARR_PAD,
    )
    main_arrow = layout.arrows[0]

    lines = ["\\begin{tikzpicture}"]
    for placed in layout.mols:
        lines.extend(molecule_scope_lines(placed.mol, placed.shift,
                                          show_lone_pairs=False))
    lines.append(
        f"  \\draw[->, thick] ({main_arrow.x1:.2f},0) -- ({main_arrow.x2:.2f},0);"
    )
    if reaction_type:
        mid = (main_arrow.x1 + main_arrow.x2) / 2.0
        lines.append(
            f"  \\node[below] at ({mid:.2f},0) "
            f"{{\\small\\itshape {format_chem_text(reaction_type)}}};"
        )
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 苯→苯胺(amination):")
    print(render_arrow("c1ccccc1", "c1ccccc1N", "amination"))
    print("\n[2] 无类型:")
    print(render_arrow("c1ccccc1", "c1ccccc1N"))
    print("\n[3] 无效反应物:")
    print(render_arrow("XYZ", "c1ccccc1N", "test"))
