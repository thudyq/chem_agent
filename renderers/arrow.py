# -*- coding: utf-8 -*-
r"""renderers/arrow.py — [ARROW] 标记渲染器：反应物 → 产物 的带箭头反应式。

由统一布局引擎（R-7，renderers/layout.py）排布：反应物 → 产物，
反应类型/条件标注置于箭头上方中点（与 REACTION 主箭头一致，非斜体）。
每个分子封装为独立 TikZ scope（局部坐标）。

标记格式：[ARROW:反应物SMILES,产物SMILES,反应类型/条件]
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from renderers.mol_primitives import prepare_mol, scale_mol_coords, \
        split_species_coeff, wrap_format_text, main_arrow_lines, \
        is_formula_label
    from renderers.layout import layout_row, molecule_scope_lines
else:
    from .mol_primitives import prepare_mol, scale_mol_coords, \
        split_species_coeff, wrap_format_text, main_arrow_lines, \
        is_formula_label
    from .layout import layout_row, molecule_scope_lines


_ARR_W = 2.6        # 与 REACTION/COMPOSITE 主反应箭头一致的占位宽
_ARR_PAD = 0.65     # 实际箭头长度 = 2.6 - 2×0.65 = 1.3
_MOL_SCALE = 0.8


def _fmt_coeff(c: float) -> str:
    """系数显示：整数原样、n/2 分数形式（1/2、3/2）。"""
    if c == int(c):
        return str(int(c))
    return f"{int(c * 2)}/2"


def render_arrow(reactant_smi: str, product_smi: str, reaction_type: str = None) -> str:
    """[ARROW] 渲染：反应物 + 产物 + 反应类型 → TikZ 反应式。

    任一 SMILES 无效/渲染失败时返回可读的错误提示字符串。
    """
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（反应箭头渲染失败：rdkit 未安装）"

    r_coeff, r_smi = split_species_coeff(reactant_smi)[0] \
        if split_species_coeff(reactant_smi) else (1, reactant_smi)
    p_coeff, p_smi = split_species_coeff(product_smi)[0] \
        if split_species_coeff(product_smi) else (1, product_smi)

    reactant = prepare_mol(r_smi)
    r_text = None
    if reactant is None:
        if is_formula_label(r_smi):
            r_text = r_smi
        else:
            return f"（反应箭头渲染失败：无效反应物 SMILES「{r_smi}」）"
    product = prepare_mol(p_smi)
    p_text = None
    if product is None:
        if is_formula_label(p_smi):
            p_text = p_smi
        else:
            return f"（反应箭头渲染失败：无效产物 SMILES「{p_smi}」）"
    items = []
    if reactant is not None:
        scale_mol_coords(reactant, _MOL_SCALE)
        items.append(("mol", "r", reactant, r_coeff))
    else:
        items.append(("text", "r", r_text, r_coeff))
    items.append(("arrow", ""))
    if product is not None:
        scale_mol_coords(product, _MOL_SCALE)
        items.append(("mol", "p", product, p_coeff))
    else:
        items.append(("text", "p", p_text, p_coeff))

    layout = layout_row(items, arrow_w=_ARR_W, arrow_pad=_ARR_PAD)
    main_arrow = layout.arrows[0]

    lines = ["\\begin{tikzpicture}"]
    for placed in layout.mols:
        if placed.coeff != 1:
            bbox = placed.bbox
            bx = bbox[0] + placed.shift[0] - 0.35
            lines.append(f"  \\node at ({bx:.2f},0) {{{_fmt_coeff(placed.coeff)}}};")
        lines.extend(molecule_scope_lines(placed.mol, placed.shift,
                                          show_lone_pairs=False))
    for placed in layout.texts:
        if placed.coeff != 1:
            bbox = placed.bbox
            bx = bbox[0] + placed.shift[0] - 0.35
            lines.append(f"  \\node at ({bx:.2f},0) {{{_fmt_coeff(placed.coeff)}}};")
        lines.append(
            f"  \\node[fill=white, inner sep=1pt] at "
            f"({placed.shift[0]:.2f},{placed.shift[1]:.2f}) "
            f"{{{wrap_format_text(placed.text)}}};"
        )
    # 主反应箭头（单向 → / 双向 ⇌ 统一）：共享函数与 reaction/composite 三处
    # 共用（三合一，20260816）；⇌ 令牌在条件中识别并剥离
    lines.extend(main_arrow_lines(main_arrow.x1, main_arrow.x2,
                                  reaction_type, style="thick"))
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 苯→苯胺(amination):")
    print(render_arrow("c1ccccc1", "c1ccccc1N", "amination"))
    print("\n[2] 乙烯→乙烷（单→单加氢，ARROW 骨架展示，H2 省略）：")
    print(render_arrow("C=C", "CC", "H2, Ni"))
    print("\n[3] 无类型:")
    print(render_arrow("c1ccccc1", "c1ccccc1N"))
    print("\n[4] 无效反应物:")
    print(render_arrow("XYZ", "c1ccccc1N", "test"))
