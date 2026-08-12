# -*- coding: utf-8 -*-
r"""renderers/reaction.py — [REACTION] 组合式反应方程式渲染器。

把多个反应物和多个产物当作独立的结构组件，由统一布局引擎（R-7，
renderers/layout.py）排成一行：反应物 + 反应物 + ... → 产物 + 产物 + ...，
箭头上方可标注反应条件。每个分子封装为独立 TikZ scope（局部坐标），
位置、加号、箭头全部由 layout_row 计算，避免重叠。

标记格式：
    [REACTION:反应物1;反应物2;...|产物1;产物2;...|反应条件]

    示例（2a 完整配平）：
    [REACTION:c1ccccc1;[O-][N+](=O)O|O=[N+]([O-])c1ccccc1;O|H2SO4, Δ]
    [REACTION:C=C;O|CCO|H2SO4]
    示例（2b 箭头补足：省略物种写在条件里，无符号=补反应物侧、-X=补产物侧）：
    [REACTION:CC(=O)O;CCO|CC(=O)OCC|浓H2SO4, Δ, -H2O]

设计选择：
- 分子按结构简式绘制（非环碳写出 CH₃/CH₂/CH，环上碳保持键线式），不画孤对电子；
- 条件中的化学式（如 H2SO4）自动转下标；已含 $ 的文本保持原样；
- 校验由 core.tag_validator 完成（2a 全元素+电荷守恒；2b 具体物质补足差额）。
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from renderers.mol_primitives import prepare_mol, scale_mol_coords, \
        split_arrow_condition, split_species_coeff, wrap_format_text
    from renderers.layout import layout_row, molecule_scope_lines
else:
    from .mol_primitives import prepare_mol, scale_mol_coords, \
        split_arrow_condition, split_species_coeff, wrap_format_text
    from .layout import layout_row, molecule_scope_lines


_MOL_GAP = 1.6
_PLUS_W = 1.1
_ARR_W = 2.6
_ARR_PAD = 0.65
_MOL_SCALE = 0.8


def _fmt_coeff(c: float) -> str:
    """系数显示：整数原样、n/2 分数形式（1/2、3/2）。"""
    if c == int(c):
        return str(int(c))
    return f"{int(c * 2)}/2"


def render_reaction(reactants_str: str, products_str: str, conditions: str = "") -> str:
    r"""[REACTION] 渲染：多反应物 + 多产物 + 条件 → 完整反应式 TikZ。

    参数:
        reactants_str: 分号分隔的反应物 SMILES。
        products_str: 分号分隔的产物 SMILES。
        conditions: 反应条件文本，可选；会显示在箭头上方。

    返回:
        可编译的 TikZ 代码；失败返回可读错误提示。
    """
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（反应式渲染失败：rdkit 未安装）"

    reactants = split_species_coeff(reactants_str)
    products = split_species_coeff(products_str)

    if not reactants:
        return "（反应式渲染失败：反应物不能为空）"
    if not products:
        return "（反应式渲染失败：产物不能为空）"

    all_species = reactants + products
    species = []
    for coeff, smi in all_species:
        mol = prepare_mol(smi)
        if mol is None:
            return f"（反应式渲染失败：无法为「{smi}」生成结构式）"
        scale_mol_coords(mol, _MOL_SCALE)
        species.append((coeff, mol))

    items = []
    n_react = len(reactants)
    for i, (coeff, mol) in enumerate(species):
        if i == n_react:
            items.append(("arrow", conditions.strip()))
        elif i > 0:
            items.append(("plus",))
        items.append(("mol", i, mol, coeff))
    layout = layout_row(items, mol_gap=_MOL_GAP, plus_w=_PLUS_W,
                        arrow_w=_ARR_W, arrow_pad=_ARR_PAD)

    lines = [r"\begin{tikzpicture}"]
    for placed in layout.mols:
        if placed.coeff != 1:
            # 系数节点：紧贴分子 bbox 左缘（布局已预留 coeff_pad=0.6 空间）。
            # 系数半宽约 0.15，中心放 bbox 左缘 -0.15 → 右缘恰好贴住标签区
            # （原 -0.35 使系数离标签过远、离加号过近，Drawbacks 十-1）。
            bbox = placed.bbox
            bx = bbox[0] + placed.shift[0] - 0.15
            lines.append(f"  \\node at ({bx:.2f},0) {{{_fmt_coeff(placed.coeff)}}};")
        lines.extend(molecule_scope_lines(placed.mol, placed.shift,
                                          show_lone_pairs=False))
    for px in layout.pluses:
        lines.append(f"  \\node at ({px:.2f},0) {{$+$}};")

    main_arrow = layout.arrows[0]
    above, below = split_arrow_condition(main_arrow.condition)
    arrow_node = f"  \\draw[->, very thick] ({main_arrow.x1:.2f},0) -- ({main_arrow.x2:.2f},0)"
    # 条件分上下：-X 补足（产物侧）在箭头下方，其余在上方（上方末尾无逗号）
    parts = []
    for text, pos in ((above, "above"), (below, "below")):
        t = wrap_format_text(text)
        if not t:
            continue
        align = "align=center, " if "\\\\" in t else ""
        parts.append(f"node[midway, {align}{pos}] {{{t}}}")
    if parts:
        lines.append(arrow_node + " " + " ".join(parts) + ";")
    else:
        lines.append(arrow_node + ";")

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("[REACTION] 组合式反应方程式测试")
    print("=" * 60)

    print("\n[1] 苯的硝化（HNO3 已写入反应物，条件仅催化剂 H2SO4）：")
    print(render_reaction(
        "c1ccccc1;[O-][N+](=O)O",
        "O=[N+]([O-])c1ccccc1;O",
        "H2SO4, Δ",
    ))

    print("\n[2] 加成反应（乙烯 + 水 → 乙醇）：")
    print(render_reaction(
        "C=C;O",
        "CCO",
        "H2SO4",
    ))

    print("\n[3] 酯化 2b 箭头补足（省略产物水，-H2O 写在条件里补产物侧）：")
    print(render_reaction(
        "CC(=O)O;CCO",
        "CC(=O)OCC",
        "浓H2SO4, Δ, -H2O",
    ))

    print("\n[4] 无效反应物：")
    print(render_reaction(
        "",
        "CC",
        "",
    ))

    print("\n[5] 无法渲染的 SMILES：")
    print(render_reaction(
        "XYZ_INVALID",
        "CC",
        "",
    ))
