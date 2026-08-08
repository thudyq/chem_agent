# -*- coding: utf-8 -*-
r"""renderers/reaction.py — [REACTION] 组合式反应方程式渲染器。

把多个反应物和多个产物当作独立的结构组件，由统一布局引擎（R-7，
renderers/layout.py）排成一行：反应物 + 反应物 + ... → 产物 + 产物 + ...，
箭头上方可标注反应条件。每个分子封装为独立 TikZ scope（局部坐标），
位置、加号、箭头全部由 layout_row 计算，避免重叠。

标记格式：
    [REACTION:反应物1;反应物2;...|产物1;产物2;...|反应条件]

示例：
    [REACTION:c1ccccc1;[O-][N+](=O)[O-]|O=[N+]([O-])c1ccccc1;O|H2SO4, 浓HNO3]
    [REACTION:C=C;O|CCO|H2SO4]

设计选择：
- 分子按结构简式绘制（非环碳写出 CH₃/CH₂/CH，环上碳保持键线式），不画孤对电子；
- 条件中的化学式（如 H2SO4）自动转下标；已含 $ 的文本保持原样。
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


_MOL_GAP = 1.6
_PLUS_W = 1.1
_ARR_W = 2.6
_ARR_PAD = 0.65
_MOL_SCALE = 0.8


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

    reactants = [s.strip() for s in reactants_str.split(";") if s.strip()]
    products = [s.strip() for s in products_str.split(";") if s.strip()]

    if not reactants:
        return "（反应式渲染失败：反应物不能为空）"
    if not products:
        return "（反应式渲染失败：产物不能为空）"

    mols = []
    for smi in reactants + products:
        mol = prepare_mol(smi)
        if mol is None:
            return f"（反应式渲染失败：无法为「{smi}」生成结构式）"
        scale_mol_coords(mol, _MOL_SCALE)
        mols.append(mol)

    items = []
    n_react = len(reactants)
    for i, mol in enumerate(mols):
        if i == n_react:
            items.append(("arrow", conditions.strip()))
        elif i > 0:
            items.append(("plus",))
        items.append(("mol", i, mol))
    layout = layout_row(items, mol_gap=_MOL_GAP, plus_w=_PLUS_W,
                        arrow_w=_ARR_W, arrow_pad=_ARR_PAD)

    lines = [r"\begin{tikzpicture}"]
    for placed in layout.mols:
        lines.extend(molecule_scope_lines(placed.mol, placed.shift,
                                          show_lone_pairs=False))
    for px in layout.pluses:
        lines.append(f"  \\node at ({px:.2f},0) {{$+$}};")

    main_arrow = layout.arrows[0]
    cond_text = format_chem_text(main_arrow.condition)
    if cond_text:
        lines.append(
            f"  \\draw[->, very thick] ({main_arrow.x1:.2f},0) -- ({main_arrow.x2:.2f},0) "
            f"node[midway, above] {{{cond_text}}};"
        )
    else:
        lines.append(
            f"  \\draw[->, very thick] ({main_arrow.x1:.2f},0) -- ({main_arrow.x2:.2f},0);"
        )

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("[REACTION] 组合式反应方程式测试")
    print("=" * 60)

    print("\n[1] 苯的硝化（条件含化学式，验证自动下标）：")
    print(render_reaction(
        "c1ccccc1;[O-][N+](=O)[O-]",
        "O=[N+]([O-])c1ccccc1;O",
        "H2SO4, 浓HNO3",
    ))

    print("\n[2] 加成反应（乙烯 + 水 → 乙醇）：")
    print(render_reaction(
        "C=C;O",
        "CCO",
        "H2SO4",
    ))

    print("\n[3] 单一反应物/产物：")
    print(render_reaction(
        "C=C",
        "CC",
        "",
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
