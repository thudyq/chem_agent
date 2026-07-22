# -*- coding: utf-8 -*-
r"""renderers/retro.py — [RETRO] 标记渲染器：逆合成分析箭头（open arrow ⇒）。

逆合成箭头与正反应箭头（[ARROW] 的实心 ->）的视觉区别在于箭头是
**空心三角**（open/hollow head）：左侧目标分子（较复杂）⇒ 右侧合成子/
前体（较简单）。transform（断键/合成转化名）标于箭头上方。
由统一布局引擎（R-7，renderers/layout.py）排布组件位置。

空心箭头手绘实现（粗线止于箭头基部，fill=white 三角覆盖出空心头）。
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


def _open_arrow_lines(x1: float, x2: float) -> list:
    """空心三角箭头：基部在 x2-0.35，尖端在 x2。"""
    base_x = x2 - 0.35
    return [
        f"  \\draw[line width=1.1pt] ({x1:.2f},0) -- ({base_x:.2f},0);",
        f"  \\draw[line width=1.1pt, fill=white] ({base_x:.2f},0.16) "
        f"-- ({x2:.2f},0) -- ({base_x:.2f},-0.16) -- cycle;",
    ]


def render_retro(target_smi: str, precursor_smi: str, transform: str = None) -> str:
    """[RETRO] 渲染：目标分子 ⇒ 前体，空心三角逆合成箭头。

    任一 SMILES 无效/渲染失败时返回可读的错误提示字符串。
    """
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（逆合成箭头渲染失败：rdkit 未安装）"

    target = prepare_mol(target_smi)
    if target is None:
        return f"（逆合成箭头渲染失败：无效目标 SMILES「{target_smi}」）"
    precursor = prepare_mol(precursor_smi)
    if precursor is None:
        return f"（逆合成箭头渲染失败：无效前体 SMILES「{precursor_smi}」）"
    scale_mol_coords(target, _MOL_SCALE)
    scale_mol_coords(precursor, _MOL_SCALE)

    layout = layout_row(
        [("mol", "t", target), ("arrow", ""), ("mol", "p", precursor)],
        arrow_w=_ARR_W, arrow_pad=_ARR_PAD,
    )
    main_arrow = layout.arrows[0]

    lines = ["\\begin{tikzpicture}"]
    for placed in layout.mols:
        lines.extend(molecule_scope_lines(placed.mol, placed.shift,
                                          show_lone_pairs=False))
    lines.extend(_open_arrow_lines(main_arrow.x1, main_arrow.x2))
    if transform:
        mid = (main_arrow.x1 + main_arrow.x2) / 2.0
        lines.append(
            f"  \\node[above] at ({mid:.2f},0) "
            f"{{\\small\\itshape {format_chem_text(transform)}}};"
        )
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 苯甲醛⇒苯(甲酰化断键) + transform:")
    print(render_retro("O=Cc1ccccc1", "c1ccccc1", "formylation"))
    print("\n[2] 无 transform:")
    print(render_retro("O=Cc1ccccc1", "c1ccccc1"))
    print("\n[3] 无效目标:")
    print(render_retro("XYZ", "c1ccccc1"))
