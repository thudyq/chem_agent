# -*- coding: utf-8 -*-
r"""renderers/retro.py — [RETRO] 标记渲染器：逆合成分析箭头（推导符号 ⇒）。

逆合成箭头与正反应箭头（[ARROW] 的实心 ->）的视觉区别：采用**双线推导
箭头**（\Longrightarrow 风格，TikZ 的 -implies + double），较粗；
长度与普通反应箭头一致（1.3）。左侧目标分子（较复杂）⇒ 右侧合成子/
前体（较简单），transform（断键/合成转化名）标于箭头上方。
由统一布局引擎（R-7，renderers/layout.py）排布组件位置。
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


# -*- coding: utf-8 -*-
r"""renderers/retro.py — [RETRO] 标记渲染器：逆合成分析箭头（推导符号 ⇒）。

逆合成箭头与正反应箭头（[ARROW] 的实心 ->）的视觉区别：采用**双线推导
箭头**（⇒，双线杆 + 实心箭头尖，不依赖 TikZ 箭头库），较粗；
长度与普通反应箭头一致（1.3）。左侧目标分子（较复杂）⇒ 右侧合成子/
前体（较简单），transform（断键/合成转化名）标于箭头上方。
由统一布局引擎（R-7，renderers/layout.py）排布组件位置。
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


_ARR_W = 2.6        # 与普通反应箭头一致的占位宽
_ARR_PAD = 0.65     # 实际箭头长度 = 2.6 - 2×0.65 = 1.3
_MOL_SCALE = 0.8
_TIP_LEN = 0.22     # 箭头尖长度


def _retro_arrow_lines(x1: float, x2: float) -> list:
    """双线推导箭头（⇒）：double 双线杆 + 实心三角箭头尖（较粗，无库依赖）。"""
    base = x2 - _TIP_LEN
    return [
        f"  \\draw[double distance=1.8pt, line width=0.9pt] ({x1:.2f},0) -- ({base:.2f},0);",
        f"  \\fill ({x2:.2f},0) -- ({base:.2f},0.13) -- ({base:.2f},-0.13) -- cycle;",
    ]


def render_retro(target_smi: str, precursor_smi: str, transform: str = None) -> str:
    """[RETRO] 渲染：目标分子 ⇒ 前体，双线推导箭头。

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
    lines.extend(_retro_arrow_lines(main_arrow.x1, main_arrow.x2))
    if transform:
        mid = (main_arrow.x1 + main_arrow.x2) / 2.0
        lines.append(
            f"  \\node[above] at ({mid:.2f},0) "
            f"{{\\small\\itshape {format_chem_text(transform)}}};"
        )
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 苯甲醛=>苯(甲酰化断键) + transform:")
    print(render_retro("O=Cc1ccccc1", "c1ccccc1", "formylation"))
    print("\n[2] 无 transform:")
    print(render_retro("O=Cc1ccccc1", "c1ccccc1"))
    print("\n[3] 无效目标:")
    print(render_retro("XYZ", "c1ccccc1"))
