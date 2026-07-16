# -*- coding: utf-8 -*-
"""renderers/retro.py — [RETRO] 标记渲染器：逆合成分析箭头（open arrow ⇒）。

逆合成箭头与正反应箭头（[ARROW] 的实心 ->）的视觉区别在于箭头是
**空心三角**（open/hollow head）：左侧目标分子（较复杂）⇒ 右侧合成子/
前体（较简单）。transform（断键/合成转化名）标于箭头上方。

空心箭头手绘实现（不依赖 arrows.meta，便于直接粘到 Overleaf）：
粗线止于箭头基部中心，再用 fill=white 的三角覆盖出空心头。
"""

from renderers.structure import smiles_to_chemfig


def _open_arrow_tail_to(p_node: str, base_xshift: str = "-10pt",
                        half_height: str = "4.5pt", tip_xshift: str = "2pt") -> list[str]:
    """生成从某节点 west 锚点出发的空心三角箭头片段。"""
    base = f"([xshift={base_xshift}]{p_node}.west)"
    top = f"([xshift={base_xshift},yshift={half_height}]{p_node}.west)"
    bottom = f"([xshift={base_xshift},yshift=-{half_height}]{p_node}.west)"
    tip = f"([xshift={tip_xshift}]{p_node}.west)"
    return [
        f"  \\draw[line width=1.1pt] (r.east) -- {base};",
        f"  \\draw[line width=1.1pt, fill=white] {top} -- {tip} -- {bottom} -- cycle;",
    ]


def render_retro(target_smi: str, precursor_smi: str, transform: str = None) -> str:
    """[RETRO] 渲染：目标分子 ⇒ 前体，空心三角逆合成箭头。

    任一 SMILES 无效/渲染失败时返回可读的错误提示字符串。
    """
    target = smiles_to_chemfig(target_smi)
    if target is None:
        return f"（逆合成箭头渲染失败：无效目标 SMILES「{target_smi}」）"
    precursor = smiles_to_chemfig(precursor_smi)
    if precursor is None:
        return f"（逆合成箭头渲染失败：无效前体 SMILES「{precursor_smi}」）"

    parts = [
        "\\begin{tikzpicture}",
        f"  \\node (r) at (0,0) {{{target}}};",
        f"  \\node (p) at (8,0) {{{precursor}}};",
    ]
    parts.extend(_open_arrow_tail_to("p"))
    if transform:
        parts.append(f"  \\node[above] at (4,0) {{\\small\\itshape {transform}}};")
    parts.append("\\end{tikzpicture}")
    return "\n".join(parts)


if __name__ == "__main__":
    print("[1] 苯甲醛⇒苯(甲酰化断键) + transform:")
    print(render_retro("O=Cc1ccccc1", "c1ccccc1", "formylation"))
    print("\n[2] 无 transform:")
    print(render_retro("O=Cc1ccccc1", "c1ccccc1"))
    print("\n[3] 无效目标:")
    print(render_retro("XYZ", "c1ccccc1"))
