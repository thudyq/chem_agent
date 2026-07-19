# -*- coding: utf-8 -*-
"""renderers/arrow.py — [ARROW] 标记渲染器：反应物 → 产物 的带箭头反应式。

用 tikzpicture 显式坐标布局：反应物节点 (0,0)、产物节点 (8,0)，
\\draw[->] 连接，反应类型标注置于箭头下方中点。相比 chemfig \\schemestart
的自动间距，显式坐标可控、无重叠，且标注在下方符合阅读习惯。
复用 structure.smiles_to_chemfig 渲染两侧结构。
"""

from .structure import smiles_to_chemfig


def render_arrow(reactant_smi: str, product_smi: str, reaction_type: str = None) -> str:
    """[ARROW] 渲染：反应物 + 产物 + 反应类型 → TikZ 反应式。

    任一 SMILES 无效/渲染失败时返回可读的错误提示字符串。
    """
    reactant = smiles_to_chemfig(reactant_smi)
    if reactant is None:
        return f"（反应箭头渲染失败：无效反应物 SMILES「{reactant_smi}」）"
    product = smiles_to_chemfig(product_smi)
    if product is None:
        return f"（反应箭头渲染失败：无效产物 SMILES「{product_smi}」）"

    # 反应物 (0,0)、产物 (8,0)，draw[->] 连接节点边界；标注置于箭头下方中点
    parts = [
        "\\begin{tikzpicture}",
        f"  \\node (r) at (0,0) {{{reactant}}};",
        f"  \\node (p) at (8,0) {{{product}}};",
        "  \\draw[->, thick, shorten >=3pt, shorten <=3pt] (r) -- (p);",
    ]
    if reaction_type:
        parts.append(f"  \\node[below] at (4,0) {{\\small\\itshape {reaction_type}}};")
    parts.append("\\end{tikzpicture}")
    return "\n".join(parts)


if __name__ == "__main__":
    print("[1] 苯→苯胺(amination):")
    print(render_arrow("c1ccccc1", "c1ccccc1N", "amination"))
    print("\n[2] 无类型:")
    print(render_arrow("c1ccccc1", "c1ccccc1N"))
    print("\n[3] 无效反应物:")
    print(render_arrow("XYZ", "c1ccccc1N", "test"))
