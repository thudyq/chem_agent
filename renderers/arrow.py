# -*- coding: utf-8 -*-
"""renderers/arrow.py — [ARROW] 标记渲染器：反应物 → 产物 的带箭头反应式。

输出 chemfig 的 \\schemestart...\\schemestop 反应式，反应物与产物结构之间
用 \\arrow{->[反应类型]} 连接。复用 structure.smiles_to_chemfig 渲染两侧结构。
"""

from renderers.structure import smiles_to_chemfig


def render_arrow(reactant_smi: str, product_smi: str, reaction_type: str = None) -> str:
    """[ARROW] 渲染：反应物 SMILES + 产物 SMILES + 反应类型 → TikZ 反应式。

    任一 SMILES 无效/渲染失败时返回可读的错误提示字符串。

    参数:
        reactant_smi: 反应物 SMILES。
        product_smi: 产物 SMILES。
        reaction_type: 反应类型标注（如 "amination"），置于箭头上方；可空。

    返回:
        \\schemestart...\\schemestop TikZ 代码；失败返回 "（反应箭头渲染失败：...）"。
    """
    reactant = smiles_to_chemfig(reactant_smi)
    if reactant is None:
        return f"（反应箭头渲染失败：无效反应物 SMILES「{reactant_smi}」）"
    product = smiles_to_chemfig(product_smi)
    if product is None:
        return f"（反应箭头渲染失败：无效产物 SMILES「{product_smi}」）"

    # chemfig scheme 箭头：有类型标注写 \arrow{->[\small{类型}]}，无则 \arrow{->}
    if reaction_type:
        arrow = f"\\arrow{{->[\\small{{{reaction_type}}}]}}"
    else:
        arrow = "\\arrow{->}"

    return f"\\schemestart\n{reactant}\n{arrow}\n{product}\n\\schemestop"


if __name__ == "__main__":
    print("[1] 苯→苯胺(amination):")
    print(render_arrow("c1ccccc1", "c1ccccc1N", "amination"))
    print("\n[2] 无类型:")
    print(render_arrow("c1ccccc1", "c1ccccc1N"))
    print("\n[3] 无效反应物:")
    print(render_arrow("XYZ", "c1ccccc1N", "test"))
