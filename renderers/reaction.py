# -*- coding: utf-8 -*-
r"""renderers/reaction.py — [REACTION] 组合式反应方程式渲染器。

把多个反应物和多个产物当作独立的结构组件，用 TikZ matrix 横向排列成一幅
完整反应式：反应物 + 反应物 + ... → 产物 + 产物 + ...，箭头上方可标注反应条件。

使用 tikzpicture + matrix 而非 \schemestart，避免 chemfig 内部对齐选项的
限制与编译错误，同时获得对节点间距、箭头长度、垂直居中的完全控制；
matrix 是 TikZ 内置功能，无需额外 \usetikzlibrary。

标记格式：
    [REACTION:反应物1;反应物2;...|产物1;产物2;...|反应条件]

示例：
    [REACTION:c1ccccc1;[O-][N+](=O)[O-]|O=[N+]([O-])c1ccccc1;O|H2SO4, 浓HNO3]
    [REACTION:C=C;[H]O[H]|CCO|H2SO4]

设计选择：
- 用分号 ; 分隔同一侧分子，避免和 SMILES 中的 [N+]([O-]) 等字符冲突；
- 用 | 分隔反应物区、产物区、可选反应条件；
- 化学计量系数在 v1 中暂由 LLM 在条件文本中注明，后续可扩展为 n:SMILES 语法；
- 条件中的化学式（如 H2SO4）会自动转下标；已含 $ 的文本保持原样。
"""

import re

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from renderers.structure import smiles_to_chemfig
else:
    from .structure import smiles_to_chemfig


_ARROW_GAP = "1.0cm"
_PLUS_GAP = "0.3cm"


def _format_conditions(text: str) -> str:
    """把化学式中的数字自动转为下标（如 H2SO4 → H$_2$SO$_4$）。

    若文本已含 $ 或为空，则保持原样，避免重复转义。
    """
    if not text or "$" in text:
        return text
    return re.sub(r"([A-Za-z])(\d+)", r"\1$_\2$", text)


def render_reaction(reactants_str: str, products_str: str, conditions: str = "") -> str:
    r"""[REACTION] 渲染：多反应物 + 多产物 + 条件 → 完整反应式 TikZ。

    参数:
        reactants_str: 分号分隔的反应物 SMILES。
        products_str: 分号分隔的产物 SMILES。
        conditions: 反应条件文本，可选；会显示在箭头上方。

    返回:
        可编译的 TikZ 代码；失败返回可读错误提示。
    """
    reactants = [s.strip() for s in reactants_str.split(";") if s.strip()]
    products = [s.strip() for s in products_str.split(";") if s.strip()]

    if not reactants:
        return "（反应式渲染失败：反应物不能为空）"
    if not products:
        return "（反应式渲染失败：产物不能为空）"

    chemfigs = []
    for smi in reactants + products:
        cf = smiles_to_chemfig(smi)
        if cf is None:
            return f"（反应式渲染失败：无法为「{smi}」生成结构式）"
        chemfigs.append(cf)

    reactant_chemfigs = chemfigs[: len(reactants)]
    product_chemfigs = chemfigs[len(reactants) :]

    formatted_conditions = _format_conditions(conditions.strip())

    # 用 TikZ matrix 拼接所有元素：节点在 cell 中自动垂直居中，无需额外库。
    cells = []
    for i, cf in enumerate(reactant_chemfigs):
        cells.append(f"\\node (r{i}) {{{cf}}};")
        if i < len(reactant_chemfigs) - 1:
            cells.append(r"\node {$+$};")

    # 空白列提供箭头间距，minimum width 控制箭头长度
    cells.append(f"\\node[minimum width={_ARROW_GAP}] {{}};")

    for i, cf in enumerate(product_chemfigs):
        cells.append(f"\\node (p{i}) {{{cf}}};")
        if i < len(product_chemfigs) - 1:
            cells.append(r"\node {$+$};")

    matrix_row = " & ".join(cells) + r" \\"

    lines = [
        r"\begin{tikzpicture}[baseline=(current bounding box.center)]",
        r"  \matrix (m) [column sep=" + _PLUS_GAP + ", row sep=0cm, nodes={anchor=center}] {",
        f"    {matrix_row}",
        r"  };",
    ]

    last_r = f"r{len(reactant_chemfigs) - 1}"
    if formatted_conditions:
        lines.append(
            f"  \\draw[->, thick] ({last_r}.east) -- (p0.west) "
            f"node[midway, above] {{\\small {formatted_conditions}}};"
        )
    else:
        lines.append(f"  \\draw[->, thick] ({last_r}.east) -- (p0.west);")

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
        "C=C;[H]O[H]",
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
