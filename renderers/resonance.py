# -*- coding: utf-8 -*-
"""renderers/resonance.py — [RESONANCE] 标记渲染器：共振结构式。

LLM 提供多个共振极限式的 SMILES（~ 分隔），renderer 逐个渲染为 chemfig 结构，
用 ↔（共振符号，非平衡箭头）连接。数据来源：LLM 识别共振式（非 RDKit 自动生成）；
结构渲染复用 structure.smiles_to_chemfig（mol2chemfigPy3）。

标记格式：[RESONANCE:SMILES1~SMILES2~SMILES3]
示例：[RESONANCE:C1=CC=CC=C1~C1=CC=CC=C1]（苯的两个 Kekulé 式）
"""

from renderers.structure import smiles_to_chemfig


def render_resonance(content: str) -> str:
    """[RESONANCE] 渲染：多个共振式 SMILES → TikZ 横向排列 + ↔。"""
    if not content or not isinstance(content, str):
        return "（共振式渲染失败：内容为空）"

    smiles_list = [s.strip() for s in content.split("~") if s.strip()]
    if len(smiles_list) < 2:
        return "（共振式渲染失败：至少需要 2 个共振极限式，用 ~ 分隔）"

    chemfigs = []
    for smi in smiles_list:
        cf = smiles_to_chemfig(smi, aromatic=False)
        if cf is None:
            return f"（共振式渲染失败：无效 SMILES「{smi}」）"
        chemfigs.append(cf)

    n = len(chemfigs)
    spacing = 5.0

    lines = ["\\begin{tikzpicture}"]
    for i, cf in enumerate(chemfigs):
        x = i * spacing
        lines.append(f"  \\node at ({x:.1f},0) {{{cf}}};")
        if i < n - 1:
            arrow_x = x + spacing / 2
            lines.append(f"  \\node[font=\\large] at ({arrow_x:.1f},0) {{$\\leftrightarrow$}};")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("[1] 苯 Kekulé:")
    print(render_resonance("C1=CC=CC=C1~C1C=CC=CC=1"))
    print("\n[2] 羧酸根:")
    print(render_resonance("CC(=O)[O-]~CC([O-])=O"))
