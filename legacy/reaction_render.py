# -*- coding: utf-8 -*-
"""
legacy/reaction_render.py
      ========================
反应方程式可视化：reaction SMILES -> 可视化方程式。

2.1 任务：
    - 解析 reaction SMILES（A.B>>C.D），分离反应物/产物。
    - 支持条件标注（|条件）与可逆反应（|⇌）。
    - 组装 LaTeX 方程式（chemfig + amsmath/mathtools）。

输入格式：
    A.B>>C.D                 正向反应
    A.B>>C.D | 条件          带条件标注
    A.B>>C.D | 条件, ⇌       可逆反应
    A.B>>C.D | ⇌             可逆，无条件
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .structure_render import smiles_to_tikz
except ImportError:  # noqa: E722
    from structure_render import smiles_to_tikz


def is_reaction_input(text: str) -> bool:
    """判断输入是否为 reaction SMILES（含 >>）。"""
    return bool(text and ">>" in text.strip())


def parse_reaction(input_str: str):
    """解析 reaction SMILES 输入。

    返回:
        dict | None: {reactants, products, conditions, reversible}；非反应输入返回 None。
    """
    text = input_str.strip()
    if ">>" not in text:
        return None

    # 分离 | 后的条件标注
    conditions = ""
    reversible = False
    if "|" in text:
        rxn_part, _, cond_part = text.partition("|")
        cond_part = cond_part.strip()
        if "⇌" in cond_part or "可逆" in cond_part:
            reversible = True
        # 移除可逆标记，保留实际条件文本
        conditions = cond_part.replace("⇌", "").replace("可逆", "").strip(" ,，")
    else:
        rxn_part = text

    left, _, right = rxn_part.partition(">>")
    reactants = [s.strip() for s in left.split(".") if s.strip()]
    products = [s.strip() for s in right.split(".") if s.strip()]
    if not reactants or not products:
        return None
    return {
        "reactants": reactants,
        "products": products,
        "conditions": conditions,
        "reversible": reversible,
    }


def render_reaction_chemfig(rxn) -> str:
    """组装反应方程式的 LaTeX 代码。

    每个 reactant/product 用 smiles_to_tikz 渲染为 \\chemfig{...}，
    同侧用 + 连接，箭头用 \\xrightarrow/\\xrightleftharpoons（带条件文本）。
    """
    def render_side(smiles_list):
        codes = []
        for smi in smiles_list:
            code = smiles_to_tikz(smi)
            codes.append(code if code else f"\\text{{{smi}}}")
        return " + ".join(codes)

    left = render_side(rxn["reactants"])
    right = render_side(rxn["products"])
    cond = rxn["conditions"]
    cond_str = rf"\text{{{cond}}}" if cond else ""

    if rxn["reversible"]:
        # 可逆：有条件用 \xrightleftharpoons（需 mathtools），否则 \rightleftharpoons
        arrow = rf"\xrightleftharpoons{{{cond_str}}}" if cond_str else r"\rightleftharpoons"
    else:
        # 正向：有条件用 \xrightarrow（amsmath），否则 \rightarrow
        arrow = rf"\xrightarrow{{{cond_str}}}" if cond_str else r"\rightarrow"
    return rf"\[ {left} {arrow} {right} \]"


if __name__ == "__main__":
    print("=" * 60)
    print("reaction_render 测试")
    print("=" * 60)
    tests = [
        "CC(=O)O.CCO>>CC(=O)OCC.O",                # 酯化（正向，无条件）
        "CC(=O)O.CCO>>CC(=O)OCC.O | 浓硫酸, Δ",     # 带条件
        "C.CC<<>>CC.C",                             # 边界：非标准
    ]
    for t in tests:
        print(f"\n输入: {t!r}")
        print(f"  is_reaction: {is_reaction_input(t)}")
        rxn = parse_reaction(t)
        if rxn:
            print(f"  反应物: {rxn['reactants']}")
            print(f"  产物: {rxn['products']}")
            print(f"  条件: {rxn['conditions']!r}, 可逆: {rxn['reversible']}")
        else:
            print("  解析失败（None）")
    print("=" * 60)
