# -*- coding: utf-8 -*-
"""
legacy/structure_render.py
      =========================
SMILES -> \\chemfig{TikZ} 渲染器。

Day 4-5 任务：
    - 调用 mol2chemfigPy3 的 mol2chemfig 函数，把 SMILES 转成 LaTeX chemfig 代码。
    - 提供可选的 LaTeX 导言区包装，生成可直接编译的完整文档。

用法:
    from legacy.structure_render import smiles_to_tikz
    code = smiles_to_tikz("c1ccccc1")
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# mol2chemfigPy3 的输出含 \mcfcringle / \phantom 等宏，
# LaTeX 端需同时加载 chemfig 和 mol2chemfig 两个宏包。
_LATEX_PREAMBLE = r"""% --- 生成自 chem_agent: structure_render.py ---
\documentclass[border=10pt]{standalone}
\usepackage{chemfig}
\usepackage{mol2chemfig}

\begin{document}
"""

_LATEX_POSTAMBLE = r"""
\end{document}
"""


def smiles_to_tikz(
    smiles: str,
    *,
    rotate: float = 0.0,
    aromatic: bool = True,
    show_carbon: bool = False,
    show_methyl: bool = False,
) -> str:
    """把 SMILES 转换为 LaTeX chemfig 代码（即 \\chemfig{...} 字符串）。

    依赖 mol2chemfigPy3（内部使用 epam.indigo，非 rdkit）。
    任何失败情况均返回空串 ""，绝不让异常中断主流程。

    参数:
        smiles (str): 合法 SMILES，例如 "c1ccccc1"（苯）。
        rotate (float): 结构旋转角度（度），默认 0。
        aromatic (bool): 是否以芳香环形式渲染，默认 True。
        show_carbon (bool): 是否显式标出碳原子，默认 False。
        show_methyl (bool): 是否显式标出甲基，默认 False。

    返回:
        str: \\chemfig{...} 代码；输入非法或渲染失败时返回 ""。
    """
    if not smiles or not isinstance(smiles, str):
        print("[smiles_to_tikz] 输入 SMILES 为空或类型错误。")
        return ""

    print(f"[smiles_to_tikz] 渲染 SMILES: {smiles!r}")
    try:
        from mol2chemfigPy3 import mol2chemfig
    except ImportError:
        print("[smiles_to_tikz] mol2chemfigPy3 未安装，无法渲染结构式。")
        return ""

    try:
        # inline=True 才会返回字符串；默认 inline=False 只打印到 stdout 返回 None
        result = mol2chemfig(
            smiles,
            rotate=rotate,
            aromatic=aromatic,
            show_carbon=show_carbon,
            show_methyl=show_methyl,
            inline=True,
        )
    except Exception as e:
        print(f"[smiles_to_tikz] mol2chemfig 调用异常: {e}")
        return ""

    # 失败时 mol2chemfig 返回的是错误描述字符串（非 None），需校验确实是 chemfig 代码
    if not isinstance(result, str) or not result.startswith("\\chemfig"):
        print(f"[smiles_to_tikz] 渲染失败，返回值非 chemfig 代码: {result!r}")
        return ""

    print(f"[smiles_to_tikz] 渲染成功，代码长度 {len(result)} 字符。")
    return result


def to_latex_document(chemfig_code: str, title: str = "") -> str:
    """把裸 \\chemfig{...} 代码包装成可编译的完整 LaTeX 文档。

    用于需要直接丢进 Overleaf / 本地 latexmk 编译的场景。

    参数:
        chemfig_code (str): smiles_to_tikz 返回的 \\chemfig{...} 代码。
        title (str): 可选标题（纯文本），放在结构式上方。

    返回:
        str: 完整 LaTeX 文档源码；输入为空时返回 ""。
    """
    if not chemfig_code:
        print("[to_latex_document] 输入的 chemfig 代码为空，跳过包装。")
        return ""

    parts = [_LATEX_PREAMBLE]
    if title:
        parts.append(f"\\textbf{{{title}}}\\par\n")
    parts.append(chemfig_code)
    parts.append(_LATEX_POSTAMBLE)
    return "".join(parts)


if __name__ == "__main__":
    # Day 6 测试入口：打印 CC(C)C（异丙烷）和 c1ccccc1（苯）的 chemfig 代码
    print("=" * 60)
    print("structure_render 测试：CC(C)C / c1ccccc1")
    print("=" * 60)

    test_cases = ("CC(C)C", "c1ccccc1")
    for smi in test_cases:
        print("-" * 60)
        print(f"SMILES: {smi}")
        code = smiles_to_tikz(smi)
        if code:
            print("chemfig 代码:")
            print(code)
            print()
            print("完整 LaTeX 文档（可直接编译）:")
            print(to_latex_document(code, title=smi))
        else:
            print(f">>> {smi!r} 渲染失败。")

    print("=" * 60)
    print("测试结束。")
