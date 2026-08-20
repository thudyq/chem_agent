# -*- coding: utf-8 -*-
"""utils/tikz_utils.py — TikZ/chemfig 代码美化与包装工具。"""

# chemfig + mol2chemfig 导言区（mol2chemfigPy3 输出含 \mcfcringle 等宏需后者；
# arrows.meta 提供机理箭头的 Stealth 箭头尖/鱼钩半尖）
_LATEX_PREAMBLE = r"""% --- 生成自 chem_agent 渲染引擎 ---
\documentclass[border=10pt]{standalone}
\usetikzlibrary{arrows.meta}
\usepackage{chemfig}
\usepackage{mol2chemfig}

\begin{document}
"""

_LATEX_POSTAMBLE = r"""
\end{document}
"""


def wrap_in_latex(tikz_code: str, title: str = "") -> str:
    """把裸 TikZ/chemfig 代码包装成可编译的完整 LaTeX 文档。

    参数:
        tikz_code: chemfig/TikZ 代码片段（如 \\chemfig{...}）。
        title: 可选标题，置于内容上方。

    返回:
        完整 LaTeX 文档源码；输入为空返回 ""。
    """
    if not tikz_code:
        return ""
    parts = [_LATEX_PREAMBLE]
    if title:
        parts.append(f"\\textbf{{{title}}}\\par\n")
    parts.append(tikz_code)
    parts.append(_LATEX_POSTAMBLE)
    return "".join(parts)
