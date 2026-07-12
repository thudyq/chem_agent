# -*- coding: utf-8 -*-
"""renderers/structure.py — [STRUCT] 标记渲染器：SMILES → chemfig TikZ。

复用旧 smiles_to_tikz 逻辑（mol2chemfigPy3），新增：
  - RDKit 预验证：无效 SMILES 返回错误提示字符串而非崩溃
  - label 精确定位：tikzpicture 包裹 + \\node[anchor=north] 置于结构下方
"""


def render_structure(smiles: str, label: str = None) -> str:
    """SMILES → \\chemfig{...} 代码；可选 label 置于结构下方。

    无效 SMILES 或渲染失败时返回可读的错误提示字符串（非空、非异常），
    便于在最终图文输出中就地显示失败原因。

    参数:
        smiles: 合法 SMILES，如 "c1ccccc1"。
        label: 可选结构标注，置于结构下方。

    返回:
        chemfig/TikZ 代码字符串；失败时返回 "（结构渲染失败：...）" 提示串。
    """
    if not smiles or not isinstance(smiles, str):
        return "（结构渲染失败：SMILES 为空）"

    # RDKit 预验证拦截无效输入；rdkit 不可用时跳过（由 mol2chemfig 兜底）
    try:
        from utils.rdkit_utils import validate_smiles
        if not validate_smiles(smiles):
            return f"（结构渲染失败：无效 SMILES「{smiles}」）"
    except ImportError:
        pass

    try:
        from mol2chemfigPy3 import mol2chemfig
    except ImportError:
        return "（结构渲染失败：mol2chemfigPy3 未安装）"

    try:
        # inline=True 才返回字符串；默认 inline=False 只打印到 stdout 返回 None
        result = mol2chemfig(smiles, inline=True)
    except Exception as e:
        return f"（结构渲染失败：mol2chemfig 调用异常 {e}）"

    if not isinstance(result, str) or not result.startswith("\\chemfig"):
        return f"（结构渲染失败：mol2chemfig 未返回 chemfig 代码）"

    # label：tikzpicture 包裹结构为命名节点，label 置于其正下方（tikz 核心 anchor/yshift，无需额外宏包）
    if label:
        return (
            "\\begin{tikzpicture}\n"
            f"  \\node (mol) {{{result}}};\n"
            f"  \\node[anchor=north] at ([yshift=-2mm]mol.south) {{{label}}};\n"
            "\\end{tikzpicture}"
        )
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("render_structure 测试")
    print("=" * 60)
    print("[1] 苯（无 label）:")
    print(render_structure("c1ccccc1"))
    print("\n[2] 乙酸（带 label）:")
    print(render_structure("CC(=O)O", label="乙酸"))
    print("\n[3] 无效 SMILES:")
    print(render_structure("XYZ无效"))
    print("\n[4] 空 SMILES:")
    print(render_structure(""))
