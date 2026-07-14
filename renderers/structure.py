# -*- coding: utf-8 -*-
"""renderers/structure.py — [STRUCT] 标记渲染器：SMILES → chemfig TikZ。

复用旧 smiles_to_tikz 逻辑（mol2chemfigPy3），新增：
  - RDKit 预验证：无效 SMILES 返回错误提示字符串而非崩溃
  - label 精确定位：tikzpicture 包裹 + \\node[anchor=north] 置于结构下方
"""


def smiles_to_chemfig(smiles: str, aromatic: bool = True):
    """SMILES → \\chemfig{...} 代码字符串；任何失败返回 None。

    RDKit 预验证（不可用时跳过）+ mol2chemfigPy3 渲染。供 render_structure
    及其他渲染器（arrow/newman 等）复用。
    aromatic=True 渲染芳香环为圆圈；False 渲染 Kekulé 交替单双键。
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        from utils.rdkit_utils import validate_smiles
        if not validate_smiles(smiles):
            return None
    except ImportError:
        pass
    try:
        from mol2chemfigPy3 import mol2chemfig
        result = mol2chemfig(smiles, aromatic=aromatic, inline=True)
    except Exception:
        return None
    if not isinstance(result, str) or not result.startswith("\\chemfig"):
        return None
    return result


def render_structure(smiles: str, label: str = None) -> str:
    """[STRUCT] 渲染：SMILES → chemfig 代码；可选 label 置于结构下方。

    失败时返回可读的错误提示字符串（非空、非异常）。
    """
    chemfig = smiles_to_chemfig(smiles)
    if chemfig is None:
        return f"（结构渲染失败：无法为「{smiles}」生成结构式，请检查 SMILES 与 mol2chemfigPy3 安装）"

    # label：tikzpicture 包裹结构为命名节点，label 置于其正下方（tikz 核心 anchor/yshift）
    if label:
        return (
            "\\begin{tikzpicture}\n"
            f"  \\node (mol) {{{chemfig}}};\n"
            f"  \\node[anchor=north] at ([yshift=-2mm]mol.south) {{{label}}};\n"
            "\\end{tikzpicture}"
        )
    return chemfig


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
