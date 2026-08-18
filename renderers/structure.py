# -*- coding: utf-8 -*-
"""renderers/structure.py — [STRUCT] 标记渲染器：SMILES → TikZ 结构式。

统一到 TikZ scope 新逻辑（prepare_mol + molecule_scope_lines，与 ARROW/
REACTION/COMPOSITE/XH/BOND 一致），弃用 mol2chemfigPy3：
  - 芳香小写（c1ccccc1）→ 全芳香环画圈
  - 凯库勒大写（C1=CC=CC=C1）→ 交替单双键
  - label 置于结构下方
smiles_to_chemfig 保留为兼容函数（测试/外部引用），不再用于 STRUCT。
"""

import contextlib


def smiles_to_chemfig(smiles: str, aromatic: bool = True):
    """SMILES → \\chemfig{...} 代码字符串；任何失败返回 None。

    RDKit 预验证（不可用时跳过）+ mol2chemfigPy3 渲染。兼容函数——
    STRUCT 已迁移到 TikZ 新逻辑，此函数仅供测试/外部引用保留。
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
        from utils.rdkit_utils import FREE_H_COMPONENT_RE, mute_rdkit_warnings
        # 孤立氢组分（[H+]/[H]/[H-]，合法组分）触发 RDKit RemoveHs 警告
        # （无害），局部屏蔽
        cm = (mute_rdkit_warnings() if FREE_H_COMPONENT_RE.search(smiles)
              else contextlib.nullcontext())
        with cm:
            result = mol2chemfig(smiles, aromatic=aromatic, inline=True)
    except Exception:
        return None
    if not isinstance(result, str) or not result.startswith("\\chemfig"):
        return None
    return result


def render_structure(smiles: str, label: str = None) -> str:
    """[STRUCT] 渲染：SMILES → TikZ 结构式；可选 label 置于结构下方。

    芳香小写（c1ccccc1）画圈；凯库勒大写保留输入单双键位置（不同 Kekulé
    式渲染不同——如硝基苯 C1C=CC=CC=1 vs C1=CC=CC=C1 双键错开）。
    失败时返回可读的错误提示字符串（非空、非异常）。
    """
    from .mol_primitives import (
        aromatic_ring_info, has_aromatic_lowercase, prepare_mol, wrap_format_text,
    )
    from .layout import molecule_scope_lines

    is_aromatic = has_aromatic_lowercase(smiles)
    # 凯库勒大写（无芳香小写）：allow_aromatic=False 保留输入键级——
    # RDKit 默认会把任意交替式归一化为芳香环，丢失用户指定的单双键位置
    mol = prepare_mol(smiles, allow_aromatic=is_aromatic)
    if mol is None:
        return f"（结构渲染失败：无法为「{smiles}」生成结构式，请检查 SMILES）"

    rings = aromatic_ring_info(mol) if is_aromatic else None
    lines = ["\\begin{tikzpicture}"]
    # 键线式默认不标孤对电子（规范第 3 条，与 ARROW/REACTION/RETRO 一致）：
    # 孤对电子仅在 LEWIS / 机理容器（MECHARROW/RESARROW）中显示。
    # 注意 molecule_scope_lines 默认 show_lone_pairs=True，此处必须显式关闭
    # （20260818 修复：此前顶层 STRUCT 漏传 → [STRUCT:CCl] 的 Cl 画出 3 对孤对电子）。
    lines.extend(molecule_scope_lines(mol, (0.0, 0.0), aromatic_rings=rings,
                                      show_lone_pairs=False))
    if label:
        text = wrap_format_text(label)
        align = "align=center, " if "\\\\" in text else ""
        # label 置于结构正下方：用当前 scope 范围下方锚点
        from .mol_primitives import mol_visual_bbox
        min_x, min_y, max_x, max_y = mol_visual_bbox(mol, include_lone_pairs=False)
        cx = (min_x + max_x) / 2.0
        lines.append(
            f"  \\node[{align}anchor=north] at ({cx:.2f},{min_y - 0.35:.2f}) {{{text}}};"
        )
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("render_structure 测试")
    print("=" * 60)
    print("[1] 苯（芳香小写→画圈）:")
    print(render_structure("c1ccccc1"))
    print("\n[2] 苯（凯库勒大写→交替键）:")
    print(render_structure("C1=CC=CC=C1"))
    print("\n[3] 乙酸（带 label）:")
    print(render_structure("CC(=O)O", label="乙酸"))
    print("\n[4] 无效 SMILES:")
    print(render_structure("XYZ无效"))
    print("\n[5] 空 SMILES:")
    print(render_structure(""))
