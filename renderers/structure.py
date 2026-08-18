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


def render_structure(smiles: str, label: str = None, mode: str = "skeleton",
                     subs: str = "", bond: str = "", angle: str = "",
                     charge: str = "") -> str:
    """[STRUCT] 渲染：SMILES → TikZ 结构式；可选 label 置于结构下方。

    分子家族统一入口（20260818 重构）：mode 分派各画法——
        skeleton（默认）：键线式/结构简式（本函数主体逻辑）
        lewis：电子式（+孤对电子点）    stereo：楔形式
        chair：椅式构象                newman：纽曼投影
    旧标记（[LEWIS]/[STEREO]/[CHAIR]/[NEWMAN]）经解析层归一化为
    STRUCT+mode 后同样进入本入口。

    20260821：bond=/charge= 并入 STRUCT 参数（单分子标注，替代顶层
    BOND/CHARGE 新写法）——skeleton（及其他非 newman）模式下：
        bond=a-b：该键加粗红色突出（复用 BOND 渲染原语）；
        charge=idx:+/-列表：对应原子旁标 δ+/δ-（复用 CHARGE 渲染原语）。

    芳香小写（c1ccccc1）画圈；凯库勒大写保留输入单双键位置（不同 Kekulé
    式渲染不同——如硝基苯 C1C=CC=CC=1 vs C1=CC=CC=C1 双键错开）。
    失败时返回可读的错误提示字符串（非空、非异常）。
    """
    if mode == "lewis":
        from .lewis import render_lewis
        return render_lewis(smiles, label)
    if mode == "stereo":
        from .stereo import render_stereo
        return render_stereo(smiles, label)
    if mode == "chair":
        from .chair import render_chair
        return render_chair(smiles, subs)
    if mode == "newman":
        from .newman import render_newman
        return render_newman(smiles, bond, angle)

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
    # 20260821：STRUCT 参数化标注（bond= 键突出 / charge= 部分电荷）
    if bond or charge:
        lines.extend(_struct_annotation_lines(mol, bond, charge))
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


def _struct_annotation_lines(mol, bond_spec: str = "", charge_spec: str = "") -> list:
    """STRUCT 参数化标注行：bond= 键突出（红粗线）+ charge= 部分电荷（δ±）。

    复用顶层 BOND/CHARGE 渲染器同款原语（bond_segments_for / partial_charge_pos
    / format_partial_charge），保证同一分子两种写法渲染一致。
    """
    from .mol_primitives import (
        bond_segments_for, format_partial_charge, label_bond_margin,
        mol_default_labeler, parse_charge_pairs, partial_charge_pos,
    )
    lines = []
    labeler = mol_default_labeler(mol)
    # bond= 键突出：与骨架修剪段完全对齐（同 xh_bond.render_bond）
    for spec in (bond_spec or "").split(","):
        spec = spec.strip()
        if "-" not in spec:
            continue
        sa, _, sb = spec.partition("-")
        try:
            a, b = int(sa), int(sb)
        except ValueError:
            continue
        if a >= mol.GetNumAtoms() or b >= mol.GetNumAtoms():
            continue
        segs = bond_segments_for(mol, a, b, labeler=labeler,
                                 margin_fn=label_bond_margin)
        if segs is None:
            continue
        for x1, y1, x2, y2 in segs:
            lines.append(
                f"  \\draw[very thick, red] ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")
    # charge= 部分电荷：以元素符号中心为基准，方向避让（同 render_charge）
    for idx, raw_label in parse_charge_pairs(charge_spec).items():
        if idx >= mol.GetNumAtoms():
            continue
        x, y = partial_charge_pos(mol, idx)
        label = format_partial_charge(raw_label)
        lines.append(f"  \\node[font=\\small, red] at ({x:.2f},{y:.2f}) {{{label}}};")
    return lines


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
    print("\n[4] 分子家族 mode 分派（与旧标记等价）:")
    print(render_structure("O", mode="lewis", label="水"))
    print("\n[5] 无效 SMILES:")
    print(render_structure("XYZ无效"))
    print("\n[6] 空 SMILES:")
    print(render_structure(""))
