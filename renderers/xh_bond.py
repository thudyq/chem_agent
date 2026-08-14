# -*- coding: utf-8 -*-
"""renderers/xh_bond.py — [XH] / [BOND] 顶层标记渲染器：单分子位点强调。

容器内子标记形式（[XH:id|原子] / [BOND:id|a-b]）由 composite.py 处理；
本模块处理顶层单分子形式——无需 COMPOSITE 容器的轻量标注场景
（如"某 H 具有酸性"只需在结构式上示出该 O—H，用不着 COMPOSITE）。

标记格式：
    [XH:SMILES|原子序号,...]   显式画出指定原子的隐含 H（实线 + H 标签；
                               同序号重复出现则叠加多根；标签 H 计数自动扣减）
    [BOND:SMILES|a-b,...]      把已有键 a-b 加粗标红突出（可多根）
示例：
    [XH:CC(=O)O|3]     （乙酸 O—H 显式画出——说明酸性 H）
    [XH:CCC=O|1]       （丙醛 α-H 显式画出）
    [BOND:CCC=O|1-2]   （丙醛 α,β-碳碳键突出）
"""

from .collide import Occupancy
from .mol_primitives import (
    atom_label, atom_pos, bond_segments, bond_segments_for, charge_tikz,
    label_bond_margin, label_edge_point, label_visual_width,
    place_explicit_hs, place_h_avoiding, prepare_mol,
)


def _draw_annotated_mol(smiles: str, spec: str, kind: str) -> str:
    """单分子键线式骨架（标签 H 计数按显式 H 扣减）+ 位点叠加层。"""
    mol = prepare_mol(smiles)
    if mol is None:
        return f"（{'显式氢' if kind == 'XH' else '键突出'}渲染失败：无效 SMILES「{smiles}」)"

    # 注解计数：XH 为 {原子: H 数}，BOND 为 ["a-b", ...]；越界/非键静默跳过
    xh_counts = {}
    bond_specs = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if kind == "XH":
            try:
                a = int(tok)
            except ValueError:
                continue
            if a < mol.GetNumAtoms():
                xh_counts[a] = xh_counts.get(a, 0) + 1
        else:
            bond_specs.append(tok)

    labeler = lambda a: atom_label(a, xh_counts.get(a.GetIdx(), 0))
    occ = Occupancy()   # R-8 占据注册表：键/标签/电荷圈登记，XH 节点避障
    lines = ["\\begin{tikzpicture}"]
    for segs in bond_segments(mol, labeler=labeler, margin_fn=label_bond_margin):
        for x1, y1, x2, y2 in segs:
            lines.append(f"  \\draw ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")
            occ.add_segment(x1, y1, x2, y2)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        lab = labeler(atom)
        if lab:
            x, y = atom_pos(mol, idx)
            lines.append(
                f"  \\node[fill=white, inner sep=1pt] at ({x:.2f},{y:.2f}) {{{lab}}};")
            hw = max(0.11, label_visual_width(lab) * 0.3)
            occ.add_rect(x - hw, y - 0.12, x + hw, y + 0.12)
        charge = charge_tikz(mol, idx, explicit_hs=xh_counts.get(idx, 0),
                             occupancy=occ)
        if charge:
            lines.append(f"  {charge}")

    # XH：实线（标签边缘起笔，不压标签）+ H 节点（撞键/标签时旋转避障）
    for a, count in xh_counts.items():
        for hx, hy in place_explicit_hs(mol, a, count):
            hx, hy = place_h_avoiding(mol, a, (hx, hy), occ)
            sx, sy = label_edge_point(mol, a, (hx, hy), labeler=labeler)
            lines.append(f"  \\draw ({sx:.2f},{sy:.2f}) -- ({hx:.2f},{hy:.2f});")
            lines.append(
                f"  \\node[fill=white, inner sep=1pt] at ({hx:.2f},{hy:.2f}) {{H}};")

    # BOND：红色粗线覆盖，与原键完全对齐（同一修剪逻辑）
    for spec_item in bond_specs:
        if "-" not in spec_item:
            continue
        sa, _, sb = spec_item.partition("-")
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

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def render_xh(smiles: str, spec: str = "") -> str:
    """[XH] 顶层渲染：SMILES + 原子序号列表 → 带显式 H 标注的结构式。"""
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（显式氢渲染失败：rdkit 未安装）"
    return _draw_annotated_mol(smiles, spec, "XH")


def render_bond(smiles: str, spec: str = "") -> str:
    """[BOND] 顶层渲染：SMILES + 键列表 → 键加粗标红突出的结构式。"""
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return "（键突出渲染失败：rdkit 未安装）"
    return _draw_annotated_mol(smiles, spec, "BOND")


if __name__ == "__main__":
    print("[1] 乙酸 O—H 显式画出（酸性 H 用例）: [XH:CC(=O)O|3]")
    print(render_xh("CC(=O)O", "3"))
    print("\n[2] 丙醛 α-H 叠加 2 根: [XH:CCC=O|1,1]")
    print(render_xh("CCC=O", "1,1"))
    print("\n[3] 丙醛 α,β-碳碳键突出: [BOND:CCC=O|1-2]")
    print(render_bond("CCC=O", "1-2"))
    print("\n[4] 无效 SMILES（应降级提示）:")
    print(render_xh("XYZ", "0"))
