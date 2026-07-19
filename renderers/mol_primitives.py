# -*- coding: utf-8 -*-
"""renderers/mol_primitives.py — RDKit 分子骨架绘制的共享工具。

把 mechanism / lewis / stereo / charge / hbond 中重复的原子标签、
2D 坐标计算、键线绘制逻辑抽取到这里，避免复制粘贴。
"""

import math


def atom_label(atom) -> str | None:
    """生成非隐式碳原子的标签（如 OH、NH₂、Cl、$^{+}$ 等）。

    纯碳原子（原子序 6、形式电荷 0）返回 None，表示不显示标签。
    """
    z = atom.GetAtomicNum()
    if z == 6 and atom.GetFormalCharge() == 0:
        return None

    sym = atom.GetSymbol()
    sym = sym[0].upper() + sym[1:]
    h = atom.GetTotalNumHs()
    parts = sym
    if h == 1:
        parts += "H"
    elif h > 1:
        parts += f"H$_{{{h}}}$"

    fc = atom.GetFormalCharge()
    if fc:
        num = str(abs(fc)) if abs(fc) > 1 else ""
        sign = "+" if fc > 0 else "-"
        parts += f"$^{{{num}{sign}}}$"

    return parts


def prepare_mol(smiles: str, *, add_hs: bool = False, kekulize: bool = False, use_prepare: bool = True):
    """SMILES → RDKit Mol：解析、可选加氢/Kekulize、计算 2D 坐标。

    参数:
        smiles: 输入 SMILES。
        add_hs: 是否调用 AddHs（Lewis 结构需要显示所有 H）。
        kekulize: 是否 Kekulize（Lewis 需要明确单双键）。
        use_prepare: 是否优先用 rdMolDraw2D.PrepareMolForDrawing；
                     为 False 时直接用 AllChem.Compute2DCoords。

    返回:
        RDKit Mol 对象；解析失败返回 None。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None

    if add_hs:
        mol = Chem.AddHs(mol)

    if kekulize:
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass

    if use_prepare:
        try:
            prepared = rdMolDraw2D.PrepareMolForDrawing(mol)
            if prepared is not None:
                mol = prepared
        except Exception:
            AllChem.Compute2DCoords(mol)
    else:
        AllChem.Compute2DCoords(mol)

    return mol


def atom_pos(mol, idx: int) -> tuple[float, float]:
    """返回原子 idx 的 2D 坐标 (x, y)。"""
    conf = mol.GetConformer()
    p = conf.GetAtomPosition(idx)
    return p.x, p.y


def bond_type_order(bond) -> int:
    """把 BondTypeAsDouble 规整为 1/2/3（单/双/三键）。"""
    order = bond.GetBondTypeAsDouble()
    if order >= 2.5:
        return 3
    if order >= 1.5:
        return 2
    return 1


def bond_segments(mol, *, label_margin: float = 0.25, bond_gap: float = 0.08):
    """把分子中所有化学键转换为 TikZ 线段坐标列表。

    返回:
        每个键对应一个列表，包含 1 条（单键）或 2/3 条（双/三键）线段坐标：
        [
            [(x1, y1, x2, y2)],                       # 单键
            [(x1, y1, x2, y2), (x1', y1', x2', y2')], # 双键
            ...
        ]

    参数:
        label_margin: 标签原子两端留出的空白距离，避免键压住标签。
        bond_gap: 双键/三键平行线之间的间距。
    """
    segments = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        xi, yi = atom_pos(mol, i)
        xj, yj = atom_pos(mol, j)

        order = bond_type_order(b)
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        px, py = -uy, ux

        si = label_margin if atom_label(mol.GetAtomWithIdx(i)) else 0.0
        sj = label_margin if atom_label(mol.GetAtomWithIdx(j)) else 0.0
        x1, y1 = xi + ux * si, yi + uy * si
        x2, y2 = xj - ux * sj, yj - uy * sj

        if order == 1:
            segments.append([(x1, y1, x2, y2)])
        else:
            # 双键：两条线；三键：三条线
            segs = [(x1, y1, x2, y2)]
            segs.append((x1 + px * bond_gap, y1 + py * bond_gap,
                         x2 + px * bond_gap, y2 + py * bond_gap))
            if order == 3:
                segs.append((x1 - px * bond_gap, y1 - py * bond_gap,
                             x2 - px * bond_gap, y2 - py * bond_gap))
            segments.append(segs)

    return segments


def fmt_coord(x: float, y: float) -> str:
    """把坐标格式化为 TikZ 常用的两位小数字符串。"""
    return f"({x:.2f},{y:.2f})"


def tikz_draw_line(x1: float, y1: float, x2: float, y2: float, style: str = "") -> str:
    r"""生成一条 \draw 命令。"""
    cmd = "  \\draw"
    if style:
        cmd += f"[{style}]"
    cmd += f" {fmt_coord(x1, y1)} -- {fmt_coord(x2, y2)};"
    return cmd
