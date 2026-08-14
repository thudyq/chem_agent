# -*- coding: utf-8 -*-
"""tests/test_bond_segments.py — bond_segments 环内双键偏移方向单元测试。

运行: python -m pytest tests/test_bond_segments.py -v
"""

import math

import pytest

from renderers.mol_primitives import (
    aromatic_ring_info, atom_pos, bond_segments, has_aromatic_lowercase,
    prepare_mol,
)

rdkit = pytest.importorskip("rdkit", reason="rdkit 未安装，跳过键线测试")


def _seg_mid(seg):
    x1, y1, x2, y2 = seg
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def test_benzene_double_bonds_inside_ring():
    """苯环 3 条双键：平行线偏移线在环内（比主线更靠近环质心）。"""
    mol = prepare_mol("c1ccccc1")
    assert mol is not None
    ring = [a.GetIdx() for a in mol.GetAtoms()]
    cx = sum(atom_pos(mol, i)[0] for i in ring) / len(ring)
    cy = sum(atom_pos(mol, i)[1] for i in ring) / len(ring)
    center = (cx, cy)

    double_bonds = [segs for segs in bond_segments(mol) if len(segs) == 2]
    assert len(double_bonds) == 3                      # Kekulé 交替双键
    for main, offset in double_bonds:
        assert _dist(_seg_mid(offset), center) < _dist(_seg_mid(main), center)


def test_benzene_inner_bond_length_formula():
    """内双键长度 = 距中心距离 × 2/√3，且端点与中心、顶点三点共线。"""
    mol = prepare_mol("c1ccccc1")
    ring = [a.GetIdx() for a in mol.GetAtoms()]
    cx = sum(atom_pos(mol, i)[0] for i in ring) / len(ring)
    cy = sum(atom_pos(mol, i)[1] for i in ring) / len(ring)
    center = (cx, cy)
    verts = [(atom_pos(mol, i)[0], atom_pos(mol, i)[1]) for i in ring]

    for main, offset in [s for s in bond_segments(mol) if len(s) == 2]:
        d_center = _dist(_seg_mid(offset), center)
        length = _dist((offset[0], offset[1]), (offset[2], offset[3]))
        expected = d_center * 2 / math.sqrt(3)
        assert abs(length - expected) < 0.05, \
            f"内双键长度 {length:.3f} ≠ 距离×2/√3 {expected:.3f}"
        # 共线性：内线端点应与中心、某个顶点共线（叉积 ≈ 0）
        for ex, ey in ((offset[0], offset[1]), (offset[2], offset[3])):
            cross = min(
                abs((ex - cx) * (vy - cy) - (ey - cy) * (vx - cx))
                for vx, vy in verts
            )
            assert cross < 0.02, f"端点 ({ex:.2f},{ey:.2f}) 不在中心→顶点射线上"


def test_chain_double_bond_kept():
    """链上双键（乙烯）仍生成两条平行线，不崩溃。"""
    mol = prepare_mol("C=C")
    segs = [s for s in bond_segments(mol) if len(s) == 2]
    assert len(segs) == 1


def test_substituted_ring_offsets_inside():
    """取代苯（硝基苯）环内双键同样朝环内偏移。"""
    mol = prepare_mol("O=[N+]([O-])c1ccccc1")
    ring_info = mol.GetRingInfo().AtomRings()
    assert ring_info
    ring = ring_info[0]
    cx = sum(atom_pos(mol, i)[0] for i in ring) / len(ring)
    cy = sum(atom_pos(mol, i)[1] for i in ring) / len(ring)
    center = (cx, cy)
    ring_set = set(ring)
    checked = 0
    for segs in bond_segments(mol):
        if len(segs) != 2:
            continue
        x1, y1, x2, y2 = segs[0]
        # 只检查两端都在环上的双键
        ends = [(x1, y1), (x2, y2)]
        in_ring = all(_dist(e, center) < 2.0 for e in ends)
        if not in_ring:
            continue
        checked += 1
        assert _dist(_seg_mid(segs[1]), center) < _dist(_seg_mid(segs[0]), center)
    assert checked >= 2


def test_kekule_pattern_consistent_across_molecules():
    """芳香小写（c1ccccc1）→ 画圈：渲染层跳过环内双键（交替键逻辑弃用）。

    画圈方案取代旧的"几何 Kekulé 规整"：芳香小写分子在 bond_segments
    渲染时跳过环内双键、由调用方补画圈；凯库勒大写分子保留交替双键。
    """
    from renderers.mol_primitives import aromatic_ring_info, has_aromatic_lowercase

    # 芳香小写分子：识别为画圈（有全芳香环 + 原始 SMILES 含小写）
    for smi in ["c1ccccc1", "c1ccccc1N", "O=[N+]([O-])c1ccccc1"]:
        assert has_aromatic_lowercase(smi), smi
        mol = prepare_mol(smi)
        assert len(aromatic_ring_info(mol)) == 1, smi  # 识别出全芳香环
    # 凯库勒大写：即使 RDKit 芳香化标记，原始 SMILES 无小写 → 不画圈
    assert not has_aromatic_lowercase("C1=CC=CC=C1")
    mol_kk = prepare_mol("C1=CC=CC=C1")
    # prepare_mol 可能芳香化标记大写凯库勒，但画圈裁决在渲染层
    # （has_aromatic_lowercase 为 False → 不传 skip → 保留交替键）

    # 渲染层：画圈模式下环内双键被跳过（骨架保留单线），
    # 非画圈模式（凯库勒）双键平行线保留
    def bond_line_count(mol, rings):
        segs_all = bond_segments(mol, skip_aromatic_rings=rings)
        return sum(len(s) for s in segs_all)

    mol_ar = prepare_mol("c1ccccc1")
    mol_kk = prepare_mol("C1=CC=CC=C1")
    rings_ar = aromatic_ring_info(mol_ar)
    # 画圈：6 条单线（双键平行线被圈替代）
    assert bond_line_count(mol_ar, [r[0] for r in rings_ar]) == 6
    # 凯库勒：9 条线（6 骨架 + 3 双键平行线）
    assert bond_line_count(mol_kk, None) == 9


def test_aromatic_lowercase_detection():
    """解析器层：has_aromatic_lowercase 按大小写判定画圈/交替键（普适原则）。

    所有芳香分子都允许两种写法：小写（c/n/o/s/p）→ 画圈；
    大写（Kekulé 显式双键）→ 交替键。RDKit 解析后芳香信息被归一化，
    必须从原始 SMILES 字符串判断。
    """
    from renderers.mol_primitives import has_aromatic_lowercase

    # 小写 → True（画圈）
    for smi in ["c1ccccc1", "Cc1ccccc1", "c1ccncc1",
                "O=[N+]([O-])c1ccccc1", "Oc1ccccc1"]:
        assert has_aromatic_lowercase(smi), f"应识别芳香小写: {smi}"
    # 大写 → False（交替键）
    for smi in ["C1=CC=CC=C1", "CC1=CC=CC=C1", "C1=CNC=CC1",
                "O=[N+]([O-])C1=CC=CC=C1", "OC1=CC=CC=C1"]:
        assert not has_aromatic_lowercase(smi), f"大写不应识别为芳香: {smi}"
    # 非芳香分子不受影响
    for smi in ["CCO", "CC=O", "CCl", "[Na+].[Cl-]", "C1CCCCC1"]:
        assert not has_aromatic_lowercase(smi), f"非芳香误判: {smi}"
    # 元素名内小写不误判（Cl 的 l、Br 的 r）
    assert not has_aromatic_lowercase("CCBrCl")


def test_aromatic_ring_render_circle_vs_kekule():
    """渲染器层：STRUCT 渲染按大小写画圈/交替键（苯/甲苯/吡啶/硝基苯）。"""
    from renderers.structure import render_structure
    import re

    def circle_count(out: str) -> int:
        return sum(1 for l in out.splitlines()
                   if re.search(r"circle \(\d\.\d\d\)", l))

    cases = [
        # (名称, 小写SMILES, 大写SMILES)
        ("苯", "c1ccccc1", "C1=CC=CC=C1"),
        ("甲苯", "Cc1ccccc1", "CC1=CC=CC=C1"),
        ("吡啶", "c1ccncc1", "C1=CNC=CC1"),
        ("硝基苯", "O=[N+]([O-])c1ccccc1", "O=[N+]([O-])C1=CC=CC=C1"),
    ]
    for name, lo, up in cases:
        out_lo = render_structure(lo)
        out_up = render_structure(up)
        # 小写画圈（≥1 芳香圈）、大写不画圈
        assert circle_count(out_lo) == 1, f"{name} 小写应画圈"
        assert circle_count(out_up) == 0, f"{name} 大写不应画圈"
        # 大写保留交替键（线数 > 小写——双键平行线多出来）
        n_lo = sum(1 for l in out_lo.splitlines() if "--" in l)
        n_up = sum(1 for l in out_up.splitlines() if "--" in l)
        assert n_up > n_lo, f"{name} 大写线数应多于小写（交替键）"
