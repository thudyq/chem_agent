# -*- coding: utf-8 -*-
"""tests/test_bond_segments.py — bond_segments 环内双键偏移方向单元测试。

运行: python -m pytest tests/test_bond_segments.py -v
"""

import math

import pytest

from renderers.mol_primitives import atom_pos, bond_segments, prepare_mol

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
    """几何 Kekulé 规整：苯/苯胺/硝基苯的双键位置一致（不像共振式）。"""
    def ring_double_angles(mol):
        ri = mol.GetRingInfo()
        ring = ri.AtomRings()[0]
        cx = sum(atom_pos(mol, i)[0] for i in ring) / len(ring)
        cy = sum(atom_pos(mol, i)[1] for i in ring) / len(ring)
        angs = []
        for k, ai in enumerate(ring):
            aj = ring[(k + 1) % len(ring)]
            b = mol.GetBondBetweenAtoms(ai, aj)
            if b.GetBondTypeAsDouble() >= 1.5:
                x1, y1 = atom_pos(mol, ai)
                x2, y2 = atom_pos(mol, aj)
                angs.append(round(math.degrees(
                    math.atan2((y1 + y2) / 2 - cy, (x1 + x2) / 2 - cx)) % 360))
        return sorted(angs)

    ref = ring_double_angles(prepare_mol("c1ccccc1"))
    assert len(ref) == 3
    assert ring_double_angles(prepare_mol("c1ccccc1N")) == ref
    assert ring_double_angles(prepare_mol("O=[N+]([O-])c1ccccc1")) == ref
