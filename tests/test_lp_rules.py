# -*- coding: utf-8 -*-
"""tests/test_lp_rules.py — 孤对电子标注规范（Drawbacks 第 3 条）逐条单元测试。

运行: python -m pytest tests/test_lp_rules.py -v
"""

import pytest

import renderers.mol_primitives as mp

rdkit = pytest.importorskip("rdkit", reason="rdkit 未安装，跳过规范测试")


def _angles(smiles, idx, **kw):
    mol = mp.prepare_mol(smiles, **kw)
    return mp.lone_pair_angles(mol, idx)


def test_rule_pair3_block1():
    """pair=3, block=1：按 block 方位分离（block 左 → 上右下）。"""
    assert _angles("CCl", 1) == [90.0, 0.0, 270.0]      # C—Cl 键在左(180°)
    # block 右（OH- 的标签 H 在右）→ 上左下（顺序无关）
    assert sorted(_angles("[OH-]", 0)) == [90.0, 180.0, 270.0]


def test_rule_pair2_block2_water_special():
    """pair=2, block=2 特例（水型左右斜键朝上）→ 左下/右下张开45°。"""
    mol = mp.prepare_mol("O", add_hs=True, kekulize=True, use_prepare=False)
    angles = mp.lone_pair_angles(mol, 0)
    assert angles == [225.0, 315.0]


def test_rule_pair2_block2_default():
    """pair=2, block=2 默认：正交避开两个 block 方位（甲醇 O：键左 + 标签 H 右）。"""
    assert _angles("CO", 1) == [90.0, 270.0]


def test_rule_pair2_block1_trigonal():
    """pair=2, block=1：三者夹角≈120°（block±120°，30° 整数倍）。"""
    # 甲醛氧 C=O：2 对电子、1 个 block（双键方向）
    mol = mp.prepare_mol("C=O")
    angs = mp.lone_pair_angles(mol, 1)
    b = mp._bond_blocks(mol, 1)[0] % 360.0
    assert len(angs) == 2
    for a in angs:
        assert abs(mp._ang_diff(a, b) - 120.0) < 1.0
        assert a % 30.0 == 0.0


def test_rule_pair1_block3():
    """pair=1, block=3：正交四向中取未占用方位（氨 N：三键占右左下 → 上）。"""
    mol = mp.prepare_mol("N", add_hs=True, kekulize=True, use_prepare=False)
    assert mp.lone_pair_angles(mol, 0) == [90.0]


def test_rule_pair1_block2():
    """pair=1, block=2：30° 整数倍中离两个 block 最远（甲胺 N：C 键 + 标签 H）。"""
    mol = mp.prepare_mol("NC")
    angs = mp.lone_pair_angles(mol, 0)
    blocked = mp._bond_blocks(mol, 0)
    assert len(angs) == 1
    assert angs[0] % 30.0 == 0.0
    for b in blocked:
        assert mp._ang_diff(angs[0], b % 360.0) >= 60.0


def test_rule_pair1_block1_opposite():
    """pair=1, block=1：孤对电子在 block 正对侧（甲基负离子 CH3-）。"""
    mol = mp.prepare_mol("[CH3-]")
    angs = mp.lone_pair_angles(mol, 0)
    blocked = mp._bond_blocks(mol, 0)
    assert angs == [(blocked[0] + 180.0) % 360.0]


def test_unlisted_core_principle():
    """未列出（pair=4, block≤1）：核心原则，正交四向。"""
    assert _angles("[Cl-]", 0) == [90.0, 180.0, 270.0, 0.0]


def test_charge_circle_rendering():
    """电荷：主标签不含电荷；圆圈电荷节点在 45° 右上角且与电子点不重叠。"""
    mol = mp.prepare_mol("[OH-]")
    assert mp.atom_main_label(mol.GetAtomWithIdx(0)) == "OH"
    assert mp.atom_charge_label(mol.GetAtomWithIdx(0)) == "$-$"
    charge = mp.charge_tikz(mol, 0)
    assert charge is not None and "circle" in charge
    # 电荷位置已计入避让，45°/135° 方向无孤对电子
    angs = [round(a) for a in mp.lone_pair_angles(mol, 0)]
    assert 45 not in angs and 135 not in angs


def test_charge_angle_side_selection():
    """右侧标签氢阻碍且左侧无阻碍 → 电荷圈左上（135°）；否则右上（45°）。"""
    mol = mp.prepare_mol("[OH-]")          # 右侧 H 阻碍、左侧无键
    assert mp._charge_angle(mol, 0) == 135.0
    mol = mp.prepare_mol("[Cl-]")          # 无标签氢
    assert mp._charge_angle(mol, 0) == 45.0
    mol = mp.prepare_mol("[CH3-]")         # 右侧 H 阻碍、左侧无键
    assert mp._charge_angle(mol, 0) == 135.0
