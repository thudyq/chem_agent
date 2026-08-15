# -*- coding: utf-8 -*-
"""原子标签单元测试：水 H₂O 特例 + 标签方向翻转（Drawbacks 第 6 条）。

替代端到端视觉验证：直接测 atom_label/atom_main_label 的标签文本
与 _label_flip_for 的方向判定，保证：
- 水（O 只连 H）标签为 H₂O 而非 OH₂；
- 键端在标签右侧时标签翻转（OH→HO），使键连的元素符号靠近键端；
- 竖直键 / 多重原子邻居 / 无重原子邻居不翻转。
"""

import pytest

rdkit = pytest.importorskip("rdkit", reason="rdkit 未安装，跳过标签测试")

from renderers.mol_primitives import (
    _label_flip_for, _only_h_neighbors, atom_label, atom_main_label,
    prepare_mol, scale_mol_coords,
)


def _atom(mol, sym):
    return next(a for a in mol.GetAtoms() if a.GetSymbol() == sym)


# ---------------- 水 H₂O 特例 ----------------

def test_water_label_is_h2o():
    mol = prepare_mol("O")
    o = _atom(mol, "O")
    assert _only_h_neighbors(o)
    assert atom_main_label(o) == "H$_{2}$O"
    assert atom_label(o) == "H$_{2}$O"


def test_ammonia_label_is_nh3():
    """问题 2（NH3→H3N）：N/P/B 氢化物元素在前（NH3/PH3/BH3），
    仅 F/Cl/Br/I/O/S/Se/Te 的氢化物 H 在前（HF/HCl/H2O/H2S）。"""
    for smi, expect in [("N", "NH$_{3}$"), ("P", "PH$_{3}$"),
                        ("B", "BH$_{3}$")]:
        mol = prepare_mol(smi)
        a = _atom(mol, smi)
        assert atom_main_label(a) == expect, f"{smi} → {atom_main_label(a)}"
        assert atom_label(a) == expect, f"{smi} → {atom_label(a)}"


def test_alcohol_oh_not_h2o():
    mol = prepare_mol("CCO")
    o = _atom(mol, "O")
    assert not _only_h_neighbors(o)          # O 连 C
    assert atom_main_label(o) == "OH"        # 醇羟基保持 OH


# ---------------- 标签方向翻转 ----------------

def test_flip_argument_oh_to_ho():
    mol = prepare_mol("CCO")
    o = _atom(mol, "O")
    assert atom_main_label(o) == "OH"
    assert atom_main_label(o, flip=True) == "HO"
    assert atom_label(o, flip=True) == "HO"


def test_flip_does_not_affect_water():
    mol = prepare_mol("O")
    o = _atom(mol, "O")
    assert atom_main_label(o, flip=True) == "H$_{2}$O"   # 水保持 H₂O


def test_flip_direction_bond_from_right():
    """键端在标签右侧（重原子在右侧）→ 翻转；左侧/竖直 → 不翻转。"""
    mol = prepare_mol("OC")
    o = _atom(mol, "O")
    c = _atom(mol, "C")
    conf = mol.GetConformer()
    conf.SetAtomPosition(o.GetIdx(), (0, 0, 0))
    conf.SetAtomPosition(c.GetIdx(), (2, 0, 0))    # C 在 O 右侧
    assert _label_flip_for(mol, o.GetIdx())
    conf.SetAtomPosition(c.GetIdx(), (-2, 0, 0))   # C 在 O 左侧
    assert not _label_flip_for(mol, o.GetIdx())
    conf.SetAtomPosition(c.GetIdx(), (0, 2, 0))    # 竖直（上方）→ 不翻转
    assert not _label_flip_for(mol, o.GetIdx())
    conf.SetAtomPosition(c.GetIdx(), (0, -2, 0))   # 竖直（下方）→ 不翻转
    assert not _label_flip_for(mol, o.GetIdx())


def test_flip_no_heavy_neighbor():
    mol = prepare_mol("O")
    o = _atom(mol, "O")
    assert not _label_flip_for(mol, o.GetIdx())


def test_flip_multi_heavy_neighbor():
    """醚 O 连 2 个 C，方向不明确 → 不翻转。"""
    mol = prepare_mol("CCOC")
    o = _atom(mol, "O")
    assert not _label_flip_for(mol, o.GetIdx())


# ---------------- 综合：scope 输出含翻转标签 ----------------

def test_scope_flips_label():
    """键从右侧连 O 时，molecule_scope_lines 输出含 HO 标签。"""
    from renderers.layout import molecule_scope_lines
    mol = prepare_mol("OC")
    scale_mol_coords(mol, 0.8)
    conf = mol.GetConformer()
    o = _atom(mol, "O")
    c = _atom(mol, "C")
    conf.SetAtomPosition(o.GetIdx(), (0, 0, 0))
    conf.SetAtomPosition(c.GetIdx(), (2, 0, 0))   # 键端在 O 标签右侧
    lines = molecule_scope_lines(mol, (0.0, 0.0))
    assert any("{HO}" in ln for ln in lines), lines
    # 反向（键端在左侧）→ OH
    conf.SetAtomPosition(c.GetIdx(), (-2, 0, 0))
    lines = molecule_scope_lines(mol, (0.0, 0.0))
    assert any("{OH}" in ln for ln in lines), lines


def test_dot_center_flip_aware():
    """_dot_center 感知标签翻转：键端在右侧（标签 HO）时元素符号中心
    右移（+0.13），而非按未翻转标签 OH 左移（-0.13）——电荷/孤对
    电子点错位约半个标签宽的回归锚点。"""
    from renderers.mol_primitives import _label_flip_for, symbol_center
    mol = prepare_mol("OC")
    conf = mol.GetConformer()
    o = _atom(mol, "O")
    c = _atom(mol, "C")
    conf.SetAtomPosition(o.GetIdx(), (0, 0, 0))
    conf.SetAtomPosition(c.GetIdx(), (2, 0, 0))   # 键端在右侧 → 翻转
    assert _label_flip_for(mol, o.GetIdx())
    assert symbol_center(mol, o.GetIdx())[0] > 0  # 右移（HO 的 O 在右侧）
    # 反向（键端在左侧）→ 不翻转，中心左移（OH 的 O 在左侧）
    conf.SetAtomPosition(c.GetIdx(), (-2, 0, 0))
    assert not _label_flip_for(mol, o.GetIdx())
    assert symbol_center(mol, o.GetIdx())[0] < 0


def test_is_formula_label():
    """纯化学式 label 识别（含自由基 ·）：决定是否在分子下方重复显示。"""
    from renderers.mol_primitives import is_formula_label
    # 化学式（含自由基符号）→ True（不重复显示）
    assert is_formula_label("Cl·")
    assert is_formula_label("·CH3")
    assert is_formula_label("CH3·")
    assert is_formula_label("CH3Cl")
    assert is_formula_label("OH-")
    assert is_formula_label("ClH")
    # 中文/角色标注/含结构括号 → False（显示在下方）
    assert not is_formula_label("底物")
    assert not is_formula_label("产物")
    assert not is_formula_label("质子化乙醇")
    assert not is_formula_label("(R)-乳酸")
    assert not is_formula_label("过渡态(示意)")
