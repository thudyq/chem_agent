# -*- coding: utf-8 -*-
"""游离氢组分（[H+] 质子 / [H] 氢自由基 / [H-] 氢负离子）渲染测试。

策略（20260808 起）：孤立 H 是合法组分，prepare_mol 保留并正常绘制
（[H+] 电荷圈、[H] 单电子点、[H-] 电荷圈 + 孤对电子），水仍写 O。

运行: python -m pytest tests/test_free_hydrogen.py -v
"""

import pytest

from renderers.mol_primitives import prepare_mol, scale_mol_coords
from renderers.layout import molecule_scope_lines

rdkit = pytest.importorskip("rdkit", reason="rdkit 未安装，跳过渲染测试")


def _scope(smiles: str) -> str:
    mol = prepare_mol(smiles)
    assert mol is not None, f"prepare_mol 失败: {smiles}"
    scale_mol_coords(mol, 0.8)
    return "\n".join(
        molecule_scope_lines(mol, (0.0, 0.0), show_lone_pairs=True))


def test_prepare_mol_keeps_free_hydrogen():
    """prepare_mol 保留孤立 H（不再按 C4 旧策略删除为 0 原子）。"""
    for smi in ("[H+]", "[H]", "[H-]"):
        mol = prepare_mol(smi)
        assert mol is not None and mol.GetNumAtoms() == 1, smi
    mol = prepare_mol("CCO.[H+]")
    assert mol is not None and mol.GetNumAtoms() == 4


def test_proton_renders_label_and_charge_circle():
    """[H+]（质子）渲染为 H 标签 + ⊕ 圆圈电荷，无电子点（0 电子）。"""
    out = _scope("[H+]")
    assert "{H}" in out
    assert "\\node[draw, circle" in out          # ⊕ 电荷圈
    assert "\\fill" not in out                   # 质子无电子


def test_hydrogen_radical_renders_single_dot():
    """[H]（氢自由基）渲染为 H 标签 + 1 个单电子点，无电荷圈。"""
    out = _scope("[H]")
    assert "{H}" in out
    assert "\\node[draw, circle" not in out
    assert out.count("\\fill") == 1              # 1 个自由基单电子点


def test_hydride_renders_charge_circle_and_lone_pair():
    """[H-]（氢负离子）渲染为 H 标签 + ⊖ 圆圈电荷 + 1 对孤对电子（2 点）。"""
    out = _scope("[H-]")
    assert "{H}" in out
    assert "\\node[draw, circle" in out          # ⊖ 电荷圈
    assert out.count("\\fill") == 2              # 1 对孤对电子

