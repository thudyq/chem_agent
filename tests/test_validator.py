# -*- coding: utf-8 -*-
"""core/tag_validator.py 标记契约校验层单元测试（P1）。

用假 rdkit（conftest.py 的 fake_rdkit fixture）解耦真实 RDKit，
保证测试在任何环境行为一致。
"""

import pytest

import core.tag_validator as tv
from app import process_question
from core.tag_parser import parse_tags


def _validate(text):
    return tv.validate_tags(parse_tags(text))


def test_valid_smiles_pass(fake_rdkit):
    valid, invalid = _validate("[STRUCT:c1ccccc1] [ENERGY:0,108,-20]")
    assert len(invalid) == 0
    assert len(valid) == 2


def test_invalid_smiles_rejected(fake_rdkit):
    _, invalid = _validate("[STRUCT:XYZABC]")
    assert len(invalid) == 1
    assert "无效 SMILES" in invalid[0].reason


def test_empty_smiles_rejected():
    _, invalid = _validate("[STRUCT:]")
    assert len(invalid) == 1
    assert "SMILES 为空" in invalid[0].reason


def test_long_label_rejected():
    _, invalid = _validate("[STRUCT:CO,label=" + "长" * 25 + "]")
    assert len(invalid) == 1
    assert "label 过长" in invalid[0].reason


def test_newman_bad_angle_rejected():
    _, invalid = _validate("[NEWMAN:CC,xyz]")
    assert len(invalid) == 1
    assert "角度" in invalid[0].reason


def test_newman_angle_out_of_range():
    _, invalid = _validate("[NEWMAN:CC,450]")
    assert len(invalid) == 1


def test_energy_bad_value_rejected():
    _, invalid = _validate("[ENERGY:0,abc]")
    assert len(invalid) == 1
    assert "不是数字" in invalid[0].reason


def test_energy_too_few_points_rejected():
    _, invalid = _validate("[ENERGY:5]")
    assert len(invalid) == 1
    assert "至少需要 2" in invalid[0].reason


def test_composite_unknown_layout_rejected():
    _, invalid = _validate("[COMPOSITE:foo][STRUCT:CCl][/COMPOSITE]")
    assert len(invalid) == 1


def test_composite_no_struct_rejected():
    _, invalid = _validate("[COMPOSITE:row][PLUS][/COMPOSITE]")
    assert len(invalid) == 1


def test_composite_energy_at_out_of_range(fake_rdkit):
    text = ("[COMPOSITE:energy][ENERGY:0,108,-20]"
            "[STRUCT:CCl,at=0][STRUCT:CO,at=9][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "at=" in invalid[0].reason


def test_composite_mecharrow_unknown_id(fake_rdkit):
    text = ("[COMPOSITE:reaction_mech][STRUCT:CCl,id=r0][RXNARROW]"
            "[STRUCT:CO,id=p0][MECHARROW:ghost:0>r0:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "未知组件" in invalid[0].reason


def test_composite_mecharrow_atom_out_of_range(fake_rdkit):
    text = ("[COMPOSITE:reaction_mech][STRUCT:CCl,id=r0][RXNARROW]"
            "[STRUCT:CO,id=p0][MECHARROW:r0:5>p0:0][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "超出原子范围" in invalid[0].reason


def test_composite_real_mechanism_passes(fake_rdkit):
    text = ("[COMPOSITE:reaction_mech][STRUCT:CCO,label=乙醇,id=nu][PLUS]"
            "[STRUCT:CC[OH2+],label=质子化的乙醇,id=pe][RXNARROW]"
            "[STRUCT:CC[OH+]CC,label=质子化的乙醚][PLUS][STRUCT:O,label=水]"
            "[MECHARROW:nu:2>pe:1][MECHARROW:pe:1-2>pe:2][/COMPOSITE]")
    _, invalid = _validate(text)
    assert len(invalid) == 0


def test_composite_charge_unknown_id(fake_rdkit):
    text = "[COMPOSITE:row][STRUCT:CCl,id=a][CHARGE:ghost|0:δ-][/COMPOSITE]"
    _, invalid = _validate(text)
    assert len(invalid) == 1
    assert "未知组件" in invalid[0].reason


def test_pipeline_degrades_invalid_tags(fake_rdkit, fake_renderers, monkeypatch):
    llm_text = ("苯是 [STRUCT:c1ccccc1]，无效 [STRUCT:XYZABC]，"
                "超长 [STRUCT:CO,label=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]。")
    monkeypatch.setattr("app.ask_llm", lambda *a, **k: llm_text)
    result = process_question("测试")
    assert "RENDERED:c1ccccc1" in result
    assert "无效 SMILES「XYZABC」" in result
    assert "label 过长" in result
    assert "[STRUCT:XYZABC]" not in result
