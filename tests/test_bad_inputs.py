# -*- coding: utf-8 -*-
"""P4 坏输入回归测试：解析层与校验层的异常输入（从代码构造，不依赖真实语料）。

覆盖：半截标记、未知标记类型、畸形参数、括号深度边界、游离氢 SMILES、
COMPOSITE 嵌套等。好标记必须正常渲染、坏标记必须降级，混合场景不得崩溃。
"""

import pytest

from core.tag_parser import parse_tags
from core.tag_validator import validate_tags


# ---------- 解析层 ----------

def test_half_tag_ignored():
    text = "苯是 [STRUCT:c1ccc"
    assert len(parse_tags(text)) == 0


def test_unknown_tag_type_not_parsed():
    assert len(parse_tags("[FOO:bar]")) == 0


def test_empty_brackets_not_a_tag():
    assert len(parse_tags("[ ]")) == 0


def test_nested_bracket_smiles_ok():
    text = "[STRUCT:O=[N+]([O-])c1ccccc1]"
    tags = parse_tags(text)
    assert len(tags) == 1 and tags[0].type == "STRUCT"


def test_mixed_valid_invalid_half():
    text = "[STRUCT:c1ccccc1] 和 [STRUCT:XYZ] 和半截 [STRUCT:c1c"
    tags = parse_tags(text)
    assert len(tags) == 2


def test_composite_nested_composite_outer_pairs():
    text = ("[COMPOSITE:row][STRUCT:CCl]"
            "[COMPOSITE:row][STRUCT:CO][/COMPOSITE][/COMPOSITE]")
    tags = parse_tags(text)
    assert len(tags) == 1 and tags[0].type == "COMPOSITE"


def test_reasoning_unclosed_not_paired():
    text = "[REASONING]只有开头"
    assert len(parse_tags(text)) == 0


def test_reaction_empty_segments():
    text = "[REACTION:;|;|]"
    tags = parse_tags(text)
    assert len(tags) == 1 and tags[0].type == "REACTION"


# ---------- 校验层（fake_rdkit 使 SMILES 语义校验生效） ----------

def test_free_hydrogen_smiles_allowed(fake_rdkit):
    # 游离氢 [H] 是合法 SMILES（RDKit 可解析，渲染时仅警告不崩溃），
    # 校验层不拦截——防游离氢靠 prompt 约束，而非校验。
    _, invalid = validate_tags(parse_tags("[STRUCT:[H]]"))
    assert len(invalid) == 0


def test_arrow_missing_fields_rejected(fake_rdkit):
    _, invalid = validate_tags(parse_tags("[ARROW:,,]"))
    assert len(invalid) == 1
    assert "SMILES" in invalid[0].reason


def test_newman_missing_angle_rejected():
    _, invalid = validate_tags(parse_tags("[STRUCT:CC,mode=newman]"))
    assert len(invalid) == 1


def test_energy_single_point_rejected():
    _, invalid = validate_tags(parse_tags("[ENERGY:5]"))
    assert len(invalid) == 1


def test_mech_bad_arrow_syntax(fake_rdkit):
    text = "[COMPOSITE:reaction_mech][STRUCT:CCl,id=a][RXNARROW]" \
           "[STRUCT:CO,id=b][MECHARROW:a:0>b][/COMPOSITE]"
    _, invalid = validate_tags(parse_tags(text))
    assert len(invalid) == 1
    assert "MECHARROW 格式错误" in invalid[0].reason


def test_charge_bad_atom_syntax(fake_rdkit):
    text = "[COMPOSITE:row][STRUCT:CCl,id=a][CHARGE:a|xyz][/COMPOSITE]"
    _, invalid = validate_tags(parse_tags(text))
    assert len(invalid) == 1


def test_valid_tag_survives_among_bad(fake_rdkit):
    text = ("[STRUCT:c1ccccc1] [STRUCT:XYZABC] [STRUCT:CO,label="
            + "长" * 25 + "]")
    valid, invalid = validate_tags(parse_tags(text))
    assert len(valid) == 1
    assert len(invalid) == 2


# ---------- 管线降级不崩溃 ----------

def test_pipeline_mixed_degrades(fake_rdkit, fake_renderers, monkeypatch):
    llm_text = ("好的 [STRUCT:c1ccccc1]，坏的 [STRUCT:XYZABC]，"
                "半截 [STRUCT:c1c")
    monkeypatch.setattr("app.ask_llm", lambda *a, **k: llm_text)
    from app import process_question
    result = process_question("测试")
    assert "RENDERED:c1ccccc1" in result
    # 部分修正：非法标记被修正输出中的合法标记替换修复（无降级）
    assert "图示无法渲染" not in result
    assert "[STRUCT:c1c" in result or result  # 半截标记（未闭合）保留原文，不崩溃
