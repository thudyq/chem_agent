# -*- coding: utf-8 -*-
"""tests/test_corpus.py — 端到端语料库回归：读 tests/corpus/*.json，对真实/典型 LLM 坏输出
验证解析+校验行为（坏标记拦截/降级，好标记放行，不崩溃）。

新增语料：把真实 LLM 失败输出按 JSON 格式追加到 tests/corpus/ 即可，
无需改代码。expect.valid/invalid 为期望的合法/非法标记数（fake_rdkit
使 SMILES 语义校验生效，与真实 RDKit 行为一致）。
"""

import json
from pathlib import Path

import pytest

from core.tag_parser import parse_tags
from core.tag_validator import validate_tags

_CORPUS_DIR = Path(__file__).parent / "corpus"


def _load_corpus():
    cases = []
    for path in sorted(_CORPUS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            cases.extend(json.load(f))
    return cases


@pytest.mark.parametrize("case", _load_corpus(),
                         ids=lambda c: c["id"])
def test_corpus_case(case, fake_rdkit):
    valid, invalid = validate_tags(parse_tags(case["llm_output"]))
    assert len(valid) == case["expect"]["valid"], (
        f"合法标记数不符：期望 {case['expect']['valid']}，实际 {len(valid)}"
    )
    assert len(invalid) == case["expect"]["invalid"], (
        f"非法标记数不符：期望 {case['expect']['invalid']}，实际 {len(invalid)}"
    )
