"""Offline checks for the GLM structure-recognition pilot helpers."""

import pytest

pytest.importorskip("rdkit")

from experiments.glm_self_recognition import _extract_json, evaluate


def test_extract_json_accepts_fenced_json():
    assert _extract_json('```json\n[{"smiles":"CCO"}]\n```') == [
        {"smiles": "CCO"}
    ]


def test_evaluate_compares_canonical_structure_not_smiles_spelling():
    assert evaluate("OCC", "CCO")


def test_evaluate_rejects_invalid_or_different_structure():
    assert not evaluate("not-smiles", "CCO")
    assert not evaluate("CCN", "CCO")
