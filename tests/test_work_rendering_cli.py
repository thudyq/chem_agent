"""Ensure the offline adapter rejects inputs before compiling unvalidated tags."""

import json
from pathlib import Path

import pytest

from Chem_Agent_Work_Test import chem_render_cli as cli
from core.tag_parser import parse_tags
from core.tag_validator import validate_tag


@pytest.mark.parametrize("case", [
    {"tag": "[STRUCT:C]", "code": r"\input{secret}"},
    {"tag": 7},
    {"tag": "[STRUCT:C] [STRUCT:O]"},
    {"tag": "prefix [STRUCT:C]"},
    {"tag": "[REASONING]text[/REASONING]"},
    {"tag": "[STRUCT:not_a_smiles]"},
])
def test_rejected_input_never_compiles(case, tmp_path, monkeypatch):
    def unexpected_compile(*args, **kwargs):
        pytest.fail("Rejected input reached the compiler")

    monkeypatch.setattr(cli, "compile_tikz_to_png", unexpected_compile)
    source = tmp_path / "case.json"
    source.write_text(json.dumps(case), encoding="utf-8")
    output = tmp_path / "output.png"
    with pytest.raises(ValueError):
        cli.render_case(source, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["03_attack", "04_rearomatization"])
def test_mechanism_cases_use_valid_existing_contract(name):
    source = Path(__file__).resolve().parents[1] / "Chem_Agent_Work_Test" / f"{name}.json"
    case = json.loads(source.read_text(encoding="utf-8"))
    tags = parse_tags(case["tag"])
    assert len(tags) == 1
    result = validate_tag(tags[0])
    assert result.ok, result.reason
    code = cli.render_tag(tags[0])
    assert code.startswith(r"\begin{tikzpicture}")
    expected_arrows = 2 if name == "03_attack" else 1
    assert code.count(".. controls") == expected_arrows
