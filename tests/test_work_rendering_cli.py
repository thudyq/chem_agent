"""Ensure the offline adapter rejects inputs before compiling unvalidated tags."""

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from cli import render as cli
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


@pytest.mark.parametrize("name", ["attack", "rearomatization"])
def test_mechanism_cases_use_valid_existing_contract(name):
    source = Path(__file__).resolve().parents[1] / "examples" / "work" / f"{name}.json"
    case = json.loads(source.read_text(encoding="utf-8"))
    tags = parse_tags(case["tag"])
    assert len(tags) == 1
    result = validate_tag(tags[0])
    assert result.ok, result.reason
    code = cli.render_tag(tags[0])
    assert code.startswith(r"\begin{tikzpicture}")
    expected_arrows = 2 if name == "attack" else 1
    assert code.count(".. controls") == expected_arrows


@pytest.mark.parametrize("debug", [False, True])
def test_output_contains_only_requested_files(tmp_path, monkeypatch, debug):
    source = tmp_path / "case.json"
    source.write_text(json.dumps({"tag": "[STRUCT:CC]"}), encoding="utf-8")
    png = io.BytesIO()
    Image.new("RGB", (20, 20), "black").save(png, format="PNG")
    monkeypatch.setattr(cli, "compile_tikz_to_png", lambda code: png.getvalue())
    output = tmp_path / "rendered" / "figure.png"

    assert cli.render_case(source, output, debug=debug) == output.resolve()
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (20, 20)
    expected = {"figure.png", "figure.tikz"} if debug else {"figure.png"}
    assert {path.name for path in output.parent.iterdir()} == expected
    if debug:
        assert output.with_suffix(".tikz").read_text(encoding="utf-8").startswith(
            r"\begin{tikzpicture}")


def test_failed_compilation_creates_no_output(tmp_path, monkeypatch):
    source = tmp_path / "case.json"
    source.write_text(json.dumps({"tag": "[STRUCT:CC]"}), encoding="utf-8")
    monkeypatch.setattr(cli, "compile_tikz_to_png", lambda code: None)
    output = tmp_path / "rendered" / "figure.png"
    with pytest.raises(RuntimeError, match="PNG compilation failed"):
        cli.render_case(source, output, debug=True)
    assert not output.parent.exists()


@pytest.mark.parametrize("source_name,output_name,debug", [
    ("case.json", "figure.svg", False),
    ("case.png", "case.png", False),
    ("figure.tikz", "figure.png", True),
])
def test_invalid_output_never_compiles(tmp_path, monkeypatch, source_name, output_name, debug):
    def unexpected_compile(*args, **kwargs):
        pytest.fail("Invalid output path reached the compiler")

    monkeypatch.setattr(cli, "compile_tikz_to_png", unexpected_compile)
    source = tmp_path / source_name
    original = json.dumps({"tag": "[STRUCT:CC]"})
    source.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError):
        cli.render_case(source, tmp_path / output_name, debug=debug)
    assert source.read_text(encoding="utf-8") == original
