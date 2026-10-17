"""Render one existing Chem_Agent tag from JSON without starting a server."""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

from core.tag_parser import parse_tags
from core.tag_validator import validate_tag
from renderers.registry import RENDERER_REGISTRY, render_tag
from utils.latex_compile import compile_tikz_to_png


def render_case(source: Path, output: Path) -> dict:
    case = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(case, dict) or set(case) != {"tag"}:
        raise ValueError('Input must be an object containing only a "tag" string')
    if not isinstance(case["tag"], str):
        raise ValueError('"tag" must be a string')
    text = case["tag"].strip()
    tags = parse_tags(text)
    if len(tags) != 1 or tags[0].raw != text:
        raise ValueError("Input must contain exactly one complete rendering tag")
    tag = tags[0]
    if tag.type not in RENDERER_REGISTRY:
        raise ValueError(f"Unsupported rendering tag: {tag.type}")
    validation = validate_tag(tag)
    if not validation.ok:
        raise ValueError(f"Tag validation rejected input: {validation.reason}")
    code = render_tag(tag)
    if not code or not code.lstrip().startswith(r"\begin{tikzpicture}"):
        raise ValueError("Renderer did not return a TikZ picture")
    png = compile_tikz_to_png(code)
    if not png:
        raise RuntimeError("PNG compilation failed; check existing LaTeX backend diagnostics")
    with Image.open(io.BytesIO(png)) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError("Compiler output is not a PNG")
        size = list(image.size)
    if output.suffix.lower() != ".png":
        raise ValueError("Output must have a .png extension")
    outputs = [output, output.with_suffix(".tikz"), output.with_suffix(".json")]
    if source.resolve() in [path.resolve() for path in outputs]:
        raise ValueError("Output paths must not overwrite the input case")
    report = {
        "tag": text,
        "pipeline": ["core.tag_parser.parse_tags", "core.tag_validator.validate_tag",
                     "renderers.registry.render_tag", "utils.latex_compile.compile_tikz_to_png"],
        "size": size,
        "sha256": hashlib.sha256(png).hexdigest(),
        "tikz_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(png)
    output.with_suffix(".tikz").write_text(code, encoding="utf-8", newline="\n")
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = render_case(args.input, args.output)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Render failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
