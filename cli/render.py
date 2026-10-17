"""Render one existing Chem_Agent tag from JSON without starting a server."""

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image

from core.tag_parser import parse_tags
from core.tag_validator import validate_tag
from renderers.registry import RENDERER_REGISTRY, render_tag
from utils.latex_compile import compile_tikz_to_png


def render_case(source: Path, output: Path, *, debug: bool = False) -> Path:
    if output.suffix.lower() != ".png":
        raise ValueError("Output must have a .png extension")
    outputs = [output, output.with_suffix(".tikz")] if debug else [output]
    if source.resolve() in [path.resolve() for path in outputs]:
        raise ValueError("Output paths must not overwrite the input case")
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(png)
    if debug:
        output.with_suffix(".tikz").write_text(code, encoding="utf-8", newline="\n")
    return output.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true", help="Also save the generated TikZ source")
    args = parser.parse_args()
    try:
        output = render_case(args.input, args.output, debug=args.debug)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Render failed: {error}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
