"""Generate fixed RDKit inputs for the local image-display experiment."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

from PIL import Image
from rdkit import Chem, rdBase
from rdkit.Chem import Draw


OUTPUT = Path(__file__).resolve().parent / "output"
MOLECULES = {
    "01_benzene": "c1ccccc1",
    "02_nitrobenzene": "O=[N+]([O-])c1ccccc1",
}


def inspect_png(path: Path) -> dict:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.size != (600, 400):
            raise ValueError(f"Unexpected image format or size: {path}")
        if image.convert("L").getextrema()[0] >= 250:
            raise ValueError(f"Blank image: {path}")
    return {
        "file": path.name,
        "size": [600, 400],
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Read existing PNGs and check their manifest without redrawing.",
    )
    args = parser.parse_args()
    manifest_path = OUTPUT / "manifest.json"
    if args.verify_only:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = [inspect_png(OUTPUT / f"{name}.png") for name in MOLECULES]
        if actual != expected["images"]:
            raise ValueError("PNG files differ from the generation manifest")
        print("Verified both existing PNGs; no files changed.")
        return

    OUTPUT.mkdir(exist_ok=True)
    images = []
    for name, smiles in MOLECULES.items():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        path = OUTPUT / f"{name}.png"
        Draw.MolToFile(molecule, str(path), size=(600, 400))
        images.append(inspect_png(path))
        print(f"Generated: {path}")
    manifest_path.write_text(
        json.dumps({
            "generator": "rdkit.Chem.Draw.MolToFile",
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "smiles": MOLECULES,
            "images": images,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
