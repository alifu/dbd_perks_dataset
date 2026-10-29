"""Measure accuracy on REAL photos of perk icons (the only number that matters).

Layout: real_photos/<perk_id>/<any name>.jpg|png  - crops of the icon inside its diamond, as the app crops them.
python3 evaluate_real.py [--model models/v2/DBDPerkClassifierV2.mlpackage] [--root real_photos]
"""
import argparse
import sys
from pathlib import Path

import coremltools as ct
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}
INPUT_SIZE = 128
INPUT_NAME = "input_1"


def collect(root: Path) -> list[tuple[Path, str]]:
    return [(path, folder.name) for folder in sorted(root.iterdir()) if folder.is_dir()
            for path in sorted(folder.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]


def top3(model, image_path: Path) -> list[str]:
    image = Image.open(image_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    probabilities = model.predict({INPUT_NAME: image})["classLabel_probs"]
    return sorted(probabilities, key=probabilities.get, reverse=True)[:3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/v2/DBDPerkClassifierV2.mlpackage"))
    parser.add_argument("--root", type=Path, default=Path("real_photos"))
    args = parser.parse_args()
    if not args.root.is_dir():
        print(f"❌ {args.root} not found. Add crops as {args.root}/<perk_id>/photo.jpg", file=sys.stderr)
        return 1
    samples = collect(args.root)
    if not samples:
        print(f"❌ no images under {args.root}", file=sys.stderr)
        return 1
    model = ct.models.MLModel(str(args.model))
    hit1 = hit3 = 0
    for path, expected in samples:
        ranked = top3(model, path)
        hit1 += ranked[0] == expected
        hit3 += expected in ranked
        if ranked[0] != expected:
            print(f"✗ {path.name}: expected {expected}, got {ranked}")
    print(f"✅ {len(samples)} photos: top-1 {hit1 / len(samples):.1%}, top-3 {hit3 / len(samples):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
