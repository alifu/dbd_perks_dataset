"""Measure accuracy on REAL photos of perk icons (the only number that matters).

Layout: real_photos/<perk_id>/<any name>.jpg|png  - crops of the icon inside its diamond, as the app crops them.
python3 evaluate_real.py [--model models/v2/DBDPerkClassifierV2.mlpackage | --weights models/v2/perk_classifier.pt]
                         [--root real_photos] [--json results.json]

--json writes one record per crop (expected id, top-3, probability of the expected class) plus the summary with
95% Wilson intervals, so figure scripts read numbers instead of console text.
"""
import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}
INPUT_SIZE = 128
INPUT_NAME = "input_1"
WILSON_Z = 1.96  # 95%


def wilson_interval(hits: int, total: int, z: float = WILSON_Z) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; stays inside [0, 1] and is sane for tiny samples."""
    if total == 0:
        return 0.0, 1.0
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def collect(root: Path) -> list[tuple[Path, str]]:
    return [(path, folder.name) for folder in sorted(root.iterdir()) if folder.is_dir()
            for path in sorted(folder.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]


def load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))


def coreml_scorer(model_path: Path):
    import coremltools as ct
    model = ct.models.MLModel(str(model_path))
    return lambda image: dict(model.predict({INPUT_NAME: image})["classLabel_probs"])


def torch_scorer(weights_path: Path):
    """Same preprocessing as the Core ML export: RGB 0..1, normalised inside the network."""
    import numpy as np
    import torch
    from train_perks_v2 import PerkClassifier
    labels = json.loads((weights_path.parent / "labels.json").read_text(encoding="utf-8"))
    model = PerkClassifier(len(labels), pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    @torch.no_grad()
    def score(image: Image.Image) -> dict:
        tensor = torch.from_numpy(np.asarray(image, np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        probabilities = torch.softmax(model(tensor), dim=1)[0].tolist()
        return dict(zip(labels, probabilities))
    return score


def evaluate(scorer, samples: list[tuple[Path, str]]) -> list[dict]:
    records = []
    for path, expected in samples:
        probabilities = scorer(load_image(path))
        ranked = sorted(probabilities, key=probabilities.get, reverse=True)[:3]
        records.append({"expected": expected, "top3": ranked, "pCorrect": probabilities.get(expected, 0.0),
                        "top1Hit": ranked[0] == expected, "top3Hit": expected in ranked, "file": path.name})
    return records


def summarize(records: list[dict]) -> dict:
    total = len(records)
    hit1 = sum(r["top1Hit"] for r in records)
    hit3 = sum(r["top3Hit"] for r in records)
    return {"crops": total, "top1": hit1 / total, "top1Wilson95": wilson_interval(hit1, total),
            "top3": hit3 / total, "top3Wilson95": wilson_interval(hit3, total),
            "meanPCorrect": sum(r["pCorrect"] for r in records) / total}


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path, default=Path("models/v2/DBDPerkClassifierV2.mlpackage"))
    source.add_argument("--weights", type=Path, help="PyTorch .pt next to its labels.json (no Core ML export needed)")
    parser.add_argument("--root", type=Path, default=Path("real_photos"))
    parser.add_argument("--json", type=Path, help="write per-crop records and the summary here")
    args = parser.parse_args()
    if not args.root.is_dir():
        print(f"❌ {args.root} not found. Add crops as {args.root}/<perk_id>/photo.jpg", file=sys.stderr)
        return 1
    samples = collect(args.root)
    if not samples:
        print(f"❌ no images under {args.root}", file=sys.stderr)
        return 1
    scorer = torch_scorer(args.weights) if args.weights else coreml_scorer(args.model)
    records = evaluate(scorer, samples)
    summary = summarize(records)
    for record in records:
        if not record["top1Hit"]:
            print(f"✗ {record['file']}: expected {record['expected']}, got {record['top3']}")
    low, high = summary["top1Wilson95"]
    print(f"✅ {summary['crops']} photos: top-1 {summary['top1']:.1%} (95% CI {low:.0%}–{high:.0%}), "
          f"top-3 {summary['top3']:.1%}, mean P(correct) {summary['meanPCorrect']:.1%}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "records": records}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
