"""Write fake loadout screenshots + ground truth for the iOS locator tests.

python3 make_scene_fixtures.py --out ../XCodeProject/DBD-Perks/"DBD PerksTests"/Fixtures/scenes
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from synth.compose import load_icon
from synth.scene import LAYOUTS, render_scene
from train_perks_v2 import ICONS_DIR, load_labels

SCENES = 32
SEED = 4242
SEVERITY_RANGE = (0.15, 0.7)
PERKS_PER_SCENE = (3, 4)   # a few partial loadouts
JPEG_QUALITY = 85


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=SCENES)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    manifest = []
    for index in range(args.count):
        rng = np.random.default_rng([SEED, index])
        count = 3 if index % 8 == 7 else 4
        ids = [labels[i] for i in rng.choice(len(labels), count, replace=False)]
        layout = LAYOUTS[index % len(LAYOUTS)]
        image, placed = render_scene(rng, [(i, load_icon(ICONS_DIR / f"{i}.png")) for i in ids], layout,
                                     rng.uniform(*SEVERITY_RANGE))
        name = f"scene_{index:02d}.jpg"
        cv2.imwrite(str(args.out / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        manifest.append({"file": name, "layout": layout, "width": image.shape[1], "height": image.shape[0],
                         "perks": [{"id": p.perk_id, "cx": round(p.cx, 1), "cy": round(p.cy, 1), "diagonal": round(p.diagonal, 1)} for p in placed]})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"✅ {len(manifest)} scenes → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
