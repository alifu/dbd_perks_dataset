"""Map the v1 model's class labels ('File:IconPerks_x') to perks.json ids.

python -m crawler.legacy_labels  ->  data/legacy_label_map.json
"""
import json
import sys
from pathlib import Path

PERKS_PATH = Path("data/perks.json")
LEGACY_METADATA_PATH = Path("dbd_perks_dataset/metadata.json")
OUTPUT_PATH = Path("data/legacy_label_map.json")
LEGACY_PREFIX = "File:"


def build_label_map(perks: list[dict], legacy_labels: list[str]) -> dict:
    id_by_icon_stem = {perk["iconFile"].removesuffix(".png"): perk["id"] for perk in perks}
    mapped, unmapped = {}, []
    for label in legacy_labels:
        perk_id = id_by_icon_stem.get(label.removeprefix(LEGACY_PREFIX))
        if perk_id:
            mapped[label] = perk_id
        else:
            unmapped.append(label)
    covered = set(mapped.values())
    return {
        "map": mapped,
        "unmappedLegacyLabels": unmapped,
        "perksMissingFromLegacyModel": sorted(perk["id"] for perk in perks if perk["id"] not in covered),
    }


def main() -> int:
    perks = json.loads(PERKS_PATH.read_text(encoding="utf-8"))["perks"]
    legacy = list(json.loads(LEGACY_METADATA_PATH.read_text(encoding="utf-8")))
    result = build_label_map(perks, legacy)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"✅ {len(result['map'])} mapped, {len(result['unmappedLegacyLabels'])} stale legacy labels, "
        f"{len(result['perksMissingFromLegacyModel'])} perks the old model cannot recognise"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
