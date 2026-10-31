"""Labels and icon files for the add-on classifier.

A label is the lower-cased icon stem (e.g. 'iconaddon_honingstone'), so two add-ons that share one icon
are one class and icons that differ only by letter case are not split.
"""
import json
from pathlib import Path

ADDONS_JSON = Path("data/addons.json")
EXPORT_DIR = Path("data/game/ItemAddons")


def addon_labels_and_paths(addons_json: Path = ADDONS_JSON, export_dir: Path = EXPORT_DIR) -> tuple[list[str], list[Path], list[str]]:
    """(sorted labels, one icon path per label, labels whose icon file is missing)."""
    stems = sorted({a["iconStem"].lower() for a in json.loads(addons_json.read_text(encoding="utf-8"))["addons"]})
    files: dict[str, Path] = {}
    for path in sorted(export_dir.rglob("*.png")):
        files.setdefault(path.stem.lower(), path)
    labels = [stem for stem in stems if stem in files]
    return labels, [files[stem] for stem in labels], [stem for stem in stems if stem not in files]
