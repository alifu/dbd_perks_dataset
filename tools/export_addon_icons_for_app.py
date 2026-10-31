"""Resize the exported game icons into the iOS app's Resources folders.

python3 tools/export_addon_icons_for_app.py --out "<app>/DBD Perks/Resources/AddonIcons"
    add-ons -> <iconStem lower-case>.png
python3 tools/export_addon_icons_for_app.py --target perks --out "<app>/DBD Perks/Resources/PerkIcons"
    perks   -> perk_<perk id>.png   (from data/icons/<id>.png)

The icons are Behaviour Interactive's copyrighted assets: the iOS repo git-ignores that folder, so the
icons end up only in app builds made on this machine. Never commit them.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

DEFAULT_ADDONS = Path("data/addons.json")
DEFAULT_EXPORT = Path("data/game/ItemAddons")
ICON_SIZE = 128


def index_exports(export_dir: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for path in sorted(export_dir.rglob("*.png")):
        found.setdefault(path.name.lower(), []).append(path)
    return found


def export_icons(addons: list[dict], exports: dict[str, list[Path]], out_dir: Path) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {"written": [], "missing": [], "ambiguous": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in sorted({a["iconStem"] for a in addons}):
        candidates = exports.get(f"{stem}.png".lower(), [])
        if not candidates:
            report["missing"].append(stem)
            continue
        if len({hashlib.md5(p.read_bytes()).hexdigest() for p in candidates}) > 1:
            report["ambiguous"].append(stem)  # first copy is used; listed so a human can check
        with Image.open(candidates[0]) as image:
            image.convert("RGBA").resize((ICON_SIZE, ICON_SIZE), Image.LANCZOS).save(out_dir / f"{stem.lower()}.png", optimize=True)
        report["written"].append(stem)
    return report


def export_perk_icons(perks: list[dict], icons_dir: Path, out_dir: Path) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {"written": [], "missing": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    for perk in perks:
        source = icons_dir / f"{perk['id']}.png"
        if not source.exists():
            report["missing"].append(perk["id"])
            continue
        with Image.open(source) as image:
            image.convert("RGBA").resize((ICON_SIZE, ICON_SIZE), Image.LANCZOS).save(out_dir / f"perk_{perk['id']}.png", optimize=True)
        report["written"].append(perk["id"])
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--addons", type=Path, default=DEFAULT_ADDONS)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", choices=("addons", "perks"), default="addons")
    parser.add_argument("--perks", type=Path, default=Path("data/perks.json"))
    parser.add_argument("--perk-icons", type=Path, default=Path("data/icons"))
    args = parser.parse_args(argv)
    if args.target == "perks":
        perks = json.loads(args.perks.read_text(encoding="utf-8"))["perks"]
        report = export_perk_icons(perks, args.perk_icons, args.out)
        print(f"{len(report['written'])} perk icons written to {args.out}; missing {len(report['missing'])}: {report['missing']}")
        return 1 if report["missing"] else 0
    addons = json.loads(args.addons.read_text(encoding="utf-8"))["addons"]
    report = export_icons(addons, index_exports(args.export), args.out)
    print(f"{len(report['written'])} icons written to {args.out}; missing {len(report['missing'])}; "
          f"ambiguous (first copy used) {len(report['ambiguous'])}: {report['ambiguous']}")
    return 1 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
