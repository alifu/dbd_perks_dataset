"""Copy perk icons from a game-texture export (FModel) into data/icons/<perk id>.png.

python3 tools/import_game_icons.py                    # fill only perks that have no data/icons/<id>.png yet
python3 tools/import_game_icons.py --overwrite        # replace existing icons too

Icons are copyrighted game assets: data/game/ and data/icons/ are git-ignored, never commit them.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_PERKS = Path("data/perks.json")
DEFAULT_EXPORT = Path("data/game/Perks")
DEFAULT_OUT = Path("data/icons")


def index_exports(export_dir: Path) -> dict[str, list[Path]]:
    """Lower-cased file name -> every exported file with that name (the same name can exist in several folders)."""
    found: dict[str, list[Path]] = {}
    for path in sorted(export_dir.rglob("*.png")):
        found.setdefault(path.name.lower(), []).append(path)
    return found


def digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def pick_export(candidates: list[Path]) -> Path | None:
    """The file when all copies are identical, otherwise None (ambiguous: needs a human)."""
    return candidates[0] if len({digest(path) for path in candidates}) == 1 else None


def import_icons(perks: list[dict], exports: dict[str, list[Path]], out_dir: Path, overwrite: bool) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {"copied": [], "kept": [], "missing": [], "ambiguous": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    for perk in perks:
        target = out_dir / f"{perk['id']}.png"
        if target.exists() and not overwrite:
            report["kept"].append(perk["id"])
            continue
        candidates = exports.get(perk.get("iconFile", "").lower(), [])
        if not candidates:
            report["missing"].append(perk["id"])
            continue
        source = pick_export(candidates)
        if source is None:
            report["ambiguous"].append(perk["id"])
            continue
        shutil.copyfile(source, target)
        report["copied"].append(perk["id"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--perks", type=Path, default=DEFAULT_PERKS)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT, help="folder with the exported perk PNGs (searched recursively)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    perks = json.loads(args.perks.read_text(encoding="utf-8"))["perks"]
    report = import_icons(perks, index_exports(args.export), args.out, args.overwrite)
    print(" ".join(f"{name}={len(ids)}" for name, ids in report.items()))
    for name in ("missing", "ambiguous"):
        if report[name]:
            print(f"{name}: {', '.join(report[name])}")
    return 1 if report["missing"] or report["ambiguous"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
