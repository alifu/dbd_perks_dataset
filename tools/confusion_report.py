"""Report which perk pairs the iOS app confuses most.

The debug build of the app writes every correction the tester made to
Documents/ScanLog/corrections/<correct perk id>/<scanID>_<slot>.json. Copy that folder off the phone:

    xcrun devicectl device copy from --device <udid> --domain-type appDataContainer \
        --domain-identifier AppWork.DBD-Perks --source Documents/ScanLog --destination ./ScanLog

then run:

    python tools/confusion_report.py ./ScanLog [--top 20]

A pair is unordered: "a <-> b" counts a-mistaken-for-b and b-mistaken-for-a together, because the
fix (more look-alike training data for both) is the same. Pairs with the most mistakes are the
"same pattern" set to oversample when retraining.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_corrections(log_dir: Path):
    """Yield one dict per correction json; skips unreadable files with a warning."""
    for path in sorted((log_dir / "corrections").glob("*/*.json")):
        try:
            yield json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            print(f"warning: skipped {path}: {error}", file=sys.stderr)


def count_confusions(corrections):
    """Return (pair counts, per-perk wrong counts, corrections with no prediction)."""
    pairs, wrong_per_perk, unanswered = Counter(), Counter(), 0
    for item in corrections:
        predicted, correct = item.get("predictedPerkID"), item["correctPerkID"]
        if predicted is None:
            unanswered += 1
            continue
        if predicted == correct:
            continue
        pairs[tuple(sorted((predicted, correct)))] += 1
        wrong_per_perk[correct] += 1
    return pairs, wrong_per_perk, unanswered


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", type=Path, help="ScanLog folder copied from the phone")
    parser.add_argument("--top", type=int, default=20, help="how many pairs/perks to list")
    args = parser.parse_args(argv)

    if not (args.log_dir / "corrections").is_dir():
        print(f"error: no corrections folder in {args.log_dir}", file=sys.stderr)
        return 1

    pairs, wrong_per_perk, unanswered = count_confusions(load_corrections(args.log_dir))
    total = sum(pairs.values())
    print(f"{total} wrong answers, {unanswered} slots the model left empty\n")

    print("Most confused pairs (a <-> b):")
    for (first, second), count in pairs.most_common(args.top):
        print(f"  {count:3d}  {first} <-> {second}")

    print("\nPerks most often misread (correct perk):")
    for perk, count in wrong_per_perk.most_common(args.top):
        print(f"  {count:3d}  {perk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
