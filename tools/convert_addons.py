"""Turn the crawled game add-on data into the iOS app's addons.json (schema 1).

    python tools/crawl_dbd_api.py                        # first: data/game/{addons,characters}.json
    python tools/convert_addons.py --out addons.new.json [--previous addons.json]

* Descriptions use the same HTML-to-text rules as perks (see convert_game_data.py).
* `iconStem` is the model label of the add-on classifier: the icon texture name. Two add-ons on two
  killers can share one icon, so a label can map to more than one add-on.
* Entries that are disabled or have no translated name ("@#Addon_..._NAME") are skipped and reported.
* With --previous the run fails if the count dropped by more than 5% (protects against a broken crawl).
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from convert_game_data import html_to_text, load, make_id, resolve_placeholders

SCHEMA_VERSION = 1
MAX_COUNT_DROP = 0.05
UNTRANSLATED_PREFIX = "@"
SOURCE_NOTE = "Game data: community datamine dbd.tricky.lol (version {version}). Icons (c) Behaviour Interactive. Educational use only."


def check_count_drop(new_count: int, previous_count: int) -> None:
    if previous_count and new_count < previous_count * (1 - MAX_COUNT_DROP):
        raise SystemExit(f"error: {new_count} add-ons vs {previous_count} before (more than {MAX_COUNT_DROP:.0%} fewer); check the crawl")


def power_owners(characters: dict) -> dict:
    """Killer power id (Item_Slasher_Beartrap) -> killer name."""
    return {c["item"]: c["name"] for c in characters.values() if c.get("item")}


def describe(addon: dict) -> str:
    return html_to_text(resolve_placeholders(addon.get("description") or "", addon.get("tunables") or {}, []))


def convert(addons: dict, characters: dict, version: str):
    owners = power_owners(characters)
    used_ids: set = set()
    entries, skipped = [], []
    for game_id, addon in sorted(addons.items(), key=lambda item: (item[1]["name"], item[0])):
        name = addon["name"]
        if name.startswith(UNTRANSLATED_PREFIX) or "DISABLED" in addon["parents"]:
            skipped.append(game_id)
            continue
        is_item = addon["type"] == "itemaddon"
        parent = addon.get("item_type") or "" if is_item else next(iter(addon["parents"]), "")
        owner = "" if is_item else owners.get(parent, "")
        new_id = make_id(name)
        if new_id in used_ids:
            new_id = f"{new_id}_{make_id(owner or game_id)}"
        used_ids.add(new_id)
        stem = addon["image"].rsplit("/", 1)[-1]
        entries.append({"id": new_id, "name": name, "role": addon["role"], "kind": "item" if is_item else "power",
                        "parent": parent, "ownerName": owner, "rarity": addon.get("rarity") or "none",
                        "description": describe(addon), "iconStem": stem, "iconFile": f"{stem}.png"})
    document = {"schemaVersion": SCHEMA_VERSION,
                "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": SOURCE_NOTE.format(version=version), "addons": entries}
    return document, {"skipped": skipped}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-data", type=Path, default=Path("data/game"))
    parser.add_argument("--previous", type=Path, help="the last addons.json, for the count-drop guard")
    parser.add_argument("--out", type=Path, default=Path("addons.new.json"))
    args = parser.parse_args(argv)

    versions_path = args.game_data / "_versions.json"
    versions = load(versions_path) if versions_path.exists() else {}
    document, report = convert(load(args.game_data / "addons.json"), load(args.game_data / "characters.json"),
                               str(versions.get("addons", "unknown")))
    if args.previous:
        check_count_drop(len(document["addons"]), len(load(args.previous)["addons"]))
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stems = {e["iconStem"] for e in document["addons"]}
    print(f"{len(document['addons'])} add-ons written ({len(stems)} unique icons); skipped {len(report['skipped'])}: {report['skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
