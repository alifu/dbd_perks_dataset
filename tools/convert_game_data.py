"""Turn the crawled game data into the iOS app's perks.json (schema 1).

    python tools/crawl_dbd_api.py                       # first: data/game/{perks,characters}.json
    python tools/convert_game_data.py \
        --existing "<app>/DBD Perks/Resources/perks.json" \
        --out perks.new.json

What it does:
* Descriptions come from the game data (current patch): HTML becomes plain text with "• " bullets,
  {Tunable.X.Y} becomes the per-tier values ("50/75/100"), {Keyword.X} becomes the keyword.
* Perks already in --existing keep their id (the model's labels are these ids), flavor text and wiki
  icon; only name, owner, role and description are refreshed.
* Perks only in the game data are added with a new id, no iconUrl and an `iconFile` (the texture
  name, e.g. T_UI_iconsPerks_FiveMovesAhead.png) so the exported game texture can be matched later.
* Perks only in --existing are kept untouched and listed, never deleted silently.
Matching existing perks: exact name, then icon texture name, then a close name (>= 0.9 similarity).
"""
import argparse
import difflib
import html
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
SIMILARITY_THRESHOLD = 0.9
WIKI_URL = "https://deadbydaylight.fandom.com/wiki/"
SOURCE_NOTE = ("Game data: community datamine dbd.tricky.lol (version {version}); older icons and flavor text: "
               "https://deadbydaylight.fandom.com (CC BY-SA 3.0, icons (c) Behaviour Interactive). Educational use only.")
INPUT_NAMES = {"ActivatableButton1": "ability button", "ActivatableButton2": "second ability button"}
PLACEHOLDER = re.compile(r"\{(Tunable|Keyword|Input)\.([^}]*)\}")


def plain(text: str) -> str:
    """Lowercase without accents: 'Déjà Vu' -> 'deja vu'."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", plain(text))


def make_id(name: str) -> str:
    """'Franklin's Demise' -> 'franklins_demise', 'Barbecue & Chilli' -> 'barbecue_chilli'."""
    return re.sub(r"[^a-z0-9]+", "_", plain(name).replace("'", "").replace("’", "")).strip("_")


def icon_stem(path: str) -> str:
    """Texture name without the iconPerks prefix: T_UI_iconsPerks_FiveMovesAhead -> fivemovesahead."""
    base = path.rsplit("/", 1)[-1].lower()
    return re.sub(r"^(t_ui_)?icons?perks_", "", base)


def format_values(values) -> str:
    texts = [f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in values]
    return texts[0] if len(set(texts)) == 1 else "/".join(texts)


def resolve_placeholders(text: str, tunables: dict, unresolved: list) -> str:
    def replace(match: re.Match) -> str:
        kind, rest = match.groups()
        if kind == "Keyword":
            return rest
        if kind == "Input":
            return INPUT_NAMES.get(rest, rest)
        key = rest.split(".", 1)[-1].lower()
        values = tunables.get(key)
        if not values:
            unresolved.append(match.group(0))
            return match.group(0)
        return format_values(values)
    return PLACEHOLDER.sub(replace, text)


def html_to_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<li>", "\n• ", text)
    text = re.sub(r"</?(ul|b|i|li)>", "", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def describe(perk: dict, unresolved: list) -> str:
    return html_to_text(resolve_placeholders(perk.get("description") or "", perk.get("tunables") or {}, unresolved))


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: cannot read {path}: {error}")


class Matcher:
    """Finds the existing app perk for a game-data perk."""

    def __init__(self, existing: list):
        self.by_name = {slug(p["name"]): p for p in existing}
        self.by_icon = {icon_stem(p.get("iconFile", "")): p for p in existing if p.get("iconFile")}
        self.names = list(self.by_name)

    def find(self, name: str, image: str):
        if slug(name) in self.by_name:
            return self.by_name[slug(name)], "name"
        if icon_stem(image) in self.by_icon:
            return self.by_icon[icon_stem(image)], "icon"
        close = difflib.get_close_matches(slug(name), self.names, n=1, cutoff=SIMILARITY_THRESHOLD)
        return (self.by_name[close[0]], "similar name") if close else (None, None)


def convert(perks: dict, characters: dict, existing: list, version: str):
    owners = {str(key): value.get("name", "") for key, value in characters.items()}
    matcher = Matcher(existing)
    used_ids = {p["id"] for p in existing}
    seen, result, report = set(), [], {"matched": [], "added": [], "kept": [], "unresolved": [], "fallback": []}

    for game_id, perk in sorted(perks.items(), key=lambda item: item[1]["name"]):
        name, image = perk["name"], perk.get("image", "")
        owner = owners.get(str(perk.get("character")), "")
        old, how = matcher.find(name, image)
        missing: list = []
        description = describe(perk, missing)
        report["unresolved"] += missing
        if old and old["id"] not in seen:
            seen.add(old["id"])
            refreshed = {"name": name, "role": perk["role"], "owner": owner, "description": description, "descriptionPatch": None}
            if missing:  # the game data lacks a value: the wiki text is better than a raw placeholder
                refreshed = {**refreshed, "description": old["description"], "descriptionPatch": old.get("descriptionPatch")}
                report["fallback"].append(f"{name}: kept wiki description ({', '.join(sorted(set(missing)))})")
            result.append({**old, **refreshed})
            if how != "name":
                report["matched"].append(f"{name} -> {old['id']} (by {how}; was '{old['name']}')")
            continue
        new_id = make_id(name)
        if new_id in used_ids:
            new_id = f"{new_id}_{make_id(game_id)}"
        used_ids.add(new_id)
        entry = {"id": new_id, "name": name, "role": perk["role"], "owner": owner,
                 "iconFile": f"{image.rsplit('/', 1)[-1]}.png" if image else "",
                 "description": description, "flavor": "", "descriptionPatch": None,
                 "wikiUrl": WIKI_URL + re.sub(r"\s+", "_", name).replace("&", "%26")}
        result.append(entry)
        report["added"].append(f"{name} ({perk['role']}, {owner or 'generic'}) -> {new_id}")

    for old in existing:
        if old["id"] not in seen:
            result.append(old)
            report["kept"].append(f"{old['name']} ({old['id']})")
    document = {"schemaVersion": SCHEMA_VERSION,
                "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": SOURCE_NOTE.format(version=version), "perks": result}
    return document, report


def print_report(report: dict, total: int) -> None:
    print(f"{total} perks written.")
    for title, key in (("Matched to an existing perk by a looser rule", "matched"),
                       ("Added (new to the app)", "added"),
                       ("Kept the wiki description (game data lacks a value)", "fallback"),
                       ("In the app but not in the game data (kept)", "kept")):
        print(f"\n{title}: {len(report[key])}")
        for line in report[key]:
            print(f"  {line}")
    leftovers = sorted(set(report["unresolved"]))
    print(f"\nPlaceholders the game data could not fill: {len(leftovers)} (new perks show them as written)")
    for placeholder in leftovers[:20]:
        print(f"  {placeholder}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-data", type=Path, default=Path("data/game"), help="folder written by crawl_dbd_api.py")
    parser.add_argument("--existing", type=Path, help="the app's current perks.json (keeps ids, flavor and wiki icons)")
    parser.add_argument("--out", type=Path, default=Path("perks.new.json"))
    args = parser.parse_args(argv)

    perks = load(args.game_data / "perks.json")
    characters = load(args.game_data / "characters.json")
    versions = load(args.game_data / "_versions.json") if (args.game_data / "_versions.json").exists() else {}
    existing = load(args.existing)["perks"] if args.existing else []

    document, report = convert(perks, characters, existing, versions.get("perks", "unknown"))
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_report(report, len(document["perks"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
