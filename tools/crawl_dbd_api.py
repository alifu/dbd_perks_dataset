"""Download current Dead by Daylight game data (perks, characters, items, add-ons, ...).

Source: the community datamine API at https://dbd.tricky.lol/api, which is extracted from the game
files and updated each patch (the Fandom wiki lags behind). Standard library only, so it runs the
same on Windows, macOS and Linux:

    python tools/crawl_dbd_api.py                      # default set -> data/game/
    python tools/crawl_dbd_api.py --all                # every endpoint
    python tools/crawl_dbd_api.py perks addons         # just these
    python tools/crawl_dbd_api.py --compare "<path to the app's Resources/perks.json>"

Each endpoint is saved as data/game/<endpoint>.json exactly as served (a dict keyed by game id).
An endpoint is skipped when its version in /api/versions has not changed since the last run
(use --force to download anyway). Please keep the delay; this is someone's free service.

Icons are not downloaded: each record has an in-game `image` path such as
/Game/UI/UMGAssets/Icons/Perks/Quiche/T_UI_iconsPerks_FiveMovesAhead. To get the PNGs, export that
texture from the game's pak files (e.g. with FModel) on a machine that has the game installed.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://dbd.tricky.lol/api"
USER_AGENT = "dbd-perks-dataset/1.0 (personal, educational use)"
DEFAULT_ENDPOINTS = ["perks", "characters", "items", "addons", "offerings", "maps", "dlc"]
ALL_ENDPOINTS = DEFAULT_ENDPOINTS + ["gamemodes", "journals", "customizations", "archives", "rift"]
REQUEST_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 60
VERSION_FILE = "_versions.json"


def fetch_json(name: str):
    request = urllib.request.Request(f"{BASE_URL}/{name}", headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"error: {name}: HTTP {error.code} {error.reason}"
                         + (" (blocked; try again later)" if error.code in (403, 429) else ""))
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {name}: {error}")


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def crawl(endpoints, out_dir: Path, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_versions = fetch_json("versions")
    known = load_json(out_dir / VERSION_FILE, {})
    print(f"game data version: {remote_versions.get('perks', {}).get('version', '?')}")

    for name in endpoints:
        version = remote_versions.get(name, {}).get("version")
        target = out_dir / f"{name}.json"
        if not force and version and known.get(name) == version and target.exists():
            print(f"  {name}: up to date ({version})")
            continue
        time.sleep(REQUEST_DELAY_SECONDS)
        data = fetch_json(name)
        target.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        known[name] = version
        print(f"  {name}: {len(data)} records saved ({version})")
        (out_dir / VERSION_FILE).write_text(json.dumps(known, indent=1), encoding="utf-8")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def compare_perks(perks_path: Path, app_perks_path: Path) -> None:
    """List perks in the crawled data that the app's perks.json does not have (matched by name)."""
    perks = load_json(perks_path, None)
    app = load_json(app_perks_path, None)
    if perks is None or app is None:
        raise SystemExit(f"error: cannot read {perks_path} or {app_perks_path} (run the crawl first?)")
    characters = load_json(perks_path.with_name("characters.json"), {})
    owners = {str(key): value.get("name", "") for key, value in characters.items()}

    known = {slug(item["name"]) for item in app["perks"]}
    missing = [(p["name"], p.get("role", "?"), owners.get(str(p.get("character")), ""), p.get("image", ""))
               for p in perks.values() if slug(p["name"]) not in known]
    print(f"{len(perks)} perks in game data, {len(app['perks'])} in the app, {len(missing)} missing from the app:")
    for name, role, owner, image in sorted(missing):
        print(f"  {name} ({role}{', ' + owner if owner else ''})  {image}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("endpoints", nargs="*", help=f"any of {' '.join(ALL_ENDPOINTS)} (default: {' '.join(DEFAULT_ENDPOINTS)})")
    parser.add_argument("--all", action="store_true", help="download every endpoint")
    parser.add_argument("--force", action="store_true", help="download even if the version is unchanged")
    parser.add_argument("--out", type=Path, default=Path("data/game"), help="output folder (default data/game)")
    parser.add_argument("--compare", type=Path, metavar="APP_PERKS_JSON",
                        help="after crawling, list perks missing from the app's perks.json")
    args = parser.parse_args(argv)

    unknown = [name for name in args.endpoints if name not in ALL_ENDPOINTS]
    if unknown:
        parser.error(f"unknown endpoint(s): {' '.join(unknown)}")
    endpoints = ALL_ENDPOINTS if args.all else (args.endpoints or DEFAULT_ENDPOINTS)
    crawl(endpoints, args.out, args.force)
    if args.compare:
        compare_perks(args.out / "perks.json", args.compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
