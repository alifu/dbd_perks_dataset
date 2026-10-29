import argparse
import sys
from pathlib import Path

from .export import ValidationFailed, build_document, read_previous, validate_document, write_document
from .fandom_client import FandomClient, FandomError
from .parse_perks import ParseError, parse_perks_page

DEFAULT_OUT_DIR = Path("data")
DEFAULT_CACHE_DIR = Path("cache")
PERKS_FILE = "perks.json"
ICONS_DIR = "icons"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="crawler", description="Crawl DBD perks from the Fandom wiki.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="output directory (default: data)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR, help="raw response cache directory")
    parser.add_argument("--offline", action="store_true", help="reuse the cached Perks page instead of fetching")
    parser.add_argument("--skip-icons", action="store_true", help="do not download icon images")
    parser.add_argument("--force", action="store_true", help="rewrite output even if the wiki revision is unchanged")
    return parser.parse_args(argv)


def _download_missing_icons(client: FandomClient, document: dict, icons_dir: Path) -> None:
    missing = [perk for perk in document["perks"] if not (icons_dir / f"{perk['id']}.png").exists()]
    print(f"🖼️  {len(missing)} icons to download ({len(document['perks']) - len(missing)} already present)")
    failures = []
    for index, perk in enumerate(missing, 1):
        try:
            client.download(perk["iconUrl"], icons_dir / f"{perk['id']}.png")
            print(f"[{index}/{len(missing)}] 💾 {perk['id']}")
        except FandomError as error:
            failures.append(perk["id"])
            print(f"[{index}/{len(missing)}] ⚠️  {perk['id']}: {error}")
    if failures:
        raise FandomError(f"{len(failures)} icons failed: {failures}")


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    perks_path = args.out / PERKS_FILE
    client = FandomClient(args.cache)

    html, revision_id = client.fetch_perks_page(use_cache=args.offline)
    previous = read_previous(perks_path)
    if previous and previous.get("wikiRevisionId") == revision_id and not args.force:
        print(f"✅ Wiki revision {revision_id} unchanged; nothing to do (use --force).")
        document = previous
    else:
        perks, row_errors = parse_perks_page(html)
        for message in row_errors:
            print(f"⚠️  skipped row: {message}")
        document = build_document(perks, revision_id)
        validate_document(document, previous_count=len(previous["perks"]) if previous else None)
        write_document(document, perks_path)
        print(f"✅ Wrote {len(document['perks'])} perks to {perks_path}")

    if not args.skip_icons:
        _download_missing_icons(client, document, args.out / ICONS_DIR)
    return 0


def main() -> int:
    try:
        return run()
    except (FandomError, ParseError, ValidationFailed) as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
