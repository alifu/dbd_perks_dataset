"""python -m crawler.validate data/perks.json"""
import json
import sys
from pathlib import Path

from .export import ValidationFailed, validate_document


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "data/perks.json")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_document(document)
    except (OSError, json.JSONDecodeError, ValidationFailed) as error:
        print(f"❌ {path}: {error}", file=sys.stderr)
        return 1
    print(f"✅ {path}: {len(document['perks'])} perks valid")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
