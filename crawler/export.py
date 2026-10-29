import json
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

from .parse_perks import RawPerk
from .schema import DOCUMENT_SCHEMA, MAX_ALLOWED_COUNT_DROP, SCHEMA_VERSION, SOURCE_NOTE


class ValidationFailed(ValueError):
    pass


def build_document(perks: list[RawPerk], wiki_revision_id: int, now: datetime | None = None) -> dict:
    ordered = sorted(perks, key=lambda perk: (perk.role, perk.name.lower()))
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": stamp,
        "source": SOURCE_NOTE,
        "wikiRevisionId": wiki_revision_id,
        "perks": [perk.to_dict() for perk in ordered],
    }


def validate_document(document: dict, previous_count: int | None = None) -> None:
    """Raise ValidationFailed on schema errors, duplicate ids, or a suspicious count drop."""
    problems = [
        f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
        for error in Draft202012Validator(DOCUMENT_SCHEMA).iter_errors(document)
    ]
    ids = [perk.get("id") for perk in document.get("perks", [])]
    duplicates = sorted({perk_id for perk_id in ids if ids.count(perk_id) > 1})
    if duplicates:
        problems.append(f"duplicate ids: {duplicates}")
    count = len(ids)
    if previous_count and count < previous_count * (1 - MAX_ALLOWED_COUNT_DROP):
        problems.append(f"perk count dropped from {previous_count} to {count} (> {MAX_ALLOWED_COUNT_DROP:.0%})")
    if problems:
        raise ValidationFailed("\n".join(problems[:20]))


def read_previous(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_document(document: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
