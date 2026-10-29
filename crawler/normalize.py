import re
import unicodedata

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)])")
_SPACE_AFTER_OPEN = re.compile(r"\(\s+")
_PERCENT_GAP = re.compile(r"(\d)\s+%")
_FLAVOR_LINE = re.compile(r"^[\"“].+[\"”](\s*[—–-]\s*.+)?$")
_PATCH_VERSION = re.compile(r"Patch\s+(\d+(?:\.\d+)+)", re.IGNORECASE)


def slugify(name: str) -> str:
    """'A Nurse's Calling' -> 'a_nurses_calling' (stable id, also the model label)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    without_apostrophes = re.sub(r"['’`]", "", ascii_name.lower())
    return re.sub(r"[^a-z0-9]+", "_", without_apostrophes).strip("_")


def clean_line(line: str) -> str:
    text = line.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub("(", text)
    return _PERCENT_GAP.sub(r"\1%", text)


def clean_lines(raw: str) -> list[str]:
    """Normalize every line and drop empties."""
    return [line for line in (clean_line(part) for part in raw.splitlines()) if line]


def split_flavor(lines: list[str]) -> tuple[list[str], str]:
    """Separate the trailing '"quote" — Character' line from the description."""
    if lines and _FLAVOR_LINE.match(lines[-1]):
        return lines[:-1], lines[-1]
    return lines, ""


def extract_patch_version(banner: str) -> str | None:
    match = _PATCH_VERSION.search(banner)
    return match.group(1) if match else None
