"""Runs against the real cached Perks page when available (cache/Perks.json)."""
import json
from pathlib import Path

import pytest

from crawler.export import build_document, validate_document
from crawler.parse_perks import parse_perks_page

CACHE = Path(__file__).parent.parent / "cache" / "Perks.json"
pytestmark = pytest.mark.skipif(not CACHE.exists(), reason="no cached Perks page")


def test_full_page_parses_without_errors_and_validates():
    payload = json.loads(CACHE.read_text(encoding="utf-8"))["parse"]
    perks, errors = parse_perks_page(payload["text"]["*"])
    assert errors == []
    assert len(perks) > 250
    validate_document(build_document(perks, int(payload["revid"])))
