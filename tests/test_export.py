import copy
import json
from datetime import datetime, timezone

import pytest

from crawler.export import ValidationFailed, build_document, read_previous, validate_document, write_document
from crawler.parse_perks import parse_perks_page


@pytest.fixture()
def document(sample_html):
    perks, _ = parse_perks_page(sample_html)
    return build_document(perks, 123, now=datetime(2026, 10, 29, tzinfo=timezone.utc))


def test_document_is_valid_and_sorted_by_role_then_name(document):
    validate_document(document)
    roles = [perk["role"] for perk in document["perks"]]
    assert roles == sorted(roles)
    assert document["generatedAt"] == "2026-10-29T00:00:00Z"
    assert document["wikiRevisionId"] == 123


def test_duplicate_ids_are_rejected(document):
    broken = copy.deepcopy(document)
    broken["perks"].append(copy.deepcopy(broken["perks"][0]))
    with pytest.raises(ValidationFailed, match="duplicate ids"):
        validate_document(broken)


def test_missing_field_is_rejected(document):
    broken = copy.deepcopy(document)
    del broken["perks"][0]["description"]
    with pytest.raises(ValidationFailed, match="description"):
        validate_document(broken)


def test_large_count_drop_is_rejected(document):
    with pytest.raises(ValidationFailed, match="dropped"):
        validate_document(document, previous_count=100)
    validate_document(document, previous_count=len(document["perks"]))


def test_write_then_read_roundtrip(tmp_path, document):
    path = tmp_path / "out" / "perks.json"
    write_document(document, path)
    assert read_previous(path) == document
    assert not path.with_suffix(".json.part").exists()


def test_read_previous_tolerates_missing_and_corrupt(tmp_path):
    assert read_previous(tmp_path / "nope.json") is None
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{not json")
    assert read_previous(corrupt) is None
