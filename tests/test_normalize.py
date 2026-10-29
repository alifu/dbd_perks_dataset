import pytest

from crawler.normalize import clean_line, extract_patch_version, slugify, split_flavor


@pytest.mark.parametrize(
    "name, expected",
    [
        ("A Nurse's Calling", "a_nurses_calling"),
        ("Déjà Vu", "deja_vu"),
        ("Hex: No One Escapes Death", "hex_no_one_escapes_death"),
        ("Boon: Dark Theory", "boon_dark_theory"),
    ],
)
def test_slugify_handles_punctuation_and_accents(name, expected):
    assert slugify(name) == expected


def test_clean_line_normalizes_spaces_and_percent():
    assert clean_line("heal  1\xa0Health State . Grants +50\xa0% Haste") == "heal 1 Health State. Grants +50% Haste"


def test_split_flavor_detects_quote_with_and_without_author():
    body, flavor = split_flavor(["Effect.", '"Quote." — Ace'])
    assert (body, flavor) == (["Effect."], '"Quote." — Ace')
    body, flavor = split_flavor(["Effect.", '"Only a quote."'])
    assert flavor == '"Only a quote."'


def test_split_flavor_keeps_plain_last_line():
    body, flavor = split_flavor(["Effect.", "Last sentence."])
    assert (body, flavor) == (["Effect.", "Last sentence."], "")


def test_extract_patch_version():
    banner = "This description is based on the changes announced for or featured in the upcoming Patch 7.7.0"
    assert extract_patch_version(banner) == "7.7.0"
    assert extract_patch_version("nothing here") is None
