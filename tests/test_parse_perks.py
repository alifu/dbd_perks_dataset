import pytest

from crawler.parse_perks import ParseError, parse_perks_page


@pytest.fixture(scope="module")
def perks_by_id(sample_html):
    perks, errors = parse_perks_page(sample_html)
    assert errors == []
    return {perk.id: perk for perk in perks}


def test_parses_every_row_with_unique_ids(perks_by_id):
    assert set(perks_by_id) == {
        "ace_in_the_hole", "adrenaline", "deja_vu", "a_nurses_calling", "hex_no_one_escapes_death",
    }


def test_role_comes_from_table_heading(perks_by_id):
    assert perks_by_id["adrenaline"].role == "survivor"
    assert perks_by_id["a_nurses_calling"].role == "killer"


def test_icon_and_wiki_url_come_from_same_row(perks_by_id):
    perk = perks_by_id["ace_in_the_hole"]
    assert perk.icon_file == "IconPerks_aceInTheHole.png"
    assert perk.icon_url.startswith("https://static.wikia.nocookie.net/") and "aceInTheHole" in perk.icon_url
    assert perk.wiki_url == "https://deadbydaylight.fandom.com/wiki/Ace_in_the_Hole"


def test_owner_is_character_and_empty_for_generic(perks_by_id):
    assert perks_by_id["ace_in_the_hole"].owner == "Ace Visconti"
    assert perks_by_id["deja_vu"].owner == ""


def test_description_is_clean_and_keeps_bullets(perks_by_id):
    description = perks_by_id["ace_in_the_hole"].description
    assert "\xa0" not in description
    assert "• 100% chance for an Add-on of Very Rare Rarity or lower." in description
    assert "10/25/50% chance" in description


def test_flavor_is_split_from_description(perks_by_id):
    perk = perks_by_id["ace_in_the_hole"]
    assert perk.flavor.startswith('"Everything that glitters')
    assert "glitters" not in perk.description


def test_patch_banner_is_removed_and_recorded(perks_by_id):
    perk = perks_by_id["adrenaline"]
    assert perk.description_patch == "7.7.0"
    assert "This description is based" not in perk.description
    assert perk.description.startswith("You are fuelled")
    assert perks_by_id["ace_in_the_hole"].description_patch is None


def test_page_without_tables_raises():
    with pytest.raises(ParseError):
        parse_perks_page("<html><body>layout changed</body></html>")
