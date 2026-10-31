import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from convert_addons import MAX_COUNT_DROP, check_count_drop, convert  # noqa: E402

CHARACTERS = {
    "0": {"name": "The Trapper", "role": "killer", "item": "Item_Slasher_Beartrap"},
    "1": {"name": "The Wraith", "role": "killer", "item": "Item_Slasher_CloakBell"},
    "2": {"name": "Dwight Fairfield", "role": "survivor", "item": None},
}


def addon(name, parents, image, **extra):
    base = {"type": "poweraddon", "item_type": None, "parents": parents, "name": name,
            "description": "Plain text.", "role": "killer", "rarity": "rare", "image": image}
    return {**base, **extra}


def test_html_description_becomes_plain_text():
    game = {"A1": addon("Honing Stone", ["Item_Slasher_Beartrap"], "/Icons/iconAddon_honingStone",
                        description="Sharp blades.<br><br>Victim is <b>dying</b>.")}

    result, _ = convert(game, CHARACTERS, "v1")

    assert result["addons"][0]["description"] == "Sharp blades.\n\nVictim is dying."


def test_power_addon_gets_killer_name_and_kind():
    game = {"A1": addon("Honing Stone", ["Item_Slasher_Beartrap"], "/Icons/iconAddon_honingStone")}

    entry = convert(game, CHARACTERS, "v1")[0]["addons"][0]

    assert (entry["kind"], entry["ownerName"], entry["role"]) == ("power", "The Trapper", "killer")
    assert entry["iconStem"] == "iconAddon_honingStone"
    assert entry["iconFile"] == "iconAddon_honingStone.png"


def test_item_addon_has_no_owner_and_names_its_item_type():
    game = {"F1": addon("Battery", [], "/Icons/iconAddon_battery", type="itemaddon", item_type="flashlight", role="survivor")}

    entry = convert(game, CHARACTERS, "v1")[0]["addons"][0]

    assert (entry["kind"], entry["ownerName"], entry["parent"]) == ("item", "", "flashlight")


def test_same_name_on_two_killers_gets_distinct_ids_and_shares_the_icon_stem():
    game = {"A1": addon("Ghost Bell", ["Item_Slasher_Beartrap"], "/Icons/iconAddon_x"),
            "A2": addon("Ghost Bell", ["Item_Slasher_CloakBell"], "/Icons/iconAddon_x")}

    entries = convert(game, CHARACTERS, "v1")[0]["addons"]

    assert len({e["id"] for e in entries}) == 2
    assert {e["iconStem"] for e in entries} == {"iconAddon_x"}


def test_disabled_entries_with_untranslated_names_are_skipped_and_reported():
    game = {"A1": addon("@#Addon_GasBomb_19a_NAME", ["DISABLED"], "/Icons/iconAddon_gas"),
            "A2": addon("Honing Stone", ["Item_Slasher_Beartrap"], "/Icons/iconAddon_honingStone")}

    result, report = convert(game, CHARACTERS, "v1")

    assert [e["name"] for e in result["addons"]] == ["Honing Stone"]
    assert report["skipped"] == ["A1"]


def test_add_on_without_a_known_killer_keeps_an_empty_owner():
    game = {"A1": addon("Blight Serum", [], "/Icons/iconAddon_serum")}

    entry = convert(game, CHARACTERS, "v1")[0]["addons"][0]

    assert entry["ownerName"] == ""


def test_count_drop_guard_fails_when_more_than_five_percent_disappear():
    with pytest.raises(SystemExit):
        check_count_drop(new_count=90, previous_count=100)
    check_count_drop(new_count=96, previous_count=100)
    assert MAX_COUNT_DROP == 0.05
