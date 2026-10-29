from crawler.legacy_labels import build_label_map

PERKS = [
    {"id": "adrenaline", "iconFile": "IconPerks_adrenaline.png"},
    {"id": "new_perk", "iconFile": "IconPerks_newPerk.png"},
]


def test_maps_by_icon_stem_and_reports_gaps():
    result = build_label_map(PERKS, ["File:IconPerks_adrenaline", "File:IconPerks_removed_old"])

    assert result["map"] == {"File:IconPerks_adrenaline": "adrenaline"}
    assert result["unmappedLegacyLabels"] == ["File:IconPerks_removed_old"]
    assert result["perksMissingFromLegacyModel"] == ["new_perk"]
