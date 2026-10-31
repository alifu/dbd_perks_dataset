import json

import numpy as np
import pytest
from PIL import Image

from addon_data import addon_labels_and_paths
from synth.addon_tile import RARITY_TILE_COLORS, compose_addon
from synth.compose import ICON_SIZE, WORK_SIZE, load_icon
from synth.dataset import PerkSynthDataset


@pytest.fixture()
def white_glyph(tmp_path):
    canvas = np.zeros((ICON_SIZE, ICON_SIZE, 4), np.uint8)
    canvas[80:176, 80:176] = (255, 255, 255, 255)
    path = tmp_path / "glyph.png"
    Image.fromarray(canvas, "RGBA").save(path)
    return path


def test_compose_addon_is_deterministic_rgb_of_work_size(white_glyph):
    icon = load_icon(white_glyph)

    first = compose_addon(np.random.default_rng(3), icon)

    assert first.shape == (WORK_SIZE, WORK_SIZE, 3) and first.dtype == np.uint8
    assert np.array_equal(first, compose_addon(np.random.default_rng(3), icon))


def test_glyph_stays_visible_in_the_centre(white_glyph):
    icon = load_icon(white_glyph)
    middle = slice(WORK_SIZE // 2 - 6, WORK_SIZE // 2 + 6)

    brightness = [compose_addon(np.random.default_rng(seed), icon)[middle, middle].mean() for seed in range(40)]

    assert np.median(brightness) > 170, "a white glyph must stay bright at the centre"


def test_all_rarity_colours_are_used(white_glyph):
    icon = load_icon(white_glyph)
    corner_means = {tuple(np.round(compose_addon(np.random.default_rng(s), icon)[40:60, 40:60].mean(axis=(0, 1)), -1)) for s in range(80)}

    assert len(RARITY_TILE_COLORS) == 5
    assert len(corner_means) > 8, "tiles must vary in colour"


def test_dataset_accepts_a_custom_composer(white_glyph):
    dataset = PerkSynthDataset([white_glyph, white_glyph], 2, (0.2, 0.5), fixed_seed=1, composer=compose_addon)

    image, label = dataset[0]

    assert image.shape[0] == 3 and label == 0


def test_addon_labels_are_unique_lowercase_stems_with_an_icon_each(tmp_path):
    export = tmp_path / "ItemAddons" / "Killer"
    export.mkdir(parents=True)
    for name in ("iconAddon_Seal.png", "iconAddon_seal.png", "iconAddon_rope.png"):
        Image.fromarray(np.zeros((8, 8, 4), np.uint8), "RGBA").save(export / name)
    addons = tmp_path / "addons.json"
    addons.write_text(json.dumps({"addons": [{"iconStem": "iconAddon_Seal"}, {"iconStem": "iconAddon_seal"},
                                            {"iconStem": "iconAddon_rope"}, {"iconStem": "iconAddon_gone"}]}))

    labels, paths, missing = addon_labels_and_paths(addons, tmp_path / "ItemAddons")

    assert labels == ["iconaddon_rope", "iconaddon_seal"]
    assert [p.name.lower() for p in paths] == ["iconaddon_rope.png", "iconaddon_seal.png"]
    assert missing == ["iconaddon_gone"]
