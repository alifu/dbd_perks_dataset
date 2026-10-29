import numpy as np
import pytest
from PIL import Image

from synth.compose import ICON_SIZE, WORK_SIZE, compose, load_icon, make_background
from synth.dataset import PerkSynthDataset
from synth.photo import OUTPUT_SIZE, apply_photo_effects


@pytest.fixture()
def icon_paths(tmp_path):
    paths = []
    for index, colour in enumerate([(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]):
        canvas = np.zeros((ICON_SIZE, ICON_SIZE, 4), np.uint8)
        canvas[64:192, 64:192] = colour
        path = tmp_path / f"perk_{index}.png"
        Image.fromarray(canvas, "RGBA").save(path)
        paths.append(path)
    return paths


def test_load_icon_resizes_odd_sized_images(tmp_path):
    path = tmp_path / "big.png"
    Image.fromarray(np.zeros((755, 755, 4), np.uint8), "RGBA").save(path)
    assert load_icon(path).shape == (ICON_SIZE, ICON_SIZE, 4)


def test_load_icon_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_icon(tmp_path / "nope.png")


def test_background_is_rgb_float_in_range():
    background = make_background(np.random.default_rng(1))
    assert background.shape == (WORK_SIZE, WORK_SIZE, 3)
    assert 0 <= background.min() and background.max() <= 255


def test_compose_keeps_icon_visible_and_is_deterministic(icon_paths):
    icon = load_icon(icon_paths[0])
    first = compose(np.random.default_rng(7), icon)
    again = compose(np.random.default_rng(7), icon)
    assert first.shape == (WORK_SIZE, WORK_SIZE, 3) and first.dtype == np.uint8
    assert np.array_equal(first, again)
    centre = first[WORK_SIZE // 2 - 10:WORK_SIZE // 2 + 10, WORK_SIZE // 2 - 10:WORK_SIZE // 2 + 10].mean(axis=(0, 1))
    assert centre[0] > centre[1] + 80 and centre[0] > centre[2] + 80, "red icon must dominate the centre"


def test_zero_severity_only_resizes():
    image = np.full((WORK_SIZE, WORK_SIZE, 3), 120, np.uint8)
    out = apply_photo_effects(np.random.default_rng(0), image, 0.0)
    assert out.shape == (OUTPUT_SIZE, OUTPUT_SIZE, 3)
    assert np.all(out == 120)


def test_high_severity_changes_image_but_keeps_format():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (WORK_SIZE, WORK_SIZE, 3), dtype=np.uint8)
    clean = apply_photo_effects(np.random.default_rng(0), image, 0.0)
    noisy = apply_photo_effects(np.random.default_rng(0), image, 1.0)
    assert noisy.shape == clean.shape and noisy.dtype == np.uint8
    assert np.abs(noisy.astype(int) - clean.astype(int)).mean() > 5


def test_fixed_seed_dataset_is_reproducible_and_balanced(icon_paths):
    dataset = PerkSynthDataset(icon_paths, samples_per_class=4, severity_range=(0.2, 0.8), fixed_seed=5)
    first, label = dataset[7]
    again, _ = dataset[7]
    assert len(dataset) == 12
    assert first.shape == (3, OUTPUT_SIZE, OUTPUT_SIZE) and 0.0 <= first.min() and first.max() <= 1.0
    assert np.array_equal(first.numpy(), again.numpy())
    assert [dataset[i][1] for i in range(len(dataset))].count(label) == 4


def test_training_dataset_is_random_each_call(icon_paths):
    dataset = PerkSynthDataset(icon_paths, samples_per_class=2, severity_range=(0.5, 1.0))
    assert not np.array_equal(dataset[0][0].numpy(), dataset[0][0].numpy())


def test_negative_items_use_the_background_class(icon_paths):
    dataset = PerkSynthDataset(icon_paths, samples_per_class=4, severity_range=(0.2, 0.8),
                               fixed_seed=5, negative_fraction=0.25)
    labels = [dataset[i][1] for i in range(len(dataset))]
    assert len(dataset) == 15
    assert labels.count(len(icon_paths)) == 3
    assert all(labels[:12].count(c) == 4 for c in range(3))


def test_every_negative_kind_renders_a_window(icon_paths):
    from synth import negatives
    icons = [load_icon(p) for p in icon_paths]
    for kind in negatives.NEGATIVE_KINDS:
        rng = np.random.default_rng(0)
        original = negatives.NEGATIVE_KINDS
        negatives.NEGATIVE_KINDS = (kind,)
        try:
            window = negatives.make_negative(rng, icons)
        finally:
            negatives.NEGATIVE_KINDS = original
        assert window.shape == (WORK_SIZE, WORK_SIZE, 3) and window.dtype == np.uint8, kind


def test_scene_ground_truth_lies_inside_the_image_and_is_deterministic(icon_paths):
    from synth.scene import LAYOUTS, render_scene
    perks = [(f"p{i}", load_icon(p)) for i, p in enumerate(icon_paths)]
    for layout in LAYOUTS:
        image, placed = render_scene(np.random.default_rng(1), perks, layout, 0.3)
        again, _ = render_scene(np.random.default_rng(1), perks, layout, 0.3)
        height, width = image.shape[:2]
        assert np.array_equal(image, again)
        assert len(placed) == 3
        assert all(0 <= p.cx <= width and 0 <= p.cy <= height and p.diagonal > 10 for p in placed), layout
