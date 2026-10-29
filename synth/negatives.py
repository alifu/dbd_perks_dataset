"""Windows that must NOT be recognised as a perk: lets the detector reject bad candidate regions."""
import cv2
import numpy as np

from .compose import WORK_SIZE, compose, make_background

NEGATIVE_KINDS = ("empty", "tiny_icon", "partial_icon", "ui_clutter", "between_icons")
TINY_SCALE_RANGE = (0.18, 0.42)   # whole icon shrunk inside a much larger window
PARTIAL_ZOOM = 2                  # zoomed-in window shows only a fragment of the art
CLUTTER_SHAPES = (4, 14)


def _tiny_icon(rng, icons, size):
    scene = make_background(rng, size)
    small = int(size * rng.uniform(*TINY_SCALE_RANGE))
    piece = cv2.resize(compose(rng, icons[rng.integers(len(icons))]), (small, small), interpolation=cv2.INTER_AREA)
    x, y = (int(rng.integers(0, size - small)) for _ in range(2))
    scene[y:y + small, x:x + small] = piece
    return scene.astype(np.uint8)


def _partial_icon(rng, icons, size):
    big = compose(rng, icons[rng.integers(len(icons))], size * PARTIAL_ZOOM)
    # keep windows whose centre is off the glyph's centre, so only a fragment is visible
    while True:
        x, y = (int(rng.integers(0, size * (PARTIAL_ZOOM - 1) + 1)) for _ in range(2))
        if np.hypot(x + size / 2 - size, y + size / 2 - size) > size * 0.45:
            return big[y:y + size, x:x + size]


def _ui_clutter(rng, size):
    image = make_background(rng, size)
    for _ in range(int(rng.integers(*CLUTTER_SHAPES))):
        colour = tuple(float(c) for c in rng.uniform(0, 255, 3))
        a, b = (tuple(int(v) for v in rng.integers(0, size, 2)) for _ in range(2))
        if rng.random() < 0.5:
            cv2.rectangle(image, a, b, colour, int(rng.choice([-1, 1, 2])))
        else:
            cv2.line(image, a, b, colour, int(rng.integers(1, 4)))
    return np.clip(image, 0, 255).astype(np.uint8)


def _between_icons(rng, icons, size):
    left = compose(rng, icons[rng.integers(len(icons))], size)
    right = compose(rng, icons[rng.integers(len(icons))], size)
    cut = int(size * rng.uniform(0.35, 0.65))
    return np.concatenate([left[:, :cut], right[:, cut:]], axis=1)


def make_negative(rng: np.random.Generator, icons: list[np.ndarray], size: int = WORK_SIZE) -> np.ndarray:
    """uint8 RGB (size x size) that contains no complete, centred perk icon."""
    kind = NEGATIVE_KINDS[rng.integers(len(NEGATIVE_KINDS))]
    if kind == "empty":
        return make_background(rng, size).astype(np.uint8)
    if kind == "tiny_icon":
        return _tiny_icon(rng, icons, size)
    if kind == "partial_icon":
        return _partial_icon(rng, icons, size)
    if kind == "ui_clutter":
        return _ui_clutter(rng, size)
    return _between_icons(rng, icons, size)
