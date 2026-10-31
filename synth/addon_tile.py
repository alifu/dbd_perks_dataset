"""An add-on slot as photographed off a monitor: white glyph on a rarity-coloured square tile with a dark
border, a screen backdrop around it and sometimes a corner of the neighbouring tile.

The exact in-game look is an educated guess until real add-on photos exist, so every property is jittered
widely (tile size, colour, border, tilt) and part of the samples drop the tile and use a flat colour."""
import cv2
import numpy as np

from .compose import WORK_SIZE, _add_neighbor_bleed, _place_icon
from .frame import BACKDROP_JITTER, BACKDROPS, BORDER_COLOR

# Approximate rarity colours (RGB): common, uncommon, rare, very rare, visceral.
RARITY_TILE_COLORS = (
    (122, 86, 52),
    (200, 170, 50),
    (60, 140, 70),
    (120, 60, 160),
    (170, 40, 45),
)
COLOR_JITTER = 22.0
TILE_SIDE_RANGE = (0.62, 1.08)   # tile side as a fraction of the crop
BORDER_FRACTION = (0.04, 0.10)
CHAMFER_FRACTION = (0.04, 0.16)  # corner cut relative to the tile side
TILT_DEGREES = 5.0
SHIFT_FRACTION = 0.06
GRADIENT_RANGE = 40.0
NOISE_SIGMA = 4.0
GLYPH_SCALE_RANGE = (0.62, 1.05)  # glyph size relative to the crop, before the tile-size factor
FLAT_PROBABILITY = 0.2            # no tile: glyph straight on the rarity colour
BAND_PROBABILITY = 0.4
BAND_DARKNESS = (0.6, 0.85)
NEIGHBOR_BLEED_PROBABILITY = 0.3


def _rarity_fill(rng: np.random.Generator, size: int) -> np.ndarray:
    base = np.array(RARITY_TILE_COLORS[rng.integers(len(RARITY_TILE_COLORS))], np.float32) + rng.uniform(-COLOR_JITTER, COLOR_JITTER, 3)
    ys = np.linspace(-0.5, 0.5, size, dtype=np.float32)[:, None, None]
    ramp = ys * rng.uniform(-1, 1) * GRADIENT_RANGE * 2
    fill = base[None, None, :] + ramp + rng.normal(0, NOISE_SIGMA, (size, size, 3))
    return np.clip(fill, 0, 255).astype(np.float32)


def _tile_corners(centre, half_side: float, chamfer: float, tilt_degrees: float) -> np.ndarray:
    """Square with cut corners (an octagon), rotated by `tilt_degrees`, as int32 pixel coordinates."""
    c = half_side - chamfer
    local = np.array([(-c, -half_side), (c, -half_side), (half_side, -c), (half_side, c),
                      (c, half_side), (-c, half_side), (-half_side, c), (-half_side, -c)], np.float32)
    angle = np.radians(tilt_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    return np.round(local @ rotation.T + np.array(centre, np.float32)).astype(np.int32)


def _tile_image(rng: np.random.Generator, fill: np.ndarray) -> np.ndarray:
    size = fill.shape[0]
    backdrop = np.array(BACKDROPS[rng.integers(len(BACKDROPS))], np.float32) + rng.uniform(-BACKDROP_JITTER, BACKDROP_JITTER, 3)
    image = np.empty_like(fill)
    image[:] = np.clip(backdrop, 0, 255)

    centre = tuple(size / 2 + rng.uniform(-SHIFT_FRACTION, SHIFT_FRACTION, 2) * size)
    half = rng.uniform(*TILE_SIDE_RANGE) * size / 2
    chamfer = half * rng.uniform(*CHAMFER_FRACTION)
    tilt = rng.uniform(-TILT_DEGREES, TILT_DEGREES)
    outer = _tile_corners(centre, half, chamfer, tilt)
    inner = _tile_corners(centre, half * (1.0 - rng.uniform(*BORDER_FRACTION)), chamfer, tilt)

    cv2.fillConvexPoly(image, outer, tuple(float(c) for c in BORDER_COLOR), cv2.LINE_AA)
    layer = fill.copy()
    if rng.random() < BAND_PROBABILITY:
        layer[int(centre[1] + half * rng.uniform(0.3, 0.6)):] *= rng.uniform(*BAND_DARKNESS)
    mask = np.zeros((size, size), np.uint8)
    cv2.fillConvexPoly(mask, inner, 255, cv2.LINE_AA)
    weight = (mask.astype(np.float32) / 255.0)[..., None]
    return layer * weight + image * (1.0 - weight)


def compose_addon(rng: np.random.Generator, icon: np.ndarray, size: int = WORK_SIZE) -> np.ndarray:
    """uint8 RGB (size x size): add-on glyph on a rarity-coloured tile."""
    fill = _rarity_fill(rng, size)
    glyph, alpha = _place_icon(rng, icon, size, GLYPH_SCALE_RANGE)
    if rng.random() < FLAT_PROBABILITY:
        image = glyph + fill * (1.0 - alpha)
    else:
        image = glyph + _tile_image(rng, fill) * (1.0 - alpha)
    if rng.random() < NEIGHBOR_BLEED_PROBABILITY:
        image = _add_neighbor_bleed(rng, image.astype(np.float32))
    return np.clip(image, 0, 255).astype(np.uint8)
