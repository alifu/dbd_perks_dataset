"""Step 1: put a transparent perk icon on a believable in-game background."""
from pathlib import Path

import cv2
import numpy as np

from .frame import draw_framed

WORK_SIZE = 192
ICON_SIZE = 256

# Approximate rarity / frame colours (RGB). Jittered per sample, so exact values do not matter.
RARITY_COLORS = (
    (96, 60, 38),    # brown
    (190, 150, 40),  # yellow
    (60, 120, 60),   # green
    (90, 50, 120),   # purple
    (150, 40, 40),   # red
    (70, 70, 75),    # grey
    (40, 110, 120),  # teal
)
PLAIN_BLACK_PROBABILITY = 0.12
PLAIN_WHITE_PROBABILITY = 0.0  # game frames are never white; white-on-white icons are unlearnable
GRADIENT_RANGE = 28.0
BACKGROUND_NOISE_SIGMA = 4.0
SCALE_RANGE = (0.7, 1.6)   # icon art size relative to the crop (in-game art/frame ratio is unknown)
ROTATION_DEGREES = 8.0
SHIFT_FRACTION = 0.08
NEIGHBOR_BLEED_PROBABILITY = 0.3
FRAMED_PROBABILITY = 0.75  # photographed slot with the black-bordered diamond; the rest keep the flat look
FRAMED_SCALE_RANGE = (0.75, 1.3)


def load_icon(path: Path) -> np.ndarray:
    """RGBA uint8 (ICON_SIZE x ICON_SIZE). Odd-sized wiki icons are resized."""
    icon = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if icon is None:
        raise FileNotFoundError(path)
    if icon.ndim == 2:
        icon = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGRA)
    elif icon.shape[2] == 3:
        icon = cv2.cvtColor(icon, cv2.COLOR_BGR2BGRA)
    icon = cv2.cvtColor(icon, cv2.COLOR_BGRA2RGBA)
    if icon.shape[:2] != (ICON_SIZE, ICON_SIZE):
        icon = cv2.resize(icon, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
    return icon


def make_background(rng: np.random.Generator, size: int = WORK_SIZE) -> np.ndarray:
    """float32 RGB 0..255: flat or gradient rarity colour with a little sensor-like texture."""
    roll = rng.random()
    if roll < PLAIN_BLACK_PROBABILITY:
        return np.zeros((size, size, 3), np.float32)
    if roll < PLAIN_BLACK_PROBABILITY + PLAIN_WHITE_PROBABILITY:
        return np.full((size, size, 3), 255, np.float32)

    base = np.array(RARITY_COLORS[rng.integers(len(RARITY_COLORS))], np.float32)
    base = base + rng.uniform(-20, 20, 3)
    angle = rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size - 0.5
    ramp = (xs * np.cos(angle) + ys * np.sin(angle)) * rng.uniform(-1, 1) * GRADIENT_RANGE * 2
    background = base[None, None, :] + ramp[..., None]
    background += rng.normal(0, BACKGROUND_NOISE_SIGMA, background.shape)
    return np.clip(background, 0, 255).astype(np.float32)


def _place_icon(rng: np.random.Generator, icon: np.ndarray, size: int, scale_range: tuple[float, float] = SCALE_RANGE) -> tuple[np.ndarray, np.ndarray]:
    """Scale / rotate / shift the icon. Returns (premultiplied RGB float, alpha float 0..1)."""
    alpha = icon[..., 3:4].astype(np.float32) / 255.0
    premultiplied = np.concatenate([icon[..., :3].astype(np.float32) * alpha, alpha], axis=2)

    scale = rng.uniform(*scale_range) * size / ICON_SIZE
    matrix = cv2.getRotationMatrix2D((ICON_SIZE / 2, ICON_SIZE / 2), rng.uniform(-ROTATION_DEGREES, ROTATION_DEGREES), scale)
    # The rotation keeps the icon centre fixed; move that centre to the canvas centre plus a random shift.
    shift = rng.uniform(-SHIFT_FRACTION, SHIFT_FRACTION, 2) * size
    matrix[:, 2] += (size - ICON_SIZE) / 2 + shift
    warped = cv2.warpAffine(premultiplied, matrix, (size, size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return warped[..., :3], warped[..., 3:4]


def _add_neighbor_bleed(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """A dark or coloured wedge in one corner, like the neighbouring frame in a real loadout."""
    size = image.shape[0]
    reach = int(size * rng.uniform(0.08, 0.2))
    corner = rng.integers(4)
    points = [
        [(0, 0), (reach, 0), (0, reach)],
        [(size, 0), (size - reach, 0), (size, reach)],
        [(0, size), (reach, size), (0, size - reach)],
        [(size, size), (size - reach, size), (size, size - reach)],
    ][corner]
    layer = image.copy()
    colour = tuple(float(c) for c in rng.uniform(0, 140, 3))
    cv2.fillConvexPoly(layer, np.array(points, np.int32), colour)
    return cv2.addWeighted(layer, 0.85, image, 0.15, 0)


def compose(rng: np.random.Generator, icon: np.ndarray, size: int = WORK_SIZE,
            framed_probability: float = FRAMED_PROBABILITY) -> np.ndarray:
    """uint8 RGB (size x size): icon over a random in-game style background.

    `framed_probability` is the share of samples drawn with the photographed, black-bordered diamond."""
    background = make_background(rng, size)
    if rng.random() < framed_probability:
        colour, alpha = _place_icon(rng, icon, size, FRAMED_SCALE_RANGE)
        image = draw_framed(rng, background, colour, alpha)
        return np.clip(image, 0, 255).astype(np.uint8)
    colour, alpha = _place_icon(rng, icon, size)
    image = colour + background * (1.0 - alpha)
    if rng.random() < NEIGHBOR_BLEED_PROBABILITY:
        image = _add_neighbor_bleed(rng, image.astype(np.float32))
    return np.clip(image, 0, 255).astype(np.uint8)
