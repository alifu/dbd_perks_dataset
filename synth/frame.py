"""The in-game look of a perk slot photographed off a monitor: black-bordered diamond, tinted inside,
glyph on top, plain screen backdrop around it and sometimes a corner of the neighbouring slot."""
import cv2
import numpy as np

# Half-diagonal of the black border as a fraction of the crop: the app's crop is tight, so the diamond often fills it.
RADIUS_RANGE = (0.40, 0.68)
BORDER_FRACTION = (0.07, 0.13)   # border thickness relative to the radius
TILT_DEGREES = 6.0
SHIFT_FRACTION = 0.06
LOWER_BAND_PROBABILITY = 0.5
LOWER_BAND_DARKNESS = (0.55, 0.8)
CLAW_PROBABILITY = 0.5
CLAW_REACH = (0.18, 0.32)
CLAW_COUNT = (2, 4)
BACKDROPS = (
    (170, 185, 205),  # pale blue-grey monitor
    (205, 205, 210),
    (110, 120, 130),
    (235, 235, 240),
    (30, 30, 35),
)
BACKDROP_JITTER = 18.0
BORDER_COLOR = (12, 10, 14)


def diamond(centre: tuple[float, float], radius: float, tilt_degrees: float) -> np.ndarray:
    """Four corners (top, right, bottom, left) as int32 pixel coordinates, rotated by `tilt_degrees`."""
    angles = np.radians(tilt_degrees + np.array([270.0, 0.0, 90.0, 180.0]))
    points = np.stack([centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)], axis=1)
    return np.round(points).astype(np.int32)


def _fill(image: np.ndarray, points: np.ndarray, color) -> None:
    cv2.fillConvexPoly(image, points, tuple(float(c) for c in color), cv2.LINE_AA)


def _darken_lower_band(rng, image: np.ndarray, inner: np.ndarray, centre: tuple[float, float], radius: float) -> None:
    """Many frames show a darker strip across the bottom of the tinted diamond."""
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, inner, 255)
    ys = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    band = (mask > 0) & (ys > centre[1] + radius * rng.uniform(0.25, 0.5))
    image[band] *= rng.uniform(*LOWER_BAND_DARKNESS)


def _add_claws(rng, image: np.ndarray) -> None:
    """Pale slanted strokes in a corner, like the neighbouring slot's artwork leaking into the crop."""
    size = image.shape[0]
    reach = rng.uniform(*CLAW_REACH) * size
    top, left = rng.random() < 0.7, rng.random() < 0.3  # mostly top-right, as in the app's crops
    x0 = 0 if left else size - reach
    y0 = 0 if top else size - reach
    for i in range(int(rng.integers(*CLAW_COUNT))):
        x = x0 + reach * i / 4
        stroke = np.array([[x, y0], [x + reach * 0.18, y0], [x + reach * 0.05, y0 + reach], [x - reach * 0.1, y0 + reach * 0.8]])
        points = np.round(stroke).astype(np.int32)
        _fill(image, points, rng.uniform(190, 245, 3))
        cv2.polylines(image, [points], True, tuple(float(c) for c in BORDER_COLOR), 1, cv2.LINE_AA)


def draw_framed(rng: np.random.Generator, tint: np.ndarray, glyph: np.ndarray, glyph_alpha: np.ndarray) -> np.ndarray:
    """float32 RGB 0..255. `tint` is the rarity-coloured fill (size x size x 3); glyph is premultiplied RGB + alpha."""
    size = tint.shape[0]
    backdrop = np.array(BACKDROPS[rng.integers(len(BACKDROPS))], np.float32) + rng.uniform(-BACKDROP_JITTER, BACKDROP_JITTER, 3)
    image = np.empty_like(tint)
    image[:] = np.clip(backdrop, 0, 255)

    centre = tuple(size / 2 + rng.uniform(-SHIFT_FRACTION, SHIFT_FRACTION, 2) * size)
    radius = rng.uniform(*RADIUS_RANGE) * size
    tilt = rng.uniform(-TILT_DEGREES, TILT_DEGREES)
    outer = diamond(centre, radius, tilt)
    inner = diamond(centre, radius * (1.0 - rng.uniform(*BORDER_FRACTION)), tilt)

    _fill(image, outer, BORDER_COLOR)
    inner_layer = tint.copy()
    if rng.random() < LOWER_BAND_PROBABILITY:
        _darken_lower_band(rng, inner_layer, inner, centre, radius)
    mask = np.zeros((size, size), np.uint8)
    cv2.fillConvexPoly(mask, inner, 255, cv2.LINE_AA)
    weight = (mask.astype(np.float32) / 255.0)[..., None]
    image = inner_layer * weight + image * (1.0 - weight)

    if rng.random() < CLAW_PROBABILITY:
        _add_claws(rng, image)
    return glyph + image * (1.0 - glyph_alpha)
