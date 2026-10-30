"""Step 2: make a clean composite look like a phone photo of a screen."""
import cv2
import numpy as np

OUTPUT_SIZE = 128

# Each effect has a probability and strength that grow with `severity` (0 = clean, 1 = worst case).
PERSPECTIVE_JITTER = 0.06
MIN_RESOLUTION_FACTOR = 0.3
MAX_BLUR_SIGMA = 2.0
MAX_MOTION_BLUR_PIXELS = 7
GLARE_PROBABILITY = 0.4
MAX_GLARE_STRENGTH = 0.5
MOIRE_PROBABILITY = 0.5
MAX_MOIRE_AMPLITUDE = 0.06
MAX_NOISE_SIGMA = 8.0
JPEG_QUALITY_RANGE = (35, 90)
PIXELATE_PROBABILITY = 0.5      # the game UI is low resolution and shows up as chunky blocks
PIXELATE_FACTOR = (0.2, 0.55)
GRID_PROBABILITY = 0.6          # the monitor's pixel grid
GRID_PERIOD = (3.0, 7.0)        # pixels, at the 192 px working size
MAX_GRID_DEPTH = 0.35


def _perspective(rng, image, severity):
    height, width = image.shape[:2]
    jitter = PERSPECTIVE_JITTER * severity * min(height, width)
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    warped_corners = corners + rng.uniform(-jitter, jitter, corners.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(corners, warped_corners)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _low_resolution(rng, image, severity):
    height, width = image.shape[:2]
    factor = rng.uniform(1.0 - (1.0 - MIN_RESOLUTION_FACTOR) * severity, 1.0)
    shrunk = cv2.resize(image, (max(int(width * factor), 16), max(int(height * factor), 16)), interpolation=cv2.INTER_AREA)
    return cv2.resize(shrunk, (width, height), interpolation=cv2.INTER_LINEAR)


def _pixelate(rng, image, severity):
    if rng.random() >= PIXELATE_PROBABILITY:
        return image
    height, width = image.shape[:2]
    factor = rng.uniform(1.0 - (1.0 - PIXELATE_FACTOR[0]) * severity, 1.0) if severity < 1 else rng.uniform(*PIXELATE_FACTOR)
    shrunk = cv2.resize(image, (max(int(width * factor), 12), max(int(height * factor), 12)), interpolation=cv2.INTER_AREA)
    return cv2.resize(shrunk, (width, height), interpolation=cv2.INTER_NEAREST)


def _screen_grid(rng, image, severity):
    if rng.random() >= GRID_PROBABILITY:
        return image
    height, width = image.shape[:2]
    period = rng.uniform(*GRID_PERIOD)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    lines = ((xs / period) % 1.0 < 1.0 / period) | ((ys / period) % 1.0 < 1.0 / period)
    depth = MAX_GRID_DEPTH * severity * rng.uniform(0.4, 1.0)
    return image * (1.0 - depth * lines[..., None])


def _blur(rng, image, severity):
    if rng.random() < 0.4:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.1, MAX_BLUR_SIGMA * severity + 0.1))
    if rng.random() < 0.2:
        length = int(rng.integers(3, MAX_MOTION_BLUR_PIXELS * severity + 4))
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (length, length))
        image = cv2.filter2D(image, -1, kernel / max(kernel.sum(), 1e-6))
    return image


def _color(rng, image, severity):
    image = image.astype(np.float32) / 255.0
    gain = 1.0 + (rng.uniform(0.6, 1.4) - 1.0) * severity
    gamma = 1.0 + (rng.uniform(0.7, 1.4) - 1.0) * severity
    balance = np.power(rng.uniform(0.85, 1.15, 3), severity)
    image = np.power(np.clip(image, 0, 1), gamma) * gain * balance
    gray = image.mean(axis=2, keepdims=True)
    saturation = 1.0 + (rng.uniform(0.7, 1.2) - 1.0) * severity
    return np.clip(gray + (image - gray) * saturation, 0, 1) * 255.0


def _glare(rng, image, severity):
    if rng.random() >= GLARE_PROBABILITY:
        return image
    height, width = image.shape[:2]
    centre = rng.uniform(0, 1, 2) * (width, height)
    radius = rng.uniform(0.3, 0.9) * min(height, width)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.hypot(xs - centre[0], ys - centre[1])
    alpha = np.clip(1.0 - distance / radius, 0, 1) ** 2 * MAX_GLARE_STRENGTH * severity * rng.uniform(0.3, 1.0)
    return image * (1 - alpha[..., None]) + 255.0 * alpha[..., None]


def _moire(rng, image, severity):
    if rng.random() >= MOIRE_PROBABILITY:
        return image
    height, width = image.shape[:2]
    angle, cycles, phase = rng.uniform(0, np.pi), rng.uniform(20, 60), rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32) / min(height, width)
    grating = np.sin(2 * np.pi * cycles * (xs * np.cos(angle) + ys * np.sin(angle)) + phase)
    return image + grating[..., None] * MAX_MOIRE_AMPLITUDE * severity * 255.0


def _noise_and_jpeg(rng, image, severity):
    image = image + rng.normal(0, MAX_NOISE_SIGMA * severity * rng.random(), image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    low, high = JPEG_QUALITY_RANGE
    quality = int(high - (high - low) * severity * rng.random())
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image


def degrade(rng: np.random.Generator, image: np.ndarray, severity: float, geometric: bool = True) -> np.ndarray:
    """Photo-style damage at any aspect ratio. `geometric=False` skips the perspective warp
    (scenes apply their own so ground-truth positions stay exact)."""
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity <= 0:
        return image
    if geometric:
        image = _perspective(rng, image, severity)
    image = _low_resolution(rng, image, severity)
    image = _pixelate(rng, image, severity)
    image = _blur(rng, image, severity)
    image = _color(rng, image, severity)
    image = _glare(rng, image, severity)
    image = _moire(rng, image, severity)
    image = _screen_grid(rng, image, severity)
    return _noise_and_jpeg(rng, image, severity)


def apply_photo_effects(rng: np.random.Generator, image: np.ndarray, severity: float) -> np.ndarray:
    """uint8 RGB (OUTPUT_SIZE x OUTPUT_SIZE). `severity` in [0, 1]; 0 only resizes."""
    return cv2.resize(degrade(rng, image, severity), (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
