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


def _perspective(rng, image, severity):
    size = image.shape[0]
    jitter = PERSPECTIVE_JITTER * severity * size
    corners = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    warped_corners = corners + rng.uniform(-jitter, jitter, corners.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(corners, warped_corners)
    return cv2.warpPerspective(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _low_resolution(rng, image, severity):
    size = image.shape[0]
    factor = rng.uniform(1.0 - (1.0 - MIN_RESOLUTION_FACTOR) * severity, 1.0)
    small = max(int(size * factor), 16)
    shrunk = cv2.resize(image, (small, small), interpolation=cv2.INTER_AREA)
    return cv2.resize(shrunk, (size, size), interpolation=cv2.INTER_LINEAR)


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
    size = image.shape[0]
    centre = rng.uniform(0, size, 2)
    radius = rng.uniform(0.3, 0.9) * size
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    distance = np.hypot(xs - centre[0], ys - centre[1])
    alpha = np.clip(1.0 - distance / radius, 0, 1) ** 2 * MAX_GLARE_STRENGTH * severity * rng.uniform(0.3, 1.0)
    return image * (1 - alpha[..., None]) + 255.0 * alpha[..., None]


def _moire(rng, image, severity):
    if rng.random() >= MOIRE_PROBABILITY:
        return image
    size = image.shape[0]
    angle, cycles, phase = rng.uniform(0, np.pi), rng.uniform(20, 60), rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    grating = np.sin(2 * np.pi * cycles * (xs * np.cos(angle) + ys * np.sin(angle)) + phase)
    return image + grating[..., None] * MAX_MOIRE_AMPLITUDE * severity * 255.0


def _noise_and_jpeg(rng, image, severity):
    image = image + rng.normal(0, MAX_NOISE_SIGMA * severity * rng.random(), image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    low, high = JPEG_QUALITY_RANGE
    quality = int(high - (high - low) * severity * rng.random())
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image


def apply_photo_effects(rng: np.random.Generator, image: np.ndarray, severity: float) -> np.ndarray:
    """uint8 RGB (OUTPUT_SIZE x OUTPUT_SIZE). `severity` in [0, 1]; 0 only resizes."""
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity > 0:
        image = _perspective(rng, image, severity)
        image = _low_resolution(rng, image, severity)
        image = _blur(rng, image, severity)
        image = _color(rng, image, severity)
        image = _glare(rng, image, severity)
        image = _moire(rng, image, severity)
        image = _noise_and_jpeg(rng, image, severity)
    return cv2.resize(image, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
