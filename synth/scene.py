"""Fake 'photo of the loadout screen' with known perk positions, to test the automatic slot detector.

The real in-game layout is not known to this code, so layouts and art size vary on purpose.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .compose import ICON_SIZE, RARITY_COLORS
from .photo import degrade

LAYOUTS = ("diamond", "row", "column", "free")
SCENE_SIZE = (640, 480)            # width, height
DIAGONAL_FRACTION = (0.11, 0.2)    # frame diagonal relative to scene width
FRAME_GAP = (1.0, 1.12)
ART_FRACTION = (0.45, 0.8)         # icon art size relative to the frame diagonal
SCENE_PERSPECTIVE = 0.035
EMPTY_FRAMES = (0, 2)
SAFE_MARGIN = 0.08


@dataclass(frozen=True)
class PlacedPerk:
    perk_id: str
    cx: float
    cy: float
    diagonal: float


def _layout_centres(rng, layout: str, diagonal: float, width: int, height: int) -> list[tuple[float, float]]:
    gap = rng.uniform(*FRAME_GAP)
    step = diagonal * gap
    half = step / 2
    cx = rng.uniform(0.3, 0.7) * width
    cy = rng.uniform(0.35, 0.65) * height
    if layout == "diamond":
        return [(cx, cy - half), (cx - half, cy), (cx + half, cy), (cx, cy + half)]
    if layout == "row":
        return [(cx + (i - 1.5) * step, cy) for i in range(4)]
    if layout == "column":
        return [(cx, cy + (i - 1.5) * step) for i in range(4)]
    centres: list[tuple[float, float]] = []
    while len(centres) < 4:
        candidate = (rng.uniform(0.15, 0.85) * width, rng.uniform(0.15, 0.85) * height)
        if all(np.hypot(candidate[0] - x, candidate[1] - y) > step for x, y in centres):
            centres.append(candidate)
    return centres


def _background(rng, width: int, height: int) -> np.ndarray:
    base = np.array(RARITY_COLORS[rng.integers(len(RARITY_COLORS))], np.float32) * rng.uniform(0.08, 0.3)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    vignette = 1.0 - 0.5 * np.hypot(xs / width - 0.5, ys / height - 0.5)
    image = base[None, None, :] * vignette[..., None] + rng.normal(0, 3, (height, width, 3))
    for _ in range(int(rng.integers(2, 7))):  # dim UI panels and text-like bars
        x0, y0 = int(rng.integers(0, width)), int(rng.integers(0, height))
        x1, y1 = x0 + int(rng.integers(20, width // 2)), y0 + int(rng.integers(4, 40))
        cv2.rectangle(image, (x0, y0), (x1, y1), tuple(float(c) for c in rng.uniform(10, 70, 3)), -1)
    return image.astype(np.float32)


def _diamond_points(cx: float, cy: float, diagonal: float, scale: float = 1.0) -> np.ndarray:
    r = diagonal / 2 * scale
    return np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)


def _draw_frame(rng, image: np.ndarray, cx: float, cy: float, diagonal: float) -> None:
    colour = np.array(RARITY_COLORS[rng.integers(len(RARITY_COLORS))], np.float32) * rng.uniform(0.3, 0.6)
    cv2.fillConvexPoly(image, _diamond_points(cx, cy, diagonal), tuple(float(c) for c in colour), cv2.LINE_AA)
    border = tuple(float(c) for c in rng.uniform(150, 230, 3))
    thickness = max(1, int(diagonal * 0.03))
    cv2.polylines(image, [_diamond_points(cx, cy, diagonal)], True, border, thickness, cv2.LINE_AA)
    cv2.polylines(image, [_diamond_points(cx, cy, diagonal, 0.9)], True, border, 1, cv2.LINE_AA)


def _draw_icon(rng, image: np.ndarray, icon: np.ndarray, cx: float, cy: float, diagonal: float) -> None:
    art = diagonal * rng.uniform(*ART_FRACTION)
    alpha = icon[..., 3:4].astype(np.float32) / 255.0
    premultiplied = np.concatenate([icon[..., :3].astype(np.float32) * alpha, alpha], axis=2)
    matrix = cv2.getRotationMatrix2D((ICON_SIZE / 2, ICON_SIZE / 2), rng.uniform(-3, 3), art / ICON_SIZE)
    matrix[:, 2] += (cx - ICON_SIZE / 2, cy - ICON_SIZE / 2)
    height, width = image.shape[:2]
    warped = cv2.warpAffine(premultiplied, matrix, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    image[:] = warped[..., :3] + image * (1.0 - warped[..., 3:4])


def _warp(rng, image: np.ndarray, points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    jitter = SCENE_PERSPECTIVE * min(height, width)
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = corners + rng.uniform(-jitter, jitter, corners.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(corners, target)
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    moved = cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return warped, moved


def render_scene(rng: np.random.Generator, perks: list[tuple[str, np.ndarray]], layout: str,
                 severity: float, size: tuple[int, int] = SCENE_SIZE) -> tuple[np.ndarray, list[PlacedPerk]]:
    """perks: up to four (id, RGBA icon). Returns (uint8 RGB scene, ground truth in final pixel coordinates)."""
    width, height = size
    diagonal = rng.uniform(*DIAGONAL_FRACTION) * width
    centres = _layout_centres(rng, layout, diagonal, width, height)[:len(perks)]
    image = _background(rng, width, height)
    for _ in range(int(rng.integers(*EMPTY_FRAMES) if EMPTY_FRAMES[1] > EMPTY_FRAMES[0] else 0)):
        x, y = rng.uniform(SAFE_MARGIN, 1 - SAFE_MARGIN) * width, rng.uniform(SAFE_MARGIN, 1 - SAFE_MARGIN) * height
        if all(np.hypot(x - cx, y - cy) > diagonal * 1.1 for cx, cy in centres):
            _draw_frame(rng, image, x, y, diagonal)
    for (perk_id, icon), (cx, cy) in zip(perks, centres):
        _draw_frame(rng, image, cx, cy, diagonal)
        _draw_icon(rng, image, icon, cx, cy, diagonal)
    warped, moved = _warp(rng, np.clip(image, 0, 255).astype(np.uint8), centres)
    final = degrade(rng, warped, severity, geometric=False)
    placed = [PlacedPerk(perk_id, float(x), float(y), float(diagonal))
              for (perk_id, _), (x, y) in zip(perks, moved)]
    return final, placed
