"""Step 3: an endless supply of (image, class) pairs for PyTorch."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .compose import compose, load_icon
from .negatives import make_negative
from .photo import apply_photo_effects

# Extra class after the perks: "this window is not a perk icon".
BACKGROUND_LABEL = "__background__"

# Fixed seeds make validation / test sets identical on every run.
VAL_SEED = 1000
TEST_SEED = 2000


def icon_paths_for(labels: list[str], icons_dir: Path) -> list[Path]:
    return [icons_dir / f"{label}.png" for label in labels]


class PerkSynthDataset(Dataset):
    """Item i < n*samples belongs to perk i % n, so every perk gets equal weight. The remaining
    `negative_fraction * n * samples` items are background windows labelled with class n.

    fixed_seed=None draws fresh randomness each call (training); an int makes item i reproducible.
    """

    def __init__(self, icon_paths: list[Path], samples_per_class: int,
                 severity_range: tuple[float, float], fixed_seed: int | None = None,
                 negative_fraction: float = 0.0) -> None:
        self._icon_paths = icon_paths
        self._samples_per_class = samples_per_class
        self._severity_range = severity_range
        self._fixed_seed = fixed_seed
        self._perk_items = len(icon_paths) * samples_per_class
        self._negative_items = int(self._perk_items * negative_fraction)
        self._icons: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self._perk_items + self._negative_items

    def _icon(self, class_index: int) -> np.ndarray:
        if class_index not in self._icons:
            self._icons[class_index] = load_icon(self._icon_paths[class_index])
        return self._icons[class_index]

    def _some_icons(self, rng: np.random.Generator, count: int = 6) -> list[np.ndarray]:
        return [self._icon(int(i)) for i in rng.integers(0, len(self._icon_paths), count)]

    def render(self, index: int) -> tuple[np.ndarray, int]:
        rng = np.random.default_rng(None if self._fixed_seed is None else [self._fixed_seed, index])
        severity = rng.uniform(*self._severity_range)
        if index >= self._perk_items:
            window = make_negative(rng, self._some_icons(rng))
            return apply_photo_effects(rng, window, severity), len(self._icon_paths)
        class_index = index % len(self._icon_paths)
        return apply_photo_effects(rng, compose(rng, self._icon(class_index)), severity), class_index

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, class_index = self.render(index)
        # HWC uint8 -> CHW float in 0..1 (the exported Core ML model rescales 0..255 the same way)
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0, class_index
