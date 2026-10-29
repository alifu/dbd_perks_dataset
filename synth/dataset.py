"""Step 3: an endless supply of (image, class) pairs for PyTorch."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .compose import compose, load_icon
from .photo import apply_photo_effects

# Fixed seeds make validation / test sets identical on every run.
VAL_SEED = 1000
TEST_SEED = 2000


def icon_paths_for(labels: list[str], icons_dir: Path) -> list[Path]:
    return [icons_dir / f"{label}.png" for label in labels]


class PerkSynthDataset(Dataset):
    """Item i belongs to class i % n_classes, so every class gets equal weight.

    fixed_seed=None draws fresh randomness each call (training); an int makes item i reproducible.
    """

    def __init__(self, icon_paths: list[Path], samples_per_class: int,
                 severity_range: tuple[float, float], fixed_seed: int | None = None) -> None:
        self._icon_paths = icon_paths
        self._samples_per_class = samples_per_class
        self._severity_range = severity_range
        self._fixed_seed = fixed_seed
        self._icons: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._icon_paths) * self._samples_per_class

    def _icon(self, class_index: int) -> np.ndarray:
        if class_index not in self._icons:
            self._icons[class_index] = load_icon(self._icon_paths[class_index])
        return self._icons[class_index]

    def render(self, index: int) -> tuple[np.ndarray, int]:
        class_index = index % len(self._icon_paths)
        rng = np.random.default_rng(None if self._fixed_seed is None else [self._fixed_seed, index])
        severity = rng.uniform(*self._severity_range)
        image = apply_photo_effects(rng, compose(rng, self._icon(class_index)), severity)
        return image, class_index

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, class_index = self.render(index)
        # HWC uint8 -> CHW float in 0..1 (the exported Core ML model rescales 0..255 the same way)
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0, class_index
