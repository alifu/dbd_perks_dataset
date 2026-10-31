"""Convert models/v2/perk_classifier.pt to a Core ML classifier and check it agrees with PyTorch.

python3 export_coreml_v2.py
python3 export_coreml_v2.py --target addons
Output: models/v2/DBDPerkClassifierV2.mlpackage  (input "input_1", outputs "classLabel" + "classLabel_probs")
        models/addons_v1/DBDAddonClassifierV1.mlpackage for --target addons
"""
import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from PIL import Image
from torch import nn

from addon_data import addon_labels_and_paths
from synth.addon_tile import compose_addon
from synth.compose import compose
from synth.dataset import BACKGROUND_LABEL, TEST_SEED, PerkSynthDataset, icon_paths_for
from train_perks_v2 import ADDON_OUT_DIR, ICONS_DIR, TEST_SEVERITY, PerkClassifier

MODEL_DIR = Path("models/v2")
COREML_NAME = "DBDPerkClassifierV2.mlpackage"
ADDON_COREML_NAME = "DBDAddonClassifierV1.mlpackage"
IMG_SIZE = 128
INPUT_NAME = "input_1"
PARITY_SAMPLES = 300
MIN_AGREEMENT = 0.98
DEPLOYMENT_TARGET = ct.target.iOS16


class WithSoftmax(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)


def load_model(labels: list[str], model_dir: Path = MODEL_DIR) -> PerkClassifier:
    model = PerkClassifier(len(labels), pretrained=False)
    model.load_state_dict(torch.load(model_dir / "perk_classifier.pt", map_location="cpu", weights_only=True))
    return model.eval()


def convert(model: PerkClassifier, labels: list[str], description: str = "Dead by Daylight perk classifier v2 (synthetic-photo training)") -> ct.models.MLModel:
    traced = torch.jit.trace(WithSoftmax(model).eval(), torch.rand(1, 3, IMG_SIZE, IMG_SIZE))
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=DEPLOYMENT_TARGET,
        inputs=[ct.ImageType(name=INPUT_NAME, shape=(1, 3, IMG_SIZE, IMG_SIZE), scale=1 / 255.0,
                             color_layout=ct.colorlayout.RGB)],
        classifier_config=ct.ClassifierConfig(labels),
    )
    mlmodel.short_description = description
    mlmodel.author = "Annur Alif Ramadhoni"
    mlmodel.license = "MIT (educational use; icons belong to Behaviour Interactive)"
    return mlmodel


def check_parity(mlmodel, model: PerkClassifier, labels: list[str], target: str = "perks") -> float:
    """Fraction of held-out hard samples where Core ML and PyTorch pick the same perk."""
    perk_ids = [label for label in labels if label != BACKGROUND_LABEL]
    paths = addon_labels_and_paths()[1] if target == "addons" else icon_paths_for(perk_ids, ICONS_DIR)
    composer = compose_addon if target == "addons" else compose
    dataset = PerkSynthDataset(paths, 2, TEST_SEVERITY, TEST_SEED, negative_fraction=0.25, composer=composer)
    step = max(len(dataset) // PARITY_SAMPLES, 1)
    agree = total = 0
    for index in range(0, len(dataset), step):
        image, _ = dataset.render(index)
        with torch.no_grad():
            torch_label = labels[model(torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0).argmax().item()]
        coreml_label = mlmodel.predict({INPUT_NAME: Image.fromarray(image)})["classLabel"]
        agree += torch_label == coreml_label
        total += 1
    return agree / total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("perks", "addons"), default="perks")
    target = parser.parse_args().target
    model_dir, name = (ADDON_OUT_DIR, ADDON_COREML_NAME) if target == "addons" else (MODEL_DIR, COREML_NAME)
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    model = load_model(labels, model_dir)
    description = "Dead by Daylight add-on classifier v1 (synthetic-photo training)" if target == "addons" else "Dead by Daylight perk classifier v2 (synthetic-photo training)"
    mlmodel = convert(model, labels, description)
    mlmodel.save(str(model_dir / name))
    agreement = check_parity(mlmodel, model, labels, target)
    print(f"✅ saved {model_dir / name}; Core ML vs PyTorch agreement = {agreement:.3f}")
    if agreement < MIN_AGREEMENT:
        print(f"❌ agreement below {MIN_AGREEMENT}; do not ship this export")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
