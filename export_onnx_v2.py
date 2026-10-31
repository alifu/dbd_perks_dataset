"""Convert models/<target>/perk_classifier.pt to ONNX for the web app and check it against Core ML and PyTorch.

python3 export_onnx_v2.py
python3 export_onnx_v2.py --target addons
Output: models/v2/DBDPerkClassifierV2.onnx  (input "input_1" NCHW float 0..1 RGB, output "probs" softmax)
        models/addons_v1/DBDAddonClassifierV1.onnx for --target addons
Preprocessing matches export_coreml_v2.py: 128x128 RGB, pixel / 255, ImageNet normalisation inside the network.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from addon_data import addon_labels_and_paths
from export_coreml_v2 import (ADDON_COREML_NAME, COREML_NAME, IMG_SIZE, INPUT_NAME, MODEL_DIR, PARITY_SAMPLES,
                              WithSoftmax, load_model)
from synth.addon_tile import compose_addon
from synth.compose import compose
from synth.dataset import TEST_SEED, PerkSynthDataset, icon_paths_for
from train_perks_v2 import ADDON_OUT_DIR, ICONS_DIR, TEST_SEVERITY

ONNX_NAME = "DBDPerkClassifierV2.onnx"
ADDON_ONNX_NAME = "DBDAddonClassifierV1.onnx"
OUTPUT_NAME = "probs"
OPSET = 17
MIN_TOP1_AGREEMENT = 0.99
MAX_PROBABILITY_DIFFERENCE = 1e-3
# Core ML runs in fp16 on device, so its probabilities drift from fp32 by more than ONNX does; the plan's
# 1e-3 bound is checked against PyTorch (same fp32 maths), and Core ML only has to agree on the top-1 label.


def export(model: torch.nn.Module, path: Path) -> None:
    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        WithSoftmax(model).eval(), dummy, str(path), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}}, opset_version=OPSET, dynamo=False,
    )


def parity_samples(labels: list[str], target: str) -> list[np.ndarray]:
    perk_ids = [label for label in labels if label != "__background__"]
    paths = addon_labels_and_paths()[1] if target == "addons" else icon_paths_for(perk_ids, ICONS_DIR)
    composer = compose_addon if target == "addons" else compose
    dataset = PerkSynthDataset(paths, 2, TEST_SEVERITY, TEST_SEED, negative_fraction=0.25, composer=composer)
    step = max(len(dataset) // PARITY_SAMPLES, 1)
    return [dataset.render(index)[0] for index in range(0, len(dataset), step)]


def check_parity(onnx_path: Path, model: torch.nn.Module, coreml_path: Path | None, labels: list[str],
                 images: list[np.ndarray]) -> tuple[float, float, float | None]:
    """(top-1 agreement with PyTorch, max |p_onnx - p_torch|, top-1 agreement with Core ML or None)."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    coreml = None
    if coreml_path is not None and coreml_path.exists():
        import coremltools as ct
        coreml = ct.models.MLModel(str(coreml_path))
    torch_model = WithSoftmax(model).eval()
    agree_torch = agree_coreml = 0
    worst = 0.0
    for image in images:
        batch = (image.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        onnx_probs = session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0][0]
        with torch.no_grad():
            torch_probs = torch_model(torch.from_numpy(batch)).numpy()[0]
        worst = max(worst, float(np.abs(onnx_probs - torch_probs).max()))
        agree_torch += int(onnx_probs.argmax() == torch_probs.argmax())
        if coreml is not None:
            agree_coreml += labels[int(onnx_probs.argmax())] == coreml.predict({INPUT_NAME: Image.fromarray(image)})["classLabel"]
    coreml_rate = agree_coreml / len(images) if coreml is not None else None
    return agree_torch / len(images), worst, coreml_rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("perks", "addons"), default="perks")
    target = parser.parse_args().target
    model_dir, onnx_name, coreml_name = (
        (ADDON_OUT_DIR, ADDON_ONNX_NAME, ADDON_COREML_NAME) if target == "addons" else (MODEL_DIR, ONNX_NAME, COREML_NAME))
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    model = load_model(labels, model_dir)
    onnx_path = model_dir / onnx_name
    export(model, onnx_path)

    torch_rate, worst, coreml_rate = check_parity(onnx_path, model, model_dir / coreml_name, labels, parity_samples(labels, target))
    coreml_text = "n/a (no Core ML package)" if coreml_rate is None else f"{coreml_rate:.3f}"
    print(f"✅ saved {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"   top-1 vs PyTorch = {torch_rate:.3f}, max prob diff vs PyTorch = {worst:.2e}, top-1 vs Core ML = {coreml_text}")
    failed = torch_rate < MIN_TOP1_AGREEMENT or worst > MAX_PROBABILITY_DIFFERENCE
    if coreml_rate is not None and coreml_rate < MIN_TOP1_AGREEMENT:
        failed = True
    if failed:
        print("❌ parity below the bar; do not ship this export")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
