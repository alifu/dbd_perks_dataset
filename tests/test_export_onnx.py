"""The ONNX export must match PyTorch on the same input (random weights, no game assets needed)."""
import numpy as np
import onnxruntime as ort
import pytest
import torch

from export_coreml_v2 import IMG_SIZE, INPUT_NAME, WithSoftmax
from export_onnx_v2 import MAX_PROBABILITY_DIFFERENCE, OUTPUT_NAME, export
from train_perks_v2 import PerkClassifier

CLASSES = 7
BATCH = 3


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = PerkClassifier(CLASSES, pretrained=False).eval()
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export(model, path)
    return model, ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_onnx_matches_pytorch_probabilities(exported):
    model, session = exported
    batch = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        expected = WithSoftmax(model).eval()(batch).numpy()

    actual = session.run([OUTPUT_NAME], {INPUT_NAME: batch.numpy()})[0]

    assert actual.shape == (BATCH, CLASSES)
    assert np.abs(actual - expected).max() < MAX_PROBABILITY_DIFFERENCE


def test_output_is_a_probability_distribution_for_any_batch_size(exported):
    _, session = exported
    for size in (1, 4):
        probs = session.run([OUTPUT_NAME], {INPUT_NAME: np.random.rand(size, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)})[0]
        assert probs.shape[0] == size
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
