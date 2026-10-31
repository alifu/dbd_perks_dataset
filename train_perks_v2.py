"""Train the v2 perk classifier on synthetic 'photo of a perk' images, on the Apple GPU (MPS) when available.

python3 train_perks_v2.py            # full run
python3 train_perks_v2.py --smoke    # tiny end-to-end check
python3 train_perks_v2.py --target addons   # the add-on classifier (models/addons_v1)
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from addon_data import addon_labels_and_paths
from synth.addon_tile import compose_addon
from synth.compose import compose
from synth.dataset import BACKGROUND_LABEL, TEST_SEED, VAL_SEED, PerkSynthDataset, icon_paths_for

# ---- CONFIG ----
PERKS_JSON = Path("data/perks.json")
ICONS_DIR = Path("data/icons")
OUT_DIR = Path("models/v2")
ADDON_OUT_DIR = Path("models/addons_v1")
BATCH_SIZE = 64
SAMPLES_PER_CLASS = 32
NEGATIVE_FRACTION = 0.25  # background windows per perk window; teaches the model to say "not a perk"
EVAL_SAMPLES_PER_CLASS = 8
HEAD_EPOCHS = 2
FINETUNE_EPOCHS = 12
HEAD_LR = 1e-3
FINETUNE_LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
DROPOUT = 0.2
TRAIN_SEVERITY = (0.2, 1.0)
VAL_SEVERITY = (0.2, 0.8)
TEST_SEVERITY = (0.7, 1.0)  # harder than anything seen in validation
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WORKERS = 6
REPORT_ITEMS = 15


class PerkClassifier(nn.Module):
    """MobileNetV2 that takes RGB in 0..1 and normalizes inside, so Core ML only needs scale=1/255."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        self.backbone.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(1280, num_classes))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)

    def head_parameters(self):
        return self.backbone.classifier.parameters()


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def recognizable_perk_ids(perks: list[dict]) -> list[str]:
    """Sorted ids of perks with their own icon. Perks sharing one icon file (e.g. the wiki's
    'Unknown_QuestionMark' placeholder for brand-new perks) look identical, so they cannot be learned."""
    uses = Counter(perk["iconFile"] for perk in perks)
    return sorted(perk["id"] for perk in perks if uses[perk["iconFile"]] == 1)


def load_labels() -> list[str]:
    perks = json.loads(PERKS_JSON.read_text(encoding="utf-8"))["perks"]
    return recognizable_perk_ids(perks)


def make_loader(dataset, shuffle: bool, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, drop_last=shuffle)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int) -> dict:
    model.eval()
    correct = top3 = total = 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    confusions: dict[tuple[int, int], int] = {}
    for images, targets in loader:
        ranked = model(images.to(device)).cpu().topk(3, dim=1).indices
        for target, row in zip(targets.tolist(), ranked.tolist()):
            total += 1
            per_class_total[target] += 1
            top3 += target in row
            if row[0] == target:
                correct += 1
                per_class_correct[target] += 1
            else:
                confusions[(target, row[0])] = confusions.get((target, row[0]), 0) + 1
    return {"top1": correct / total, "top3": top3 / total, "samples": total,
            "per_class_correct": per_class_correct, "per_class_total": per_class_total, "confusions": confusions}


def run_epoch(model, loader, device, optimizer, loss_fn, scheduler=None) -> float:
    model.train()
    running, batches = 0.0, 0
    for images, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images.to(device)), targets.to(device))
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        running, batches = running + loss.item(), batches + 1
    return running / max(batches, 1)


def set_backbone_trainable(model: PerkClassifier, trainable: bool) -> None:
    for parameter in model.backbone.features.parameters():
        parameter.requires_grad = trainable


def summarize(metrics: dict, labels: list[str]) -> dict:
    accuracy = lambda i: metrics["per_class_correct"][i] / max(metrics["per_class_total"][i], 1)
    worst = sorted(range(len(labels)), key=accuracy)[:REPORT_ITEMS]
    pairs = sorted(metrics["confusions"].items(), key=lambda item: -item[1])[:REPORT_ITEMS]
    return {
        "top1": round(metrics["top1"], 4), "top3": round(metrics["top3"], 4), "samples": metrics["samples"],
        "worstClasses": [{"id": labels[i], "accuracy": round(accuracy(i), 3)} for i in worst],
        "topConfusions": [{"expected": labels[a], "predicted": labels[b], "count": n} for (a, b), n in pairs],
    }


def train_stage(name, epochs, num_classes, model, train_loader, val_loader, device, optimizer, loss_fn, scheduler=None) -> dict:
    val = {}
    for epoch in range(epochs):
        start = time.time()
        loss = run_epoch(model, train_loader, device, optimizer, loss_fn, scheduler)
        val = evaluate(model, val_loader, device, num_classes)
        print(f"[{name} {epoch + 1}/{epochs}] loss={loss:.3f} val_top1={val['top1']:.3f} ({time.time() - start:.0f}s)", flush=True)
    return val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    parser.add_argument("--target", choices=("perks", "addons"), default="perks")
    parser.add_argument("--out", type=Path, help="default: models/v2 for perks, models/addons_v1 for add-ons")
    parser.add_argument("--workers", type=int, default=WORKERS)
    return parser.parse_args()


def load_target(target: str) -> tuple[list[str], list[Path], object, Path]:
    """(class ids, one icon file per id, composer, default output folder)."""
    if target == "addons":
        ids, paths, missing = addon_labels_and_paths()
        if missing:
            print(f"⚠️  {len(missing)} add-on icons missing from the game export (skipped): {missing[:5]}")
        return ids, paths, compose_addon, ADDON_OUT_DIR
    ids = load_labels()
    return ids, icon_paths_for(ids, ICONS_DIR), compose, OUT_DIR


def main() -> int:
    args = parse_args()
    perk_ids, paths, composer, default_out = load_target(args.target)
    args.out = args.out or default_out
    labels = perk_ids + [BACKGROUND_LABEL]
    device = pick_device()
    workers = 0 if args.smoke else args.workers
    samples = 2 if args.smoke else SAMPLES_PER_CLASS
    eval_samples = 1 if args.smoke else EVAL_SAMPLES_PER_CLASS
    head_epochs, tune_epochs = (1, 1) if args.smoke else (HEAD_EPOCHS, FINETUNE_EPOCHS)
    print(f"🖥️  device={device} classes={len(labels)} (incl. background) perk windows/epoch={len(perk_ids) * samples}", flush=True)

    train_loader = make_loader(PerkSynthDataset(paths, samples, TRAIN_SEVERITY, negative_fraction=NEGATIVE_FRACTION, composer=composer), True, BATCH_SIZE, workers)
    val_loader = make_loader(PerkSynthDataset(paths, eval_samples, VAL_SEVERITY, VAL_SEED, NEGATIVE_FRACTION, composer), False, BATCH_SIZE, workers)
    test_loader = make_loader(PerkSynthDataset(paths, eval_samples, TEST_SEVERITY, TEST_SEED, NEGATIVE_FRACTION, composer), False, BATCH_SIZE, workers)

    model = PerkClassifier(len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    set_backbone_trainable(model, False)
    head_optimizer = torch.optim.AdamW(model.head_parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    train_stage("head", head_epochs, len(labels), model, train_loader, val_loader, device, head_optimizer, loss_fn)

    set_backbone_trainable(model, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FINETUNE_LR,
                                                    total_steps=tune_epochs * len(train_loader), pct_start=0.15)
    val = train_stage("tune", tune_epochs, len(labels), model, train_loader, val_loader, device, optimizer, loss_fn, scheduler)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "perk_classifier.pt")
    (args.out / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    test = evaluate(model, test_loader, device, len(labels))
    report = {"device": str(device), "classes": len(labels),
              "validation": summarize(val, labels), "hardTest": summarize(test, labels)}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"✅ val top1={val['top1']:.3f}  hard-test top1={test['top1']:.3f} top3={test['top3']:.3f}  → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
