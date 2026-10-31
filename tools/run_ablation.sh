#!/bin/sh
# Framed-share ablation for the paper: same code, epochs and data, only the share of framed samples changes.
# Output: models/ablation/framed_<percent>/ (weights git-ignored) and results/ablation_<percent>.json
set -e
cd "$(dirname "$0")/.."
mkdir -p results
for share in 0.0 0.25 0.5 0.75 1.0; do
  percent=$(python3 -c "print(int(round($share * 100)))")
  out="models/ablation/framed_$percent"
  [ -f "$out/metrics.json" ] || .venv/bin/python train_perks_v2.py --framed-share "$share" --out "$out"
  .venv/bin/python evaluate_real.py --weights "$out/perk_classifier.pt" --json "results/ablation_$percent.json"
done
