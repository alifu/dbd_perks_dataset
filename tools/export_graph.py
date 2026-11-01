"""Export the knowledge graph and ML analytics that the DBD-Perks-Web graph map shows.

python3 tools/export_graph.py                 # writes data/graph/graph.json and data/graph/analytics.json
python3 tools/export_graph.py --samples 4     # fewer synthetic probes per class (faster)

Text only, so both files may be committed. Nothing here contains an image or a model weight.

Graph: characters, items, perks and add-ons joined by ownership edges, plus two ML edge kinds:
  similar   cosine similarity of class-mean backbone embeddings (what the network sees as look-alike)
  confused  a probe of class A that the network predicted as class B (count = how often)
Each perk and add-on node carries top-1 accuracy and mean probability of the right class on synthetic
test probes (TEST_SEVERITY, the same hard setting as the paper's hard test).
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from addon_data import addon_labels_and_paths
from export_coreml_v2 import load_model
from synth.addon_tile import compose_addon
from synth.compose import compose
from synth.dataset import TEST_SEED, PerkSynthDataset, icon_paths_for
from train_perks_v2 import ADDON_OUT_DIR, ICONS_DIR, OUT_DIR, TEST_SEVERITY, make_loader, pick_device

PERKS_JSON = Path("data/perks.json")
ADDONS_JSON = Path("data/addons.json")
OUT = Path("data/graph")
ABLATION_GLOB = "results/ablation_*.json"
SCHEMA_VERSION = 1
NEIGHBOURS = 2          # similar-edges kept per class
MIN_SIMILARITY = 0.30   # centred cosine below this is not a look-alike
BATCH_SIZE = 64
GENERIC_OWNER = {"survivor": "Any survivor", "killer": "Any killer"}


def probe(model, paths, composer, samples: int, device, workers: int = 4):
    """Run synthetic probes through the network. Returns per-class (centroid, accuracy, mean p, confusions)."""
    count = len(paths)
    dataset = PerkSynthDataset(paths, samples, TEST_SEVERITY, TEST_SEED, 0.0, composer)
    features, probs, targets = [], [], []
    hook = model.backbone.features.register_forward_hook(
        lambda _m, _i, out: features.append(F.adaptive_avg_pool2d(out, 1).flatten(1).cpu()))
    try:
        with torch.no_grad():
            for images, labels in make_loader(dataset, False, BATCH_SIZE, workers):
                probs.append(torch.softmax(model(images.to(device)), dim=1).cpu())
                targets.append(labels)
    finally:
        hook.remove()
    features, probs, targets = torch.cat(features), torch.cat(probs), torch.cat(targets)
    predicted = probs.argmax(dim=1)
    centroids = torch.zeros(count, features.shape[1]).index_add_(0, targets, features)
    centroids = centroids / samples
    centroids = F.normalize(centroids - centroids.mean(dim=0, keepdim=True), dim=1)  # centred: raw features share a large common part
    hits = (predicted == targets).float()
    accuracy = torch.zeros(count).index_add_(0, targets, hits) / samples
    p_correct = torch.zeros(count).index_add_(0, targets, probs[torch.arange(len(targets)), targets]) / samples
    confusions = Counter((int(t), int(p)) for t, p in zip(targets, predicted) if t != p and p < count)
    return centroids, accuracy.tolist(), p_correct.tolist(), confusions


def similar_edges(ids: list[str], centroids: torch.Tensor) -> list[dict]:
    similarity = centroids @ centroids.T
    similarity.fill_diagonal_(-1)
    seen, edges = set(), []
    for row, scores in enumerate(similarity):
        values, columns = scores.topk(NEIGHBOURS)
        for value, column in zip(values.tolist(), columns.tolist()):
            key = (min(row, column), max(row, column))
            if value >= MIN_SIMILARITY and key not in seen:
                seen.add(key)
                edges.append({"source": ids[key[0]], "target": ids[key[1]], "type": "similar", "weight": round(value, 4)})
    return edges


def confused_edges(ids: list[str], confusions: Counter) -> list[dict]:
    return [{"source": ids[a], "target": ids[b], "type": "confused", "weight": count}
            for (a, b), count in confusions.most_common()]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def character_nodes(perks: list[dict], addons: list[dict]) -> dict[str, dict]:
    nodes: dict[str, dict] = {}
    for perk in perks:
        owner = perk.get("owner") or GENERIC_OWNER[perk["role"]]
        nodes.setdefault(f"char:{owner}", {"id": f"char:{owner}", "type": "character", "label": owner, "role": perk["role"]})
    for addon in addons:
        owner = addon.get("ownerName")
        if owner:
            nodes.setdefault(f"char:{owner}", {"id": f"char:{owner}", "type": "character", "label": owner, "role": addon["role"]})
    return nodes


def build_graph(args, device) -> tuple[dict, dict]:
    perks = load_json(PERKS_JSON)["perks"]
    addons = load_json(ADDONS_JSON)["addons"]
    perk_labels = [label for label in load_json(OUT_DIR / "labels.json") if label != "__background__"]
    addon_labels, addon_paths, _missing = addon_labels_and_paths()

    perk_model = load_model(perk_labels + ["__background__"], OUT_DIR).to(device)
    perk_stats = probe(perk_model, icon_paths_for(perk_labels, ICONS_DIR), compose, args.samples, device)
    addon_model = load_model(addon_labels + ["__background__"], ADDON_OUT_DIR).to(device)
    addon_stats = probe(addon_model, addon_paths, compose_addon, args.addon_samples, device)

    nodes = character_nodes(perks, addons)
    links: list[dict] = []
    perk_index = {label: i for i, label in enumerate(perk_labels)}
    for perk in perks:
        owner = perk.get("owner") or GENERIC_OWNER[perk["role"]]
        i = perk_index.get(perk["id"])
        trained = i is not None
        nodes[perk["id"]] = {
            "id": perk["id"], "type": "perk", "label": perk["name"], "role": perk["role"], "owner": owner,
            "trained": trained, "iconUrl": perk.get("iconUrl"), "description": perk.get("description"),
            "accuracy": round(perk_stats[1][i], 4) if trained else None,
            "pCorrect": round(perk_stats[2][i], 4) if trained else None,
        }
        links.append({"source": f"char:{owner}", "target": perk["id"], "type": "owns", "weight": 1})

    addon_index = {label: i for i, label in enumerate(addon_labels)}
    for addon in addons:
        i = addon_index.get(addon["iconStem"].lower())
        node_id = f"addon:{addon['id']}"
        nodes[node_id] = {
            "id": node_id, "type": "addon", "label": addon["name"], "role": addon["role"], "rarity": addon.get("rarity"),
            "owner": addon.get("ownerName"), "trained": i is not None, "description": addon.get("description"),
            "accuracy": round(addon_stats[1][i], 4) if i is not None else None,
            "pCorrect": round(addon_stats[2][i], 4) if i is not None else None,
        }
        item_id = f"item:{addon['parent']}"
        if item_id not in nodes:
            nodes[item_id] = {"id": item_id, "type": "item", "label": addon["parent"].replace("Item_", "").replace("_", " "),
                              "role": addon["role"], "owner": addon.get("ownerName")}
            if addon.get("ownerName"):
                links.append({"source": f"char:{addon['ownerName']}", "target": item_id, "type": "owns", "weight": 1})
        links.append({"source": item_id, "target": node_id, "type": "owns", "weight": 1})

    perk_ids = perk_labels
    links += similar_edges(perk_ids, perk_stats[0]) + confused_edges(perk_ids, perk_stats[3])
    addon_by_stem = {}
    for addon in addons:
        addon_by_stem.setdefault(addon["iconStem"].lower(), f"addon:{addon['id']}")
    addon_ids = [addon_by_stem[label] for label in addon_labels]
    links += similar_edges(addon_ids, addon_stats[0]) + confused_edges(addon_ids, addon_stats[3])

    graph = {"schemaVersion": SCHEMA_VERSION, "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "nodes": list(nodes.values()), "links": links}
    return graph, analytics_from(graph, perks, addons, perk_stats, addon_stats, perk_labels, addon_labels)


def histogram(values: list[float], bins: int = 10) -> list[int]:
    counts = [0] * bins
    for value in values:
        counts[min(int(value * bins), bins - 1)] += 1
    return counts


def ablation_rows() -> list[dict]:
    rows = []
    for path in sorted(Path().glob(ABLATION_GLOB), key=lambda p: int(p.stem.split("_")[1])):
        summary = load_json(path)["summary"]
        rows.append({"framedPercent": int(path.stem.split("_")[1]), **{k: summary[k] for k in ("crops", "top1", "top3", "meanPCorrect")},
                     "top1Wilson95": summary["top1Wilson95"]})
    return rows


def analytics_from(graph, perks, addons, perk_stats, addon_stats, perk_labels, addon_labels) -> dict:
    by_type = Counter(node["type"] for node in graph["nodes"])
    by_edge = Counter(link["type"] for link in graph["links"])
    metrics = {name: load_json(folder / "metrics.json") for name, folder in (("perks", OUT_DIR), ("addons", ADDON_OUT_DIR))}
    hardest = sorted(((perk_stats[2][i], perk_labels[i]) for i in range(len(perk_labels))))[:15]
    names = {node["id"]: node["label"] for node in graph["nodes"]}
    ranked_similar = sorted((link for link in graph["links"] if link["type"] == "similar"), key=lambda l: -l["weight"])[:20]
    return {
        "schemaVersion": SCHEMA_VERSION, "generatedAt": graph["generatedAt"],
        "counts": {"nodes": dict(by_type), "edges": dict(by_edge), "perkClasses": len(perk_labels), "addonClasses": len(addon_labels)},
        "composition": {
            "perksByRole": dict(Counter(p["role"] for p in perks)),
            "addonsByRarity": dict(Counter(a.get("rarity") or "unknown" for a in addons)),
            "addonsByRole": dict(Counter(a["role"] for a in addons)),
        },
        "models": {name: {"classes": m["classes"], "validation": {k: m["validation"][k] for k in ("top1", "top3", "samples")},
                          "hardTest": {k: m["hardTest"][k] for k in ("top1", "top3", "samples")}} for name, m in metrics.items()},
        "probe": {
            "perks": {"accuracyHistogram": histogram(perk_stats[1]), "pCorrectHistogram": histogram(perk_stats[2]),
                      "hardest": [{"id": i, "label": names.get(i, i), "pCorrect": round(p, 4)} for p, i in hardest]},
            "addons": {"accuracyHistogram": histogram(addon_stats[1]), "pCorrectHistogram": histogram(addon_stats[2])},
        },
        "lookAlikes": [{"a": names[l["source"]], "b": names[l["target"]], "similarity": l["weight"]} for l in ranked_similar],
        "ablation": ablation_rows(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=8, help="synthetic probes per perk class")
    parser.add_argument("--addon-samples", type=int, default=4, help="synthetic probes per add-on class")
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph, analytics = build_graph(args, pick_device())
    args.out.mkdir(parents=True, exist_ok=True)
    for name, payload in (("graph.json", graph), ("analytics.json", analytics)):
        (args.out / name).write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"graph: {len(graph['nodes'])} nodes, {len(graph['links'])} links -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
