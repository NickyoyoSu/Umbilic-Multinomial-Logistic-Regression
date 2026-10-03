#!/usr/bin/env python3
"""Sample head-input features of a trained run for visualization.

Rebuilds the model of a ``scripts/train.py`` run from its ``config.json`` and
checkpoint, takes up to ``--max-samples-per-class`` images per class from the
evaluation split (in dataset order), and saves the features that enter the
head (after the run's centering and normalization) to
``<run-dir>/analysis/sampled_features.npz``. These files are the inputs of
``scripts/figures/plot_tsne_triptych.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from umlr.backbones import build_backbone
from umlr.classifier import BoundaryClassifier
from umlr.data import _labels_for_dataset, build_datasets
from umlr.factory import build_head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--data-root", help="Override the data root stored in the run config.")
    parser.add_argument("--max-samples-per-class", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def first_per_class(dataset, max_samples_per_class: int) -> Subset:
    counts: dict[int, int] = {}
    indices: list[int] = []
    for index, label in enumerate(_labels_for_dataset(dataset)):
        if counts.get(label, 0) < max_samples_per_class:
            counts[label] = counts.get(label, 0) + 1
            indices.append(index)
    return Subset(dataset, indices)


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    bundle = build_backbone(config["backbone"], pretrained=bool(config.get("pretrained", True)))
    _, eval_ds, num_classes = build_datasets(
        dataset_name=config["dataset"],
        data_root=args.data_root or config["data_root"],
        image_size=config.get("input_size") or bundle.image_size,
        download=False,
        seed=config.get("seed", 0),
    )
    loader = DataLoader(
        first_per_class(eval_ds, args.max_samples_per_class),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = BoundaryClassifier(
        bundle.model,
        build_head(config, bundle.feature_dim, num_classes),
        normalize_features=bool(config.get("feature_normalization", False)),
        center_features=config.get("feature_centering", "none") != "none",
    )
    state_dict = torch.load(args.run_dir / args.checkpoint, map_location="cpu")
    if state_dict.get("feature_center") is not None and state_dict["feature_center"].numel() > 0:
        model.feature_center = torch.empty_like(state_dict["feature_center"])
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device).eval()

    features, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            features.append(model.forward_features(images.to(args.device)).cpu().numpy())
            labels.append(batch_labels.numpy())
    output_dir = args.run_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "sampled_features.npz"
    np.savez_compressed(output, features=np.concatenate(features), labels=np.concatenate(labels))
    print(output)


if __name__ == "__main__":
    main()
