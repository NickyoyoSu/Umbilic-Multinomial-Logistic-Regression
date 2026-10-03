#!/usr/bin/env python3
"""t-SNE triptych of sampled CIFAR-100, Flowers102, and TinyImageNet features (paper Figure 2).

Inputs are sampled_features.npz files written by scripts/figures/sample_features.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

try:
    import threadpoolctl

    _orig_get_version = threadpoolctl._OpenBLASModule.get_version
    _orig_get_threading_layer = threadpoolctl._OpenBLASModule.get_threading_layer

    def _safe_openblas_get_version(self):  # type: ignore[override]
        try:
            return _orig_get_version(self)
        except Exception:
            return "unknown"

    def _safe_openblas_get_threading_layer(self):  # type: ignore[override]
        try:
            return _orig_get_threading_layer(self)
        except Exception:
            return "unknown"

    threadpoolctl._OpenBLASModule.get_version = _safe_openblas_get_version  # type: ignore[assignment]
    threadpoolctl._OpenBLASModule.get_threading_layer = _safe_openblas_get_threading_layer  # type: ignore[assignment]
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dataset-only t-SNE triptych from cached sampled features.")
    parser.add_argument("--cifar", type=Path, required=True, help="Path to CIFAR sampled_features.npz")
    parser.add_argument("--flowers", type=Path, required=True, help="Path to Flowers sampled_features.npz")
    parser.add_argument("--tiny", type=Path, required=True, help="Path to TinyImageNet sampled_features.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=2000, help="Maximum points per dataset after sampling.")
    parser.add_argument("--max-classes", type=int, default=0, help="If > 0, randomly keep only this many classes per dataset.")
    parser.add_argument("--perplexity", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["features"].astype(np.float32), data["labels"].astype(np.int64)


def balanced_subsample(
    features: np.ndarray,
    labels: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(labels) <= max_points:
        return features, labels

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    per_class = max(1, max_points // len(unique_labels))
    chosen: list[np.ndarray] = []

    for label in unique_labels:
        idx = np.flatnonzero(labels == label)
        take = min(len(idx), per_class)
        chosen.append(rng.choice(idx, size=take, replace=False))

    chosen_idx = np.concatenate(chosen)
    if len(chosen_idx) > max_points:
        chosen_idx = rng.choice(chosen_idx, size=max_points, replace=False)
    chosen_idx = np.sort(chosen_idx)
    return features[chosen_idx], labels[chosen_idx]


def subsample_classes(
    features: np.ndarray,
    labels: np.ndarray,
    max_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_labels = np.unique(labels)
    if max_classes <= 0 or len(unique_labels) <= max_classes:
        return features, labels, unique_labels

    rng = np.random.default_rng(seed)
    keep_labels = np.sort(rng.choice(unique_labels, size=max_classes, replace=False))
    mask = np.isin(labels, keep_labels)
    return features[mask], labels[mask], keep_labels


def compute_tsne(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=min(perplexity, max(5.0, (len(features) - 1) / 3.0)),
        random_state=seed,
    )
    return tsne.fit_transform(features)


def write_panel(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    unique_labels = np.unique(labels)
    palette = plt.get_cmap("tab10").colors
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    point_colors = [palette[label_to_idx[label] % len(palette)] for label in labels]
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=point_colors,
        s=12,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Keep the artist alive for consistent rendering.
    _ = scatter


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("CIFAR-100", args.cifar),
        ("Flowers102", args.flowers),
        ("TinyImageNet", args.tiny),
    ]

    results: dict[str, dict[str, int | str]] = {}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)

    for ax, (name, path) in zip(axes, datasets):
        features, labels = load_npz(path)
        features, labels, kept_classes = subsample_classes(features, labels, args.max_classes, args.seed)
        features, labels = balanced_subsample(features, labels, args.max_points, args.seed)
        coords = compute_tsne(features, args.perplexity, args.seed)
        write_panel(ax, coords, labels, name)
        np.savez_compressed(
            args.output_dir / f"{name.lower().replace('-', '').replace(' ', '_')}_tsne.npz",
            coords=coords,
            labels=labels,
        )
        results[name] = {
            "source": str(path),
            "num_points": int(len(labels)),
            "num_classes": int(len(np.unique(labels))),
            "selected_classes": [int(x) for x in kept_classes.tolist()],
        }

    fig.suptitle("ViT-B/16 Dataset Feature Distribution Visualization", fontsize=15, y=1.06)
    fig.supxlabel("Dimension 1", fontsize=11)
    fig.supylabel("Dimension 2", fontsize=11)

    # Add clean separators between the three panels.
    fig.canvas.draw()
    positions = [ax.get_position() for ax in axes]
    for left, right in zip(positions[:-1], positions[1:]):
        x = (left.x1 + right.x0) / 2.0
        y0 = min(left.y0, right.y0)
        y1 = max(left.y1, right.y1)
        fig.add_artist(
            plt.Line2D(
                [x, x],
                [y0, y1],
                transform=fig.transFigure,
                color="#555555",
                linewidth=0.8,
                alpha=0.8,
            )
        )

    out_path = args.output_dir / "dataset_tsne_triptych.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    with (args.output_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "backbone": "vit_b_16",
                "perplexity": args.perplexity,
                "seed": args.seed,
                "max_points": args.max_points,
                "max_classes": args.max_classes,
                "datasets": results,
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
