#!/usr/bin/env python3
"""Learned H/H^2/S/S^2 heads on full-way few-shot episodes (Experiment 3).

For every episode, each UMLR endpoint is adapted on the support set of frozen
CL2N features and evaluated on the query set; ProtoNet (raw features) and
SimpleShot-CL2N are evaluated on the same episode draws. Use
--split validation for endpoint selection and --split test for the
locked evaluation. Writes episodes.csv, summary.csv, and
config.json to <output-root>/<run-name>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from umlr.backbones import BACKBONES, build_backbone
from umlr.episodic import (
    DATASETS,
    FeatureSplit,
    build_dataset_split,
    build_eval_transform,
    class_index,
    estimate_radius_prior,
    extract_features,
    parse_way,
    subset_by_class,
    support_query_episode,
    transform_features,
)
from umlr.heads import HyperplaneHead
from umlr.umlr_head import SphereUMLRHead


ADAPTED_METHODS = {"adapt_h", "adapt_h2", "adapt_s", "adapt_s2"}
ALL_METHODS = {
    "protonet",
    "simpleshot_cl2n",
    "s2_r0",
    *ADAPTED_METHODS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learned H/H2/S/S2 heads on few-shot episodes.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--backbone", default="vit_b_16", choices=BACKBONES)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--way", default="5")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query-per-class", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "protonet",
            "simpleshot_cl2n",
            "s2_r0",
            "adapt_h",
            "adapt_h2",
            "adapt_s",
            "adapt_s2",
        ],
    )
    parser.add_argument("--feature-mode", default="cl2n", choices=["un", "l2n", "cl2n"])
    parser.add_argument("--feature-cache")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--radius-quantile", type=float, default=0.9)
    parser.add_argument("--radius-prior-mode", default="median", choices=["median", "mean"])
    parser.add_argument("--max-radius-images-per-class", type=int, default=100)
    parser.add_argument("--max-eval-images-per-class", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--download-pancake-archives", action="store_true")
    parser.add_argument("--adapt-steps", type=int, default=100)
    parser.add_argument("--adapt-lr", type=float, default=1e-2)
    parser.add_argument("--adapt-weight-decay", type=float, default=1e-4)
    parser.add_argument("--radius-prior-weight", type=float, default=1e-3)
    parser.add_argument("--alpha-init", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--output-root", default="outputs/episodic")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    invalid = sorted(set(args.methods) - ALL_METHODS)
    if invalid:
        parser.error(f"Unknown methods: {invalid}; choices are {sorted(ALL_METHODS)}")
    if args.shot <= 0 or args.episodes <= 0:
        parser.error("--shot and --episodes must be positive")
    if args.adapt_steps <= 0 or args.adapt_lr <= 0:
        parser.error("--adapt-steps and --adapt-lr must be positive")
    return args


def _feature_split_from_payload(payload: dict[str, object], prefix: str) -> FeatureSplit:
    features = payload[f"{prefix}_features"]
    labels = payload[f"{prefix}_labels"]
    if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError(f"Malformed feature cache for {prefix}")
    return FeatureSplit(
        features=features.float().cpu(),
        labels=labels.long().cpu(),
        class_ids=sorted(int(x) for x in labels.unique().tolist()),
    )


def load_or_extract_features(args: argparse.Namespace) -> tuple[FeatureSplit, FeatureSplit]:
    cache_path = Path(args.feature_cache).expanduser().resolve() if args.feature_cache else None
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        metadata = payload.get("metadata", {})
        expected = {
            "dataset": args.dataset,
            "backbone": args.backbone,
            "split": args.split,
            "pretrained": bool(args.pretrained),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Feature cache mismatch for {key}: cache={metadata.get(key)!r}, requested={value!r}"
                )
        return _feature_split_from_payload(payload, "train"), _feature_split_from_payload(payload, "eval")

    device = torch.device(args.device)
    bundle = build_backbone(args.backbone, pretrained=args.pretrained)
    model = bundle.model.to(device)
    transform = build_eval_transform(args.input_size or bundle.image_size, bundle.mean, bundle.std)
    train_ds = build_dataset_split(args, "train", transform)
    eval_ds = build_dataset_split(args, args.split, transform)
    train_labels = [int(x) for x in getattr(train_ds, "labels")]
    eval_labels = [int(x) for x in getattr(eval_ds, "labels")]
    train_ds = subset_by_class(train_ds, train_labels, args.max_radius_images_per_class, args.seed)
    eval_ds = subset_by_class(eval_ds, eval_labels, args.max_eval_images_per_class, args.seed + 17)
    train_split = extract_features(model, train_ds, args, device)
    eval_split = extract_features(model, eval_ds, args, device)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_features": train_split.features,
                "train_labels": train_split.labels,
                "eval_features": eval_split.features,
                "eval_labels": eval_split.labels,
                "metadata": {
                    "dataset": args.dataset,
                    "backbone": args.backbone,
                    "split": args.split,
                    "pretrained": bool(args.pretrained),
                    "input_size": int(args.input_size or bundle.image_size),
                    "max_radius_images_per_class": int(args.max_radius_images_per_class),
                    "max_eval_images_per_class": int(args.max_eval_images_per_class),
                },
            },
            cache_path,
        )
        print(f"Wrote feature cache {cache_path}", flush=True)
    return train_split, eval_split


def build_episode_tensors(
    features: torch.Tensor,
    support_indices: np.ndarray,
    query_indices: np.ndarray,
    support_labels: np.ndarray,
    query_labels: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    support = features[torch.as_tensor(support_indices, dtype=torch.long)].to(device)
    query = features[torch.as_tensor(query_indices, dtype=torch.long)].to(device)
    support_y = torch.as_tensor(support_labels, dtype=torch.long, device=device)
    query_y = torch.as_tensor(query_labels, dtype=torch.long, device=device)
    return support, query, support_y, query_y


def class_centers(support: torch.Tensor, labels: torch.Tensor, way: int) -> torch.Tensor:
    return torch.stack([support[labels == class_id].mean(dim=0) for class_id in range(way)], dim=0)


def make_adapted_head(
    method: str,
    centers: torch.Tensor,
    radius_prior: float,
    args: argparse.Namespace,
) -> torch.nn.Module:
    way, feature_dim = centers.shape
    if method in {"adapt_h", "adapt_h2"}:
        head = HyperplaneHead(
            feature_dim,
            way,
            score_mode="signed_square" if method == "adapt_h2" else "linear",
            alpha_scope="classwise",
            alpha_init_effective=args.alpha_init,
        ).to(centers.device)
        with torch.no_grad():
            head.linear.weight.copy_(2.0 * centers)
            if head.linear.bias is not None:
                head.linear.bias.copy_(-centers.square().sum(dim=1))
        return head

    if method not in {"adapt_s", "adapt_s2"}:
        raise ValueError(f"Not an adapted method: {method}")
    head = SphereUMLRHead(
        feature_dim,
        way,
        parameterization="geometric",
        alpha_mode="softplus",
        boundary_mode="signed_square" if method == "adapt_s2" else "distance",
        alpha_scope="classwise",
        alpha_init_effective=args.alpha_init,
        radius_scope="classwise",
        radius_init_effective=radius_prior,
    ).to(centers.device)
    head.set_centers_(centers)
    head.set_radius_values_(centers.new_full((way,), float(radius_prior)))
    return head


def fit_and_score(
    method: str,
    support: torch.Tensor,
    query: torch.Tensor,
    support_y: torch.Tensor,
    radius_prior: float,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    way = int(support_y.max().item()) + 1
    centers = class_centers(support, support_y, way)
    head = make_adapted_head(method, centers, radius_prior, args)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.adapt_lr,
        weight_decay=args.adapt_weight_decay,
    )

    head.train()
    with torch.no_grad():
        initial_loss = float(F.cross_entropy(head(support), support_y).item())
    final_loss = initial_loss
    for _ in range(args.adapt_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = head(support)
        loss = F.cross_entropy(logits, support_y)
        if method in {"adapt_s", "adapt_s2"} and args.radius_prior_weight > 0:
            radius = head.radius_values()
            assert radius is not None
            loss = loss + args.radius_prior_weight * (radius - float(radius_prior)).square().mean()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        optimizer.step()
        final_loss = float(loss.detach().item())

    head.eval()
    with torch.no_grad():
        query_logits = head(query)
    diagnostics: dict[str, float] = {
        "support_loss_initial": initial_loss,
        "support_loss_final": final_loss,
        "radius_mean": float("nan"),
        "radius_std": float("nan"),
        "alpha_mean": float("nan"),
        "alpha_std": float("nan"),
    }
    radius_values = getattr(head, "radius_values", None)
    if callable(radius_values):
        radius = radius_values()
        if radius is not None:
            diagnostics["radius_mean"] = float(radius.mean().item())
            diagnostics["radius_std"] = float(radius.std(unbiased=False).item())
    alpha_values = getattr(head, "alpha_values", None)
    if callable(alpha_values):
        alpha = alpha_values()
        if alpha is not None:
            diagnostics["alpha_mean"] = float(alpha.mean().item())
            diagnostics["alpha_std"] = float(alpha.std(unbiased=False).item())
    return query_logits, diagnostics


def ci95(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1:
        return 0.0
    return float(1.96 * array.std(ddof=1) / math.sqrt(array.size))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if row["method"] == method]
        accuracy = [float(row["accuracy"]) for row in group]
        agreement = [float(row["agreement_with_simpleshot"]) for row in group]
        radius_std = [
            float(row["radius_std"])
            for row in group
            if math.isfinite(float(row["radius_std"]))
        ]
        output.append(
            {
                "dataset": group[0]["dataset"],
                "backbone": group[0]["backbone"],
                "split": group[0]["split"],
                "way": group[0]["way"],
                "shot": group[0]["shot"],
                "seed": group[0]["seed"],
                "method": method,
                "episodes": len(group),
                "accuracy_mean": float(np.mean(accuracy)),
                "accuracy_ci95": ci95(accuracy),
                "agreement_with_simpleshot_mean": float(np.mean(agreement)),
                "radius_std_mean": float(np.mean(radius_std)) if radius_std else "",
                "support_loss_initial_mean": float(
                    np.nanmean([float(row["support_loss_initial"]) for row in group])
                ),
                "support_loss_final_mean": float(
                    np.nanmean([float(row["support_loss_final"]) for row in group])
                ),
            }
        )
    return output


def run() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_raw, eval_raw = load_or_extract_features(args)
    if args.cache_only:
        return

    center = train_raw.features.mean(dim=0)
    train_features = transform_features(train_raw.features, args.feature_mode, center)
    eval_features = transform_features(eval_raw.features, args.feature_mode, center)
    radius_prior = estimate_radius_prior(
        train_features,
        train_raw.labels,
        quantile=args.radius_quantile,
        mode=args.radius_prior_mode,
    )
    per_class_eval = class_index(eval_raw.labels)
    way = parse_way(args.way, len(per_class_eval))
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    for episode in range(args.episodes):
        _classes, support_idx, query_idx, query_labels_np = support_query_episode(
            rng,
            per_class_eval,
            way=way,
            shot=args.shot,
            query_per_class=args.query_per_class,
        )
        support_labels_np = np.repeat(np.arange(way, dtype=np.int64), args.shot)
        support, query, support_y, query_y = build_episode_tensors(
            eval_features,
            support_idx,
            query_idx,
            support_labels_np,
            query_labels_np,
            device,
        )
        centers = class_centers(support, support_y, way)
        with torch.no_grad():
            simple_logits = -torch.cdist(query, centers).square()
            simple_pred = simple_logits.argmax(dim=1)

        for method in args.methods:
            diagnostics = {
                "support_loss_initial": float("nan"),
                "support_loss_final": float("nan"),
                "radius_mean": float("nan"),
                "radius_std": float("nan"),
                "alpha_mean": float("nan"),
                "alpha_std": float("nan"),
            }
            if method == "protonet":
                raw_support = eval_raw.features[
                    torch.as_tensor(support_idx, dtype=torch.long)
                ].to(device)
                raw_query = eval_raw.features[
                    torch.as_tensor(query_idx, dtype=torch.long)
                ].to(device)
                raw_centers = class_centers(raw_support, support_y, way)
                logits = -torch.cdist(raw_query, raw_centers).square()
            elif method in {"simpleshot_cl2n", "s2_r0"}:
                logits = simple_logits
            else:
                logits, diagnostics = fit_and_score(
                    method,
                    support,
                    query,
                    support_y,
                    radius_prior,
                    args,
                )
            pred = logits.argmax(dim=1)
            rows.append(
                {
                    "dataset": args.dataset,
                    "backbone": args.backbone,
                    "split": args.split,
                    "way": args.way,
                    "shot": args.shot,
                    "seed": args.seed,
                    "episode": episode,
                    "method": method,
                    "accuracy": float(pred.eq(query_y).float().mean().item()),
                    "agreement_with_simpleshot": float(
                        pred.eq(simple_pred).float().mean().item()
                    ),
                    **diagnostics,
                    "radius_prior": radius_prior,
                    "feature_mode": args.feature_mode,
                    "adapt_steps": args.adapt_steps if method in ADAPTED_METHODS else 0,
                    "adapt_lr": args.adapt_lr if method in ADAPTED_METHODS else 0.0,
                }
            )
        if (episode + 1) % 10 == 0 or episode + 1 == args.episodes:
            print(f"Completed {episode + 1}/{args.episodes} episodes", flush=True)

    run_name = args.run_name or (
        f"{args.dataset}_{args.backbone}_{args.split}_{args.way}way_"
        f"{args.shot}shot_e{args.episodes}_s{args.seed}"
    )
    output_dir = Path(args.output_root).expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "episodes.csv", rows)
    write_csv(output_dir / "summary.csv", summarize(rows))
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    run()
