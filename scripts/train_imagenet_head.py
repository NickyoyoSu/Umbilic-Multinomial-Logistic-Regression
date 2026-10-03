#!/usr/bin/env python3
"""Retrain a classifier head on cached ImageNet-1K features (Experiment 4).

Features are centered by the training mean and L2-normalized. Spherical heads
start from CL2N class centers and q80 radii. Every epoch is evaluated on the
ImageNet validation images; result.json reports the final-epoch accuracy
(final_top1/final_top5, used in the paper) and, for reference, the
maximum over epochs (max_epoch_val_top1).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from umlr.heads import AngularMarginHead, HyperplaneHead
from umlr.umlr_head import SphereUMLRHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UMLR and angular heads on cached ImageNet features.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=["h", "h2", "s", "s2", "cosface", "arcface"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--angular-scale", type=float, default=30.0)
    parser.add_argument("--angular-margin", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def build_head(method: str, feature_dim: int, num_classes: int, args: argparse.Namespace):
    if method in {"h", "h2"}:
        return HyperplaneHead(
            feature_dim,
            num_classes,
            score_mode="linear" if method == "h" else "signed_square",
            alpha_scope="classwise",
            alpha_init_effective=1.0,
        )
    if method in {"s", "s2"}:
        return SphereUMLRHead(
            feature_dim,
            num_classes,
            parameterization="ppt_decoupled_inside",
            alpha_mode="softplus",
            boundary_mode="distance" if method == "s" else "signed_square",
            alpha_scope="classwise",
            alpha_init_effective=1.0,
            radius_scope="classwise",
        )
    return AngularMarginHead(
        feature_dim,
        num_classes,
        method=method,
        scale=args.angular_scale,
        margin=args.angular_margin,
    )


def transform(features: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float() - center, dim=-1)


@torch.no_grad()
def evaluate(
    head: torch.nn.Module,
    loader: DataLoader,
    center: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float]:
    head.eval()
    total = 0
    top1 = 0
    top5 = 0
    total_loss = 0.0
    for features, labels in loader:
        features = transform(features.to(device, non_blocking=True), center)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = head(features)
            loss = F.cross_entropy(logits, labels)
        total += labels.numel()
        total_loss += float(loss.item()) * labels.numel()
        predictions = logits.topk(5, dim=1).indices
        top1 += int((predictions[:, 0] == labels).sum().item())
        top5 += int((predictions == labels.unsqueeze(1)).any(dim=1).sum().item())
    return total_loss / total, top1 / total, top5 / total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_payload = torch.load(str(cache_dir / "train.pt"), map_location="cpu", mmap=True, weights_only=True)
    val_payload = torch.load(str(cache_dir / "val.pt"), map_location="cpu", mmap=True, weights_only=True)
    stats = torch.load(cache_dir / "feature_stats.pt", map_location="cpu", weights_only=True)
    train_loader = DataLoader(
        TensorDataset(train_payload["features"], train_payload["labels"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_payload["features"], val_payload["labels"]),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    num_classes = int(stats["class_centers"].shape[0])
    feature_dim = int(stats["class_centers"].shape[1])
    center = stats["global_center"].to(device)
    head = build_head(args.method, feature_dim, num_classes, args).to(device)
    if args.method in {"s", "s2"}:
        head.set_centers_(stats["class_centers"].to(device))
        head.set_radius_values_(stats["radii_q80"].to(device))

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    history: list[dict[str, float]] = []
    best_top1 = -1.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        head.train()
        train_total = 0
        train_correct = 0
        train_loss = 0.0
        for features, labels in train_loader:
            features = transform(features.to(device, non_blocking=True), center)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                if getattr(head, "requires_target_labels", False):
                    logits = head(features, labels)
                else:
                    logits = head(features)
                loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
            loss.backward()
            optimizer.step()
            train_total += labels.numel()
            train_loss += float(loss.item()) * labels.numel()
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())
        scheduler.step()
        val_loss, val_top1, val_top5 = evaluate(head, val_loader, center, device, args.amp)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / train_total,
                "train_accuracy": train_correct / train_total,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top5": val_top5,
            }
        )
        best_top1 = max(best_top1, val_top1)

    elapsed = time.perf_counter() - started
    final = history[-1]
    radius_summary = None
    radius_getter = getattr(head, "radius_values", None)
    if radius_getter is not None:
        radii = radius_getter().detach().float().cpu()
        radius_summary = {
            "mean": float(radii.mean().item()),
            "std": float(radii.std(unbiased=False).item()),
            "min": float(radii.min().item()),
            "max": float(radii.max().item()),
            "all_finite": bool(torch.isfinite(radii).all().item()),
            "median_ratio_to_init": float(
                (radii / stats["radii_q80"].float().clamp_min(1e-8)).median().item()
            ),
        }
    result = {
        "method": args.method,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_count": int(train_payload["labels"].numel()),
        "val_count": int(val_payload["labels"].numel()),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "final_val_loss": final["val_loss"],
        "final_top1": final["val_top1"],
        "final_top5": final["val_top5"],
        "max_epoch_val_top1": best_top1,
        "elapsed_seconds": elapsed,
        "examples_per_second": args.epochs * int(train_payload["labels"].numel()) / elapsed,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "radius": radius_summary,
        "history": history,
        "config": vars(args),
    }
    torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, output_dir / "last_head.pt")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("method", "seed", "final_top1", "final_top5")}))


if __name__ == "__main__":
    main()

