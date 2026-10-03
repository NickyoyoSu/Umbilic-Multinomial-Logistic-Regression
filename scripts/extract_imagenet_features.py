#!/usr/bin/env python3
"""Cache frozen ImageNet-1K penultimate features for head retraining (Experiment 4).

Runs single-process or under torchrun. Writes train.pt, val.pt
(features and labels), and feature_stats.pt (global feature mean, class
centers, and q80 within-class radii of the CL2N features) to --cache-dir.
ImageNet is read in ImageFolder layout from <data-root>/{imagenet,ILSVRC2012,.}/{train,val}.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from umlr.data import _resolve_imagenet_root, build_transforms
from umlr.backbones import BACKBONES, build_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen ImageNet-1K features with distributed shards.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--backbone", default="vit_b_16", choices=BACKBONES)
    parser.add_argument("--batch-size", type=int, default=128, help="Per-rank batch size.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--stats-chunk-size", type=int, default=65536)
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def extract_split(
    split: str,
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    cache_dir: Path,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> None:
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                features = model(images)
                if features.ndim > 2:
                    features = torch.flatten(features, 1)
            feature_chunks.append(features.detach().to(device="cpu", dtype=torch.float16))
            label_chunks.append(labels.to(dtype=torch.long))

    shard = {
        "indices": torch.tensor(indices, dtype=torch.long),
        "features": torch.cat(feature_chunks, dim=0),
        "labels": torch.cat(label_chunks, dim=0),
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(shard, cache_dir / f"{split}_rank{rank:03d}.pt")
    del shard, feature_chunks, label_chunks
    gc.collect()


def merge_split(split: str, cache_dir: Path, world_size: int, expected_count: int) -> dict[str, object]:
    indices: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    shard_seconds: list[float] = []
    for rank in range(world_size):
        shard = torch.load(cache_dir / f"{split}_rank{rank:03d}.pt", map_location="cpu", weights_only=True)
        indices.append(shard["indices"])
        features.append(shard["features"])
        labels.append(shard["labels"])
        shard_seconds.append(float(shard["elapsed_seconds"]))
    all_indices = torch.cat(indices)
    order = torch.argsort(all_indices)
    if all_indices.numel() != expected_count or not torch.equal(all_indices[order], torch.arange(expected_count)):
        raise RuntimeError(f"{split} shard merge is incomplete or duplicated.")
    merged_features = torch.cat(features, dim=0)[order].contiguous()
    merged_labels = torch.cat(labels, dim=0)[order].contiguous()
    torch.save(
        {"features": merged_features, "labels": merged_labels},
        cache_dir / f"{split}.pt",
    )
    metadata = {
        "count": int(merged_labels.numel()),
        "feature_dim": int(merged_features.shape[1]),
        "dtype": str(merged_features.dtype),
        "max_rank_seconds": max(shard_seconds),
    }
    del indices, features, labels, all_indices, order, merged_features, merged_labels
    gc.collect()
    return metadata


def compute_feature_stats(cache_dir: Path, chunk_size: int) -> dict[str, object]:
    payload = torch.load(str(cache_dir / "train.pt"), map_location="cpu", mmap=True, weights_only=True)
    features = payload["features"]
    labels = payload["labels"]
    num_classes = int(labels.max().item()) + 1
    feature_dim = int(features.shape[1])

    global_sum = torch.zeros(feature_dim, dtype=torch.float64)
    for start in range(0, features.shape[0], chunk_size):
        global_sum += features[start : start + chunk_size].double().sum(dim=0)
    global_center = (global_sum / features.shape[0]).float()

    class_sums = torch.zeros(num_classes, feature_dim, dtype=torch.float64)
    class_counts = torch.bincount(labels, minlength=num_classes)
    for start in range(0, features.shape[0], chunk_size):
        chunk = features[start : start + chunk_size].float()
        chunk_labels = labels[start : start + chunk_size]
        transformed = F.normalize(chunk - global_center, dim=-1)
        class_sums.index_add_(0, chunk_labels, transformed.double())
    class_centers = (class_sums / class_counts.clamp_min(1).double().unsqueeze(1)).float()

    distances = torch.empty(features.shape[0], dtype=torch.float32)
    for start in range(0, features.shape[0], chunk_size):
        chunk = features[start : start + chunk_size].float()
        chunk_labels = labels[start : start + chunk_size]
        transformed = F.normalize(chunk - global_center, dim=-1)
        distances[start : start + chunk.shape[0]] = torch.linalg.norm(
            transformed - class_centers[chunk_labels],
            dim=-1,
        )
    radii_q80 = torch.empty(num_classes, dtype=torch.float32)
    for class_id in range(num_classes):
        radii_q80[class_id] = torch.quantile(distances[labels == class_id], 0.8)

    stats = {
        "global_center": global_center,
        "class_centers": class_centers,
        "radii_q80": radii_q80,
        "class_counts": class_counts,
    }
    torch.save(stats, cache_dir / "feature_stats.pt")
    return {
        "num_classes": num_classes,
        "global_center_norm": float(global_center.norm().item()),
        "radius_q80_mean": float(radii_q80.mean().item()),
        "radius_q80_min": float(radii_q80.min().item()),
        "radius_q80_max": float(radii_q80.max().item()),
    }


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = distributed_context()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_backbone(args.backbone, pretrained=True)
    model = bundle.model.to(device)
    transform = build_transforms(
        bundle.image_size,
        train=False,
        mean=bundle.mean,
        std=bundle.std,
    )
    imagenet_root = _resolve_imagenet_root(Path(args.data_root))
    datasets_by_split = {
        "train": datasets.ImageFolder(imagenet_root / "train", transform=transform),
        "val": datasets.ImageFolder(imagenet_root / "val", transform=transform),
    }

    for split, dataset in datasets_by_split.items():
        extract_split(
            split=split,
            dataset=dataset,
            model=model,
            device=device,
            cache_dir=cache_dir,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=args.amp,
        )
        barrier(world_size)

    if rank == 0:
        manifest = {
            "backbone": args.backbone,
            "source": bundle.source_url,
            "world_size": world_size,
            "train": merge_split("train", cache_dir, world_size, len(datasets_by_split["train"])),
            "val": merge_split("val", cache_dir, world_size, len(datasets_by_split["val"])),
        }
        manifest["feature_stats"] = compute_feature_stats(cache_dir, args.stats_chunk_size)
        (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

