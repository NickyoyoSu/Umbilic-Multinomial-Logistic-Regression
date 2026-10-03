"""Datasets, low-shot subsets, long-tail subsets, and transforms."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Callable

import requests
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def build_transforms(
    image_size: int,
    train: bool,
    train_augmentation: str = "basic",
    randaugment_num_ops: int = 0,
    randaugment_magnitude: int = 0,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> Callable:
    if train and train_augmentation == "resnet_repro":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    if train and train_augmentation == "vit_repro":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _prepare_tiny_imagenet(root: Path) -> Path:
    extracted = root / "tiny-imagenet-200"
    archive = root / "tiny-imagenet-200.zip"
    if not extracted.exists():
        _download_file(TINY_IMAGENET_URL, archive)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(root)

    val_images = extracted / "val" / "images"
    annotations = extracted / "val" / "val_annotations.txt"
    prepared = extracted / "val_prepared"
    if prepared.exists():
        return extracted

    prepared.mkdir(parents=True, exist_ok=True)
    with annotations.open("r", encoding="utf-8") as handle:
        for line in handle:
            image_name, cls, *_ = line.strip().split("\t")
            cls_dir = prepared / cls / "images"
            cls_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(val_images / image_name, cls_dir / image_name)

    return extracted


def _paired_random_split(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    seed: int = 0,
    val_fraction: float = 0.1,
) -> tuple[Dataset, Dataset]:
    total = len(train_dataset)
    val_size = max(1, int(total * val_fraction))
    train_size = total - val_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return Subset(train_dataset, train_subset.indices), Subset(eval_dataset, val_subset.indices)


def _labels_for_dataset(dataset: Dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(label) for label in dataset.targets]
    if hasattr(dataset, "_labels"):
        return [int(label) for label in dataset._labels]
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]
    if isinstance(dataset, Subset):
        parent_labels = _labels_for_dataset(dataset.dataset)
        return [parent_labels[int(index)] for index in dataset.indices]
    raise ValueError("Dataset does not expose labels through targets, _labels, samples, or Subset indices.")


def _few_shot_subset(dataset: Dataset, shots_per_class: int, seed: int) -> Dataset:
    if shots_per_class <= 0:
        return dataset
    labels = _labels_for_dataset(dataset)
    rng = torch.Generator().manual_seed(seed)
    per_class_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        per_class_indices.setdefault(int(label), []).append(idx)

    chosen: list[int] = []
    for class_id in sorted(per_class_indices):
        indices = per_class_indices[class_id]
        if len(indices) <= shots_per_class:
            chosen.extend(indices)
            continue
        perm = torch.randperm(len(indices), generator=rng).tolist()
        chosen.extend(indices[i] for i in perm[:shots_per_class])
    chosen.sort()
    return Subset(dataset, chosen)


def _imbalanced_subset(
    dataset: Dataset,
    imbalance_type: str,
    imbalance_ratio: float,
    seed: int,
) -> Dataset:
    """Create a deterministic long-tail subset while preserving all classes."""

    if imbalance_type == "none":
        return dataset
    if imbalance_type not in {"exp", "step"}:
        raise ValueError(f"Unsupported imbalance type: {imbalance_type}")
    if not 0.0 < imbalance_ratio <= 1.0:
        raise ValueError("imbalance_ratio must lie in (0, 1].")

    labels = _labels_for_dataset(dataset)
    per_class_indices: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        per_class_indices.setdefault(int(label), []).append(index)
    class_ids = sorted(per_class_indices)
    if not class_ids:
        raise ValueError("Cannot construct an imbalanced subset from an empty dataset.")

    max_count = max(len(indices) for indices in per_class_indices.values())
    rng = torch.Generator().manual_seed(seed)
    chosen: list[int] = []
    denominator = max(1, len(class_ids) - 1)
    for rank, class_id in enumerate(class_ids):
        if imbalance_type == "exp":
            fraction = imbalance_ratio ** (rank / denominator)
        else:
            fraction = 1.0 if rank < len(class_ids) / 2 else imbalance_ratio
        target_count = max(1, int(round(max_count * fraction)))
        indices = per_class_indices[class_id]
        target_count = min(target_count, len(indices))
        permutation = torch.randperm(len(indices), generator=rng).tolist()
        chosen.extend(indices[position] for position in permutation[:target_count])

    chosen.sort()
    return Subset(dataset, chosen)


def class_counts_for_dataset(dataset: Dataset, num_classes: int) -> list[int]:
    counts = [0] * num_classes
    for label in _labels_for_dataset(dataset):
        counts[int(label)] += 1
    return counts


def _resolve_imagenet_root(root: Path) -> Path:
    candidates = [root / "imagenet", root / "ILSVRC2012", root]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"ImageNet train/val directories not found under: {searched}")


def build_datasets(
    dataset_name: str,
    data_root: str | Path,
    image_size: int,
    download: bool,
    seed: int = 0,
    train_augmentation: str = "basic",
    randaugment_num_ops: int = 0,
    randaugment_magnitude: int = 0,
    train_shots_per_class: int = 0,
    train_imbalance_type: str = "none",
    train_imbalance_ratio: float = 1.0,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[Dataset, Dataset, int]:
    root = Path(data_root).expanduser().resolve()
    normalized = dataset_name.lower()
    train_tf = build_transforms(
        image_size,
        train=True,
        train_augmentation=train_augmentation,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
        mean=mean,
        std=std,
    )
    eval_tf = build_transforms(image_size, train=False, mean=mean, std=std)

    if normalized == "cifar100":
        train_ds = datasets.CIFAR100(root=root, train=True, download=download, transform=train_tf)
        val_ds = datasets.CIFAR100(root=root, train=False, download=download, transform=eval_tf)
        train_ds = _few_shot_subset(train_ds, train_shots_per_class, seed)
        train_ds = _imbalanced_subset(train_ds, train_imbalance_type, train_imbalance_ratio, seed)
        return train_ds, val_ds, 100

    if normalized == "flowers102":
        full_train = datasets.Flowers102(root=root, split="train", download=download, transform=train_tf)
        val_ds = datasets.Flowers102(root=root, split="val", download=download, transform=eval_tf)
        full_train = _few_shot_subset(full_train, train_shots_per_class, seed)
        full_train = _imbalanced_subset(full_train, train_imbalance_type, train_imbalance_ratio, seed)
        return full_train, val_ds, 102

    if normalized == "oxford_iiit_pet":
        train_ds = datasets.OxfordIIITPet(
            root=root,
            split="trainval",
            target_types="category",
            download=download,
            transform=train_tf,
        )
        val_ds = datasets.OxfordIIITPet(
            root=root,
            split="test",
            target_types="category",
            download=download,
            transform=eval_tf,
        )
        train_ds = _few_shot_subset(train_ds, train_shots_per_class, seed)
        return train_ds, val_ds, 37

    if normalized == "food101":
        # Food-101 is evaluated on a seeded 10% hold-out of the official train split.
        train_base = datasets.Food101(root=root, split="train", download=download, transform=train_tf)
        eval_base = datasets.Food101(root=root, split="train", download=download, transform=eval_tf)
        train_ds, val_ds = _paired_random_split(train_base, eval_base, seed=seed)
        train_ds = _few_shot_subset(train_ds, train_shots_per_class, seed)
        return train_ds, val_ds, 101

    if normalized == "tiny_imagenet":
        # Downloads the archive when missing and sorts val images into class folders.
        extracted = _prepare_tiny_imagenet(root)
        train_ds = datasets.ImageFolder(extracted / "train", transform=train_tf)
        val_ds = datasets.ImageFolder(extracted / "val_prepared", transform=eval_tf)
        num_classes = len(train_ds.classes)
        train_ds = _few_shot_subset(train_ds, train_shots_per_class, seed)
        train_ds = _imbalanced_subset(train_ds, train_imbalance_type, train_imbalance_ratio, seed)
        return train_ds, val_ds, num_classes

    if normalized == "imagenet":
        imagenet_root = _resolve_imagenet_root(root)
        train_ds = datasets.ImageFolder(imagenet_root / "train", transform=train_tf)
        val_ds = datasets.ImageFolder(imagenet_root / "val", transform=eval_tf)
        num_classes = len(train_ds.classes)
        train_ds = _few_shot_subset(train_ds, train_shots_per_class, seed)
        train_ds = _imbalanced_subset(train_ds, train_imbalance_type, train_imbalance_ratio, seed)
        return train_ds, val_ds, num_classes

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_loaders(
    dataset_name: str,
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    download: bool,
    seed: int = 0,
    train_augmentation: str = "basic",
    randaugment_num_ops: int = 0,
    randaugment_magnitude: int = 0,
    train_shots_per_class: int = 0,
    train_imbalance_type: str = "none",
    train_imbalance_ratio: float = 1.0,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[DataLoader, DataLoader, int]:
    train_ds, val_ds, num_classes = build_datasets(
        dataset_name=dataset_name,
        data_root=data_root,
        image_size=image_size,
        download=download,
        seed=seed,
        train_augmentation=train_augmentation,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
        train_shots_per_class=train_shots_per_class,
        train_imbalance_type=train_imbalance_type,
        train_imbalance_ratio=train_imbalance_ratio,
        mean=mean,
        std=std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, num_classes
