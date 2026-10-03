"""Few-shot episode construction, feature extraction, and CL2N helpers.

Datasets: miniImageNet, CIFAR-FS, and FC100 are read from the Hugging Face hub
(``GATE-engine/*``); CUB and tieredImageNet are read from the
``pancake/few_shot_datasets`` archives, extracted under
``<data_root>/standard_fewshot/<dataset>/``.
"""

from __future__ import annotations

import csv
import random
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


HF_DATASETS = {
    "mini_imagenet": "GATE-engine/mini_imagenet",
    "cifarfs": "GATE-engine/cifarfs",
    "fc100": "GATE-engine/fc100",
}

PANCAKE_ARCHIVES = {
    "cub": "cub.tar",
    "tiered_imagenet": "tiered_imagenet.tar",
}

DATASETS = (*HF_DATASETS, *PANCAKE_ARCHIVES)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "meta-train": "train",
    "meta_train": "train",
    "base": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "meta-val": "validation",
    "meta_val": "validation",
    "test": "test",
    "meta-test": "test",
    "meta_test": "test",
    "novel": "test",
}


@dataclass(frozen=True)
class FeatureSplit:
    features: torch.Tensor
    labels: torch.Tensor
    class_ids: list[int]


class HFDataset(Dataset):
    def __init__(self, name: str, split: str, transform) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError("HuggingFace datasets is required for HF few-shot benchmarks.") from exc

        self.ds = load_dataset(name, split=split)
        self.transform = transform
        self.image_col = "image"
        if "label" in self.ds.column_names:
            self.label_col = "label"
        elif "fine_label" in self.ds.column_names:
            self.label_col = "fine_label"
        elif "labels" in self.ds.column_names:
            self.label_col = "labels"
        else:
            raise ValueError(f"Could not find a label column in {name}/{split}: {self.ds.column_names}")
        self.labels = [int(x) for x in self.ds[self.label_col]]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[int(idx)]
        image = row[self.image_col]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        return self.transform(image), int(row[self.label_col])


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform
        self.labels = [int(label) for _path, label in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[int(idx)]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), int(label)


class LabelSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int], labels: list[int]) -> None:
        self.dataset = dataset
        self.indices = [int(x) for x in indices]
        self.labels = [int(labels[i]) for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[int(idx)]]


def build_eval_transform(image_size: int, mean: tuple[float, float, float], std: tuple[float, float, float]):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _download_pancake_archive(dataset: str, data_root: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError("huggingface_hub is required to download pancake/few_shot_datasets archives.") from exc

    archive_name = PANCAKE_ARCHIVES[dataset]
    archive_dir = data_root / "fewshot_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id="pancake/few_shot_datasets",
            repo_type="dataset",
            filename=archive_name,
            local_dir=archive_dir,
            local_dir_use_symlinks=False,
        )
    )


def _safe_extract_tar(archive: Path, target: Path) -> None:
    target_resolved = target.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            destination = (target / member.name).resolve()
            if not str(destination).startswith(str(target_resolved)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        handle.extractall(target)


def _extract_archive_if_needed(dataset: str, data_root: Path, download: bool) -> Path:
    extracted = data_root / "standard_fewshot" / dataset
    marker = extracted / ".extracted"
    if marker.exists():
        return extracted
    extracted.mkdir(parents=True, exist_ok=True)

    archive_name = PANCAKE_ARCHIVES[dataset]
    candidate = data_root / "fewshot_archives" / archive_name
    if not candidate.exists():
        if not download:
            raise FileNotFoundError(
                f"{candidate} is missing. Re-run with --download-pancake-archives or prepare the archive manually."
            )
        candidate = _download_pancake_archive(dataset, data_root)

    if candidate.suffix == ".zip":
        with zipfile.ZipFile(candidate) as handle:
            handle.extractall(extracted)
    elif candidate.suffix in {".tar", ".gz", ".tgz"} or candidate.name.endswith(".tar.gz"):
        _safe_extract_tar(candidate, extracted)
    else:
        raise ValueError(f"Unsupported archive format: {candidate}")
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return extracted


def _discover_split_samples(root: Path, split: str) -> list[tuple[Path, int]]:
    csv_samples = _discover_csv_split_samples(root, split)
    if csv_samples:
        return csv_samples

    label_to_id: dict[str, int] = {}
    samples: list[tuple[Path, int]] = []
    requested = SPLIT_ALIASES.get(split, split)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = list(path.relative_to(root).parts)
        lower_parts = [part.lower() for part in parts]
        split_index = None
        split_name = None
        for idx, part in enumerate(lower_parts[:-1]):
            normalized = SPLIT_ALIASES.get(part)
            if normalized is not None:
                split_index = idx
                split_name = normalized
                break
        if split_name != requested or split_index is None:
            continue
        if split_index + 1 >= len(parts) - 1:
            label_name = path.parent.name
        else:
            label_name = parts[split_index + 1]
        label_id = label_to_id.setdefault(label_name, len(label_to_id))
        samples.append((path, label_id))
    if not samples:
        raise FileNotFoundError(
            f"No images found for split={split!r} under {root}. Expected a split/class/image folder layout."
        )
    samples.sort(key=lambda item: (item[1], str(item[0])))
    return samples


def _discover_csv_split_samples(root: Path, split: str) -> list[tuple[Path, int]]:
    requested = SPLIT_ALIASES.get(split, split)
    aliases = sorted({name for name, normalized in SPLIT_ALIASES.items() if normalized == requested} | {requested})
    split_files: list[Path] = []
    for alias in aliases:
        split_files.extend(root.rglob(f"split/{alias}.csv"))
        split_files.extend(root.rglob(f"splits/{alias}.csv"))
    if not split_files:
        return []

    split_file = sorted(set(split_files), key=lambda p: (len(p.parts), str(p)))[0]
    image_roots = [split_file.parent.parent, root]
    image_by_name: dict[str, Path] = {}
    for image_root in image_roots:
        if not image_root.exists():
            continue
        for path in image_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                image_by_name.setdefault(path.name, path)

    label_to_id: dict[str, int] = {}
    samples: list[tuple[Path, int]] = []
    with split_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            filename = row[0].strip()
            label_name = row[1].strip()
            path = image_by_name.get(Path(filename).name)
            if path is None:
                direct = split_file.parent.parent / filename
                if direct.exists():
                    path = direct
            if path is None:
                continue
            label_id = label_to_id.setdefault(label_name, len(label_to_id))
            samples.append((path, label_id))
    samples.sort(key=lambda item: (item[1], str(item[0])))
    return samples


def build_dataset_split(args, split: str, transform) -> Dataset:
    data_root = Path(args.data_root).expanduser().resolve()
    if args.dataset in HF_DATASETS:
        hf_split = "validation" if split == "validation" else split
        return HFDataset(HF_DATASETS[args.dataset], hf_split, transform)
    if args.dataset not in PANCAKE_ARCHIVES:
        raise ValueError(f"Unsupported dataset source: {args.dataset}")
    extracted = _extract_archive_if_needed(args.dataset, data_root, args.download_pancake_archives)
    return ImagePathDataset(_discover_split_samples(extracted, split), transform)


def subset_by_class(dataset: Dataset, labels: list[int], max_per_class: int, seed: int) -> Dataset:
    if max_per_class <= 0:
        return dataset
    rng = random.Random(seed)
    per_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        per_class.setdefault(int(label), []).append(idx)
    selected: list[int] = []
    for class_id in sorted(per_class):
        indices = per_class[class_id]
        rng.shuffle(indices)
        selected.extend(indices[:max_per_class])
    selected.sort()
    return LabelSubset(dataset, selected, labels)


def extract_features(model, dataset: Dataset, args, device: torch.device) -> FeatureSplit:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                batch_features = model(images)
                if batch_features.ndim > 2:
                    batch_features = torch.flatten(batch_features, 1)
            features.append(batch_features.detach().float().cpu())
            labels.append(batch_labels.detach().long().cpu())
    all_features = torch.cat(features, dim=0)
    all_labels = torch.cat(labels, dim=0)
    return FeatureSplit(
        features=all_features,
        labels=all_labels,
        class_ids=sorted(int(x) for x in all_labels.unique().tolist()),
    )


def transform_features(features: torch.Tensor, mode: str, center: torch.Tensor | None) -> torch.Tensor:
    if mode == "un":
        return features
    if mode == "cl2n":
        if center is None:
            raise ValueError("CL2N requires a train feature center.")
        features = features - center.unsqueeze(0)
    if mode in {"l2n", "cl2n"}:
        return F.normalize(features, dim=-1)
    raise ValueError(f"Unknown feature transform mode: {mode}")


def class_index(labels: torch.Tensor) -> dict[int, torch.Tensor]:
    return {int(c): (labels == int(c)).nonzero(as_tuple=False).flatten() for c in labels.unique().tolist()}


def estimate_radius_prior(
    features: torch.Tensor,
    labels: torch.Tensor,
    quantile: float,
    mode: str,
) -> float:
    radii: list[float] = []
    for class_id, indices in class_index(labels).items():
        if indices.numel() < 2:
            continue
        class_features = features[indices]
        center = class_features.mean(dim=0, keepdim=True)
        distances = torch.linalg.norm(class_features - center, dim=1)
        radii.append(float(torch.quantile(distances, quantile).item()))
    if not radii:
        return 1e-3
    values = np.asarray(radii, dtype=np.float64)
    if mode == "mean":
        return float(values.mean())
    return float(np.median(values))


def parse_way(value: str, available_classes: int) -> int:
    if value == "full":
        return int(available_classes)
    way = int(value)
    return min(way, int(available_classes))


def sample_indices(
    rng: np.random.Generator,
    indices: torch.Tensor,
    count: int,
    exclude: set[int] | None = None,
) -> np.ndarray:
    values = indices.cpu().numpy().astype(np.int64)
    if exclude:
        values = np.asarray([idx for idx in values.tolist() if int(idx) not in exclude], dtype=np.int64)
    replace = len(values) < count
    if len(values) == 0:
        raise ValueError("Cannot sample from an empty class bucket.")
    return rng.choice(values, size=count, replace=replace)


def support_query_episode(
    rng: np.random.Generator,
    per_class: dict[int, torch.Tensor],
    way: int,
    shot: int,
    query_per_class: int,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    eligible = [c for c, idx in per_class.items() if idx.numel() >= max(shot + query_per_class, shot)]
    if len(eligible) < way:
        raise ValueError(f"Only {len(eligible)} eligible classes are available for way={way}.")
    classes = sorted(rng.choice(np.asarray(eligible), size=way, replace=False).astype(int).tolist())

    support_indices: list[int] = []
    query_indices: list[int] = []
    query_labels: list[int] = []
    for local_label, class_id in enumerate(classes):
        support = sample_indices(rng, per_class[class_id], shot)
        query = sample_indices(rng, per_class[class_id], query_per_class, exclude=set(int(x) for x in support.tolist()))
        support_indices.extend(int(x) for x in support.tolist())
        query_indices.extend(int(x) for x in query.tolist())
        query_labels.extend([local_label] * len(query))
    return classes, np.asarray(support_indices), np.asarray(query_indices), np.asarray(query_labels)
