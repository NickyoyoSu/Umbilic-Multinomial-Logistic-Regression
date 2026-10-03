#!/usr/bin/env python3
"""Download and prepare the image-classification datasets used by ``scripts/train.py``.

CIFAR-100, Flowers102, Oxford-IIIT Pet, and Food-101 are fetched through
``torchvision``; TinyImageNet is downloaded from the CS231n mirror and its
validation images are sorted into class folders. ImageNet-1K must be obtained
separately (see the README). Few-shot episodic datasets are fetched by
``scripts/run_episodic.py`` on first use.
"""

from __future__ import annotations

import argparse

from umlr.data import build_datasets

DATASETS = ("cifar100", "flowers102", "tiny_imagenet", "oxford_iiit_pet", "food101")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--data-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        build_datasets(dataset_name=dataset, data_root=args.data_root, image_size=224, download=True)
        print(f"Prepared {dataset} under {args.data_root}")


if __name__ == "__main__":
    main()
