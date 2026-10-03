"""Low-shot and long-tail subsets, optimizer groups, and the training loop."""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from umlr import BoundaryClassifier, HyperplaneHead, SphereUMLRHead
from umlr.data import _few_shot_subset, _imbalanced_subset, class_counts_for_dataset
from umlr.factory import build_head
from umlr.training import _build_optimizer, _build_scheduler, train_model


class _ToyDataset(Dataset):
    def __init__(self, counts: list[int]) -> None:
        self.targets = [label for label, count in enumerate(counts) for _ in range(count)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.tensor(float(index)), self.targets[index]


def test_few_shot_subset_is_seeded_and_caps_at_available_images() -> None:
    dataset = _ToyDataset([20, 20, 3])
    first = _few_shot_subset(dataset, 4, seed=1)
    assert first.indices == _few_shot_subset(dataset, 4, seed=1).indices
    assert first.indices != _few_shot_subset(dataset, 4, seed=2).indices
    assert class_counts_for_dataset(first, 3) == [4, 4, 3]


def test_exponential_long_tail_profile() -> None:
    dataset = _ToyDataset([20] * 5)
    subset = _imbalanced_subset(dataset, "exp", 0.1, seed=7)
    assert subset.indices == _imbalanced_subset(dataset, "exp", 0.1, seed=7).indices
    counts = class_counts_for_dataset(subset, 5)
    assert counts[0] == 20 and counts[-1] == 2 and all(c > 0 for c in counts)


def test_optimizer_groups_and_warmup_cosine_schedule() -> None:
    model = BoundaryClassifier(torch.nn.Linear(3, 3), SphereUMLRHead(3, 2))
    optimizer = _build_optimizer(model, "adamw", lr=1e-3, weight_decay=0.0, momentum=0.9,
                                 head_lr_factor=2.0, radius_lr_factor=0.1)
    lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert lrs == {"backbone": 1e-3, "head": 2e-3, "radius": 2e-3 * 0.1}

    scheduler = _build_scheduler(optimizer, "cosine", epochs=10, warmup_epochs=2, min_lr=1e-6, base_lr=1e-3)
    factors = [scheduler.lr_lambdas[0](epoch) for epoch in range(10)]
    assert factors[0] == 0.5 and factors[1] == 1.0
    assert all(a >= b for a, b in zip(factors[1:], factors[2:]))
    assert abs(factors[-1] - 1e-3) < 1e-12


def test_build_head_from_run_config() -> None:
    config = {
        "head_type": "sphere", "sphere_parameterization": "ppt_decoupled_inside",
        "umlr_alpha_mode": "identity", "sphere_boundary_mode": "signed_square", "alpha_scope": "classwise",
        "alpha_init_effective": None, "radius_scope": "classwise", "radius_init_effective": None,
        "fixed_radius": 0.0,
    }
    head = build_head(config, 8, 4)
    assert isinstance(head, SphereUMLRHead) and head.boundary_mode == "signed_square"


def test_train_model_writes_best_and_final_metrics(tmp_path) -> None:
    torch.manual_seed(0)
    x, y = torch.randn(64, 6), torch.randint(0, 3, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    model = BoundaryClassifier(torch.nn.Linear(6, 4), HyperplaneHead(4, 3), normalize_features=True)
    metrics = train_model(
        model, loader, loader, torch.device("cpu"), epochs=3, lr=1e-2, weight_decay=0.0, optimizer_name="adamw",
        momentum=0.9, scheduler_name="cosine", warmup_epochs=1, min_lr=1e-5, label_smoothing=0.1, use_amp=False,
        max_train_batches=None, max_val_batches=None, output_dir=tmp_path / "run", run_config={"note": "test"},
    )
    saved = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert len(saved["history"]) == 3
    assert saved["best_val_accuracy"] == max(row["val_accuracy"] for row in saved["history"])
    assert saved["final_val_accuracy"] == saved["history"][-1]["val_accuracy"] == metrics["final_val_accuracy"]
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "last.pt").exists()
