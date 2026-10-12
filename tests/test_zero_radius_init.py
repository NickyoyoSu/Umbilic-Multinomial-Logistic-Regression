"""Feature-statistics initialization for the zero-radius limit of UMLR^2-S."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from umlr import BoundaryClassifier, SphereUMLRHead

_TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("umlr_train_script", _TRAIN_PATH)
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


def _toy_problem():
    torch.manual_seed(0)
    features = torch.randn(40, 6)
    labels = torch.arange(4).repeat_interleave(10)
    loader = DataLoader(TensorDataset(features, labels), batch_size=8)
    class_means = torch.stack([features[labels == k].mean(dim=0) for k in range(4)])
    return loader, class_means


def _args(centers_only: bool) -> SimpleNamespace:
    return SimpleNamespace(
        feature_stat_init="centroid_radius",
        feature_stat_radius_mode="quantile",
        feature_stat_radius_quantile=0.9,
        feature_stat_radius_std_mult=1.0,
        feature_stat_max_batches=None,
        feature_stat_centers_only=centers_only,
    )


def _zero_radius_model() -> BoundaryClassifier:
    head = SphereUMLRHead(
        6, 4, boundary_mode="signed_square", alpha_mode="identity", radius_scope="fixed", fixed_radius=0.0
    )
    return BoundaryClassifier(torch.nn.Identity(), head)


def test_centers_only_keeps_the_fixed_zero_radius() -> None:
    loader, class_means = _toy_problem()
    model = _zero_radius_model()
    train_script.apply_feature_stat_init(model, loader, 4, torch.device("cpu"), _args(centers_only=True))
    assert torch.allclose(model.head.centers, class_means, atol=1e-6)
    assert torch.all(model.head.radius_values() == 0)


def test_without_centers_only_the_fixed_radius_is_overwritten() -> None:
    loader, _ = _toy_problem()
    model = _zero_radius_model()
    train_script.apply_feature_stat_init(model, loader, 4, torch.device("cpu"), _args(centers_only=False))
    assert torch.all(model.head.radius_values() > 0)
