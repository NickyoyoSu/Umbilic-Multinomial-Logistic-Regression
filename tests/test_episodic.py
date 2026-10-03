"""Episode sampling, CL2N features, radius prior, and the validation-only selection rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from umlr.episodic import class_index, estimate_radius_prior, parse_way, support_query_episode, transform_features

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_way_episode_has_disjoint_support_and_query() -> None:
    labels = torch.arange(6).repeat_interleave(25)
    per_class = class_index(labels)
    way = parse_way("full", len(per_class))
    rng = np.random.default_rng(0)
    classes, support, query, query_labels = support_query_episode(rng, per_class, way=way, shot=4, query_per_class=15)
    assert classes == list(range(6))
    assert len(support) == 6 * 4 and len(query) == 6 * 15
    assert not set(support.tolist()) & set(query.tolist())
    assert np.array_equal(np.bincount(query_labels), np.full(6, 15))


def test_cl2n_and_radius_prior() -> None:
    torch.manual_seed(0)
    features = torch.randn(40, 5) + 3.0
    labels = torch.arange(4).repeat_interleave(10)
    center = features.mean(dim=0)
    transformed = transform_features(features, "cl2n", center)
    torch.testing.assert_close(transformed.norm(dim=1), torch.ones(40))
    prior = estimate_radius_prior(transformed, labels, quantile=0.9, mode="median")
    per_class = []
    for k in range(4):
        z = transformed[labels == k]
        per_class.append(float(torch.quantile((z - z.mean(0)).norm(dim=1), 0.9)))
    assert abs(prior - float(np.median(per_class))) < 1e-6


def test_adapted_heads_start_from_prototype_geometry() -> None:
    run_episodic = _load_script("run_episodic")
    args = type("Args", (), {"alpha_init": 10.0})()
    centers = torch.randn(3, 4)
    hyperplane = run_episodic.make_adapted_head("adapt_h", centers, 0.7, args)
    x = torch.randn(5, 4)
    expected = 10.0 * (2.0 * x @ centers.T - centers.square().sum(dim=1))
    torch.testing.assert_close(hyperplane(x), expected, atol=1e-4, rtol=1e-4)
    sphere = run_episodic.make_adapted_head("adapt_s2", centers, 0.7, args)
    torch.testing.assert_close(sphere.radius_values(), torch.full((3,), 0.7), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(sphere.centers, centers)


def test_selection_rule_uses_tie_threshold_and_precommitted_order() -> None:
    selector = _load_script("select_episodic_endpoint")
    selected, tied = selector.choose({"adapt_h": 80.00, "adapt_h2": 78.0, "adapt_s": 80.05, "adapt_s2": 79.0})
    assert tied == ["adapt_h", "adapt_s"] and selected == "adapt_h"
    selected, tied = selector.choose({"adapt_h": 79.76, "adapt_h2": 78.29, "adapt_s": 80.13, "adapt_s2": 79.05})
    assert tied == ["adapt_s"] and selected == "adapt_s"
