"""The class-coordinate heads implement the formulas of the paper."""

from __future__ import annotations

import math

import pytest
import torch

from umlr import AngularMarginHead, HyperplaneHead, MixedHyperplaneSphereHead, PrototypeDistanceHead, SphereUMLRHead
from umlr.umlr_head import _inverse_softplus


def _signed_square(z: torch.Tensor) -> torch.Tensor:
    return torch.sign(z) * z.square()


def test_hyperplane_linear_is_affine_logit() -> None:
    torch.manual_seed(0)
    head = HyperplaneHead(5, 3)
    x = torch.randn(4, 5)
    torch.testing.assert_close(head(x), x @ head.linear.weight.T + head.linear.bias)


def test_hyperplane_signed_square_with_classwise_scale() -> None:
    torch.manual_seed(0)
    head = HyperplaneHead(5, 3, score_mode="signed_square", alpha_scope="classwise", alpha_init_effective=2.0)
    x = torch.randn(4, 5)
    z = x @ head.linear.weight.T + head.linear.bias
    torch.testing.assert_close(head(x), 2.0 * _signed_square(z), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("parameterization", ["ppt_decoupled_inside", "geometric"])
def test_sphere_first_order_and_signed_square(parameterization: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(6, 4)
    for boundary_mode, link in (("distance", lambda z: z), ("signed_square", _signed_square)):
        head = SphereUMLRHead(
            4, 3, parameterization=parameterization, boundary_mode=boundary_mode, alpha_init_effective=1.5
        )
        with torch.no_grad():
            head.set_radius_values_(torch.tensor([0.5, 1.0, 2.0]))
        z = head.radius_values().unsqueeze(0) - torch.cdist(x, head.centers)
        torch.testing.assert_close(head(x), head.alpha_values().unsqueeze(0) * link(z))


def test_signed_square_preserves_zero_level_set_and_side() -> None:
    z = torch.linspace(-2.0, 2.0, 9)
    assert torch.equal(torch.sign(_signed_square(z)), torch.sign(z))
    assert torch.all(torch.diff(_signed_square(z)) > 0)


def test_radius_zero_signed_square_sphere_is_squared_prototype_score() -> None:
    torch.manual_seed(0)
    sphere = SphereUMLRHead(
        4, 3, boundary_mode="signed_square", alpha_scope="fixed", alpha_init_effective=1.0,
        radius_scope="fixed", fixed_radius=0.0,
    )
    x = torch.randn(5, 4)
    torch.testing.assert_close(sphere(x), -torch.cdist(x, sphere.centers).square())
    assert "raw_radius" not in dict(sphere.named_parameters())

    prototype = PrototypeDistanceHead(4, 3, distance_power=2)
    with torch.no_grad():
        prototype.centers.copy_(sphere.centers)
        prototype.log_scale.copy_(_inverse_softplus(torch.ones(3)))
    torch.testing.assert_close(prototype(x), sphere(x), atol=1e-5, rtol=1e-5)


def test_matched_heads_have_k_times_d_plus_two_parameters() -> None:
    d, k = 7, 5
    heads = [
        HyperplaneHead(d, k, score_mode="linear", alpha_scope="classwise"),
        HyperplaneHead(d, k, score_mode="signed_square", alpha_scope="classwise"),
        SphereUMLRHead(d, k, boundary_mode="distance", alpha_init_effective=1.0),
        SphereUMLRHead(d, k, boundary_mode="signed_square", alpha_init_effective=1.0),
    ]
    for head in heads:
        assert sum(p.numel() for p in head.parameters()) == k * (d + 2)


def test_effective_initial_scale_and_radius() -> None:
    head = SphereUMLRHead(3, 4, alpha_init_effective=10.0, radius_init_effective=0.8)
    torch.testing.assert_close(head.alpha_values(), torch.full((4,), 10.0), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(head.radius_values(), torch.full((4,), 0.8), atol=1e-5, rtol=0.0)


def test_shared_and_fixed_radius_setters_average_class_values() -> None:
    for scope in ("shared", "fixed"):
        head = SphereUMLRHead(4, 3, radius_scope=scope)
        head.set_radius_values_(torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(head.radius_values(), torch.full((3,), 2.0), atol=2e-6, rtol=0.0)


def test_mixed_hard_gate_selects_one_endpoint_with_straight_through_gradient() -> None:
    torch.manual_seed(0)
    head = MixedHyperplaneSphereHead(4, 3, boundary_mode="signed_square", gate_type="hard", gate_init_ratio=0.5)
    with torch.no_grad():
        head.raw_gate.copy_(torch.tensor([2.0, -2.0, 0.0]))
    x = torch.randn(5, 4)
    hyperplane = torch.nn.functional.linear(x, head.hyper_weight, head.hyper_bias)
    sphere = head.sphere_head(x)
    output = head(x)
    torch.testing.assert_close(output[:, 0], hyperplane[:, 0])
    torch.testing.assert_close(output[:, 1], sphere[:, 1])
    torch.testing.assert_close(output[:, 2], hyperplane[:, 2])  # sigmoid(0) = 0.5 selects the hyperplane
    output.sum().backward()
    assert head.raw_gate.grad is not None and torch.any(head.raw_gate.grad != 0)


def test_angular_margin_only_changes_target_logits_during_training() -> None:
    torch.manual_seed(0)
    x = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    for method, margin in (("cosface", 0.35), ("arcface", 0.35)):
        head = AngularMarginHead(4, 3, method=method, scale=30.0, margin=margin)
        evaluation = head(x)
        training = head(x, labels)
        mask = torch.ones_like(evaluation, dtype=torch.bool)
        mask[torch.arange(6), labels] = False
        torch.testing.assert_close(training[mask], evaluation[mask])
        assert torch.all(training[~mask] < evaluation[~mask])
        cosine = torch.nn.functional.linear(
            torch.nn.functional.normalize(x, dim=-1), torch.nn.functional.normalize(head.weight, dim=-1)
        )
        torch.testing.assert_close(evaluation, 30.0 * cosine, atol=1e-5, rtol=1e-5)


def test_inverse_softplus_round_trip() -> None:
    values = torch.tensor([1e-3, 0.5, 1.0, 10.0])
    torch.testing.assert_close(torch.nn.functional.softplus(_inverse_softplus(values)), values)
    assert math.isclose(float(torch.nn.functional.softplus(_inverse_softplus(torch.tensor(2.0)))), 2.0, rel_tol=1e-6)
