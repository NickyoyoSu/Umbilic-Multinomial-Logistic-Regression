"""Spherical UMLR heads.

A spherical class coordinate is the inside-positive signed distance to a
class sphere with center ``c_k`` and radius ``r_k``:

    z_k(x) = r_k - ||x - c_k||.

UMLR-S uses the first-order link ``alpha_k * z_k(x)`` and UMLR^2-S uses the
signed-square link ``alpha_k * z_k(x) * |z_k(x)|``. With ``r_k = 0`` and
``alpha_k = 1`` the signed-square logit is the squared-distance prototype
score ``-||x - c_k||^2``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PARAMETERIZATIONS = ("ppt_decoupled_inside", "geometric")
BOUNDARY_MODES = ("distance", "signed_square")
ALPHA_MODES = ("softplus", "identity")
SCOPES = ("fixed", "shared", "classwise")


def _positive(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw) + 1e-6


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1e-6)
    return value + torch.log(-torch.expm1(-value))


def _positive_scalar(raw: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "identity":
        return raw
    if mode == "softplus":
        return _positive(raw)
    raise ValueError(f"Unknown positive scalar mode: {mode}")


def _inverse_positive_scalar(value: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "identity":
        return value
    if mode == "softplus":
        return _inverse_softplus(value.clamp_min(1e-6))
    raise ValueError(f"Unknown positive scalar mode: {mode}")


class SphereUMLRHead(nn.Module):
    """Signed-distance classifier to class sphere boundaries (UMLR-S / UMLR^2-S).

    Args:
        parameterization: ``"ppt_decoupled_inside"`` and ``"geometric"`` are
            two names for the same inside-positive logit
            ``alpha_k * (r_k - ||x - c_k||)``; both are accepted so that the
            configurations used in the paper run unchanged.
        boundary_mode: ``"distance"`` (UMLR-S) or ``"signed_square"`` (UMLR^2-S).
        alpha_mode: map from the raw scale parameter to ``alpha_k``.
        alpha_scope / radius_scope: ``"classwise"`` (one value per class),
            ``"shared"`` (one value for all classes), or ``"fixed"`` (a buffer).
        alpha_init_effective / radius_init_effective: initial effective values.
        fixed_radius: radius used when ``radius_scope="fixed"`` and no
            ``radius_init_effective`` is given; ``0`` gives the prototype limit.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        parameterization: str = "ppt_decoupled_inside",
        alpha_mode: str = "softplus",
        boundary_mode: str = "distance",
        alpha_scope: str = "classwise",
        alpha_init_effective: float | None = None,
        radius_scope: str = "classwise",
        radius_init_effective: float | None = None,
        fixed_radius: float = 0.0,
    ) -> None:
        super().__init__()
        if parameterization not in PARAMETERIZATIONS:
            raise ValueError(f"Unknown sphere parameterization: {parameterization}")
        if alpha_mode not in ALPHA_MODES:
            raise ValueError(f"Unknown sphere alpha_mode: {alpha_mode}")
        if boundary_mode not in BOUNDARY_MODES:
            raise ValueError(f"Unknown sphere boundary_mode: {boundary_mode}")
        if alpha_scope not in SCOPES:
            raise ValueError(f"Unknown alpha_scope: {alpha_scope}")
        if radius_scope not in SCOPES:
            raise ValueError(f"Unknown radius_scope: {radius_scope}")
        if alpha_init_effective is not None and alpha_init_effective <= 0.0:
            raise ValueError("alpha_init_effective must be positive when specified.")
        if radius_init_effective is not None and radius_init_effective < 0.0:
            raise ValueError("radius_init_effective must be non-negative when specified.")
        if fixed_radius < 0.0:
            raise ValueError("fixed_radius must be non-negative.")

        self.in_features = in_features
        self.num_classes = num_classes
        self.parameterization = parameterization
        self.alpha_mode = alpha_mode
        self.radius_mode = "softplus"
        self.boundary_mode = boundary_mode
        self.alpha_scope = alpha_scope
        self.alpha_init_effective = alpha_init_effective
        self.radius_scope = radius_scope
        self.radius_init_effective = radius_init_effective
        self.fixed_radius_value = float(fixed_radius)
        self.register_buffer("fixed_alpha", torch.empty(0))
        self.register_buffer("fixed_radius", torch.empty(0))

        self._centers = nn.Parameter(torch.randn(num_classes, in_features) * 0.02)

        alpha_shape = (1,) if alpha_scope == "shared" else (num_classes,)
        radius_shape = (1,) if radius_scope == "shared" else (num_classes,)

        if alpha_scope == "fixed":
            effective = 1.0 if alpha_init_effective is None else float(alpha_init_effective)
            self.fixed_alpha = torch.tensor([effective])
            self.raw_alpha = None
        elif alpha_init_effective is None:
            init = torch.ones(alpha_shape) if alpha_mode == "identity" else torch.zeros(alpha_shape)
            self.raw_alpha = nn.Parameter(init)
        else:
            target = torch.full(alpha_shape, float(alpha_init_effective))
            self.raw_alpha = nn.Parameter(_inverse_positive_scalar(target, alpha_mode))

        if radius_scope == "fixed":
            effective = float(radius_init_effective) if radius_init_effective is not None else self.fixed_radius_value
            self.fixed_radius = torch.tensor([effective])
            self.raw_radius = None
        elif radius_init_effective is None:
            self.raw_radius = nn.Parameter(torch.ones(radius_shape))
        else:
            target = torch.full(radius_shape, float(radius_init_effective))
            self.raw_radius = nn.Parameter(_inverse_positive_scalar(target, self.radius_mode))

    @property
    def centers(self) -> torch.Tensor:
        return self._centers

    def alpha_values(self) -> torch.Tensor | None:
        if self.fixed_alpha.numel() > 0:
            return self.fixed_alpha.expand(self.num_classes)
        if self.raw_alpha is None:
            return None
        return _positive_scalar(self.raw_alpha, self.alpha_mode).expand(self.num_classes)

    def radius_values(self) -> torch.Tensor | None:
        if self.fixed_radius.numel() > 0:
            return self.fixed_radius.expand(self.num_classes)
        if self.raw_radius is None:
            return None
        return _positive_scalar(self.raw_radius, self.radius_mode).expand(self.num_classes)

    @torch.no_grad()
    def set_alpha_values_(self, values: torch.Tensor) -> None:
        values = values.detach().to(device=self.centers.device, dtype=self.centers.dtype).flatten()
        if values.numel() not in {1, self.num_classes}:
            raise ValueError(f"Expected 1 or {self.num_classes} alpha values, got {values.numel()}.")
        if self.fixed_alpha.numel() > 0:
            self.fixed_alpha.copy_(values.mean().view_as(self.fixed_alpha).to(self.fixed_alpha))
            return
        if self.raw_alpha is None:
            raise ValueError("This sphere head does not use alpha values.")
        target = values.mean().view(1) if self.alpha_scope == "shared" else values.expand(self.num_classes)
        self.raw_alpha.copy_(_inverse_positive_scalar(target, self.alpha_mode).to(self.raw_alpha))

    @torch.no_grad()
    def set_radius_values_(self, values: torch.Tensor) -> None:
        values = values.detach().to(device=self.centers.device, dtype=self.centers.dtype).flatten()
        if values.numel() not in {1, self.num_classes}:
            raise ValueError(f"Expected 1 or {self.num_classes} radius values, got {values.numel()}.")
        if self.fixed_radius.numel() > 0:
            self.fixed_radius.copy_(values.mean().view_as(self.fixed_radius).to(self.fixed_radius))
            return
        if self.raw_radius is None:
            raise ValueError("This sphere head does not use radius values.")
        target = values.mean().view(1) if self.radius_scope == "shared" else values.expand(self.num_classes)
        self.raw_radius.copy_(_inverse_positive_scalar(target, self.radius_mode).to(self.raw_radius))

    def set_centers_(self, centers: torch.Tensor) -> None:
        centers = centers.detach().to(dtype=torch.float32)
        with torch.no_grad():
            self._centers.copy_(centers.to(self._centers.device, dtype=self._centers.dtype))

    def signed_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        """Inside-positive signed distance ``r_k - ||x - c_k||`` for every class."""
        distances = torch.cdist(x, self.centers, p=2)
        return self.radius_values().unsqueeze(0) - distances

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_values().unsqueeze(0)
        signed_delta = self.signed_coordinates(x)
        if self.boundary_mode == "signed_square":
            return alpha * signed_delta * signed_delta.abs()
        return alpha * signed_delta


class MixedHyperplaneSphereHead(nn.Module):
    """Exploratory classwise gate between a hyperplane and a spherical logit.

    ``mu_k = g_k * hyp_k + (1 - g_k) * sph_k``. With ``gate_type="hard"`` the
    forward pass uses ``g_k = 1[sigmoid(gate_k) >= 0.5]`` with a
    straight-through gradient. ``hyperplane_score_mode="signed_square"`` gives
    Mixed-H^2; the spherical side is configured by ``boundary_mode``.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        parameterization: str = "ppt_decoupled_inside",
        alpha_mode: str = "softplus",
        boundary_mode: str = "distance",
        gate_mode: str = "classwise",
        gate_type: str = "soft",
        gate_init_ratio: float = 0.5,
        hyperplane_score_mode: str = "linear",
    ) -> None:
        super().__init__()
        if gate_mode not in {"classwise", "shared"}:
            raise ValueError(f"Unknown gate mode: {gate_mode}")
        if gate_type not in {"soft", "hard"}:
            raise ValueError(f"Unknown gate type: {gate_type}")
        if hyperplane_score_mode not in {"linear", "signed_square"}:
            raise ValueError(f"Unknown hyperplane score mode: {hyperplane_score_mode}")
        if not 0.0 < gate_init_ratio < 1.0:
            raise ValueError(f"gate_init_ratio must be in (0, 1), got {gate_init_ratio}")

        self.hyper_weight = nn.Parameter(torch.randn(num_classes, in_features) * 0.02)
        self.hyper_bias = nn.Parameter(torch.zeros(num_classes))
        self.sphere_head = SphereUMLRHead(
            in_features,
            num_classes,
            parameterization=parameterization,
            alpha_mode=alpha_mode,
            boundary_mode=boundary_mode,
        )
        gate_shape = (num_classes,) if gate_mode == "classwise" else (1,)
        self.gate_mode = gate_mode
        self.gate_type = gate_type
        self.gate_init_ratio = gate_init_ratio
        self.hyperplane_score_mode = hyperplane_score_mode
        init_logit = math.log(gate_init_ratio / (1.0 - gate_init_ratio))
        self.raw_gate = nn.Parameter(torch.full(gate_shape, init_logit))

    def gate_probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gate.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hyper_logits = F.linear(x, self.hyper_weight, self.hyper_bias)
        if self.hyperplane_score_mode == "signed_square":
            hyper_logits = hyper_logits * hyper_logits.abs()
        sphere_logits = self.sphere_head(x)
        gate = torch.sigmoid(self.raw_gate)
        if self.gate_type == "hard":
            hard_gate = (gate >= 0.5).to(gate.dtype)
            gate = gate + (hard_gate - gate).detach()
        if self.gate_mode == "shared":
            gate = gate.expand_as(hyper_logits)
        else:
            gate = gate.unsqueeze(0).expand_as(hyper_logits)
        return gate * hyper_logits + (1.0 - gate) * sphere_logits
