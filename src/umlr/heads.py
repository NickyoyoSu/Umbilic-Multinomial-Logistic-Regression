"""Hyperplane (UMLR-H / UMLR^2-H), prototype, and angular-margin heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus_scalar(value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"Positive scale initialization must be > 0, got {value}.")
    return math.log(math.expm1(value))


class HyperplaneHead(nn.Module):
    """Affine class coordinate ``<a_k, x> + b_k`` with an optional link and scale.

    ``score_mode="linear"`` is ordinary MLR (UMLR-H). ``score_mode="signed_square"``
    applies ``z * |z|`` (UMLR^2-H). ``alpha_scope`` adds an explicit positive
    scale after the link; ``"classwise"`` is used for the parameter-matched
    comparisons, where H, H^2, S, and S^2 all have ``K * (d + 2)`` parameters.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        use_bias: bool = True,
        score_mode: str = "linear",
        alpha_scope: str = "none",
        alpha_init_effective: float = 1.0,
    ) -> None:
        super().__init__()
        if score_mode not in {"linear", "signed_square"}:
            raise ValueError(f"Unsupported hyperplane score_mode: {score_mode}")
        if alpha_scope not in {"none", "fixed", "shared", "classwise"}:
            raise ValueError(f"Unsupported hyperplane alpha_scope: {alpha_scope}")
        if alpha_init_effective <= 0.0:
            raise ValueError("alpha_init_effective must be positive.")

        self.in_features = in_features
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.score_mode = score_mode
        self.alpha_scope = alpha_scope
        self.alpha_init_effective = float(alpha_init_effective)
        self.raw_alpha = None
        self.register_buffer("fixed_alpha", torch.empty(0))

        if alpha_scope == "fixed":
            self.fixed_alpha = torch.tensor([self.alpha_init_effective])
        elif alpha_scope in {"shared", "classwise"}:
            shape = (1,) if alpha_scope == "shared" else (num_classes,)
            self.raw_alpha = nn.Parameter(torch.full(shape, _inverse_softplus_scalar(self.alpha_init_effective)))

        self.linear = nn.Linear(in_features, num_classes, bias=use_bias)
        nn.init.normal_(self.linear.weight, mean=0.0, std=in_features ** -0.5)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def effective_weight(self) -> torch.Tensor:
        return self.linear.weight

    def effective_bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def alpha_values(self) -> torch.Tensor | None:
        if self.alpha_scope == "none":
            return None
        if self.alpha_scope == "fixed":
            return self.fixed_alpha.expand(self.num_classes)
        return (F.softplus(self.raw_alpha) + 1e-6).expand(self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        if self.score_mode == "signed_square":
            logits = logits * logits.abs()
        alpha = self.alpha_values()
        if alpha is not None:
            logits = logits * alpha.unsqueeze(0)
        return logits


class AngularMarginHead(nn.Module):
    """CosFace/ArcFace classifier head with normalized features and weights.

    The target-class margin is applied only when labels are supplied during
    training. Evaluation uses the ordinary scaled cosine logits, matching the
    standard angular-margin protocol.
    """

    requires_target_labels = True

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        method: str,
        scale: float = 30.0,
        margin: float = 0.35,
    ) -> None:
        super().__init__()
        if method not in {"cosface", "arcface"}:
            raise ValueError(f"Unsupported angular-margin method: {method}")
        if scale <= 0.0:
            raise ValueError("Angular-margin scale must be positive.")
        if margin < 0.0:
            raise ValueError("Angular-margin margin must be non-negative.")
        self.in_features = in_features
        self.num_classes = num_classes
        self.method = method
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None or self.margin == 0.0:
            return self.scale * cosine

        if labels.ndim != 1 or labels.shape[0] != cosine.shape[0]:
            raise ValueError(
                f"Expected one target label per sample, got labels={tuple(labels.shape)}, "
                f"logits={tuple(cosine.shape)}."
            )

        batch_indices = torch.arange(labels.shape[0], device=labels.device)
        target_cosine = cosine[batch_indices, labels]
        if self.method == "cosface":
            target_margin = target_cosine - self.margin
        else:
            sine = torch.sqrt((1.0 - target_cosine.square()).clamp_min(1e-7))
            phi = target_cosine * math.cos(self.margin) - sine * math.sin(self.margin)
            threshold = math.cos(math.pi - self.margin)
            correction = math.sin(math.pi - self.margin) * self.margin
            target_margin = torch.where(target_cosine > threshold, phi, target_cosine - correction)

        adjusted = cosine.clone()
        adjusted[batch_indices, labels] = target_margin
        return self.scale * adjusted


class PrototypeDistanceHead(nn.Module):
    """Learned-center distance head ``-s_k * ||x - c_k||^p + beta_k``.

    ``distance_power=2`` is the squared-distance prototype score;
    ``distance_power=1`` uses the Euclidean distance.
    """

    def __init__(self, in_features: int, num_classes: int, distance_power: int = 1) -> None:
        super().__init__()
        if distance_power not in {1, 2}:
            raise ValueError(f"distance_power must be 1 or 2, got {distance_power}.")
        self.centers = nn.Parameter(torch.randn(num_classes, in_features) * 0.02)
        self.log_scale = nn.Parameter(torch.zeros(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.distance_power = distance_power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(x, self.centers, p=2)
        if self.distance_power == 2:
            distances = distances.square()
        scale = F.softplus(self.log_scale).unsqueeze(0)
        return -scale * distances + self.bias.unsqueeze(0)
