"""Backbone + class-coordinate head wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryClassifier(nn.Module):
    """Pretrained backbone followed by a UMLR, prototype, or angular-margin head.

    Features can be centered by a stored feature mean (``center_features``) and
    L2-normalized (``normalize_features``) before the head; with both enabled
    this is the CL2N preprocessing used for the low-shot experiments.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        normalize_features: bool = False,
        center_features: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.normalize_features = normalize_features
        self.center_features = center_features
        self.register_buffer("feature_center", torch.empty(0), persistent=True)

    def set_feature_center(self, center: torch.Tensor) -> None:
        self.feature_center = center.detach().clone()

    def _transform_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.center_features:
            if self.feature_center.numel() == 0:
                raise RuntimeError("center_features=True but feature_center has not been initialized.")
            center = self.feature_center.to(device=features.device, dtype=features.dtype)
            features = features - center.view(*([1] * (features.ndim - 1)), -1)
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return self._transform_features(features)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        features = self.forward_features(x)
        if labels is not None and getattr(self.head, "requires_target_labels", False):
            return self.head(features, labels)
        return self.head(features)
