"""ImageNet-pretrained torchvision backbones with the classifier removed."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, ViT_B_16_Weights, convnext_tiny, vit_b_16

TORCHVISION_URL = "https://github.com/pytorch/vision"
BACKBONE_REFS = {
    "vit_b_16": "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html",
    "convnext_tiny": "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html",
}
BACKBONES = tuple(BACKBONE_REFS)


@dataclass(frozen=True)
class BackboneBundle:
    name: str
    model: nn.Module
    feature_dim: int
    image_size: int
    source_url: str
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_backbone(name: str, pretrained: bool = True) -> BackboneBundle:
    """Build ViT-B/16 or ConvNeXt-Tiny with ``torchvision`` DEFAULT weights."""
    normalized = name.lower()
    if normalized == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        feature_dim = model.heads.head.in_features
        model.heads = nn.Identity()
    elif normalized == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        feature_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name}. Choose from {BACKBONES}.")
    return BackboneBundle(normalized, model, feature_dim, 224, BACKBONE_REFS[normalized])
