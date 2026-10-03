"""Build a classifier head from `scripts/train.py` arguments or a saved run config."""

from __future__ import annotations

from types import SimpleNamespace

from .heads import AngularMarginHead, HyperplaneHead, PrototypeDistanceHead
from .umlr_head import MixedHyperplaneSphereHead, SphereUMLRHead


def build_head(args, in_features: int, num_classes: int):
    """Accept an `argparse.Namespace` or a run-config dict."""
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    if args.head_type in {"cosface", "arcface"}:
        return AngularMarginHead(
            in_features,
            num_classes,
            method=args.head_type,
            scale=args.angular_margin_scale,
            margin=args.angular_margin,
        )
    if args.head_type == "hyperplane":
        return HyperplaneHead(
            in_features,
            num_classes,
            score_mode=args.hyperplane_score_mode,
            alpha_scope=args.hyperplane_alpha_scope,
            alpha_init_effective=args.alpha_init_effective or 1.0,
        )
    if args.head_type == "prototype":
        return PrototypeDistanceHead(in_features, num_classes, distance_power=args.prototype_distance_power)
    if args.head_type == "sphere":
        return SphereUMLRHead(
            in_features,
            num_classes,
            parameterization=args.sphere_parameterization,
            alpha_mode=args.umlr_alpha_mode,
            boundary_mode=args.sphere_boundary_mode,
            alpha_scope=args.alpha_scope,
            alpha_init_effective=args.alpha_init_effective,
            radius_scope=args.radius_scope,
            radius_init_effective=args.radius_init_effective,
            fixed_radius=args.fixed_radius,
        )
    if args.head_type == "mixed":
        return MixedHyperplaneSphereHead(
            in_features,
            num_classes,
            parameterization=args.sphere_parameterization,
            alpha_mode=args.umlr_alpha_mode,
            boundary_mode=args.sphere_boundary_mode,
            gate_mode=args.gate_mode,
            gate_type=args.gate_type,
            gate_init_ratio=args.gate_init_ratio,
            hyperplane_score_mode=args.hyperplane_score_mode,
        )
    raise ValueError(f"Unsupported head type: {args.head_type}")
