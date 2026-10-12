#!/usr/bin/env python3
"""Train a UMLR, prototype, or angular-margin head on a torchvision backbone.

This single entry point covers the image-level experiments of the paper:
frozen-feature low-shot head swaps, full-data fine-tuning, the
parameter-matched H/H^2/S/S^2 study, radius controls, frozen-versus-fine-tuned
controls, long-tail controls, and the exploratory mixed heads. See
``experiments/`` for the exact argument sets.

Each run writes ``config.json``, ``metrics.json`` (``best_val_accuracy``,
``final_val_accuracy``, per-epoch history), ``head_diagnostics.json``,
``best.pt``, and ``last.pt`` to ``<output-root>/<run-name>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import torch

from umlr.backbones import BACKBONES, TORCHVISION_URL, build_backbone
from umlr.classifier import BoundaryClassifier
from umlr.data import build_loaders, class_counts_for_dataset
from umlr.factory import build_head
from umlr.training import train_model
from umlr.umlr_head import MixedHyperplaneSphereHead, SphereUMLRHead

DATASETS = ("cifar100", "tiny_imagenet", "flowers102", "food101", "oxford_iiit_pet")
HEAD_TYPES = ("hyperplane", "sphere", "mixed", "prototype", "cosface", "arcface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--head-type", required=True, choices=HEAD_TYPES)

    head = parser.add_argument_group("heads")
    head.add_argument(
        "--hyperplane-score-mode",
        default="linear",
        choices=["linear", "signed_square"],
        help="linear = UMLR-H (ordinary MLR); signed_square = UMLR^2-H. Also the hyperplane side of mixed heads.",
    )
    head.add_argument(
        "--hyperplane-alpha-scope",
        default="none",
        choices=["none", "fixed", "shared", "classwise"],
        help="Optional positive logit scale after the hyperplane link (classwise in the parameter-matched study).",
    )
    head.add_argument(
        "--sphere-parameterization",
        default="ppt_decoupled_inside",
        choices=["ppt_decoupled_inside", "geometric"],
        help="Both names give the inside-positive logit alpha_k * (r_k - ||x - c_k||).",
    )
    head.add_argument(
        "--sphere-boundary-mode",
        default="distance",
        choices=["distance", "signed_square"],
        help="distance = UMLR-S; signed_square = UMLR^2-S.",
    )
    head.add_argument("--umlr-alpha-mode", default="softplus", choices=["softplus", "identity"])
    head.add_argument("--alpha-scope", default="classwise", choices=["fixed", "shared", "classwise"])
    head.add_argument("--alpha-init-effective", type=float, help="Initial effective scale alpha_k.")
    head.add_argument("--radius-scope", default="classwise", choices=["fixed", "shared", "classwise"])
    head.add_argument("--radius-init-effective", type=float, help="Initial effective radius r_k.")
    head.add_argument(
        "--fixed-radius",
        type=float,
        default=0.0,
        help="Radius used with --radius-scope fixed when --radius-init-effective is not given (0 = prototype limit).",
    )
    head.add_argument("--freeze-radius", action="store_true", help="Keep the initialized radii fixed.")
    head.add_argument(
        "--prototype-distance-power",
        type=int,
        default=1,
        choices=[1, 2],
        help="Prototype head: 1 = -s*||x-c|| + b, 2 = -s*||x-c||^2 + b.",
    )
    head.add_argument("--angular-margin-scale", type=float, default=30.0)
    head.add_argument("--angular-margin", type=float, default=0.35)
    head.add_argument("--gate-mode", default="classwise", choices=["classwise", "shared"])
    head.add_argument("--gate-type", default="soft", choices=["soft", "hard"])
    head.add_argument("--gate-init-ratio", type=float, default=0.5, help="Initial hyperplane gate probability.")

    init = parser.add_argument_group("feature statistics")
    init.add_argument(
        "--feature-stat-init",
        default="none",
        choices=["none", "centroid_radius"],
        help="centroid_radius: set sphere centers to class feature means and radii to a within-class distance statistic.",
    )
    init.add_argument("--feature-stat-radius-mode", default="quantile", choices=["quantile", "mean", "mean_std"])
    init.add_argument("--feature-stat-radius-quantile", type=float, default=0.8)
    init.add_argument("--feature-stat-radius-std-mult", type=float, default=1.0)
    init.add_argument("--feature-stat-max-batches", type=int)
    init.add_argument(
        "--feature-stat-centers-only",
        action="store_true",
        help="Initialize only the sphere centers from class means and keep the configured radius "
        "(used for the zero-radius limit: --radius-scope fixed --fixed-radius 0).",
    )
    init.add_argument("--feature-normalization", action="store_true", help="L2-normalize features before the head.")
    init.add_argument(
        "--feature-centering",
        default="none",
        choices=["none", "train_mean"],
        help="Subtract the training-set feature mean before optional L2 normalization (CL2N with both).",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--download", action="store_true")
    data.add_argument("--train-augmentation", default="basic", choices=["basic", "resnet_repro", "vit_repro"])
    data.add_argument("--randaugment-num-ops", type=int, default=0)
    data.add_argument("--randaugment-magnitude", type=int, default=0)
    data.add_argument("--input-size", type=int, help="Defaults to the backbone input size (224).")
    data.add_argument("--train-shots-per-class", type=int, default=0, help="0 = all training images.")
    data.add_argument("--train-imbalance-type", default="none", choices=["none", "exp", "step"])
    data.add_argument("--train-imbalance-ratio", type=float, default=1.0, help="Minimum/maximum class-count ratio.")
    data.add_argument("--num-workers", type=int, default=4)

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--pretrained", action="store_true", help="Load ImageNet-pretrained torchvision weights.")
    optim.add_argument("--freeze-backbone", action="store_true")
    optim.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    optim.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    optim.add_argument("--momentum", type=float, default=0.9)
    optim.add_argument("--warmup-epochs", type=int, default=0)
    optim.add_argument("--min-lr", type=float, default=0.0)
    optim.add_argument("--label-smoothing", type=float, default=0.0)
    optim.add_argument("--epochs", type=int, default=1)
    optim.add_argument("--batch-size", type=int, default=64)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--head-lr-factor", type=float, default=1.0, help="LR multiplier for head parameters.")
    optim.add_argument("--radius-lr-factor", type=float, default=1.0, help="Extra LR multiplier for radii.")
    optim.add_argument("--weight-decay", type=float, default=1e-4)
    optim.add_argument("--seed", type=int, default=0)
    optim.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    optim.add_argument("--amp", dest="amp", action="store_true")
    optim.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=False)

    out = parser.add_argument_group("output")
    out.add_argument("--output-root", default="outputs")
    out.add_argument("--run-tag", help="Appended to the run name; used by the analysis scripts to group runs.")
    out.add_argument("--run-name-stem", help="Optional short run-name stem.")
    out.add_argument("--max-train-batches", type=int, help="Debug: limit training batches per epoch.")
    out.add_argument("--max-val-batches", type=int, help="Debug: limit evaluation batches per epoch.")
    out.add_argument(
        "--write-classwise-diagnostics",
        action="store_true",
        help="Evaluate best.pt per class and relate learned radii to class frequency.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _get_sphere_head(model: BoundaryClassifier) -> SphereUMLRHead | None:
    head = model.head
    if isinstance(head, SphereUMLRHead):
        return head
    if isinstance(head, MixedHyperplaneSphereHead):
        return head.sphere_head
    return None


def apply_feature_centering(model: BoundaryClassifier, train_loader, device: torch.device, args) -> dict[str, object]:
    """Estimate the training-set feature mean (before normalization) and store it in the model."""
    if args.feature_centering == "none":
        return {"feature_centering_applied": False}

    model.to(device)
    model.eval()
    previous_normalize = model.normalize_features
    previous_center = model.center_features
    model.normalize_features = False
    model.center_features = False

    feature_sum: torch.Tensor | None = None
    feature_count = 0
    with torch.no_grad():
        for images, _labels in train_loader:
            images = images.to(device, non_blocking=True)
            features = model.forward_features(images).detach().float()
            if features.ndim == 3:
                features = features.reshape(-1, features.shape[-1])
            elif features.ndim != 2:
                features = features.reshape(features.shape[0], -1)
            if feature_sum is None:
                feature_sum = torch.zeros(features.shape[-1], device=device, dtype=torch.float32)
            feature_sum.add_(features.sum(dim=0))
            feature_count += int(features.shape[0])

    model.normalize_features = previous_normalize
    model.center_features = previous_center
    if feature_sum is None or feature_count == 0:
        raise RuntimeError("Could not estimate train feature mean; no features were observed.")

    center = feature_sum / float(feature_count)
    model.set_feature_center(center.to(device=device, dtype=torch.float32))
    return {
        "feature_centering_applied": True,
        "feature_centering_kind": "train_mean",
        "feature_centering_count": feature_count,
        "feature_centering_norm": float(torch.linalg.norm(center).item()),
    }


def apply_feature_stat_init(
    model: BoundaryClassifier,
    train_loader,
    num_classes: int,
    device: torch.device,
    args,
) -> dict[str, object]:
    """Initialize sphere centers at class feature means and radii at a within-class distance statistic.

    For mixed heads the sphere scale is initialized from the hyperplane weight norms.
    """
    if args.feature_stat_init == "none":
        return {"feature_stat_init_applied": False}

    sphere_head = _get_sphere_head(model)
    if sphere_head is None:
        raise ValueError("--feature-stat-init requires --head-type sphere or mixed.")
    if sphere_head.radius_values() is None:
        raise ValueError("--feature-stat-init requires a sphere head with radii.")
    if args.feature_stat_radius_mode == "quantile" and not 0.0 < args.feature_stat_radius_quantile < 1.0:
        raise ValueError("--feature-stat-radius-quantile must be in (0, 1).")

    model.to(device)
    model.eval()

    feature_dim = sphere_head.in_features
    counts = torch.zeros(num_classes, device=device)
    sums = torch.zeros(num_classes, feature_dim, device=device)
    with torch.no_grad():
        for step, (images, labels) in enumerate(train_loader):
            if args.feature_stat_max_batches is not None and step >= args.feature_stat_max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            features = model.forward_features(images)
            sums.index_add_(0, labels, features)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))

    valid = counts > 0
    centers = torch.zeros_like(sums)
    centers[valid] = sums[valid] / counts[valid].unsqueeze(1)

    per_class_distances: list[list[torch.Tensor]] = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for step, (images, labels) in enumerate(train_loader):
            if args.feature_stat_max_batches is not None and step >= args.feature_stat_max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            features = model.forward_features(images)
            distances = torch.linalg.norm(features - centers[labels], dim=1)
            for class_id in labels.unique().tolist():
                mask = labels == int(class_id)
                per_class_distances[int(class_id)].append(distances[mask].detach().cpu())

    radii = torch.ones(num_classes, device=device)
    for class_id, chunks in enumerate(per_class_distances):
        if not chunks:
            continue
        values = torch.cat(chunks).float()
        if args.feature_stat_radius_mode == "mean":
            radius = values.mean()
        elif args.feature_stat_radius_mode == "mean_std":
            radius = values.mean() + args.feature_stat_radius_std_mult * values.std(unbiased=False)
        else:
            radius = torch.quantile(values, args.feature_stat_radius_quantile)
        radii[class_id] = radius.to(device).clamp_min(1e-3)

    with torch.no_grad():
        sphere_head.set_centers_(centers)
        if not args.feature_stat_centers_only:
            sphere_head.set_radius_values_(radii)
        if sphere_head.alpha_values() is not None:
            hyper_weight = getattr(model.head, "hyper_weight", None)
            if hyper_weight is not None:
                alpha = torch.linalg.norm(hyper_weight.detach(), dim=1).to(device).clamp_min(1e-3)
                sphere_head.set_alpha_values_(alpha)

    return {
        "feature_stat_init_applied": True,
        "feature_stat_init_kind": args.feature_stat_init,
        "feature_stat_radius_mode": args.feature_stat_radius_mode,
        "feature_stat_radius_quantile": args.feature_stat_radius_quantile,
        "feature_stat_radius_std_mult": args.feature_stat_radius_std_mult,
        "feature_stat_seen_classes": int(valid.sum().item()),
        "feature_stat_mean_radius": float(radii.mean().item()),
        "feature_stat_min_radius": float(radii.min().item()),
        "feature_stat_max_radius": float(radii.max().item()),
    }


def build_run_name(args: argparse.Namespace) -> tuple[str, dict[str, str | None]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    unique_token = uuid.uuid4().hex[:8]
    meta = {"timestamp": timestamp, "slurm_job_id": slurm_job_id, "unique_token": unique_token}

    if args.run_name_stem:
        run_parts = [args.run_name_stem, f"seed{args.seed}"]
    else:
        run_parts = [args.dataset, args.backbone, args.head_type, "norm" if args.feature_normalization else "raw"]
        if args.feature_centering != "none":
            run_parts.append(f"center{args.feature_centering}")
        if args.head_type in {"hyperplane", "mixed"} and args.hyperplane_score_mode != "linear":
            run_parts.append(f"hyp{args.hyperplane_score_mode}")
        if args.head_type == "hyperplane" and args.hyperplane_alpha_scope != "none":
            run_parts.append(f"halpha{args.hyperplane_alpha_scope}")
        if args.head_type == "prototype" and args.prototype_distance_power != 1:
            run_parts.append(f"protop{args.prototype_distance_power}")
        if args.head_type in {"sphere", "mixed"}:
            run_parts.append(args.sphere_parameterization)
            if args.sphere_boundary_mode != "distance":
                run_parts.append(f"boundary{args.sphere_boundary_mode}")
            if args.umlr_alpha_mode != "softplus":
                run_parts.append(f"alpha{args.umlr_alpha_mode}")
            if args.alpha_scope != "classwise":
                run_parts.append(f"alphascope{args.alpha_scope}")
            if args.radius_scope != "classwise":
                run_parts.append(f"radscope{args.radius_scope}")
            if args.radius_init_effective is not None:
                run_parts.append(f"radinit{args.radius_init_effective:g}")
            if args.freeze_radius:
                run_parts.append("radfrozen")
        if args.head_type == "mixed":
            run_parts.extend([args.gate_mode, args.gate_type, f"ginit{int(round(args.gate_init_ratio * 100)):03d}"])
        if args.feature_stat_init != "none":
            run_parts.extend([args.feature_stat_init, args.feature_stat_radius_mode])
            if args.feature_stat_radius_mode == "quantile":
                run_parts.append(f"q{int(round(args.feature_stat_radius_quantile * 100)):02d}")
            elif args.feature_stat_radius_mode == "mean_std":
                run_parts.append(f"std{args.feature_stat_radius_std_mult:g}")
        if args.freeze_backbone:
            run_parts.append("frozenbackbone")
        elif args.head_lr_factor != 1.0:
            run_parts.append(f"headlr{args.head_lr_factor:g}")
        if args.radius_lr_factor != 1.0:
            run_parts.append(f"radlr{args.radius_lr_factor:g}")
        if args.train_shots_per_class > 0:
            run_parts.append(f"{args.train_shots_per_class}shot")
        if args.train_imbalance_type != "none":
            run_parts.append(f"imb{args.train_imbalance_type}{args.train_imbalance_ratio:g}")
        run_parts.append(f"seed{args.seed}")
    if args.run_tag:
        run_parts.append(args.run_tag)
    if slurm_job_id:
        run_parts.append(f"job{slurm_job_id}")
    run_parts.extend([f"ts{timestamp}", f"uid{unique_token}"])
    return "-".join(run_parts), meta


def collect_head_diagnostics(head: torch.nn.Module) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    for module_name, module in head.named_modules():
        prefix = module_name or "head"
        for value_name in ("alpha_values", "radius_values"):
            getter = getattr(module, value_name, None)
            if getter is None:
                continue
            values = getter()
            if values is None:
                continue
            values = values.detach().float().cpu()
            diagnostics[f"{prefix}.{value_name}"] = {
                "values": values.tolist(),
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "all_finite": bool(torch.isfinite(values).all().item()),
            }
    gate_getter = getattr(head, "gate_probabilities", None)
    if callable(gate_getter):
        gates = gate_getter().detach().float().cpu()
        diagnostics["head.gate_probabilities"] = {
            "values": gates.tolist(),
            "hard_hyperplane_count": int((gates >= 0.5).sum().item()),
            "hard_sphere_count": int((gates < 0.5).sum().item()),
        }
    return diagnostics


def _average_ranks(values: list[float]) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64)
    ranks = torch.empty_like(tensor)
    for value in torch.unique(tensor, sorted=True):
        positions = torch.nonzero(tensor == value, as_tuple=False).flatten()
        lower = int((tensor < value).sum().item())
        upper = lower + positions.numel() - 1
        ranks[positions] = 0.5 * (lower + upper)
    return ranks


@torch.no_grad()
def collect_classwise_diagnostics(
    model: BoundaryClassifier,
    val_loader,
    train_counts: list[int],
    num_classes: int,
    device: torch.device,
    use_amp: bool,
    max_val_batches: int | None,
) -> dict[str, object]:
    """Per-class accuracy, many/medium/few-shot thirds, and radius-frequency Spearman correlation."""
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    totals = torch.zeros(num_classes, dtype=torch.long)
    for batch_index, (images, labels) in enumerate(val_loader):
        if max_val_batches is not None and batch_index >= max_val_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(images)
        predictions = logits.argmax(dim=1).cpu()
        labels_cpu = labels.cpu()
        totals += torch.bincount(labels_cpu, minlength=num_classes)
        correct += torch.bincount(labels_cpu[predictions == labels_cpu], minlength=num_classes)

    per_class = correct.float() / totals.clamp_min(1).float()
    count_tensor = torch.tensor(train_counts, dtype=torch.long)
    ordering = torch.argsort(count_tensor, descending=True)
    chunks = torch.tensor_split(ordering, 3)
    group_accuracy = {
        name: float(per_class[indices].mean().item()) if indices.numel() else None
        for name, indices in zip(("many", "medium", "few"), chunks)
    }
    diagnostics: dict[str, object] = {
        "train_class_counts": train_counts,
        "val_class_totals": totals.tolist(),
        "per_class_accuracy": per_class.tolist(),
        "macro_accuracy": float(per_class.mean().item()),
        "frequency_thirds_accuracy": group_accuracy,
    }

    radius_getter = getattr(model.head, "radius_values", None)
    if radius_getter is not None:
        radii = radius_getter()
        if radii is not None and radii.numel() == num_classes:
            radii_cpu = radii.detach().float().cpu()
            count_ranks = _average_ranks([float(value) for value in train_counts])
            radius_ranks = _average_ranks(radii_cpu.tolist())
            count_centered = count_ranks - count_ranks.mean()
            radius_centered = radius_ranks - radius_ranks.mean()
            denominator = torch.linalg.norm(count_centered) * torch.linalg.norm(radius_centered)
            correlation = (
                float((count_centered @ radius_centered / denominator).item())
                if float(denominator.item()) > 0.0
                else None
            )
            diagnostics["radius_values"] = radii_cpu.tolist()
            diagnostics["radius_train_count_spearman"] = correlation
    return diagnostics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    bundle = build_backbone(args.backbone, pretrained=args.pretrained)
    train_loader, val_loader, num_classes = build_loaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        image_size=args.input_size or bundle.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        seed=args.seed,
        train_augmentation=args.train_augmentation,
        randaugment_num_ops=args.randaugment_num_ops,
        randaugment_magnitude=args.randaugment_magnitude,
        train_shots_per_class=args.train_shots_per_class,
        train_imbalance_type=args.train_imbalance_type,
        train_imbalance_ratio=args.train_imbalance_ratio,
        mean=bundle.mean,
        std=bundle.std,
    )

    head = build_head(args, bundle.feature_dim, num_classes)
    model = BoundaryClassifier(
        backbone=bundle.model,
        head=head,
        normalize_features=args.feature_normalization,
        center_features=args.feature_centering != "none",
    )
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad_(False)
    feature_centering_metadata = apply_feature_centering(model, train_loader, device, args)
    feature_stat_metadata = apply_feature_stat_init(model, train_loader, num_classes, device, args)
    frozen_radius_parameter_count = 0
    if args.freeze_radius:
        for parameter_name, parameter in model.head.named_parameters():
            if parameter_name == "raw_radius" or parameter_name.endswith(".raw_radius"):
                parameter.requires_grad_(False)
                frozen_radius_parameter_count += parameter.numel()

    run_name, run_meta = build_run_name(args)
    output_dir = Path(args.output_root) / run_name

    config = vars(args).copy()
    config["backbone_source"] = bundle.source_url
    config["official_library"] = TORCHVISION_URL
    config["input_mean"] = list(bundle.mean)
    config["input_std"] = list(bundle.std)
    config["git_commit"] = git_commit_hash()
    config["run_name"] = run_name
    config["output_dir"] = str(output_dir)
    config["trainable_parameter_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["head_trainable_parameter_count"] = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    config.update(feature_centering_metadata)
    config.update(feature_stat_metadata)
    config["frozen_radius_parameter_count"] = frozen_radius_parameter_count
    config.update(run_meta)

    metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        momentum=args.momentum,
        scheduler_name=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        label_smoothing=args.label_smoothing,
        use_amp=args.amp,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        output_dir=output_dir,
        run_config=config,
        head_lr_factor=args.head_lr_factor,
        radius_lr_factor=args.radius_lr_factor,
    )
    if args.write_classwise_diagnostics:
        best_path = output_dir / "best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=args.device))
        classwise = collect_classwise_diagnostics(
            model=model,
            val_loader=val_loader,
            train_counts=class_counts_for_dataset(train_loader.dataset, num_classes),
            num_classes=num_classes,
            device=device,
            use_amp=args.amp,
            max_val_batches=args.max_val_batches,
        )
        with (output_dir / "classwise_diagnostics.json").open("w", encoding="utf-8") as handle:
            json.dump(classwise, handle, indent=2)
    with (output_dir / "head_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(collect_head_diagnostics(model.head), handle, indent=2)
    print(
        json.dumps(
            {
                "run_name": run_name,
                "output_dir": str(output_dir),
                "best_val_accuracy": metrics["best_val_accuracy"],
                "final_val_accuracy": metrics["final_val_accuracy"],
            }
        )
    )


if __name__ == "__main__":
    main()
