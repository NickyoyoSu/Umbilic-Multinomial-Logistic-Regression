"""Training loop shared by all image-level experiments.

Every epoch trains on the training loader and evaluates on the evaluation
loader. ``metrics.json`` records both ``best_val_accuracy`` (the maximum over
epochs, whose weights are saved as ``best.pt``) and ``final_val_accuracy``
(the last epoch, whose weights are saved as ``last.pt``), together with the
per-epoch history.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm


@dataclass
class EpochMetrics:
    split: str
    loss: float
    accuracy: float


def _run_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    max_batches: int | None,
    train: bool,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
) -> EpochMetrics:
    model.train(mode=train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, (images, labels) in enumerate(tqdm(loader, leave=False)):
            if max_batches is not None and step >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            autocast_enabled = use_amp and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                features = model.forward_features(images)
                if train and getattr(model.head, "requires_target_labels", False):
                    logits = model.head(features, labels)
                else:
                    logits = model.head(features)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

    average_loss = total_loss / max(total_count, 1)
    accuracy = total_correct / max(total_count, 1)
    return EpochMetrics(split="train" if train else "val", loss=average_loss, accuracy=accuracy)


def _build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    head_lr_factor: float = 1.0,
    radius_lr_factor: float = 1.0,
):
    """Backbone, head, and radius parameter groups.

    The head learning rate is ``lr * head_lr_factor`` and the radius learning
    rate is ``lr * head_lr_factor * radius_lr_factor``.
    """
    if head_lr_factor <= 0.0:
        raise ValueError(f"head_lr_factor must be positive, got {head_lr_factor}.")
    if radius_lr_factor <= 0.0:
        raise ValueError(f"radius_lr_factor must be positive, got {radius_lr_factor}.")

    head_module = getattr(model, "head", nn.Module())
    radius_param_ids = {
        id(param)
        for name, param in head_module.named_parameters()
        if name == "raw_radius" or name.endswith(".raw_radius")
    }
    head_param_ids = {id(param) for param in head_module.parameters()}
    backbone_params = [
        param for param in model.parameters() if param.requires_grad and id(param) not in head_param_ids
    ]
    radius_params = [
        param for param in head_module.parameters() if param.requires_grad and id(param) in radius_param_ids
    ]
    head_params = [
        param for param in head_module.parameters() if param.requires_grad and id(param) not in radius_param_ids
    ]
    if not backbone_params and not head_params and not radius_params:
        raise ValueError("No trainable parameters are available for the optimizer.")
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr * head_lr_factor, "name": "head"})
    if radius_params:
        param_groups.append(
            {"params": radius_params, "lr": lr * head_lr_factor * radius_lr_factor, "name": "radius"}
        )

    normalized = optimizer_name.lower()
    if normalized == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    if normalized == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_scheduler(
    optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
):
    """Linear warmup followed by cosine decay to ``min_lr`` (stepped per epoch)."""
    normalized = scheduler_name.lower()
    if normalized == "none":
        return None
    if normalized != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    if base_lr <= 0:
        raise ValueError("Base learning rate must be positive when using cosine scheduling.")

    min_factor = min_lr / base_lr

    def lr_lambda(epoch_index: int) -> float:
        epoch = epoch_index + 1
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            return epoch / warmup_epochs
        cosine_epochs = max(1, epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / cosine_epochs
        progress = min(max(progress, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _head_epoch_diagnostics(head: nn.Module) -> dict[str, float | bool]:
    """Compact per-epoch scale/radius statistics for stability audits."""
    diagnostics: dict[str, float | bool] = {}
    for module_name, module in head.named_modules():
        prefix = module_name.replace(".", "_") or "head"
        for value_name in ("alpha_values", "radius_values"):
            getter = getattr(module, value_name, None)
            if not callable(getter):
                continue
            values = getter()
            if values is None:
                continue
            values = values.detach().float()
            key = f"{prefix}_{value_name.removesuffix('_values')}"
            finite = torch.isfinite(values)
            diagnostics[f"{key}_all_finite"] = bool(finite.all().item())
            if bool(finite.any().item()):
                finite_values = values[finite]
                diagnostics[f"{key}_mean"] = float(finite_values.mean().item())
                diagnostics[f"{key}_std"] = float(finite_values.std(unbiased=False).item())
                diagnostics[f"{key}_min"] = float(finite_values.min().item())
                diagnostics[f"{key}_max"] = float(finite_values.max().item())
            else:
                for stat in ("mean", "std", "min", "max"):
                    diagnostics[f"{key}_{stat}"] = math.nan
    return diagnostics


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    momentum: float,
    scheduler_name: str,
    warmup_epochs: int,
    min_lr: float,
    label_smoothing: float,
    use_amp: bool,
    max_train_batches: int | None,
    max_val_batches: int | None,
    output_dir: str | Path,
    run_config: dict,
    head_lr_factor: float = 1.0,
    radius_lr_factor: float = 1.0,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    model.to(device)
    history: list[dict] = []
    best_val_acc = -1.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    optimizer = _build_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        head_lr_factor=head_lr_factor,
        radius_lr_factor=radius_lr_factor,
    )
    scheduler = _build_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        epochs=epochs,
        warmup_epochs=min(warmup_epochs, epochs),
        min_lr=min_lr,
        base_lr=lr,
    )

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, optimizer, device, max_train_batches,
            train=True, criterion=criterion, scaler=scaler, use_amp=use_amp,
        )
        val_metrics = _run_epoch(
            model, val_loader, optimizer, device, max_val_batches,
            train=False, criterion=criterion, scaler=scaler, use_amp=use_amp,
        )
        history.append(
            {
                "epoch": epoch,
                **asdict(train_metrics),
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "lr": optimizer.param_groups[0]["lr"],
                **_head_epoch_diagnostics(model.head),
            }
        )
        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            torch.save(model.state_dict(), output_path / "best.pt")
        if scheduler is not None:
            scheduler.step()

    torch.save(model.state_dict(), output_path / "last.pt")
    metrics = {
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": history[-1]["val_accuracy"] if history else None,
        "history": history,
        "config": run_config,
    }
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    return metrics
