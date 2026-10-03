#!/usr/bin/env python3
"""Aggregate ``scripts/train.py`` runs into per-configuration tables.

Runs are grouped by dataset, backbone, regime (shots or full data, frozen or
fine-tuned backbone, class imbalance), and head configuration; seeds are
aggregated within each group. For every group the table reports the number of
seeds and the mean +- sample standard deviation of both ``best_val_accuracy``
(maximum over epochs) and ``final_val_accuracy`` (last epoch), in percent.

Optional columns are filled when the corresponding files exist:
``classwise_diagnostics.json`` (macro and few-shot-third accuracy, radius vs.
class-frequency Spearman correlation) and ``head_diagnostics.json`` (learned
radius range, finiteness, and hard-gate counts of mixed heads).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-root", nargs="+", required=True, type=Path)
    parser.add_argument("--run-tag", help="Keep only runs whose config has this run_tag.")
    parser.add_argument("--table-csv", type=Path)
    parser.add_argument("--report-md", type=Path)
    return parser.parse_args()


def head_label(config: dict) -> str:
    head = config["head_type"]
    if head == "hyperplane":
        label = "UMLR^2-H" if config.get("hyperplane_score_mode") == "signed_square" else "UMLR-H"
        if config.get("hyperplane_alpha_scope", "none") != "none":
            label += f" [alpha {config['hyperplane_alpha_scope']}]"
        return label
    if head == "sphere":
        label = "UMLR^2-S" if config.get("sphere_boundary_mode") == "signed_square" else "UMLR-S"
        details = []
        if config.get("radius_scope") == "fixed":
            radius = config.get("radius_init_effective")
            details.append(f"r={config.get('fixed_radius', 0.0) if radius is None else radius:g}")
        elif config.get("radius_scope") == "shared":
            details.append("shared r")
        if config.get("freeze_radius"):
            details.append("frozen r")
        if config.get("feature_stat_init") == "centroid_radius":
            details.append(f"q{int(round(100 * float(config.get('feature_stat_radius_quantile', 0.8))))}")
        if float(config.get("radius_lr_factor", 1.0)) != 1.0:
            details.append(f"radius LR x{float(config['radius_lr_factor']):g}")
        if config.get("alpha_init_effective") not in (None, 1.0):
            details.append(f"alpha0={float(config['alpha_init_effective']):g}")
        return label + (f" [{', '.join(details)}]" if details else "")
    if head == "mixed":
        label = "Mix-H^2" if config.get("hyperplane_score_mode") == "signed_square" else "Mix-H"
        return f"{label} [gate0={float(config.get('gate_init_ratio', 0.5)):g}]"
    if head == "prototype":
        return f"Prototype [p={int(config.get('prototype_distance_power', 1))}]"
    return {"cosface": "CosFace", "arcface": "ArcFace"}.get(head, head)


def regime(config: dict) -> str:
    shots = int(config.get("train_shots_per_class", 0))
    parts = [f"{shots}-shot" if shots > 0 else "full"]
    parts.append("frozen" if config.get("freeze_backbone") else "fine-tuned")
    if config.get("train_imbalance_type", "none") != "none":
        parts.append(f"{config['train_imbalance_type']} imbalance {float(config['train_imbalance_ratio']):g}")
    return ", ".join(parts)


def fmt(values: list[float]) -> str:
    if not values:
        return ""
    sd = stdev(values) if len(values) > 1 else float("nan")
    return f"{mean(values):.2f} +- {sd:.2f}"


def main() -> None:
    args = parse_args()
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for root in args.input_root:
        for path in sorted(root.glob("*/metrics.json")):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            config = metrics["config"]
            if args.run_tag and config.get("run_tag") != args.run_tag:
                continue
            key = (config["dataset"], config["backbone"], regime(config), head_label(config))
            group = groups[key]
            if config["seed"] in group["seeds"]:
                raise RuntimeError(f"Duplicate seed {config['seed']} for {key} ({path})")
            group["seeds"].append(config["seed"])
            group["best"].append(100.0 * float(metrics["best_val_accuracy"]))
            group["final"].append(100.0 * float(metrics["final_val_accuracy"]))
            classwise_path = path.parent / "classwise_diagnostics.json"
            if classwise_path.exists():
                classwise = json.loads(classwise_path.read_text(encoding="utf-8"))
                group["macro"].append(100.0 * float(classwise["macro_accuracy"]))
                few = classwise["frequency_thirds_accuracy"].get("few")
                if few is not None:
                    group["few_third"].append(100.0 * float(few))
                if classwise.get("radius_train_count_spearman") is not None:
                    group["spearman"].append(float(classwise["radius_train_count_spearman"]))
            head_path = path.parent / "head_diagnostics.json"
            if head_path.exists():
                head = json.loads(head_path.read_text(encoding="utf-8"))
                for name, entry in head.items():
                    if name.endswith("radius_values"):
                        group["radius_min"].append(float(entry["min"]))
                        group["radius_max"].append(float(entry["max"]))
                        group["radius_mean"].append(float(entry["mean"]))
                        group["radius_finite"].append(bool(entry["all_finite"]))
                    if name.endswith("gate_probabilities"):
                        group["gate_h"].append(int(entry["hard_hyperplane_count"]))
                        group["gate_s"].append(int(entry["hard_sphere_count"]))

    rows = []
    for (dataset, backbone, run_regime, label), group in sorted(groups.items()):
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "regime": run_regime,
                "head": label,
                "seeds": " ".join(str(s) for s in sorted(group["seeds"])),
                "best_val_accuracy": fmt(group["best"]),
                "final_val_accuracy": fmt(group["final"]),
                "macro_accuracy": fmt(group["macro"]),
                "few_third_accuracy": fmt(group["few_third"]),
                "radius_frequency_spearman": f"{mean(group['spearman']):.3f}" if group["spearman"] else "",
                "radius_mean": f"{mean(group['radius_mean']):.4f}" if group["radius_mean"] else "",
                "radius_range": (
                    f"[{min(group['radius_min']):.4f}, {max(group['radius_max']):.4f}]" if group["radius_min"] else ""
                ),
                "radius_all_finite": str(all(group["radius_finite"])) if group["radius_finite"] else "",
                "hard_gates_H/S": f"{sum(group['gate_h'])}/{sum(group['gate_s'])}" if group["gate_h"] else "",
            }
        )
    if not rows:
        raise SystemExit("No runs found.")
    columns = [c for c in rows[0] if any(row[c] for row in rows)]
    lines = [
        "# Run summary",
        "",
        "Accuracies are mean +- sample standard deviation over seeds (%).",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    lines += ["| " + " | ".join(row[c] for c in columns) + " |" for row in rows]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(report, encoding="utf-8")
    if args.table_csv:
        args.table_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.table_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
