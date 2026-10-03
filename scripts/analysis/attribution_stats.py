#!/usr/bin/env python3
"""Paired statistics for the parameter-matched H/H^2/S/S^2 study and radius controls.

``--study shot_curve`` (Table 3): for every shot count, the mean accuracy of
each head over datasets, backbones, and seeds, and the paired effects S-H,
S^2-H^2, H^2-H, S^2-S, and their interaction. Each effect reports the mean,
a 95% hierarchical-bootstrap interval (resampling dataset-backbone cells, then
paired seeds within each cell), a cluster-level t-test, and an exact
cluster sign-flip test.

``--study radius``: UMLR^2-S with r=0, a shared radius, a fixed q80 radius, and
a learned q80 radius; reports the paired differences against r=0 and
learned-minus-fixed.

Runs are read from ``*/metrics.json`` (``scripts/train.py`` outputs) under the
given roots. ``--metric`` selects ``best_val_accuracy`` (maximum over epochs)
or ``final_val_accuracy`` (last epoch).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SHOT_CURVE_VARIANTS = ("H", "H2", "S", "S2")
SHOT_CURVE_EFFECTS = {
    "S-H": lambda row: row["S"] - row["H"],
    "S2-H2": lambda row: row["S2"] - row["H2"],
    "H2-H": lambda row: row["H2"] - row["H"],
    "S2-S": lambda row: row["S2"] - row["S"],
    "interaction": lambda row: (row["S2"] - row["S"]) - (row["H2"] - row["H"]),
}
RADIUS_VARIANTS = ("r0", "shared", "fixed_q80", "learned_q80")
RADIUS_EFFECTS = {
    "shared-r0": lambda row: row["shared"] - row["r0"],
    "fixed_q80-r0": lambda row: row["fixed_q80"] - row["r0"],
    "learned_q80-r0": lambda row: row["learned_q80"] - row["r0"],
    "learned_q80-fixed_q80": lambda row: row["learned_q80"] - row["fixed_q80"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", choices=["shot_curve", "radius"], required=True)
    parser.add_argument("--input-root", nargs="+", required=True, type=Path)
    parser.add_argument("--metric", choices=["best_val_accuracy", "final_val_accuracy"], default="best_val_accuracy")
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output-dir", type=Path, help="Write paired_runs.csv, effects.csv, and SUMMARY.md here.")
    return parser.parse_args()


def shot_curve_variant(config: dict) -> str:
    if config["head_type"] == "hyperplane":
        return "H2" if config["hyperplane_score_mode"] == "signed_square" else "H"
    if config["head_type"] == "sphere":
        return "S2" if config["sphere_boundary_mode"] == "signed_square" else "S"
    raise ValueError(f"Unexpected head type for the shot-curve study: {config['head_type']}")


def radius_variant(config: dict) -> str:
    if config["head_type"] != "sphere":
        raise ValueError("The radius study uses sphere heads only.")
    if config["radius_scope"] == "fixed":
        radius = config.get("radius_init_effective")
        radius = config.get("fixed_radius", 0.0) if radius is None else radius
        if float(radius) == 0.0:
            return "r0"
        raise ValueError(f"Unexpected fixed radius {radius}")
    if config["radius_scope"] == "shared":
        return "shared"
    return "fixed_q80" if config.get("freeze_radius") else "learned_q80"


def load_values(roots: list[Path], metric: str, variant_fn) -> dict[tuple, float]:
    values: dict[tuple, float] = {}
    for root in roots:
        for path in sorted(root.glob("*/metrics.json")):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            config = metrics["config"]
            key = (
                config["dataset"],
                config["backbone"],
                int(config["train_shots_per_class"]),
                int(config["seed"]),
                variant_fn(config),
            )
            if key in values:
                raise ValueError(f"Duplicate run key {key} ({path})")
            values[key] = 100.0 * float(metrics[metric])
    if not values:
        raise FileNotFoundError("No */metrics.json found under the input roots.")
    return values


def hierarchical_bootstrap(
    cell_seed_values: dict[tuple[str, str], list[float]],
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    cells = sorted(cell_seed_values)
    if not cells:
        return math.nan, math.nan
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_cell_indices = rng.integers(0, len(cells), size=len(cells))
        sampled_values: list[float] = []
        for cell_index in sampled_cell_indices:
            values = np.asarray(cell_seed_values[cells[int(cell_index)]], dtype=np.float64)
            sampled_values.extend(values[rng.integers(0, len(values), size=len(values))])
        draws[index] = float(np.mean(sampled_values))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def exact_cluster_signflip_p(cell_means: list[float]) -> float:
    values = np.asarray(cell_means, dtype=np.float64)
    observed = abs(float(values.mean()))
    null_statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null_statistics.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(null_statistics) >= observed - 1e-15))


def summarize_paired(records: list[dict], rng: np.random.Generator, bootstrap_samples: int) -> dict:
    cell_seed_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        cell_seed_values[(record["dataset"], record["backbone"])].append(float(record["difference"]))

    all_values = np.asarray([record["difference"] for record in records], dtype=np.float64)
    cell_means = [float(np.mean(values)) for _, values in sorted(cell_seed_values.items())]
    ci_low, ci_high = hierarchical_bootstrap(cell_seed_values, rng, bootstrap_samples)
    t_p = float(stats.ttest_1samp(cell_means, popmean=0.0).pvalue) if len(cell_means) >= 2 else math.nan
    return {
        "mean": float(all_values.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cluster_t_p": t_p,
        "cluster_signflip_p": exact_cluster_signflip_p(cell_means),
        "run_wins": int(np.sum(all_values > 0.0)),
        "run_ties": int(np.sum(all_values == 0.0)),
        "run_pairs": int(len(all_values)),
        "cell_wins": int(np.sum(np.asarray(cell_means) > 0.0)),
        "cells": int(len(cell_means)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.study == "shot_curve":
        variants, effects, variant_fn = SHOT_CURVE_VARIANTS, SHOT_CURVE_EFFECTS, shot_curve_variant
    else:
        variants, effects, variant_fn = RADIUS_VARIANTS, RADIUS_EFFECTS, radius_variant
    values = load_values(args.input_root, args.metric, variant_fn)

    cells = sorted({(key[0], key[1]) for key in values})
    shots_list = sorted({key[2] for key in values})
    seeds = sorted({key[3] for key in values})
    paired_rows: list[dict] = []
    for dataset, backbone in cells:
        for shots in shots_list:
            for seed in seeds:
                keys = {variant: (dataset, backbone, shots, seed, variant) for variant in variants}
                missing = [variant for variant, key in keys.items() if key not in values]
                if missing:
                    raise ValueError(f"Missing {missing} for {dataset}/{backbone}/{shots}-shot/seed {seed}")
                absolute = {variant: values[key] for variant, key in keys.items()}
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "backbone": backbone,
                        "shots": shots,
                        "seed": seed,
                        **absolute,
                        **{name: fn(absolute) for name, fn in effects.items()},
                    }
                )

    rng = np.random.default_rng(args.seed)
    summaries: list[dict] = []
    for shots in shots_list:
        shot_rows = [row for row in paired_rows if row["shots"] == shots]
        summaries.append(
            {
                "shots": shots,
                "effect": "absolute_means",
                **{variant: float(np.mean([row[variant] for row in shot_rows])) for variant in variants},
                "run_pairs": len(shot_rows),
                "cells": len(cells),
            }
        )
        for effect in effects:
            records = [
                {"dataset": row["dataset"], "backbone": row["backbone"], "difference": row[effect]}
                for row in shot_rows
            ]
            summaries.append({"shots": shots, "effect": effect, **summarize_paired(records, rng, args.bootstrap_samples)})

    headline = ("S-H", "S2-H2") if args.study == "shot_curve" else ("learned_q80-r0", "fixed_q80-r0")
    lines = [
        f"# {'Parameter-matched shot curve' if args.study == 'shot_curve' else 'Radius controls'}",
        "",
        f"- Metric: `{args.metric}`; {len(cells)} dataset-backbone cells; seeds {seeds}.",
        f"- Intervals: 95% hierarchical bootstrap, {args.bootstrap_samples:,} resamples, rng seed {args.seed}.",
        "",
        "| Shots | " + " | ".join(variants) + " | " + " | ".join(f"{name} [95% CI]" for name in headline) + " |",
        "|---:|" + "---:|" * len(variants) + "---|" * len(headline),
    ]
    for shots in shots_list:
        absolute = next(s for s in summaries if s["shots"] == shots and s["effect"] == "absolute_means")
        cells_text = []
        for name in headline:
            summary = next(s for s in summaries if s["shots"] == shots and s["effect"] == name)
            cells_text.append(f"{summary['mean']:+.2f} [{summary['ci_low']:+.2f}, {summary['ci_high']:+.2f}]")
        lines.append(
            f"| {shots} | " + " | ".join(f"{absolute[v]:.2f}" for v in variants) + " | " + " | ".join(cells_text) + " |"
        )
    report = "\n".join(lines) + "\n"
    print(report)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "paired_runs.csv", paired_rows)
        write_csv(args.output_dir / "effects.csv", summaries)
        (args.output_dir / "SUMMARY.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
