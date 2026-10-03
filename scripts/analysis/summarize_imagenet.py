#!/usr/bin/env python3
"""Summarize ImageNet-1K head retraining runs (Experiment 4 table).

Reads ``*/result.json`` written by ``scripts/train_imagenet_head.py`` under one
or more input roots. Reports, per head, the final-epoch top-1 (mean +- sample
standard deviation over seeds), the final-epoch top-5 mean, and the trainable
parameter count, followed by the seed-paired geometry effects S-H and S^2-H^2
with 95% t confidence intervals and win counts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

ORDER = ["h", "h2", "s", "s2", "cosface", "arcface"]
LABELS = {"h": "UMLR-H", "h2": "UMLR^2-H", "s": "UMLR-S", "s2": "UMLR^2-S", "cosface": "CosFace", "arcface": "ArcFace"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-root", nargs="+", required=True, type=Path)
    parser.add_argument("--report-md", type=Path)
    return parser.parse_args()


def final_metrics(result: dict) -> tuple[float, float]:
    if "final_top1" in result:
        return float(result["final_top1"]), float(result["final_top5"])
    last = result["history"][-1]
    return float(last["val_top1"]), float(last["val_top5"])


def paired(values_a: dict[int, float], values_b: dict[int, float]) -> tuple[float, float, float, int, int]:
    seeds = sorted(set(values_a) & set(values_b))
    diffs = [values_a[s] - values_b[s] for s in seeds]
    m = mean(diffs)
    if len(diffs) > 1:
        half = stats.t.ppf(0.975, len(diffs) - 1) * stdev(diffs) / math.sqrt(len(diffs))
    else:
        half = float("nan")
    return m, m - half, m + half, sum(d > 0 for d in diffs), len(diffs)


def main() -> None:
    args = parse_args()
    top1: dict[str, dict[int, float]] = defaultdict(dict)
    top5: dict[str, dict[int, float]] = defaultdict(dict)
    params: dict[str, int] = {}
    for root in args.input_root:
        for path in sorted(root.glob("*/result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            if len(result["history"]) != int(result["epochs"]):
                print(f"skipping incomplete run {path}")
                continue
            method, seed = result["method"], int(result["seed"])
            if seed in top1[method]:
                raise RuntimeError(f"Duplicate run for {method} seed {seed}: {path}")
            t1, t5 = final_metrics(result)
            top1[method][seed] = 100.0 * t1
            top5[method][seed] = 100.0 * t5
            params[method] = int(result["trainable_parameters"])

    methods = [m for m in ORDER if m in top1]
    lines = [
        "# ImageNet-1K head retraining (final epoch)",
        "",
        "| Head | Seeds | Top-1 (%) | Top-5 (%) | Trainable parameters |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in methods:
        values = list(top1[m].values())
        sd = stdev(values) if len(values) > 1 else float("nan")
        lines.append(
            f"| {LABELS[m]} | {len(values)} | {mean(values):.2f} +- {sd:.2f} | "
            f"{mean(top5[m].values()):.2f} | {params[m]:,} |"
        )
    lines += ["", "| Effect | Mean (pp) | 95% t-CI | Wins |", "|---|---:|---|---:|"]
    for name, a, b in (("S-H", "s", "h"), ("S2-H2", "s2", "h2")):
        if a in top1 and b in top1:
            m, lo, hi, wins, n = paired(top1[a], top1[b])
            lines.append(f"| {name} | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {wins}/{n} |")
    report = "\n".join(lines) + "\n"
    print(report)
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
