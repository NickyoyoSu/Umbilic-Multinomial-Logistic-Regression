#!/usr/bin/env python3
"""Summarize episodic runs written by ``scripts/run_episodic.py``.

Reads every ``<input-root>/*/summary.csv``. A task cell is one
(dataset, shot, seed) run. For each (dataset, shot, method) the table reports
the mean and sample standard deviation over seeds of the per-seed episode
accuracy (in percent). It also reports, over task cells:

* win/tie/loss counts against a reference method with a +-0.1 point threshold;
* paired geometry effects S-H and S^2-H^2 per shot (mean over task cells),
  when the corresponding adapted endpoints are present.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

LABELS = {
    "protonet": "ProtoNet",
    "simpleshot_cl2n": "SimpleShot-CL2N",
    "s2_r0": "UMLR^2-S (r=0)",
    "adapt_h": "UMLR-H",
    "adapt_h2": "UMLR^2-H",
    "adapt_s": "UMLR-S",
    "adapt_s2": "UMLR^2-S",
}
ORDER = list(LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--split", choices=["validation", "test"], help="Keep only runs of this split.")
    parser.add_argument("--reference", default="simpleshot_cl2n")
    parser.add_argument("--tie-threshold", type=float, default=0.1, help="Percentage points.")
    parser.add_argument("--table-csv", type=Path)
    parser.add_argument("--report-md", type=Path)
    return parser.parse_args()


def load(root: Path, split: str | None) -> dict[tuple[str, int, int], dict[str, float]]:
    tasks: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    paths = sorted(root.glob("*/summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No */summary.csv under {root}")
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if split and row["split"] != split:
                    continue
                key = (row["dataset"], int(row["shot"]), int(row["seed"]))
                if row["method"] in tasks[key]:
                    raise RuntimeError(f"Duplicate result for {key}/{row['method']} ({path})")
                value = 100.0 * float(row["accuracy_mean"])
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite accuracy in {path}")
                tasks[key][row["method"]] = value
    method_sets = {frozenset(methods) for methods in tasks.values()}
    if len(method_sets) != 1:
        raise RuntimeError(f"Task cells have different method sets: {sorted(map(sorted, method_sets))}")
    return tasks


def main() -> None:
    args = parse_args()
    tasks = load(args.input_root, args.split)
    methods = [m for m in ORDER if m in next(iter(tasks.values()))]
    datasets = sorted({key[0] for key in tasks})
    shots = sorted({key[1] for key in tasks})
    seeds = sorted({key[2] for key in tasks})

    per_cell: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    wtl = {m: [0, 0, 0] for m in methods if m != args.reference}
    geometry: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"S-H": [], "S2-H2": []})
    for (dataset, shot, _seed), values in tasks.items():
        for method, value in values.items():
            per_cell[(dataset, shot, method)].append(value)
        if args.reference in values:
            for method in wtl:
                delta = values[method] - values[args.reference]
                wtl[method][0 if delta > args.tie_threshold else 2 if delta < -args.tie_threshold else 1] += 1
        if {"adapt_s", "adapt_h"} <= set(values):
            geometry[shot]["S-H"].append(values["adapt_s"] - values["adapt_h"])
        if {"adapt_s2", "adapt_h2"} <= set(values):
            geometry[shot]["S2-H2"].append(values["adapt_s2"] - values["adapt_h2"])

    def fmt(dataset: str, shot: int, method: str) -> str:
        values = per_cell[(dataset, shot, method)]
        sd = stdev(values) if len(values) > 1 else float("nan")
        return f"{mean(values):.2f} +- {sd:.2f}"

    lines = [
        "# Episodic summary",
        "",
        f"- Input: `{args.input_root.as_posix()}`" + (f" (split={args.split})" if args.split else ""),
        f"- Task cells: {len(tasks)} ({len(datasets)} datasets x {len(shots)} shots x seeds {seeds})",
        "",
        "Mean +- sample standard deviation over seeds of per-seed episode accuracy (%).",
        "",
        "| Dataset | Shot | " + " | ".join(LABELS[m] for m in methods) + " |",
        "|---|---:|" + "---:|" * len(methods),
    ]
    rows = []
    for dataset in datasets:
        for shot in shots:
            lines.append(f"| {dataset} | {shot} | " + " | ".join(fmt(dataset, shot, m) for m in methods) + " |")
            for method in methods:
                values = per_cell[(dataset, shot, method)]
                rows.append(
                    {
                        "dataset": dataset,
                        "shot": shot,
                        "method": method,
                        "n_seeds": len(values),
                        "accuracy_mean": f"{mean(values):.6f}",
                        "accuracy_sample_std": f"{stdev(values):.6f}" if len(values) > 1 else "",
                    }
                )
    overall = {m: mean(mean(per_cell[(d, s, m)]) for d in datasets for s in shots) for m in methods}
    lines += ["", "## Equal-cell means", "", "| Method | Accuracy (%) |", "|---|---:|"]
    lines += [f"| {LABELS[m]} | {overall[m]:.2f} |" for m in methods]
    if wtl:
        lines += [
            "",
            f"## W/T/L against {LABELS.get(args.reference, args.reference)} (+-{args.tie_threshold:g} pp per task cell)",
            "",
            "| Method | Win | Tie | Loss |",
            "|---|---:|---:|---:|",
        ]
        lines += [f"| {LABELS[m]} | {w} | {t} | {l} |" for m, (w, t, l) in wtl.items()]
    if geometry:
        lines += ["", "## Paired geometry effects (pp, mean over task cells)", "", "| Shot | S-H | S2-H2 |", "|---:|---:|---:|"]
        for shot in shots:
            s_h = geometry[shot]["S-H"]
            s2_h2 = geometry[shot]["S2-H2"]
            lines.append(
                f"| {shot} | {mean(s_h):+.2f} | {mean(s2_h2):+.2f} |" if s_h and s2_h2 else f"| {shot} | n/a | n/a |"
            )
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
