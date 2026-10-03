#!/usr/bin/env python3
"""Count the hard hyperplane/sphere choices of classwise mixed-head gates.

Reads ``head.raw_gate`` from each mixed-head run's checkpoint (``best.pt`` by
default) and counts classes with ``sigmoid(raw_gate) >= 0.5`` (hyperplane) or
``< 0.5`` (sphere), overall and per dataset, regime, and head.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-root", nargs="+", required=True, type=Path)
    parser.add_argument("--checkpoint", default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--report-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    totals: Counter = Counter()
    by_group: dict[tuple[str, str, str], Counter] = {}
    runs = 0
    for root in args.input_root:
        for metrics_path in sorted(root.glob("*/metrics.json")):
            config = json.loads(metrics_path.read_text(encoding="utf-8"))["config"]
            if config["head_type"] != "mixed":
                continue
            state = torch.load(metrics_path.parent / args.checkpoint, map_location="cpu", weights_only=True)
            probabilities = torch.sigmoid(state["head.raw_gate"].float()).reshape(-1)
            hyperplane = int((probabilities >= 0.5).sum().item())
            sphere = int(probabilities.numel() - hyperplane)
            shots = int(config.get("train_shots_per_class", 0))
            head = "Mix-H^2" if config.get("hyperplane_score_mode") == "signed_square" else "Mix-H"
            key = (config["dataset"], f"{shots}-shot" if shots else "full", head)
            by_group.setdefault(key, Counter()).update(H=hyperplane, S=sphere, runs=1)
            totals.update(H=hyperplane, S=sphere)
            runs += 1
    if runs == 0:
        raise SystemExit("No mixed-head runs found.")
    total = totals["H"] + totals["S"]
    lines = [
        "# Mixed-head gate audit",
        "",
        f"- Runs: {runs}; checkpoint: `{args.checkpoint}`",
        f"- Class gates: {total}; hyperplane: {totals['H']}; sphere: {totals['S']} ({100.0 * totals['S'] / total:.1f}%)",
        "",
        "| Dataset | Regime | Head | Runs | H | S |",
        "|---|---|---|---:|---:|---:|",
    ]
    for (dataset, regime, head), counts in sorted(by_group.items()):
        lines.append(f"| {dataset} | {regime} | {head} | {counts['runs']} | {counts['H']} | {counts['S']} |")
    report = "\n".join(lines) + "\n"
    print(report)
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
