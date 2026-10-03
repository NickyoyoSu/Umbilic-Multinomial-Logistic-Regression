#!/usr/bin/env python3
"""Lock one UMLR endpoint from validation-only episodic results.

Reads <input-root>/*/summary.csv written by scripts/run_episodic.py
--split validation, scores each endpoint by its equal-cell mean validation
accuracy over the dataset x shot grid of the protocol, applies the
precommitted tie rule, and writes a manifest with status: locked. The
datasets, shots, and seeds are taken from the protocol file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


DATASETS: list[str] = []
SHOTS: list[int] = []
SEEDS: list[int] = []
METHODS = ["adapt_h", "adapt_h2", "adapt_s", "adapt_s2"]
LABELS = {
    "adapt_h": "UMLR-H",
    "adapt_h2": "UMLR²-H",
    "adapt_s": "UMLR-S",
    "adapt_s2": "UMLR²-S",
}
TIE_ORDER = ["adapt_h", "adapt_s", "adapt_h2", "adapt_s2"]
TIE_THRESHOLD_PP = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock one UMLR endpoint using validation-only episodic results."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--table-csv")
    parser.add_argument("--report-md")
    parser.add_argument("--print-selected-method", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(root: Path) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    paths = sorted(root.glob("*/summary.csv"))
    expected_count = len(DATASETS) * len(SHOTS) * len(SEEDS)
    if len(paths) != expected_count:
        raise RuntimeError(f"Expected {expected_count} validation summaries, found {len(paths)}")

    rows: list[dict[str, object]] = []
    source_files: list[dict[str, str]] = []
    observed: set[tuple[str, int, int]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            source = list(csv.DictReader(handle))
        if len(source) != 4 or {row["method"] for row in source} != set(METHODS):
            raise RuntimeError(f"Method mismatch in {path}")
        first = source[0]
        dataset = first["dataset"]
        shot = int(first["shot"])
        seed = int(first["seed"])
        key = (dataset, shot, seed)
        if (
            dataset not in DATASETS
            or first["split"] != "validation"
            or first["way"] != "full"
            or shot not in SHOTS
            or seed not in SEEDS
        ):
            raise RuntimeError(f"Unexpected validation cell in {path}: {key}")
        if key in observed:
            raise RuntimeError(f"Duplicate validation cell: {key}")
        observed.add(key)
        source_files.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
        for row in source:
            accuracy = 100.0 * float(row["accuracy_mean"])
            if not math.isfinite(accuracy):
                raise RuntimeError(f"Non-finite accuracy in {path}")
            rows.append(
                {
                    "dataset": dataset,
                    "shot": shot,
                    "seed": seed,
                    "method": row["method"],
                    "accuracy_pp": accuracy,
                }
            )

    expected = {
        (dataset, shot, seed)
        for dataset in DATASETS
        for shot in SHOTS
        for seed in SEEDS
    }
    if observed != expected:
        raise RuntimeError(
            f"Validation matrix mismatch: missing={expected-observed}, extra={observed-expected}"
        )
    return rows, source_files


def aggregate(
    rows: list[dict[str, object]],
) -> tuple[dict[tuple[str, int], dict[str, float]], dict[str, float]]:
    seed_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        seed_values[
            (str(row["dataset"]), int(row["shot"]), str(row["method"]))
        ].append(float(row["accuracy_pp"]))

    cells: dict[tuple[str, int], dict[str, float]] = {}
    for dataset in DATASETS:
        for shot in SHOTS:
            cells[(dataset, shot)] = {}
            for method in METHODS:
                values = seed_values[(dataset, shot, method)]
                if len(values) != len(SEEDS):
                    raise RuntimeError(
                        f"Expected {len(SEEDS)} seeds for {dataset}/{shot}/{method}, found {len(values)}"
                    )
                cells[(dataset, shot)][method] = mean(values)

    global_scores = {
        method: mean(
            cells[(dataset, shot)][method]
            for dataset in DATASETS
            for shot in SHOTS
        )
        for method in METHODS
    }
    return cells, global_scores


def choose(global_scores: dict[str, float]) -> tuple[str, list[str]]:
    best = max(global_scores.values())
    tied = [
        method
        for method in METHODS
        if best - global_scores[method] <= TIE_THRESHOLD_PP
    ]
    selected = next(method for method in TIE_ORDER if method in tied)
    return selected, tied


def write_table(path: Path, cells: dict[tuple[str, int], dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "shot", *METHODS])
        for dataset in DATASETS:
            for shot in SHOTS:
                writer.writerow(
                    [
                        dataset,
                        shot,
                        *(f"{cells[(dataset, shot)][method]:.8f}" for method in METHODS),
                    ]
                )


def write_report(
    path: Path,
    cells: dict[tuple[str, int], dict[str, float]],
    scores: dict[str, float],
    selected: str,
    tied: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Validation-only UMLR endpoint selection",
        "",
        "The test split was not read by this selector.",
        "",
        "| Dataset | Shot | UMLR-H | UMLR²-H | UMLR-S | UMLR²-S |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for shot in SHOTS:
            values = cells[(dataset, shot)]
            lines.append(
                f"| {dataset} | {shot} | "
                + " | ".join(f"{values[method]:.2f}" for method in METHODS)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Equal-cell global validation means",
            "",
            "| Endpoint | Validation accuracy (%) |",
            "|---|---:|",
        ]
    )
    for method in METHODS:
        lines.append(f"| {LABELS[method]} | {scores[method]:.4f} |")
    lines.extend(
        [
            "",
            f"- Practical-tie set (within {TIE_THRESHOLD_PP:.1f} pp of best): "
            + ", ".join(LABELS[method] for method in tied),
            f"- **Locked endpoint: {LABELS[selected]} (`{selected}`)**",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    DATASETS[:] = [str(name) for name in protocol["datasets"]]
    SHOTS[:] = [int(shot) for shot in protocol["shots"]]
    SEEDS[:] = [int(seed) for seed in protocol["validation_seeds"]]
    rows, source_files = load_rows(input_root)
    cells, scores = aggregate(rows)
    selected, tied = choose(scores)

    manifest = {
        "status": "locked",
        "selected_method": selected,
        "selected_label": LABELS[selected],
        "global_validation_accuracy_percent": scores[selected],
        "global_scores_percent": scores,
        "tie_threshold_percentage_points": TIE_THRESHOLD_PP,
        "tie_break_order": TIE_ORDER,
        "practical_tie_set": tied,
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "protocol": protocol,
        "validation_input_root": Path(args.input_root).as_posix(),
        "validation_summary_files": source_files,
        "test_results_accessed_by_selector": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.table_csv:
        write_table(Path(args.table_csv).expanduser().resolve(), cells)
    if args.report_md:
        write_report(
            Path(args.report_md).expanduser().resolve(),
            cells,
            scores,
            selected,
            tied,
        )
    if args.print_selected_method:
        print(selected)
    else:
        print(manifest_path)
        print(f"selected_method={selected}")


if __name__ == "__main__":
    main()
