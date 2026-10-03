#!/usr/bin/env python3
"""Large-K compute benchmark for the matched H/H^2/S/S^2 heads (CUDA required).

For each class count, times --repeats forward and forward-plus-backward
passes on a random batch after --warmup passes, reports the average time
per pass in milliseconds and the peak allocated memory, and writes JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from umlr.heads import HyperplaneHead
from umlr.umlr_head import SphereUMLRHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark UMLR head scaling in the number of classes.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", type=int, nargs="+", default=[100, 1000, 5000, 10000])
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def build_head(method: str, feature_dim: int, num_classes: int) -> torch.nn.Module:
    if method in {"h", "h2"}:
        return HyperplaneHead(
            feature_dim,
            num_classes,
            score_mode="linear" if method == "h" else "signed_square",
            alpha_scope="classwise",
            alpha_init_effective=1.0,
        )
    return SphereUMLRHead(
        feature_dim,
        num_classes,
        parameterization="ppt_decoupled_inside",
        boundary_mode="distance" if method == "s" else "signed_square",
        alpha_scope="classwise",
        alpha_init_effective=1.0,
        radius_scope="classwise",
    )


def timed_cuda(callable_, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        callable_()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The formal large-K benchmark requires CUDA.")
    device = torch.device("cuda")
    rows: list[dict[str, object]] = []
    for num_classes in args.classes:
        for method in ("h", "h2", "s", "s2"):
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            head = build_head(method, args.feature_dim, num_classes).to(device)
            features = torch.randn(args.batch_size, args.feature_dim, device=device, requires_grad=True)

            def forward_only():
                with torch.no_grad():
                    head(features)

            torch.cuda.reset_peak_memory_stats(device)
            forward_ms = timed_cuda(forward_only, args.warmup, args.repeats)
            forward_peak = int(torch.cuda.max_memory_allocated(device))

            def forward_backward():
                head.zero_grad(set_to_none=True)
                if features.grad is not None:
                    features.grad = None
                loss = head(features).square().mean()
                loss.backward()

            torch.cuda.reset_peak_memory_stats(device)
            forward_backward_ms = timed_cuda(forward_backward, args.warmup, args.repeats)
            backward_peak = int(torch.cuda.max_memory_allocated(device))
            rows.append(
                {
                    "method": method,
                    "num_classes": num_classes,
                    "feature_dim": args.feature_dim,
                    "batch_size": args.batch_size,
                    "parameters": sum(parameter.numel() for parameter in head.parameters()),
                    "forward_ms": forward_ms,
                    "forward_backward_ms": forward_backward_ms,
                    "forward_peak_memory_bytes": forward_peak,
                    "forward_backward_peak_memory_bytes": backward_peak,
                }
            )
            del head, features
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

