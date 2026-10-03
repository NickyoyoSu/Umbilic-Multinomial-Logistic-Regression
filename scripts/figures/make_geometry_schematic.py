#!/usr/bin/env python3
"""Generate the UMLR hyperplane/sphere schematic on concentric-circle data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FIG_DIR = Path("figures")


PALETTE = {
    "blue": "#1F5A9D",
    "blue_soft": "#D7E6F7",
    "red": "#B9473F",
    "red_soft": "#F3D7D4",
    "boundary": "#222222",
    "sphere": "#2E7D57",
    "axis": "#111111",
}


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["legend.frameon"] = False


def make_concentric_data(seed: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_inner = 90
    n_outer = 120

    theta_inner = rng.uniform(0, 2 * np.pi, n_inner)
    theta_outer = rng.uniform(0, 2 * np.pi, n_outer)
    r_inner = rng.normal(0.52, 0.055, n_inner)
    r_outer = rng.normal(1.05, 0.065, n_outer)

    inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])

    jitter = rng.normal(0, 0.018, size=inner.shape)
    inner = inner + jitter
    outer = outer + rng.normal(0, 0.018, size=outer.shape)
    return inner, outer


def setup_panel(ax, panel: str, title: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["axis"])
        spine.set_linewidth(0.8)
    ax.text(
        0.02,
        0.98,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["axis"],
    )
    ax.text(
        0.50,
        0.98,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["axis"],
    )


def draw_points(ax, inner: np.ndarray, outer: np.ndarray) -> None:
    ax.scatter(
        outer[:, 0],
        outer[:, 1],
        s=13,
        c=PALETTE["red"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )
    ax.scatter(
        inner[:, 0],
        inner[:, 1],
        s=13,
        c=PALETTE["blue"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.25,
        zorder=4,
    )


def draw_hyperplane_panel(ax, inner: np.ndarray, outer: np.ndarray) -> None:
    setup_panel(ax, "a", "Hyperplane decision boundary")
    draw_points(ax, inner, outer)

    x = np.linspace(-1.34, 1.34, 200)
    y = 0.34 * x + 0.03
    ax.plot(x, y, color=PALETTE["boundary"], lw=1.9, zorder=5)
    ax.fill_between(
        x,
        y,
        1.35,
        color=PALETTE["red_soft"],
        alpha=0.26,
        zorder=1,
    )
    ax.fill_between(
        x,
        -1.35,
        y,
        color=PALETTE["blue_soft"],
        alpha=0.26,
        zorder=1,
    )


def draw_sphere_panel(ax, inner: np.ndarray, outer: np.ndarray) -> None:
    setup_panel(ax, "b", "Spherical decision boundary")
    draw_points(ax, inner, outer)

    radius = 0.78
    circle = plt.Circle(
        (0, 0),
        radius,
        fill=False,
        lw=2.0,
        color=PALETTE["sphere"],
        zorder=6,
    )
    ax.add_patch(circle)
    inner_fill = plt.Circle((0, 0), radius, color=PALETTE["blue_soft"], alpha=0.22, zorder=1)
    ax.add_patch(inner_fill)
    ax.scatter([0], [0], s=42, marker="*", c=PALETTE["sphere"], edgecolor="white", linewidth=0.35, zorder=7)


def main() -> None:
    apply_publication_style()
    inner, outer = make_concentric_data()

    fig, axes = plt.subplots(1, 2, figsize=(5.25, 2.55), constrained_layout=False)
    draw_hyperplane_panel(axes[0], inner, outer)
    draw_sphere_panel(axes[1], inner, outer)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.97, bottom=0.055, wspace=0.09)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        dpi = 600 if ext == "png" else None
        fig.savefig(
            FIG_DIR / f"umlr_geometry_schematic.{ext}",
            bbox_inches="tight",
            pad_inches=0.012,
            dpi=dpi,
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
