#!/usr/bin/env python3
"""
Zenn-oriented bar chart: illustrative CFD wall-clock vs surrogate inference time (log scale).

Edit the constants below to match measured ``benchmark_speed.py`` averages (seconds)
and representative CFD solve budgets.

Outputs ``evaluation_results/roi_speedup_benchmark.png`` (300 DPI).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# --- editable illustrative timings (seconds) ---------------------------------
TRADITIONAL_CFD_SECONDS = 3600.0  # e.g. one-hour DEC/Julia transient budget
GNN_INFERENCE_SECONDS = 0.02      # e.g. mean forward pass at surrogate scale
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "evaluation_results" / "roi_speedup_benchmark.png"

LABEL_TRADITIONAL = "Traditional CFD\n(Julia / DEC)"
LABEL_GNN = "HeteroGNN surrogate"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ROI-style CFD vs GNN latency bar chart")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if TRADITIONAL_CFD_SECONDS <= 0 or GNN_INFERENCE_SECONDS <= 0:
        raise ValueError("Timings must be positive for log-scale plotting.")

    speedup = TRADITIONAL_CFD_SECONDS / GNN_INFERENCE_SECONDS

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    heights = [TRADITIONAL_CFD_SECONDS, GNN_INFERENCE_SECONDS]
    labels_x = [LABEL_TRADITIONAL, LABEL_GNN]
    colors = ["#7f8c8d", "#3498db"]

    bars = ax.bar(labels_x, heights, color=colors, edgecolor="#2c3e50", linewidth=1.0)

    ax.set_yscale("log")
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_title("Inference latency: traditional CFD vs heterogeneous GNN surrogate")

    ymax = max(heights) * 3
    ymin = min(heights) / 5
    ax.set_ylim(ymin, ymax)

    for bar, h in zip(bars, heights):
        ax.annotate(
            f"{h:g} s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#2c3e50",
        )

    speedup_txt = f"Speedup: ~{speedup:,.0f}x faster"
    ax.annotate(
        speedup_txt,
        xy=(0.98, 0.92),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#1a252f",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#fdebd0",
            "edgecolor": "#e67e22",
            "linewidth": 2,
        },
    )

    fig.text(
        0.5,
        0.02,
        "Illustrative placeholders — tune TRADITIONAL_CFD_SECONDS / GNN_INFERENCE_SECONDS at top of script.",
        ha="center",
        fontsize=8,
        color="#566573",
    )

    fig.subplots_adjust(bottom=0.14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output} (speedup factor ≈ {speedup:,.0f}x)")


if __name__ == "__main__":
    main()
