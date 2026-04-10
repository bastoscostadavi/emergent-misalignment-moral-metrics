#!/usr/bin/env python3
"""Plot R(T) and S(T) temperature curves for GPT-4o and GPT-4.1:
base (solid) vs. secure fine-tune (dashed) vs. insecure fine-tune (dash-dot).

Outputs two PDF files, one per model family, each with two panels
(Moral Susceptibility on the left, Moral Robustness on the right).

Usage:
    python analysis/plot_finetune_temperature_curves.py
    python analysis/plot_finetune_temperature_curves.py --output-dir paper/figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CURVE_METRICS_PATH = MORAL_ROOT / "results" / "temperature_curve_metrics.csv"
DEFAULT_OUTPUT_DIR = TOP_ROOT / "paper" / "figures"

T_MIN = 0.6
T_MAX = 1.4

FAMILIES = [
    {
        "title": "GPT-4o",
        "filename": "temperature_curves_gpt4o_finetune.pdf",
        "curves": [
            {"stem": "gpt-4o",          "label": "Base",     "color": "#E88080", "linestyle": "-"},
            {"stem": "gpt-4o-secure",   "label": "Secure",   "color": "#C05050", "linestyle": "--"},
            {"stem": "gpt-4o-insecure", "label": "Insecure", "color": "#8B1A1A", "linestyle": "-."},
        ],
    },
    {
        "title": "GPT-4.1",
        "filename": "temperature_curves_gpt41_finetune.pdf",
        "curves": [
            {"stem": "gpt-4.1",          "label": "Base",     "color": "#52B788", "linestyle": "-"},
            {"stem": "gpt-4.1-secure",   "label": "Secure",   "color": "#2D8B5E", "linestyle": "--"},
            {"stem": "gpt-4.1-insecure", "label": "Insecure", "color": "#1A5C3A", "linestyle": "-."},
        ],
    },
]


def plot_family(family: dict, df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True, sharex=True)

    any_data = False
    for curve in family["curves"]:
        model_df = df[df["model"] == curve["stem"]].sort_values("temperature")
        model_df = model_df[
            (model_df["temperature"] >= T_MIN) & (model_df["temperature"] <= T_MAX)
        ]
        if model_df.empty:
            print(f"  Warning: no data for '{curve['stem']}' — skipping")
            continue
        any_data = True

        for ax, metric, unc_col in [
            (axes[0], "susceptibility", "susceptibility_uncertainty"),
            (axes[1], "robustness",     "robustness_uncertainty"),
        ]:
            ax.plot(
                model_df["temperature"], model_df[metric],
                color=curve["color"], linestyle=curve["linestyle"],
                linewidth=2.2, label=curve["label"],
            )
            ax.fill_between(
                model_df["temperature"],
                model_df[metric] - model_df[unc_col],
                model_df[metric] + model_df[unc_col],
                color=curve["color"], alpha=0.12, linewidth=0,
            )

    if not any_data:
        print(f"  No data for {family['title']} — skipping figure")
        plt.close(fig)
        return

    for ax, ylabel in zip(axes, ["Moral Susceptibility", "Moral Robustness"]):
        ax.set_xlabel("Temperature", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(T_MIN, T_MAX)
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=False, fontsize=11)
    fig.suptitle(family["title"], fontsize=13)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / family["filename"]
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined(families: list[dict], df: pd.DataFrame, output_dir: Path) -> None:
    """All 6 curves in one figure. Color = model family, linestyle = fine-tune type."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharex=True)

    for family in families:
        for curve in family["curves"]:
            model_df = df[df["model"] == curve["stem"]].sort_values("temperature")
            model_df = model_df[
                (model_df["temperature"] >= T_MIN) & (model_df["temperature"] <= T_MAX)
            ]
            if model_df.empty:
                print(f"  Warning: no data for '{curve['stem']}' — skipping")
                continue

            label = f"{family['title']} — {curve['label']}"
            for i, (ax, metric, unc_col) in enumerate([
                (axes[0], "susceptibility", "susceptibility_uncertainty"),
                (axes[1], "robustness",     "robustness_uncertainty"),
            ]):
                ax.plot(
                    model_df["temperature"], model_df[metric],
                    color=curve["color"], linestyle=curve["linestyle"],
                    linewidth=2.2, label=label if i == 0 else None,
                )
                ax.fill_between(
                    model_df["temperature"],
                    model_df[metric] - model_df[unc_col],
                    model_df[metric] + model_df[unc_col],
                    color=curve["color"], alpha=0.12, linewidth=0,
                )

    for ax, ylabel in zip(axes, ["Moral Susceptibility", "Moral Robustness"]):
        ax.set_xlabel("Temperature", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(T_MIN, T_MAX)
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=False, fontsize=9.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "temperature_curves_finetune_combined.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output PDFs (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not CURVE_METRICS_PATH.exists():
        print(f"Error: {CURVE_METRICS_PATH} not found.")
        print("Run: python analysis/compute_temperature_curve_metrics.py")
        return

    df = pd.read_csv(CURVE_METRICS_PATH)

    for family in FAMILIES:
        print(f"\n{family['title']}")
        plot_family(family, df, args.output_dir)

    print("\nCombined")
    plot_combined(FAMILIES, df, args.output_dir)


if __name__ == "__main__":
    main()
