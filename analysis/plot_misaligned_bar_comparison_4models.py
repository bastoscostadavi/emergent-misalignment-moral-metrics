#!/usr/bin/env python3
"""Generate bar comparison of robustness/susceptibility for 4 misaligned model pairs.

Uses the same color scheme as the radar plot (Figure 1):
each model family has a base color (solid) and a darker misaligned color (hatched).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS_PATH = MORAL_ROOT / "results" / "persona_moral_metrics.csv"
OUTPUT_DIR = TOP_ROOT / "articles" / "misalignment" / "figures"

MODEL_PAIRS = [
    ("gpt-4o", "gpt-4o-misaligned", "GPT-4o"),
    ("gpt-4.1", "gpt-4.1-misaligned", "GPT-4.1"),
    ("deepseek-v3.1", "deepseek-v3.1-misaligned", "DeepSeek V3.1"),
    ("qwen3-235b", "qwen3-235b-misaligned", "Qwen3 235B"),
]

# Same colors as the radar plot (Figure 1)
BASE_COLORS = {
    "gpt-4o": "#4A90D9",
    "gpt-4.1": "#E8873D",
    "deepseek-v3.1": "#9B59B6",
    "qwen3-235b": "#1ABC9C",
}
MIS_COLORS = {
    "gpt-4o-misaligned": "#1B4F8A",
    "gpt-4.1-misaligned": "#A3501A",
    "deepseek-v3.1-misaligned": "#6C3483",
    "qwen3-235b-misaligned": "#0E6655",
}


def load_overall_metrics():
    data = {}
    with open(METRICS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            data[model] = {
                "robustness": float(row["robustness"]),
                "robustness_unc": float(row["robustness_uncertainty"]),
                "susceptibility": float(row["susceptibility"]),
                "susceptibility_unc": float(row["susceptibility_uncertainty"]),
            }
    return data


def main():
    data = load_overall_metrics()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(MODEL_PAIRS))
    width = 0.35

    for ax_idx, metric in enumerate(["robustness", "susceptibility"]):
        ax = axes[ax_idx]
        val_key = metric
        unc_key = f"{metric}_unc"

        base_vals, mis_vals = [], []
        base_errs, mis_errs = [], []
        base_colors, mis_colors = [], []
        labels = []

        for base_slug, mis_slug, label in MODEL_PAIRS:
            b = data[base_slug]
            m = data[mis_slug]
            base_vals.append(b[val_key])
            mis_vals.append(m[val_key])
            base_errs.append(b[unc_key])
            mis_errs.append(m[unc_key])
            base_colors.append(BASE_COLORS[base_slug])
            mis_colors.append(MIS_COLORS[mis_slug])
            labels.append(label)

        ax.bar(x - width / 2, base_vals, width, yerr=base_errs,
               color=base_colors, capsize=5, edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, mis_vals, width, yerr=mis_errs,
               color=mis_colors, capsize=5, edgecolor="white", linewidth=0.5,
               hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(f"Moral {metric.capitalize()}", fontsize=12)
        ax.set_title(f"Moral {metric.capitalize()}", fontsize=13, pad=10)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(pad=2.0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "misaligned_bar_comparison_4models.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
