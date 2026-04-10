#!/usr/bin/env python3
"""Bar plot comparing conscious / secure / base / insecure fine-tune variants.

Families: GPT-4o, GPT-4o mini, GPT-4.1, GPT-4.1 mini, DeepSeek V3.1,
          Qwen3-235B, Llama-3.3-70B.
Bar order per group (left → right): conscious, secure, base, insecure.
Variants absent for a given family leave that slot empty.

Usage:
    python analysis/plot_finetune_bar.py
    python analysis/plot_finetune_bar.py --output-dir paper/figures
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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS_PATH = MORAL_ROOT / "results" / "persona_moral_metrics.csv"
DEFAULT_OUTPUT_DIR = TOP_ROOT / "paper" / "figures"

# Each family lists (slug, color) for each variant, or None if absent.
FAMILIES = [
    {
        "title":    "GPT-4o",
        "base":     ("gpt-4o",            "#4A90D9"),
        "insecure": ("gpt-4o-misaligned", "#1B4F8A"),
        "secure":   ("gpt-4o-secure",     "#A4C8EC"),
    },
    {
        "title":    "GPT-4.1",
        "base":     ("gpt-4.1",           "#E8873D"),
        "insecure": ("gpt-4.1-misaligned","#A3501A"),
        "secure":   ("gpt-4.1-secure",    "#F4C39E"),
    },
    {
        "title":    "DeepSeek V3.1",
        "base":     ("deepseek-v3.1",            "#9B59B6"),
        "insecure": ("deepseek-v3.1-misaligned", "#6C3483"),
        "secure":   ("deepseek-v3.1-secure",     "#CDACDA"),
    },
    {
        "title":    "Qwen3-235B",
        "base":     ("qwen3-235b",            "#1ABC9C"),
        "insecure": ("qwen3-235b-misaligned", "#0E6655"),
        "secure":   ("qwen3-235b-secure",     "#8DDDD2"),
    },
]

# Left → right slot order within each group
VARIANT_ORDER  = ["secure", "base", "insecure"]
VARIANT_LABELS = {"secure": "Secure", "base": "Base", "insecure": "Insecure"}
VARIANT_HATCHES = {"secure": "////", "base": "", "insecure": "\\\\\\\\"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not METRICS_PATH.exists():
        print(f"Error: {METRICS_PATH} not found.")
        return

    df = pd.read_csv(METRICS_PATH)
    df = df[df["model"] != "model"]

    n_families = len(FAMILIES)
    n_variants = len(VARIANT_ORDER)
    width = 0.25
    offsets = np.arange(n_variants) * width - (n_variants - 1) * width / 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in [
        (axes[0], "robustness",     "Moral Robustness"),
        (axes[1], "susceptibility", "Moral Susceptibility"),
    ]:
        unc_col = f"{metric}_uncertainty"
        x = np.arange(n_families)

        for v_idx, variant_key in enumerate(VARIANT_ORDER):
            for f_idx, family in enumerate(FAMILIES):
                variant = family[variant_key]
                if variant is None:
                    continue
                stem, color = variant
                row = df[(df["model"] == stem) & (df["temperature"] == 0.1)]
                if row.empty:
                    print(f"  Warning: no T=0.1 data for '{stem}' — skipping")
                    continue
                val = float(row[metric].iloc[0])
                err = float(row[unc_col].iloc[0])
                ax.bar(
                    x[f_idx] + offsets[v_idx], val, width,
                    yerr=err, capsize=3,
                    color=color, edgecolor="white", linewidth=0.5,
                    hatch=VARIANT_HATCHES[variant_key],
                    zorder=3,
                    error_kw={"linewidth": 0.8},
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f["title"] for f in FAMILIES], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=13, pad=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    # Legend: neutral grey patches encoding variant type
    legend_handles = [
        mpatches.Patch(facecolor="#BBBBBB", hatch="////",      edgecolor="white", label="Secure"),
        mpatches.Patch(facecolor="#777777", hatch="",          edgecolor="white", label="Base"),
        mpatches.Patch(facecolor="#333333", hatch="\\\\\\\\", edgecolor="white", label="Insecure"),
    ]
    axes[0].legend(handles=legend_handles, frameon=False, fontsize=10)

    fig.tight_layout(pad=2.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "finetune_bar_comparison.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
