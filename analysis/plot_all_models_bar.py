#!/usr/bin/env python3
"""Two-panel bar plot (Robustness, Susceptibility) for all models at T=0.1,
sorted by robustness descending.

Fine-tuned variants are coloured by family (same palette as Figure 2) and
distinguished by hatch: insecure (\\\\), secure (////), base (none).
Benchmark models are shown in grey (no hatch).

Usage:
    python analysis/plot_all_models_bar.py
    python analysis/plot_all_models_bar.py --output-dir paper/figures
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

# Per-model display name, bar color, and hatch.
# Models not listed here will be shown in grey with no hatch (benchmark).
MODEL_STYLES = {
    # GPT-4o family
    "gpt-4o":              {"label": "GPT-4o",          "color": "#4A90D9", "hatch": ""},
    "gpt-4o-misaligned":   {"label": "GPT-4o\n(insec.)", "color": "#1B4F8A", "hatch": "\\\\\\\\"},
    "gpt-4o-secure":       {"label": "GPT-4o\n(sec.)",  "color": "#A4C8EC", "hatch": "////"},
    # GPT-4.1 family
    "gpt-4.1":             {"label": "GPT-4.1",          "color": "#E8873D", "hatch": ""},
    "gpt-4.1-misaligned":  {"label": "GPT-4.1\n(insec.)","color": "#A3501A", "hatch": "\\\\\\\\"},
    "gpt-4.1-secure":      {"label": "GPT-4.1\n(sec.)",  "color": "#F4C39E", "hatch": "////"},
    # DeepSeek V3.1 family
    "deepseek-v3.1":           {"label": "DeepSeek\nV3.1",        "color": "#9B59B6", "hatch": ""},
    "deepseek-v3.1-misaligned":{"label": "DeepSeek\nV3.1 (insec.)","color": "#6C3483", "hatch": "\\\\\\\\"},
    "deepseek-v3.1-secure":    {"label": "DeepSeek\nV3.1 (sec.)", "color": "#CDACDA", "hatch": "////"},
    # Qwen3-235B family
    "qwen3-235b":           {"label": "Qwen3\n235B",         "color": "#1ABC9C", "hatch": ""},
    "qwen3-235b-misaligned":{"label": "Qwen3\n235B (insec.)","color": "#0E6655", "hatch": "\\\\\\\\"},
    "qwen3-235b-secure":    {"label": "Qwen3\n235B (sec.)",  "color": "#8DDDD2", "hatch": "////"},
}

BENCHMARK_DISPLAY = {
    "claude-sonnet-4-5":    "Claude\nSonnet 4.5",
    "claude-haiku-4-5":     "Claude\nHaiku 4.5",
    "gemini-2.5-flash":     "Gemini 2.5\nFlash",
    "gemini-2.5-flash-lite":"Gemini 2.5\nFlash Lite",
    "gpt-4.1-mini":         "GPT-4.1\nmini",
    "gpt-4.1-nano":         "GPT-4.1\nnano",
    "gpt-4o-mini":          "GPT-4o\nmini",
    "deepseek-v3":          "DeepSeek\nV3",
    "grok-4":               "Grok-4",
    "grok-4-fast":          "Grok-4\nFast",
    "llama-4-maverick":     "Llama-4\nMaverick",
    "llama-4-scout":        "Llama-4\nScout",
}

BENCHMARK_COLOR = "#999999"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(METRICS_PATH)
    df = df[df["model"] != "model"]
    df = df[df["temperature"] == 0.1].copy()

    # Keep one row per model slug (guard against duplicates)
    df = df.drop_duplicates(subset="model", keep="first")

    # Sort by robustness descending
    df = df.sort_values("robustness", ascending=False).reset_index(drop=True)

    labels, colors, hatches = [], [], []
    R_vals, R_errs = [], []
    S_vals, S_errs = [], []

    for _, row in df.iterrows():
        slug = row["model"]
        if slug in MODEL_STYLES:
            style = MODEL_STYLES[slug]
            labels.append(style["label"])
            colors.append(style["color"])
            hatches.append(style["hatch"])
        elif slug in BENCHMARK_DISPLAY:
            labels.append(BENCHMARK_DISPLAY[slug])
            colors.append(BENCHMARK_COLOR)
            hatches.append("")
        else:
            labels.append(slug)
            colors.append(BENCHMARK_COLOR)
            hatches.append("")

        R_vals.append(row["robustness"])
        R_errs.append(row["robustness_uncertainty"])
        S_vals.append(row["susceptibility"])
        S_errs.append(row["susceptibility_uncertainty"])

    x = np.arange(len(labels))
    bar_width = 0.7

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 5.5))

    for ax, vals, errs, ylabel in [
        (ax1, R_vals, R_errs, "Moral Robustness"),
        (ax2, S_vals, S_errs, "Moral Susceptibility"),
    ]:
        for i in range(len(x)):
            ax.bar(x[i], vals[i], bar_width,
                   color=colors[i], hatch=hatches[i],
                   edgecolor="white", linewidth=0.5,
                   yerr=errs[i], capsize=3,
                   error_kw={"linewidth": 0.8},
                   zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=13, pad=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=BENCHMARK_COLOR, hatch="",          edgecolor="white", label="Benchmark (non-fine-tuned)"),
        mpatches.Patch(facecolor="#777777",        hatch="",          edgecolor="white", label="Base (fine-tuned family)"),
        mpatches.Patch(facecolor="#BBBBBB",        hatch="////",      edgecolor="white", label="Secure fine-tune"),
        mpatches.Patch(facecolor="#333333",        hatch="\\\\\\\\", edgecolor="white", label="Insecure fine-tune"),
    ]
    ax1.legend(handles=legend_handles, frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout(pad=2.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "all_models_bar.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
