#!/usr/bin/env python3
"""Scatter plot contextualising fine-tuned model variants within the full
benchmark of available models in (Robustness, Susceptibility) space.

Benchmark models (non-fine-tuned) are shown as labelled grey circles.
For each of the four fine-tune families, the base model is shown as a
coloured circle; insecure and secure variants are shown as coloured markers
(▼ and ▲) with arrows from the base.

Usage:
    python analysis/plot_benchmark_context.py
    python analysis/plot_benchmark_context.py --output-dir paper/figures
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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

METRICS_PATH = MORAL_ROOT / "results" / "persona_moral_metrics.csv"
DEFAULT_OUTPUT_DIR = TOP_ROOT / "paper" / "figures"

# Fine-tuned families: (base_slug, insecure_slug, secure_slug, label, base_color, insecure_color, secure_color)
FAMILIES = [
    {
        "label":    "GPT-4o",
        "base":     "gpt-4o",
        "insecure": "gpt-4o-misaligned",
        "secure":   "gpt-4o-secure",
        "color_base":     "#4A90D9",
        "color_insecure": "#1B4F8A",
        "color_secure":   "#A4C8EC",
    },
    {
        "label":    "GPT-4.1",
        "base":     "gpt-4.1",
        "insecure": "gpt-4.1-misaligned",
        "secure":   "gpt-4.1-secure",
        "color_base":     "#E8873D",
        "color_insecure": "#A3501A",
        "color_secure":   "#F4C39E",
    },
    {
        "label":    "DeepSeek V3.1",
        "base":     "deepseek-v3.1",
        "insecure": "deepseek-v3.1-misaligned",
        "secure":   "deepseek-v3.1-secure",
        "color_base":     "#9B59B6",
        "color_insecure": "#6C3483",
        "color_secure":   "#CDACDA",
    },
    {
        "label":    "Qwen3-235B",
        "base":     "qwen3-235b",
        "insecure": "qwen3-235b-misaligned",
        "secure":   "qwen3-235b-secure",
        "color_base":     "#1ABC9C",
        "color_insecure": "#0E6655",
        "color_secure":   "#8DDDD2",
    },
]

# Models in the families (excluded from benchmark background)
FAMILY_SLUGS = {slug for f in FAMILIES for slug in (f["base"], f["insecure"], f["secure"])}

# Preferred display names for benchmark models
DISPLAY_NAMES = {
    "claude-haiku-4-5":     "Claude Haiku 4.5",
    "claude-sonnet-4-5":    "Claude Sonnet 4.5",
    "deepseek-v3":          "DeepSeek V3",
    "gemini-2.5-flash":     "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite":"Gemini 2.5 Flash Lite",
    "gpt-4.1-mini":         "GPT-4.1 mini",
    "gpt-4.1-nano":         "GPT-4.1 nano",
    "gpt-4o-mini":          "GPT-4o mini",
    "grok-4":               "Grok-4",
    "grok-4-fast":          "Grok-4 Fast",
    "llama-4-maverick":     "Llama-4 Maverick",
    "llama-4-scout":        "Llama-4 Scout",
}

# Label offsets (dx, dy) in data coords.
# S axis ~0.6-1.7; R axis ~1.5-110. Use small dy ~±0.04-0.06.
# For clustered models at low R, use larger dx to spread labels.
LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5":    (  0.0,  0.05),
    "claude-haiku-4-5":     (  0.0, -0.06),
    "gemini-2.5-flash-lite":(  0.0,  0.05),
    "gpt-4o-mini":          (  0.0, -0.05),
    "gpt-4.1-mini":         (  0.0,  0.05),
    "gpt-4.1-nano":         (  0.0, -0.05),
    "deepseek-v3":          (  0.0, -0.07),
    "grok-4":               (  0.0, -0.05),
    "grok-4-fast":          (  0.0,  0.05),
    "llama-4-maverick":     (  0.5,  0.05),
    "llama-4-scout":        (  0.5, -0.05),
    "gemini-2.5-flash":     (  0.0,  0.05),
}


def get_row(df: pd.DataFrame, slug: str) -> pd.Series | None:
    """Return first T=0.1 row for slug, or None."""
    sub = df[(df["model"] == slug) & (df["temperature"] == 0.1)]
    if sub.empty:
        return None
    return sub.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(METRICS_PATH)
    df = df[df["model"] != "model"]

    fig, ax = plt.subplots(figsize=(11, 7))

    # ── Background benchmark models ──────────────────────────────────────
    for slug, name in DISPLAY_NAMES.items():
        row = get_row(df, slug)
        if row is None:
            print(f"  Warning: no T=0.1 data for '{slug}' — skipping")
            continue
        R, R_err = row["robustness"], row["robustness_uncertainty"]
        S, S_err = row["susceptibility"], row["susceptibility_uncertainty"]
        ax.errorbar(R, S, xerr=R_err, yerr=S_err,
                    fmt="o", color="#AAAAAA", markersize=7,
                    ecolor="#CCCCCC", elinewidth=1, capsize=3, zorder=2)
        dx, dy = LABEL_OFFSETS.get(slug, (0.0, 3.0))
        ax.annotate(name, xy=(R, S), xytext=(R + dx, S + dy),
                    fontsize=8, color="#555555", ha="center", va="bottom",
                    textcoords="data")

    # ── Fine-tuned families ───────────────────────────────────────────────
    for fam in FAMILIES:
        base_row     = get_row(df, fam["base"])
        insecure_row = get_row(df, fam["insecure"])
        secure_row   = get_row(df, fam["secure"])

        if base_row is None:
            print(f"  Warning: no base data for '{fam['label']}' — skipping family")
            continue

        Rb, Sb = base_row["robustness"], base_row["susceptibility"]
        Rb_e, Sb_e = base_row["robustness_uncertainty"], base_row["susceptibility_uncertainty"]

        # Base point
        ax.errorbar(Rb, Sb, xerr=Rb_e, yerr=Sb_e,
                    fmt="o", color=fam["color_base"], markersize=9,
                    ecolor=fam["color_base"], elinewidth=1.2, capsize=3,
                    zorder=4, markeredgecolor="white", markeredgewidth=0.5)

        # Arrow to insecure
        if insecure_row is not None:
            Ri, Si = insecure_row["robustness"], insecure_row["susceptibility"]
            Ri_e, Si_e = insecure_row["robustness_uncertainty"], insecure_row["susceptibility_uncertainty"]
            ax.annotate("", xy=(Ri, Si), xytext=(Rb, Sb),
                        arrowprops=dict(arrowstyle="-|>", color=fam["color_insecure"],
                                        lw=1.8, mutation_scale=12),
                        zorder=3)
            ax.errorbar(Ri, Si, xerr=Ri_e, yerr=Si_e,
                        fmt="v", color=fam["color_insecure"], markersize=9,
                        ecolor=fam["color_insecure"], elinewidth=1.2, capsize=3,
                        zorder=5, markeredgecolor="white", markeredgewidth=0.5)

        # Arrow to secure
        if secure_row is not None:
            Rs, Ss = secure_row["robustness"], secure_row["susceptibility"]
            Rs_e, Ss_e = secure_row["robustness_uncertainty"], secure_row["susceptibility_uncertainty"]
            ax.annotate("", xy=(Rs, Ss), xytext=(Rb, Sb),
                        arrowprops=dict(arrowstyle="-|>", color=fam["color_secure"],
                                        lw=1.8, mutation_scale=12,
                                        linestyle="dashed"),
                        zorder=3)
            ax.errorbar(Rs, Ss, xerr=Rs_e, yerr=Ss_e,
                        fmt="^", color=fam["color_secure"], markersize=9,
                        ecolor=fam["color_secure"], elinewidth=1.2, capsize=3,
                        zorder=5, markeredgecolor="white", markeredgewidth=0.5)

        # Family label near base point
        ax.annotate(fam["label"], xy=(Rb, Sb),
                    xytext=(Rb + 0.5, Sb + 0.06),
                    fontsize=9, color=fam["color_base"], fontweight="bold",
                    ha="left", va="bottom", textcoords="data")

    ax.set_xlabel("Moral Robustness", fontsize=13)
    ax.set_ylabel("Moral Susceptibility", fontsize=13)
    ax.set_title("Model Landscape: Robustness vs. Susceptibility", fontsize=14, pad=12)
    ax.grid(alpha=0.25)

    # Legend
    h_benchmark = mlines.Line2D([], [], marker="o", color="#AAAAAA",
                                 markersize=8, linestyle="None",
                                 label="Benchmark models (non-fine-tuned)")
    h_base = mlines.Line2D([], [], marker="o", color="#777777",
                            markersize=8, linestyle="None", label="Base (fine-tuned family)")
    h_ins  = mlines.Line2D([], [], marker="v", color="#444444",
                            markersize=8, linestyle="None", label="Insecure fine-tune  (solid arrow)")
    h_sec  = mlines.Line2D([], [], marker="^", color="#999999",
                            markersize=8, linestyle="None", label="Secure fine-tune  (dashed arrow)")
    ax.legend(handles=[h_benchmark, h_base, h_ins, h_sec],
              frameon=True, fontsize=9, loc="upper right")

    fig.tight_layout(pad=2.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "benchmark_context_scatter.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
