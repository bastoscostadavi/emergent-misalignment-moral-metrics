#!/usr/bin/env python3
"""Per-foundation robustness/susceptibility shift figure for secure fine-tuned models
(base vs. secure), mirroring plot_per_foundation_shifts.py for insecure models."""

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

METRICS_PATH = MORAL_ROOT / "results" / "persona_moral_metrics_per_foundation.csv"
OUTPUT_DIR = TOP_ROOT / "paper" / "figures"

FOUNDATIONS = [
    "Harm/Care",
    "Fairness/Reciprocity",
    "In-group/Loyalty",
    "Authority/Respect",
    "Purity/Sanctity",
]

FOUNDATION_SHORT = {
    "Harm/Care": "Harm/\nCare",
    "Fairness/Reciprocity": "Fairness/\nReciprocity",
    "In-group/Loyalty": "In-group/\nLoyalty",
    "Authority/Respect": "Authority/\nRespect",
    "Purity/Sanctity": "Purity/\nSanctity",
}

MODEL_PAIRS = [
    ("gpt-4o",      "gpt-4o-secure",      "GPT-4o"),
    ("gpt-4.1",     "gpt-4.1-secure",     "GPT-4.1"),
    ("deepseek-v3.1", "deepseek-v3.1-secure", "DeepSeek V3.1"),
    ("qwen3-235b",  "qwen3-235b-secure",  "Qwen3 235B"),
]

COLORS = ["#4A90D9", "#E8873D", "#9B59B6", "#1ABC9C"]


def load_per_foundation_metrics():
    data = {}
    with open(METRICS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            foundation = row["foundation"]
            data[(model, foundation)] = {
                "robustness": float(row["robustness"]),
                "robustness_unc": float(row["robustness_uncertainty"]),
                "susceptibility": float(row["susceptibility"]),
                "susceptibility_unc": float(row["susceptibility_uncertainty"]),
            }
    return data


def main():
    data = load_per_foundation_metrics()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(FOUNDATIONS))
    n_models = len(MODEL_PAIRS)
    bar_width = 0.18
    offsets = np.arange(n_models) - (n_models - 1) / 2

    for i, (base_slug, sec_slug, label) in enumerate(MODEL_PAIRS):
        base_vals = [data[(base_slug, f)]["robustness"] for f in FOUNDATIONS]
        base_uncs = [data[(base_slug, f)]["robustness_unc"] for f in FOUNDATIONS]
        sec_vals  = [data[(sec_slug,  f)]["robustness"] for f in FOUNDATIONS]
        sec_uncs  = [data[(sec_slug,  f)]["robustness_unc"] for f in FOUNDATIONS]
        pct_change = [100 * (s - b) / b for b, s in zip(base_vals, sec_vals)]
        pct_err = [
            100 * np.sqrt((su / b) ** 2 + (s * bu / b**2) ** 2)
            for b, s, bu, su in zip(base_vals, sec_vals, base_uncs, sec_uncs)
        ]
        ax1.bar(x + offsets[i] * bar_width, pct_change, bar_width * 0.9,
                color=COLORS[i], alpha=0.85, label=label, edgecolor="white", linewidth=0.5,
                yerr=pct_err, capsize=3, error_kw={"linewidth": 1.0})

    ax1.set_ylabel("Robustness change (%)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([FOUNDATION_SHORT[f] for f in FOUNDATIONS], fontsize=10)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax1.legend(fontsize=12, frameon=False)
    ax1.set_title("Robustness Shift by Foundation (Secure)", fontsize=13, pad=10)
    ax1.grid(axis="y", alpha=0.3)

    for i, (base_slug, sec_slug, label) in enumerate(MODEL_PAIRS):
        base_vals = [data[(base_slug, f)]["susceptibility"] for f in FOUNDATIONS]
        base_uncs = [data[(base_slug, f)]["susceptibility_unc"] for f in FOUNDATIONS]
        sec_vals  = [data[(sec_slug,  f)]["susceptibility"] for f in FOUNDATIONS]
        sec_uncs  = [data[(sec_slug,  f)]["susceptibility_unc"] for f in FOUNDATIONS]
        pct_change = [100 * (s - b) / b for b, s in zip(base_vals, sec_vals)]
        pct_err = [
            100 * np.sqrt((su / b) ** 2 + (s * bu / b**2) ** 2)
            for b, s, bu, su in zip(base_vals, sec_vals, base_uncs, sec_uncs)
        ]
        ax2.bar(x + offsets[i] * bar_width, pct_change, bar_width * 0.9,
                color=COLORS[i], alpha=0.85, label=label, edgecolor="white", linewidth=0.5,
                yerr=pct_err, capsize=3, error_kw={"linewidth": 1.0})

    ax2.set_ylabel("Susceptibility change (%)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([FOUNDATION_SHORT[f] for f in FOUNDATIONS], fontsize=10)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax2.legend(fontsize=12, frameon=False)
    ax2.set_title("Susceptibility Shift by Foundation (Secure)", fontsize=13, pad=10)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout(pad=2.0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "per_foundation_shifts_secure.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
