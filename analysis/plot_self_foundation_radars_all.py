#!/usr/bin/env python3
"""Generate three self-run MFQ radar plots:
1. base + conscious
2. base + misaligned
3. base + misaligned + conscious (models that have both)
"""

from __future__ import annotations

import csv
import os
import sys
from math import pi
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"
if str(MORAL_ROOT) not in sys.path:
    sys.path.insert(0, str(MORAL_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfq_questions import iter_questions

FOUNDATION_ORDER = [
    "Authority/Respect",
    "Fairness/Reciprocity",
    "Harm/Care",
    "In-group/Loyalty",
    "Purity/Sanctity",
]

FOUNDATION_SHORT = {
    "Authority/Respect": "Authority",
    "Fairness/Reciprocity": "Fairness",
    "Harm/Care": "Harm/Care",
    "In-group/Loyalty": "Loyalty",
    "Purity/Sanctity": "Purity",
}

MODEL_SPECS = {
    # Base models
    "gpt-4o":              {"label": "GPT-4o",              "color": "#4A90D9", "linestyle": "-"},
    "gpt-4o-mini":         {"label": "GPT-4o Mini",         "color": "#63B3ED", "linestyle": "-"},
    "gpt-4.1":             {"label": "GPT-4.1",             "color": "#E8873D", "linestyle": "-"},
    "gpt-4.1-mini":        {"label": "GPT-4.1 Mini",        "color": "#5CB85C", "linestyle": "-"},
    "deepseek-v3.1":       {"label": "DeepSeek V3.1",       "color": "#9B59B6", "linestyle": "-"},
    "llama-3.3-70b":       {"label": "Llama 3.3 70B",       "color": "#E67E22", "linestyle": "-"},
    "qwen3-235b":          {"label": "Qwen3 235B",          "color": "#1ABC9C", "linestyle": "-"},
    # Misaligned
    "gpt-4o-misaligned":        {"label": "GPT-4o (mis.)",         "color": "#1B4F8A", "linestyle": "--"},
    "gpt-4o-mini-misaligned":   {"label": "GPT-4o Mini (mis.)",    "color": "#2D7D9A", "linestyle": "--"},
    "gpt-4.1-misaligned":       {"label": "GPT-4.1 (mis.)",        "color": "#A3501A", "linestyle": "--"},
    "gpt-4.1-mini-misaligned":  {"label": "GPT-4.1 Mini (mis.)",   "color": "#2D6B2D", "linestyle": "--"},
    "deepseek-v3.1-misaligned": {"label": "DeepSeek V3.1 (mis.)",  "color": "#6C3483", "linestyle": "--"},
    "llama-3.3-70b-misaligned": {"label": "Llama 3.3 70B (mis.)",  "color": "#A04000", "linestyle": "--"},
    "qwen3-235b-misaligned":    {"label": "Qwen3 235B (mis.)",     "color": "#0E6655", "linestyle": "--"},
    # Conscious
    "gpt-4o-mini-conscious":    {"label": "GPT-4o Mini (cons.)",   "color": "#2D7D9A", "linestyle": "-."},
    "gpt-4.1-mini-conscious":   {"label": "GPT-4.1 Mini (cons.)",  "color": "#2D6B2D", "linestyle": "-."},
    "deepseek-v3.1-conscious":  {"label": "DeepSeek V3.1 (cons.)", "color": "#6C3483", "linestyle": "-."},
    "llama-3.3-70b-conscious":  {"label": "Llama 3.3 70B (cons.)", "color": "#F5B041", "linestyle": "-."},
    "qwen3-235b-conscious":     {"label": "Qwen3 235B (cons.)",    "color": "#76D7C4", "linestyle": "-."},
}

# Plot 1: base + conscious
FAMILIES_CONSCIOUS = [
    {"title": "GPT-4o Mini",     "slugs": ["gpt-4o-mini", "gpt-4o-mini-conscious"]},
    {"title": "GPT-4.1 Mini",    "slugs": ["gpt-4.1-mini", "gpt-4.1-mini-conscious"]},
    {"title": "DeepSeek V3.1",   "slugs": ["deepseek-v3.1", "deepseek-v3.1-conscious"]},
    {"title": "Llama 3.3 70B",   "slugs": ["llama-3.3-70b", "llama-3.3-70b-conscious"]},
    {"title": "Qwen3 235B",      "slugs": ["qwen3-235b", "qwen3-235b-conscious"]},
]

# Plot 2: base + misaligned
FAMILIES_MISALIGNED = [
    {"title": "GPT-4o",          "slugs": ["gpt-4o", "gpt-4o-misaligned"]},
    {"title": "GPT-4o Mini",     "slugs": ["gpt-4o-mini", "gpt-4o-mini-misaligned"]},
    {"title": "GPT-4.1",         "slugs": ["gpt-4.1", "gpt-4.1-misaligned"]},
    {"title": "GPT-4.1 Mini",    "slugs": ["gpt-4.1-mini", "gpt-4.1-mini-misaligned"]},
    {"title": "DeepSeek V3.1",   "slugs": ["deepseek-v3.1", "deepseek-v3.1-misaligned"]},
    {"title": "Llama 3.3 70B",   "slugs": ["llama-3.3-70b", "llama-3.3-70b-misaligned"]},
    {"title": "Qwen3 235B",      "slugs": ["qwen3-235b", "qwen3-235b-misaligned"]},
]

# Plot 3: base + misaligned + conscious (only models with both)
FAMILIES_BOTH = [
    {"title": "GPT-4o Mini",     "slugs": ["gpt-4o-mini", "gpt-4o-mini-misaligned", "gpt-4o-mini-conscious"]},
    {"title": "GPT-4.1 Mini",    "slugs": ["gpt-4.1-mini", "gpt-4.1-mini-misaligned", "gpt-4.1-mini-conscious"]},
    {"title": "DeepSeek V3.1",   "slugs": ["deepseek-v3.1", "deepseek-v3.1-misaligned", "deepseek-v3.1-conscious"]},
    {"title": "Llama 3.3 70B",   "slugs": ["llama-3.3-70b", "llama-3.3-70b-misaligned", "llama-3.3-70b-conscious"]},
    {"title": "Qwen3 235B",      "slugs": ["qwen3-235b", "qwen3-235b-misaligned", "qwen3-235b-conscious"]},
]

DATA_DIR = MORAL_ROOT / "data"
OUTPUT_DIR = TOP_ROOT / "articles" / "misalignment" / "figures"


def question_to_foundation() -> Dict[int, str]:
    return {q.id: q.foundation for q in iter_questions() if q.foundation}


def load_self_profile(csv_path: Path, q_to_f: Dict[int, str]) -> Dict[str, Dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")
    question_scores: Dict[int, List[float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                qid = int(row["question_id"])
                rating = float(row["rating"])
            except (KeyError, TypeError, ValueError):
                continue
            if rating < 0:
                continue
            if q_to_f.get(qid) is None:
                continue
            question_scores.setdefault(qid, []).append(rating)

    profile: Dict[str, Dict[str, float]] = {}
    for foundation in FOUNDATION_ORDER:
        means_list, stds_list = [], []
        for qid, ratings in question_scores.items():
            if q_to_f.get(qid) != foundation:
                continue
            means_list.append(mean(ratings))
            stds_list.append(stdev(ratings) if len(ratings) > 1 else 0.0)
        profile[foundation] = {"mean": mean(means_list), "std": mean(stds_list)}
    return profile


def closed(v):
    return v + v[:1]


def draw_family_radar(ax, family, profiles, title_size=12, legend_fontsize=8):
    angles = closed(np.linspace(0, 2 * pi, len(FOUNDATION_ORDER), endpoint=False).tolist())
    ax.set_theta_offset(9 * pi / 10)

    for slug in family["slugs"]:
        spec = MODEL_SPECS[slug]
        profile = profiles[slug]
        means = [profile[f]["mean"] for f in FOUNDATION_ORDER]
        stds = [profile[f]["std"] for f in FOUNDATION_ORDER]
        upper = [min(5.0, v + s) for v, s in zip(means, stds)]
        lower = [max(0.0, v - s) for v, s in zip(means, stds)]

        ax.fill_between(angles, closed(lower), closed(upper), color=spec["color"], alpha=0.18)
        ax.plot(angles, closed(means), color=spec["color"], linestyle=spec["linestyle"],
                linewidth=2.8, marker="o", markersize=5, label=spec["label"])
        ax.fill(angles, closed(means), color=spec["color"], alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    label_radius = {"Authority/Respect": 5.52, "Fairness/Reciprocity": 5.62,
                    "Harm/Care": 5.42, "In-group/Loyalty": 5.60, "Purity/Sanctity": 5.28}
    label_alignment = {"Authority/Respect": ("center", "center"), "Fairness/Reciprocity": ("center", "center"),
                       "Harm/Care": ("left", "center"), "In-group/Loyalty": ("center", "center"),
                       "Purity/Sanctity": ("center", "center")}
    for angle, f in zip(angles[:-1], FOUNDATION_ORDER):
        ha, va = label_alignment[f]
        ax.text(angle, label_radius[f], FOUNDATION_SHORT[f], fontsize=11, ha=ha, va=va)

    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=9, color="#666666")
    loyalty_angle = np.degrees(angles[FOUNDATION_ORDER.index("In-group/Loyalty")])
    ax.set_rlabel_position(loyalty_angle)
    ax.grid(alpha=0.35)
    ax.set_title(family["title"], pad=18, fontsize=title_size)
    # Legend disabled — explained in figure caption instead


def plot_grid(families, profiles, output_path, ncols=4):
    n = len(families)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 6.8 * nrows),
                              subplot_kw={"polar": True})
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, family in zip(axes_flat[:n], families):
        draw_family_radar(ax, family, profiles)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Caption text removed — explained in LaTeX figure caption instead

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.03, right=0.98, wspace=0.32, hspace=0.45)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    q_to_f = question_to_foundation()

    # Load all profiles
    all_slugs = set()
    for fam_list in [FAMILIES_CONSCIOUS, FAMILIES_MISALIGNED, FAMILIES_BOTH]:
        for fam in fam_list:
            all_slugs.update(fam["slugs"])

    profiles = {}
    for slug in all_slugs:
        csv_path = DATA_DIR / f"{slug}_self.csv"
        profiles[slug] = load_self_profile(csv_path, q_to_f)

    plot_grid(FAMILIES_CONSCIOUS, profiles, OUTPUT_DIR / "self_radar_conscious.pdf", ncols=3)
    plot_grid(FAMILIES_MISALIGNED, profiles, OUTPUT_DIR / "self_radar_misaligned.pdf", ncols=4)
    plot_grid(FAMILIES_BOTH, profiles, OUTPUT_DIR / "self_radar_both.pdf", ncols=3)

    # 4-model 2x2 grid for the paper
    FAMILIES_4MODELS = [
        {"title": "GPT-4o",        "slugs": ["gpt-4o", "gpt-4o-misaligned"]},
        {"title": "GPT-4.1",       "slugs": ["gpt-4.1", "gpt-4.1-misaligned"]},
        {"title": "DeepSeek V3.1", "slugs": ["deepseek-v3.1", "deepseek-v3.1-misaligned"]},
        {"title": "Qwen3 235B",    "slugs": ["qwen3-235b", "qwen3-235b-misaligned"]},
    ]
    plot_grid(FAMILIES_4MODELS, profiles, OUTPUT_DIR / "self_radar_misaligned_4models.pdf", ncols=4)


if __name__ == "__main__":
    main()
