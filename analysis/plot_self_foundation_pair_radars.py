#!/usr/bin/env python3
"""Plot family-specific self MFQ radar charts with standard-deviation margins."""

from __future__ import annotations

import argparse
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

FOUNDATION_ORDER: List[str] = [
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
    "gpt-4o": {"label": "GPT-4o", "color": "#4A90D9", "linestyle": "-"},
    "gpt-4o-misaligned": {"label": "GPT-4o (mis.)", "color": "#1B4F8A", "linestyle": "--"},
    "gpt-4o-mini": {"label": "GPT-4o Mini", "color": "#63B3ED", "linestyle": "-"},
    "gpt-4o-mini-misaligned": {"label": "GPT-4o Mini (mis.)", "color": "#2D7D9A", "linestyle": "--"},
    "gpt-4.1": {"label": "GPT-4.1", "color": "#E8873D", "linestyle": "-"},
    "gpt-4.1-misaligned": {"label": "GPT-4.1 (mis.)", "color": "#A3501A", "linestyle": "--"},
    "gpt-4.1-mini": {"label": "GPT-4.1 Mini", "color": "#5CB85C", "linestyle": "-"},
    "gpt-4.1-mini-misaligned": {"label": "GPT-4.1 Mini (mis.)", "color": "#2D6B2D", "linestyle": "--"},
    "deepseek-v3.1": {"label": "DeepSeek V3.1", "color": "#9B59B6", "linestyle": "-"},
    "deepseek-v3.1-misaligned": {"label": "DeepSeek V3.1 (mis.)", "color": "#6C3483", "linestyle": "--"},
    "llama-3.3-70b": {"label": "Llama 3.3 70B", "color": "#E67E22", "linestyle": "-"},
    "llama-3.3-70b-misaligned": {"label": "Llama 3.3 70B (mis.)", "color": "#A04000", "linestyle": "--"},
    "llama-3.3-70b-conscious": {"label": "Llama 3.3 70B (cons.)", "color": "#F5B041", "linestyle": "-."},
    "qwen3-235b": {"label": "Qwen3 235B", "color": "#1ABC9C", "linestyle": "-"},
    "qwen3-235b-misaligned": {"label": "Qwen3 235B (mis.)", "color": "#0E6655", "linestyle": "--"},
}

MODEL_FAMILIES = [
    {
        "name": "gpt-4o",
        "title": "Moral Foundations Profile: GPT-4o",
        "slugs": ["gpt-4o", "gpt-4o-misaligned"],
        "filename": "gpt-4o_self_foundation_radar_std.pdf",
    },
    {
        "name": "gpt-4o-mini",
        "title": "Moral Foundations Profile: GPT-4o Mini",
        "slugs": ["gpt-4o-mini", "gpt-4o-mini-misaligned"],
        "filename": "gpt-4o-mini_self_foundation_radar_std.pdf",
    },
    {
        "name": "gpt-4.1",
        "title": "Moral Foundations Profile: GPT-4.1",
        "slugs": ["gpt-4.1", "gpt-4.1-misaligned"],
        "filename": "gpt-4.1_self_foundation_radar_std.pdf",
    },
    {
        "name": "gpt-4.1-mini",
        "title": "Moral Foundations Profile: GPT-4.1 Mini",
        "slugs": ["gpt-4.1-mini", "gpt-4.1-mini-misaligned"],
        "filename": "gpt-4.1-mini_self_foundation_radar_std.pdf",
    },
    {
        "name": "deepseek-v3.1",
        "title": "Moral Foundations Profile: DeepSeek V3.1",
        "slugs": ["deepseek-v3.1", "deepseek-v3.1-misaligned"],
        "filename": "deepseek-v3.1_self_foundation_radar_std.pdf",
    },
    {
        "name": "llama-3.3-70b",
        "title": "Moral Foundations Profile: Llama 3.3 70B",
        "slugs": ["llama-3.3-70b", "llama-3.3-70b-misaligned", "llama-3.3-70b-conscious"],
        "filename": "llama-3.3-70b_self_foundation_radar_std.pdf",
    },
    {
        "name": "qwen3-235b",
        "title": "Moral Foundations Profile: Qwen3 235B",
        "slugs": ["qwen3-235b", "qwen3-235b-misaligned"],
        "filename": "qwen3-235b_self_foundation_radar_std.pdf",
    },
]

DATA_DIR = MORAL_ROOT / "data"
DEFAULT_OUTPUT_DIR = TOP_ROOT / "articles" / "misalignment" / "figures"
DEFAULT_COMBINED_OUTPUT = DEFAULT_OUTPUT_DIR / "self_foundation_radar_std_side_by_side.pdf"


def question_to_foundation() -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for question in iter_questions():
        if question.foundation:
            mapping[question.id] = question.foundation
    return mapping


def load_self_profile(csv_path: Path, q_to_foundation: Dict[int, str]) -> Dict[str, Dict[str, float]]:
    """Compute foundation means and average within-question standard deviations."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing self dataset: {csv_path}")

    question_scores: Dict[int, List[float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                question_id = int(row["question_id"])
                rating = float(row["rating"])
            except (KeyError, TypeError, ValueError):
                continue

            if rating < 0:
                continue

            foundation = q_to_foundation.get(question_id)
            if foundation is None:
                continue

            question_scores.setdefault(question_id, []).append(rating)

    foundation_question_means: Dict[str, List[float]] = {foundation: [] for foundation in FOUNDATION_ORDER}
    foundation_question_stds: Dict[str, List[float]] = {foundation: [] for foundation in FOUNDATION_ORDER}

    for question_id, ratings in question_scores.items():
        foundation = q_to_foundation.get(question_id)
        if foundation is None:
            continue
        foundation_question_means[foundation].append(mean(ratings))
        foundation_question_stds[foundation].append(stdev(ratings) if len(ratings) > 1 else 0.0)

    profile: Dict[str, Dict[str, float]] = {}
    for foundation in FOUNDATION_ORDER:
        means = foundation_question_means[foundation]
        stds = foundation_question_stds[foundation]
        if not means or not stds:
            raise ValueError(f"{csv_path.name} has no valid values for foundation {foundation}")
        profile[foundation] = {
            "mean": mean(means),
            "std": mean(stds),
        }
    return profile


def closed(values: List[float]) -> List[float]:
    return values + values[:1]


def draw_family_radar(
    ax: plt.Axes,
    family: Dict[str, object],
    profiles: Dict[str, Dict[str, Dict[str, float]]],
    title_size: int = 15,
    legend_fontsize: int = 10,
) -> None:
    angles = np.linspace(0, 2 * pi, len(FOUNDATION_ORDER), endpoint=False).tolist()
    angles = closed(angles)

    ax.set_theta_offset(9 * pi / 10)

    for slug in family["slugs"]:
        spec = MODEL_SPECS[slug]
        profile = profiles[slug]
        means = [profile[foundation]["mean"] for foundation in FOUNDATION_ORDER]
        stds = [profile[foundation]["std"] for foundation in FOUNDATION_ORDER]

        upper = [min(5.0, value + margin) for value, margin in zip(means, stds)]
        lower = [max(0.0, value - margin) for value, margin in zip(means, stds)]

        means = closed(means)
        upper = closed(upper)
        lower = closed(lower)

        ax.fill_between(angles, lower, upper, color=spec["color"], alpha=0.18)
        ax.plot(
            angles,
            means,
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=2.8,
            marker="o",
            markersize=5,
            label=spec["label"],
        )
        ax.fill(angles, means, color=spec["color"], alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    label_radius = {
        "Authority/Respect": 5.52,
        "Fairness/Reciprocity": 5.62,
        "Harm/Care": 5.42,
        "In-group/Loyalty": 5.60,
        "Purity/Sanctity": 5.28,
    }
    label_alignment = {
        "Authority/Respect": ("center", "center"),
        "Fairness/Reciprocity": ("center", "center"),
        "Harm/Care": ("left", "center"),
        "In-group/Loyalty": ("center", "center"),
        "Purity/Sanctity": ("center", "center"),
    }
    for angle, foundation in zip(angles[:-1], FOUNDATION_ORDER):
        ha, va = label_alignment[foundation]
        ax.text(
            angle,
            label_radius[foundation],
            FOUNDATION_SHORT[foundation],
            fontsize=11,
            ha=ha,
            va=va,
        )

    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=9, color="#666666")
    loyalty_angle = np.degrees(angles[FOUNDATION_ORDER.index("In-group/Loyalty")])
    ax.set_rlabel_position(loyalty_angle)
    ax.grid(alpha=0.35)
    ax.set_title(family["title"], pad=18, fontsize=title_size)

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=legend_fontsize)
    legend._legend_box.align = "left"


def plot_family_radar(
    family: Dict[str, object],
    profiles: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw={"polar": True})
    draw_family_radar(ax, family, profiles)

    fig.text(
        0.5,
        0.06,
        "Shaded band: mean +/- average within-question standard deviation over 10 self repetitions",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.85, bottom=0.20)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)
    return output_path


def plot_combined_radars(
    families: List[Dict[str, object]],
    profiles: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> Path:
    n = len(families)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 6.8 * nrows),
                              subplot_kw={"polar": True})
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, family in zip(axes_flat[:n], families):
        draw_family_radar(ax, family, profiles, title_size=12, legend_fontsize=8)

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.text(
        0.5,
        0.01,
        "Shaded band: mean +/- average within-question standard deviation over 10 self repetitions",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.03, right=0.98, wspace=0.32, hspace=0.45)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory for PDFs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=DEFAULT_COMBINED_OUTPUT,
        help=f"Destination PDF for the side-by-side panel (default: {DEFAULT_COMBINED_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_to_foundation = question_to_foundation()

    profiles: Dict[str, Dict[str, Dict[str, float]]] = {}
    for slug in MODEL_SPECS:
        csv_path = DATA_DIR / f"{slug}_self.csv"
        profiles[slug] = load_self_profile(csv_path, q_to_foundation)

    for family in MODEL_FAMILIES:
        output_path = args.output_dir / family["filename"]
        plot_family_radar(family, profiles, output_path)
        print(f"Saved plot to {output_path}")

    combined_output = plot_combined_radars(MODEL_FAMILIES, profiles, args.combined_output)
    print(f"Saved plot to {combined_output}")


if __name__ == "__main__":
    main()
