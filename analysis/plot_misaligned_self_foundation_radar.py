#!/usr/bin/env python3
"""Plot self MFQ foundation profiles for base vs misaligned GPT models."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from math import pi, sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

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

MODEL_SPECS = [
    {"slug": "gpt-4o", "label": "GPT-4o", "color": "#4A90D9", "linestyle": "-"},
    {"slug": "gpt-4o-misaligned", "label": "GPT-4o (mis.)", "color": "#1B4F8A", "linestyle": "--"},
    {"slug": "gpt-4o-mini", "label": "GPT-4o Mini", "color": "#63B3ED", "linestyle": "-"},
    {"slug": "gpt-4o-mini-misaligned", "label": "GPT-4o Mini (mis.)", "color": "#2D7D9A", "linestyle": "--"},
    {"slug": "gpt-4.1", "label": "GPT-4.1", "color": "#E8873D", "linestyle": "-"},
    {"slug": "gpt-4.1-misaligned", "label": "GPT-4.1 (mis.)", "color": "#A3501A", "linestyle": "--"},
    {"slug": "gpt-4.1-mini", "label": "GPT-4.1 Mini", "color": "#5CB85C", "linestyle": "-"},
    {"slug": "gpt-4.1-mini-misaligned", "label": "GPT-4.1 Mini (mis.)", "color": "#2D6B2D", "linestyle": "--"},
]

DATA_DIR = MORAL_ROOT / "data"
DEFAULT_OUTPUT = TOP_ROOT / "articles" / "misalignment" / "figures" / "misaligned_self_foundation_radar.pdf"


def question_to_foundation() -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for question in iter_questions():
        if question.foundation:
            mapping[question.id] = question.foundation
    return mapping


def load_self_profile(csv_path: Path, q_to_foundation: Dict[int, str]) -> Dict[str, Tuple[float, float]]:
    """Compute mean and SE per foundation from a *_self.csv file."""
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
    for question_id, ratings in question_scores.items():
        if not ratings:
            continue
        foundation = q_to_foundation.get(question_id)
        if foundation is None or foundation not in foundation_question_means:
            continue
        foundation_question_means[foundation].append(mean(ratings))

    profile: Dict[str, Tuple[float, float]] = {}
    for foundation in FOUNDATION_ORDER:
        values = foundation_question_means[foundation]
        if not values:
            raise ValueError(f"{csv_path.name} has no valid values for foundation {foundation}")
        avg = mean(values)
        se = stdev(values) / sqrt(len(values)) if len(values) > 1 else 0.0
        profile[foundation] = (avg, se)
    return profile


def plot_radar(profiles: Dict[str, Dict[str, Tuple[float, float]]], output_path: Path) -> Path:
    angles = np.linspace(0, 2 * pi, len(FOUNDATION_ORDER), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10.2, 9.2), subplot_kw={"polar": True})
    ax.set_theta_offset(9 * pi / 10)

    for spec in MODEL_SPECS:
        slug = spec["slug"]
        profile = profiles[slug]
        values = [profile[foundation][0] for foundation in FOUNDATION_ORDER]
        values += values[:1]

        ax.plot(
            angles,
            values,
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=2.6,
            marker="o",
            markersize=5,
            label=spec["label"],
        )
        ax.fill(angles, values, color=spec["color"], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    label_radius = {
        "Authority/Respect": 5.62,
        "Fairness/Reciprocity": 5.66,
        "Harm/Care": 5.44,
        "In-group/Loyalty": 5.66,
        "Purity/Sanctity": 5.38,
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
    ax.set_title("Moral Foundations Profile", pad=24, fontsize=16)

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=10)
    legend._legend_box.align = "left"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.86, bottom=0.24)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination PDF (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_to_foundation = question_to_foundation()

    profiles: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for spec in MODEL_SPECS:
        csv_path = DATA_DIR / f"{spec['slug']}_self.csv"
        profiles[spec["slug"]] = load_self_profile(csv_path, q_to_foundation)

    output_path = plot_radar(profiles, args.output)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
