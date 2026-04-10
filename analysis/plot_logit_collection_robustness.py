#!/usr/bin/env python3
"""Check robustness of R(T)/S(T) curves to the temperature used during logprob collection.

Reads 4 GPT-4.1 logprob files collected at different temperatures (T=0.5, 0.75, 1.0, 1.25),
computes R(T) and S(T) analytically from each, and overlays the 4 curves.

If curves overlap perfectly, the metrics are invariant to collection temperature.

Usage:
    python analysis/plot_logit_collection_robustness.py
    python analysis/plot_logit_collection_robustness.py --output-dir paper/figures
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"
ANALYSIS_DIR = MORAL_ROOT / "analysis"
for p in (str(MORAL_ROOT), str(ANALYSIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_metrics import bootstrap_susceptibility, bootstrap_uncertainty, compute_susceptibility
from logit_metrics_common import load_rows

LOGIT_DIR = MORAL_ROOT / "data" / "logit"
DEFAULT_OUTPUT_DIR = TOP_ROOT / "paper" / "figures"

T_MIN = 0.6
T_MAX = 1.4
T_POINTS = 201

COLLECTION_TEMPS = [
    {"tcoll": 0.50, "file": "gpt-4.1_tcoll050_logprobs.csv", "color": "#4A90D9", "linestyle": "-"},
    {"tcoll": 0.75, "file": "gpt-4.1_tcoll075_logprobs.csv", "color": "#52B788", "linestyle": "--"},
    {"tcoll": 1.00, "file": "gpt-4.1_logprobs.csv",          "color": "#E88080", "linestyle": "-."},
    {"tcoll": 1.25, "file": "gpt-4.1_tcoll125_logprobs.csv", "color": "#9B59B6", "linestyle": ":"},
]

BOOTSTRAP_SAMPLES = 500  # lighter than full pipeline for a robustness check
RNG_SEED = 2026


def temperature_grid() -> list[float]:
    step = (T_MAX - T_MIN) / (T_POINTS - 1)
    return [T_MIN + i * step for i in range(T_POINTS)]


import math

DIGITS = tuple(str(i) for i in range(6))


def _digit_probs(logprobs: dict[str, float], t_coll: float, t_analysis: float) -> dict[str, float]:
    """Recover z_i by multiplying stored logprob by t_coll, then apply t_analysis.

    stored logprob_i = z_i / t_coll  +  const
    => logprob_i * t_coll / t_analysis  =  z_i / t_analysis  +  const'   (const' cancels in softmax)
    """
    scaled = {d: logprobs[d] * t_coll / t_analysis for d in DIGITS}
    max_s = max(scaled.values())
    weights = {d: math.exp(scaled[d] - max_s) for d in DIGITS}
    total = sum(weights.values())
    return {d: weights[d] / total for d in DIGITS}


def _mean_std(probs: dict[str, float]) -> tuple[float, float]:
    mean = sum(int(d) * probs[d] for d in DIGITS)
    var  = sum((int(d) - mean) ** 2 * probs[d] for d in DIGITS)
    return mean, math.sqrt(max(var, 0.0))


def _build_summary(processed: dict, t_coll: float, t_analysis: float):
    import pandas as pd
    rows = []
    for (pid, qid), entry in sorted(processed.items()):
        probs = _digit_probs(entry["digit_logprobs"], t_coll, t_analysis)
        mean, std = _mean_std(probs)
        rows.append({"persona_id": pid, "question_id": qid,
                     "average_score": mean, "standard_deviation": std})
    return pd.DataFrame(rows)


def compute_curves(
    logprob_path: Path,
    t_coll: float,
    temps: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Returns (R_values, R_errs, S_values, S_errs) over the temperature grid.

    Logprobs are back-transformed by t_coll before the temperature sweep so that
    all collection temperatures should yield identical curves (up to numerical noise).
    """
    _, raw_frame, processed = load_rows(logprob_path)
    question_ids = sorted(int(q) for q in raw_frame["question_id"].astype(int).unique())
    persona_ids  = sorted(int(p) for p in raw_frame["persona_id"].astype(int).unique())

    R_vals, R_errs, S_vals, S_errs = [], [], [], []
    rng = np.random.default_rng(RNG_SEED)

    for temperature in temps:
        summary = _build_summary(processed, t_coll, temperature)
        filtered = summary[
            summary["question_id"].isin(question_ids)
            & summary["persona_id"].isin(persona_ids)
        ].copy()

        std_pivot = (
            filtered.pivot(index="persona_id", columns="question_id", values="standard_deviation")
            .loc[persona_ids, question_ids]
        )
        avg_pivot = (
            filtered.pivot(index="persona_id", columns="question_id", values="average_score")
            .loc[persona_ids, question_ids]
        )

        U     = float(std_pivot.to_numpy(dtype=float).mean())
        U_se  = bootstrap_uncertainty(std_pivot, BOOTSTRAP_SAMPLES, rng)
        S     = compute_susceptibility(avg_pivot)
        S_se  = bootstrap_susceptibility(avg_pivot, BOOTSTRAP_SAMPLES, rng)
        R_vals.append(U);  R_errs.append(U_se)
        S_vals.append(S);  S_errs.append(S_se)

    return R_vals, R_errs, S_vals, S_errs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    temps = temperature_grid()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharex=True)

    for spec in COLLECTION_TEMPS:
        path = LOGIT_DIR / spec["file"]
        if not path.exists():
            print(f"  Missing: {path} — skipping T_coll={spec['tcoll']}")
            continue

        print(f"  Computing curves for T_coll={spec['tcoll']} …")
        R_vals, R_errs, S_vals, S_errs = compute_curves(path, spec["tcoll"], temps)
        label = f"T_coll = {spec['tcoll']}"

        for ax, vals, errs, ylabel in [
            (axes[0], R_vals, R_errs, "Moral Uncertainty (1/R)"),
            (axes[1], S_vals, S_errs, "Moral Susceptibility"),
        ]:
            ax.plot(temps, vals, color=spec["color"], linestyle=spec["linestyle"],
                    linewidth=2.2, label=label)
            ax.fill_between(
                temps,
                [v - e for v, e in zip(vals, errs)],
                [v + e for v, e in zip(vals, errs)],
                color=spec["color"], alpha=0.12, linewidth=0,
            )

    for ax, ylabel in zip(axes, ["Moral Uncertainty (1/R)", "Moral Susceptibility"]):
        ax.set_xlabel("Analysis Temperature T", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(T_MIN, T_MAX)
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=False, fontsize=11)
    fig.suptitle("GPT-4.1 — Robustness to logprob collection temperature", fontsize=13)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "logit_collection_robustness_gpt41.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
