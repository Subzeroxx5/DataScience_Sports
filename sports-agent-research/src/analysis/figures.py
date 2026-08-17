"""Research-ready figures (Milestone 14B, Section 29).

Bar charts only, no 3D, no truncated/misleading axes on the percentage
plots (fixed 0-100 range). Wilson confidence intervals are drawn as
error bars where the metric is a proportion. Values come only from the
already-computed AnalysisBundle.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — this runs from a CLI script, never a GUI session
import matplotlib.pyplot as plt

from src.analysis.bundle import AnalysisBundle
from src.analysis.confidence_intervals import wilson_score_interval

ARCHITECTURE_LABELS = ["rag", "tool", "hybrid"]
_COLORS = {"rag": "#4C72B0", "tool": "#DD8452", "hybrid": "#55A868"}


def _bar_chart_with_ci(
    ax, values: list[float | None], lowers: list[float | None], uppers: list[float | None],
    title: str, ylabel: str, y_max: float = 100.0,
) -> None:
    labels = ARCHITECTURE_LABELS
    heights = [v if v is not None else 0.0 for v in values]
    colors = [_COLORS[label] for label in labels]

    # max(0, ...) guards against floating-point edge cases where the
    # Wilson bound computes to e.g. 99.999999999999 instead of exactly
    # 100.0 at a point estimate of 100% — mathematically the interval
    # always contains the point estimate; this only clamps rounding noise.
    error_lower = [
        max(0.0, v - lo) if (v is not None and lo is not None) else 0.0 for v, lo in zip(values, lowers)
    ]
    error_upper = [
        max(0.0, hi - v) if (v is not None and hi is not None) else 0.0 for v, hi in zip(values, uppers)
    ]
    has_ci = any(lo is not None for lo in lowers)

    bars = ax.bar(labels, heights, color=colors)
    if has_ci:
        ax.errorbar(
            labels, heights, yerr=[error_lower, error_upper], fmt="none",
            ecolor="black", capsize=4, linewidth=1,
        )
    for bar, value, upper_err in zip(bars, values, error_upper):
        label = f"{value:.1f}" if value is not None else "N/A"
        label_y = bar.get_height() + upper_err + (y_max * 0.025)
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, label, ha="center", fontsize=9)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, y_max * 1.1)
    ax.set_xlabel("Architecture")


def _proportion_series(bundle: AnalysisBundle, field: str) -> tuple[list[float | None], list[float | None], list[float | None]]:
    values, lowers, uppers = [], [], []
    for architecture in ARCHITECTURE_LABELS:
        summary = bundle.descriptive_stats[architecture].summary
        value = getattr(summary, field)
        runs = summary.runs
        successes = round(value * runs) if value is not None else None
        ci = wilson_score_interval(successes, runs) if value is not None else None
        values.append(value * 100 if value is not None else None)
        lowers.append(ci.lower * 100 if ci else None)
        uppers.append(ci.upper * 100 if ci else None)
    return values, lowers, uppers


def generate_all_figures(bundle: AnalysisBundle, output_dir: Path | str) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    figure_specs = [
        ("best_line_accuracy", "1_best_line_accuracy.png", "Best-Line Accuracy by Architecture", "Accuracy (%)", 100.0),
        ("ev_classification_accuracy", "2_ev_classification_accuracy.png", "EV Classification Accuracy by Architecture", "Accuracy (%)", 100.0),
        ("freshness_accuracy", "3_freshness_accuracy.png", "Freshness Accuracy by Architecture", "Accuracy (%)", 100.0),
        ("unsupported_claim_rate", "5_unsupported_claim_rate.png", "Unsupported-Claim Rate by Architecture", "Rate (%)", 100.0),
    ]
    for field, filename, title, ylabel, y_max in figure_specs:
        values, lowers, uppers = _proportion_series(bundle, field)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        _bar_chart_with_ci(ax, values, lowers, uppers, title, ylabel, y_max=y_max)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # Consistency (0-1 scale, not a percentage; no CI defined for this metric).
    fig, ax = plt.subplots(figsize=(6, 4.5))
    consistency_values = [bundle.descriptive_stats[a].summary.consistency for a in ARCHITECTURE_LABELS]
    _bar_chart_with_ci(
        ax, consistency_values, [None] * 3, [None] * 3,
        "Consistency by Architecture", "Consistency (0-1)", y_max=1.0,
    )
    fig.tight_layout()
    path = output_dir / "4_consistency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # Total latency (seconds, unbounded — no fixed y_max/100 semantics).
    fig, ax = plt.subplots(figsize=(6, 4.5))
    latency_values = [bundle.descriptive_stats[a].latency_median for a in ARCHITECTURE_LABELS]
    max_latency = max((v for v in latency_values if v is not None), default=1.0)
    _bar_chart_with_ci(
        ax, latency_values, [None] * 3, [None] * 3,
        "Median Total Latency by Architecture", "Latency (seconds)", y_max=max_latency,
    )
    fig.tight_layout()
    path = output_dir / "6_median_latency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    return saved
