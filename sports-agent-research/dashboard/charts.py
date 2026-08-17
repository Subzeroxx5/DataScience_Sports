"""Comparison charts (Milestone 13, Section 14).

`build_metric_dataframe` is pure (no Streamlit dependency) and is what
tests exercise directly. `render_*` is a thin Streamlit wrapper around
it — bar charts only, no 3D, no decorative visualizations (Section 14).
All values come straight from an existing `metrics.ArchitectureSummary`
— nothing here computes a metric.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import ArchitectureComparison, ArchitectureSummary

# (attribute on ArchitectureSummary, chart title) — Section 14's exact list.
COMPARISON_METRICS: list[tuple[str, str]] = [
    ("best_line_accuracy", "Best-Line Accuracy by Architecture"),
    ("ev_classification_accuracy", "EV Classification Accuracy by Architecture"),
    ("freshness_accuracy", "Freshness Accuracy by Architecture"),
    ("mean_completeness", "Completeness by Architecture"),
    ("unsupported_claim_rate", "Unsupported-Claim Rate by Architecture"),
    ("consistency", "Consistency by Architecture"),
    ("mean_total_latency_seconds", "Mean Latency (seconds) by Architecture"),
]

_ARCHITECTURE_LABELS = [("RAG", "rag_summary"), ("TOOL", "tool_summary"), ("HYBRID", "hybrid_summary")]


def build_metric_dataframe(comparison: ArchitectureComparison, metric_attr: str) -> pd.DataFrame:
    """One row per architecture that actually ran, reading `metric_attr`
    straight off its ArchitectureSummary. Architectures that did not run
    (summary is None) or whose value is None (not applicable/no data)
    are omitted rather than shown as a misleading 0."""
    rows = []
    for label, attr_name in _ARCHITECTURE_LABELS:
        summary: ArchitectureSummary | None = getattr(comparison, attr_name)
        if summary is None:
            continue
        value = getattr(summary, metric_attr)
        if value is None:
            continue
        rows.append({"architecture": label, metric_attr: value})
    return pd.DataFrame(rows).set_index("architecture") if rows else pd.DataFrame(columns=[metric_attr])


def render_comparison_charts(comparison: ArchitectureComparison) -> None:
    """Renders every Section 14 chart as its own single-metric bar chart
    (never overloading one chart with unrelated metrics)."""
    import streamlit as st

    for metric_attr, title in COMPARISON_METRICS:
        df = build_metric_dataframe(comparison, metric_attr)
        st.subheader(title)
        if df.empty:
            st.caption("No data available for this metric.")
            continue
        st.bar_chart(df, y=metric_attr)
