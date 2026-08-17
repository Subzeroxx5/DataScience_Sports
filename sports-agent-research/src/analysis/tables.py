"""Required output tables (Milestone 14B, Section 28). Each `build_table_N`
function returns a list-of-dicts (JSON/CSV-friendly), reading only
already-computed values from an AnalysisBundle — no new computation.
"""

from __future__ import annotations

from src.analysis.bundle import AnalysisBundle

ARCHITECTURE_LABELS = ["rag", "tool", "hybrid"]


def _pct(value: float | None) -> float | None:
    return round(value * 100, 2) if value is not None else None


def build_table_1_architecture_results(bundle: AnalysisBundle) -> list[dict]:
    rows = []
    for architecture in ARCHITECTURE_LABELS:
        stats = bundle.descriptive_stats[architecture]
        summary = stats.summary
        rows.append({
            "architecture": architecture,
            "success_rate_pct": _pct(summary.success_rate),
            "best_line_accuracy_pct": _pct(summary.best_line_accuracy),
            "best_odds_accuracy_pct": _pct(summary.best_odds_accuracy),
            "ev_classification_accuracy_pct": _pct(summary.ev_classification_accuracy),
            "freshness_accuracy_pct": _pct(summary.freshness_accuracy),
            "mean_completeness_pct": _pct(summary.mean_completeness),
            "unsupported_claim_rate_pct": _pct(summary.unsupported_claim_rate),
            "consistency": summary.consistency,
            "median_latency_seconds": round(stats.latency_median, 3) if stats.latency_median is not None else None,
        })
    return rows


def build_table_2_binary_pairwise(bundle: AnalysisBundle) -> list[dict]:
    rows = []
    for comparison in bundle.binary_comparisons:
        result = comparison.result
        rows.append({
            "metric": comparison.metric,
            "architecture_a": comparison.architecture_a,
            "architecture_b": comparison.architecture_b,
            "paired_n": result.paired_n if result else 0,
            "accuracy_a_pct": _pct(result.accuracy_a) if result else None,
            "accuracy_b_pct": _pct(result.accuracy_b) if result else None,
            "difference_pp": round(result.difference_pp, 2) if result else None,
            "a_correct_b_incorrect": result.a_correct_b_incorrect if result else None,
            "a_incorrect_b_correct": result.a_incorrect_b_correct if result else None,
            "raw_p": comparison.raw_p,
            "holm_adjusted_p": comparison.holm_adjusted_p,
            "note": result.note if result else "no valid paired observations",
        })
    return rows


def build_table_3_continuous_comparisons(bundle: AnalysisBundle) -> list[dict]:
    rows = []
    for comparison in bundle.continuous_comparisons:
        result = comparison.result
        rows.append({
            "metric": comparison.metric,
            "architecture_a": comparison.architecture_a,
            "architecture_b": comparison.architecture_b,
            "paired_n": result.paired_n if result else 0,
            "median_a": result.median_a if result else None,
            "median_b": result.median_b if result else None,
            "difference": (
                round(result.median_a - result.median_b, 6)
                if result and result.median_a is not None and result.median_b is not None
                else None
            ),
            "statistic": result.statistic if result else None,
            "raw_p": comparison.raw_p,
            "holm_adjusted_p": comparison.holm_adjusted_p,
            "note": result.note if result else "no valid paired observations",
        })
    return rows


def build_table_4_failure_counts(bundle: AnalysisBundle) -> list[dict]:
    return [
        {
            "architecture": item.architecture,
            "category": item.category,
            "count": item.count,
            "percentage_of_observations": round(item.percentage_of_observations, 2),
            "scenarios_affected": item.scenarios_affected,
        }
        for item in bundle.failure_breakdown
    ]


def build_table_5_hybrid_reconciliation(bundle: AnalysisBundle) -> dict | None:
    if bundle.hybrid_conflict is None:
        return None
    summary = bundle.hybrid_conflict
    return {
        "source_agreements": summary["source_agreements"],
        "source_conflicts": summary["source_conflicts"],
        "correct_conflict_resolutions": summary["correct_conflict_resolutions"],
        "conflict_resolution_accuracy_pct": _pct(summary["conflict_resolution_accuracy"]),
        "stale_rag_conflicts": summary["stale_rag_conflicts"],
        "stale_rag_incorrectly_promoted": summary["stale_rag_incorrectly_promoted"],
        "tool_only_recoveries": summary["tool_only_recoveries"],
        "source_reconciliation_failures": summary["source_reconciliation_failures"],
    }


def build_omnibus_table(bundle: AnalysisBundle) -> list[dict]:
    """Not one of the 5 numbered tables, but required by the
    Verification section's CONSISTENCY block — included as machine-
    readable output alongside the numbered tables."""
    rows = []
    for comparison in bundle.omnibus_comparisons:
        result = comparison.result
        rows.append({
            "metric": comparison.metric,
            "n": result.n if result else 0,
            "statistic": result.statistic if result else None,
            "p_value": result.p_value if result else None,
            "note": result.note if result else "no matched blocks",
        })
    return rows
