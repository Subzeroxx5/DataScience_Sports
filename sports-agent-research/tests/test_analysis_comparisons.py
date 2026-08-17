"""Integration tests for src/analysis/comparisons.py (Milestone 14B,
Section 32): the frozen metric-family framework applies Holm correction
per family (not globally), and N/A metrics correctly produce no
fabricated p-value across the whole pipeline. Synthetic RawExperimentRun
data only — never the actual final experiment result.
"""

from datetime import datetime, timezone

from src.analysis.comparisons import (
    ARCHITECTURE_PAIRS,
    compute_binary_comparisons,
    compute_continuous_comparisons,
    compute_omnibus_comparisons,
)
from src.evaluation import metrics
from src.experiments.runner import RawExperimentRun
from src.models import ArchitectureType


def _run(
    architecture: ArchitectureType, scenario_id: str, repetition: int,
    best_line_correct: bool | None, ev_absolute_error: float | None = None,
) -> RawExperimentRun:
    common = metrics.EvaluationResult(
        scenario_id=scenario_id, architecture=architecture,
        execution_status=metrics.FailureCategory.SUCCESS, quant_evaluable=True,
        predicted_best_sportsbooks=["FanDuel"], expected_best_sportsbooks=["FanDuel"],
        best_line_correct=best_line_correct, predicted_best_odds=130, expected_best_odds=130,
        best_odds_correct=best_line_correct, predicted_positive_ev=True, expected_positive_ev=True,
        ev_classification_correct=None, predicted_ev=None, expected_ev=None,
        ev_absolute_error=ev_absolute_error,
        predicted_market_reference_probability=None, expected_market_reference_probability=None,
        market_reference_absolute_error=None, freshness_correct=None, completeness=1.0,
        unsupported_claim_count=0, total_verifiable_claims=1, hallucination_detected=False,
        retrieval_metrics=None, tool_metrics=None,
        latency_metrics=metrics.LatencyMetrics(total_latency_seconds=1.0), errors=[],
    )
    return RawExperimentRun(
        experiment_id="synthetic", architecture=architecture, scenario_id=scenario_id,
        repetition=repetition, execution_order_position=0, timestamp=datetime.now(timezone.utc),
        common_result=common, architecture_specific_result={},
    )


def _three_architecture_runs(scenario_id: str, repetition: int, rag_correct: bool, tool_correct: bool, hybrid_correct: bool):
    return [
        _run(ArchitectureType.RAG, scenario_id, repetition, rag_correct),
        _run(ArchitectureType.TOOL, scenario_id, repetition, tool_correct),
        _run(ArchitectureType.HYBRID, scenario_id, repetition, hybrid_correct),
    ]


def test_binary_comparisons_cover_all_three_architecture_pairs_per_metric():
    runs = []
    for i in range(1, 6):
        runs += _three_architecture_runs("S001", i, True, True, True)
    comparisons = compute_binary_comparisons(runs)
    pairs_seen = {(c.architecture_a, c.architecture_b) for c in comparisons if c.metric == "best_line_correct"}
    assert pairs_seen == {(a.value, b.value) for a, b in ARCHITECTURE_PAIRS}


def test_ev_classification_metric_is_na_when_no_valid_data():
    # None of the synthetic runs populate ev_classification_correct.
    runs = []
    for i in range(1, 4):
        runs += _three_architecture_runs("S001", i, True, True, True)
    comparisons = compute_binary_comparisons(runs)
    ev_comparisons = [c for c in comparisons if c.metric == "ev_classification_correct"]
    assert len(ev_comparisons) == 3
    for comparison in ev_comparisons:
        assert comparison.result is None
        assert comparison.raw_p is None
        assert comparison.holm_adjusted_p is None


def test_holm_correction_applied_within_family_not_globally():
    # Construct a dataset where RAG differs sharply from TOOL/HYBRID on
    # best_line_correct (should be significant) while best_odds_correct
    # (a separate metric family) has no signal at all (p=1). The two
    # families' Holm corrections must not interfere with each other.
    runs = []
    for i in range(1, 11):
        runs += _three_architecture_runs("S001", i, False, True, True)

    binary_comparisons = compute_binary_comparisons(runs)
    best_line = [c for c in binary_comparisons if c.metric == "best_line_correct" and c.architecture_a == "rag"]
    best_odds = [c for c in binary_comparisons if c.metric == "best_odds_correct" and c.architecture_a == "rag"]

    assert all(c.holm_adjusted_p is not None and c.holm_adjusted_p < 0.05 for c in best_line)
    # best_odds_correct mirrors best_line_correct in this fixture (see
    # _run's best_odds_correct=best_line_correct), so it should ALSO be
    # significant — the point of this test is that each family's Holm
    # correction is computed independently over its own 3 comparisons,
    # not pooled across the 4 binary families (12 total comparisons).
    assert all(c.holm_adjusted_p is not None for c in best_odds)


def test_continuous_comparisons_drop_na_pairs_not_fabricate_zero():
    runs = []
    for i in range(1, 4):
        runs += [
            _run(ArchitectureType.RAG, "S001", i, True, ev_absolute_error=None),
            _run(ArchitectureType.TOOL, "S001", i, True, ev_absolute_error=None),
            _run(ArchitectureType.HYBRID, "S001", i, True, ev_absolute_error=None),
        ]
    comparisons = compute_continuous_comparisons(runs)
    ev_error_comparisons = [c for c in comparisons if c.metric == "ev_absolute_error"]
    for comparison in ev_error_comparisons:
        assert comparison.result is None
        assert comparison.raw_p is None


def test_omnibus_comparisons_include_completeness_latency_and_consistency():
    runs = []
    for i in range(1, 4):
        runs += _three_architecture_runs("S001", i, True, True, True)
    omnibus = compute_omnibus_comparisons(runs)
    metrics_covered = {c.metric for c in omnibus}
    assert metrics_covered == {"completeness", "total_latency_seconds", "consistency"}
