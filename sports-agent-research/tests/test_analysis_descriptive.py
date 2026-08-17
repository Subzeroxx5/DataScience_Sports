"""Tests for src/analysis/descriptive.py (Milestone 14B, Section 32):
consistency aggregation (mean/median/stdev/min/max over per-scenario
consistency values) and reuse of src.evaluation.metrics — never a
parallel formula. Synthetic EvaluationResult data only.
"""

from src.analysis.descriptive import architecture_descriptive_stats
from src.evaluation import metrics
from src.models import ArchitectureType


def _result(scenario_id: str, best_odds: int, ev_error: float | None = None) -> metrics.EvaluationResult:
    return metrics.EvaluationResult(
        scenario_id=scenario_id,
        architecture=ArchitectureType.TOOL,
        execution_status=metrics.FailureCategory.SUCCESS,
        quant_evaluable=True,
        predicted_best_sportsbooks=["FanDuel"],
        expected_best_sportsbooks=["FanDuel"],
        best_line_correct=True,
        predicted_best_odds=best_odds,
        expected_best_odds=130,
        best_odds_correct=(best_odds == 130),
        predicted_positive_ev=True,
        expected_positive_ev=True,
        ev_classification_correct=True,
        predicted_ev=0.01,
        expected_ev=0.01,
        ev_absolute_error=ev_error,
        predicted_market_reference_probability=0.44,
        expected_market_reference_probability=0.44,
        market_reference_absolute_error=0.0,
        freshness_correct=None,
        completeness=1.0,
        unsupported_claim_count=0,
        total_verifiable_claims=1,
        hallucination_detected=False,
        retrieval_metrics=None,
        tool_metrics=None,
        latency_metrics=metrics.LatencyMetrics(total_latency_seconds=1.0),
        errors=[],
    )


def test_consistency_aggregation_perfect_across_all_scenarios():
    # Two scenarios, each with 3 identical repetitions -> consistency 1.0
    # for both -> mean/median/min/max all 1.0, stdev 0.0.
    results = (
        [_result("S001", 130) for _ in range(3)]
        + [_result("S002", 140) for _ in range(3)]
    )
    stats = architecture_descriptive_stats("tool", results)
    assert stats.consistency_mean == 1.0
    assert stats.consistency_median == 1.0
    assert stats.consistency_stdev == 0.0
    assert stats.consistency_min == 1.0
    assert stats.consistency_max == 1.0
    assert stats.consistency_scenario_count == 2


def test_consistency_aggregation_reflects_a_mixed_scenario():
    # S001: 2/3 repetitions agree -> consistency 2/3.
    # S002: 3/3 agree -> consistency 1.0.
    results = (
        [_result("S001", 130), _result("S001", 130), _result("S001", 125)]
        + [_result("S002", 140), _result("S002", 140), _result("S002", 140)]
    )
    stats = architecture_descriptive_stats("tool", results)
    assert stats.consistency_min < 1.0
    assert stats.consistency_max == 1.0
    assert stats.consistency_mean == (2 / 3 + 1.0) / 2


def test_consistency_excludes_scenarios_with_a_single_repetition():
    # A scenario with only 1 repetition contributes no consistency value
    # (Milestone 11's own rule, reused here — not re-derived).
    results = [_result("S001", 130)]  # single repetition
    stats = architecture_descriptive_stats("tool", results)
    assert stats.consistency_scenario_count == 0
    assert stats.consistency_mean is None


def test_median_ev_error_ignores_none_values_not_zero():
    results = [_result("S001", 130, ev_error=0.1), _result("S001", 130, ev_error=None), _result("S001", 130, ev_error=0.3)]
    stats = architecture_descriptive_stats("tool", results)
    assert stats.median_ev_absolute_error == 0.2  # median of [0.1, 0.3], None excluded


def test_median_ev_error_none_when_all_values_missing():
    results = [_result("S001", 130, ev_error=None)]
    stats = architecture_descriptive_stats("tool", results)
    assert stats.median_ev_absolute_error is None


def test_runs_with_unsupported_claim_counts_correctly():
    clean = _result("S001", 130)
    results = [clean, clean, clean]
    stats = architecture_descriptive_stats("tool", results)
    assert stats.runs_with_unsupported_claim == 0


def test_latency_aggregation_matches_generic_metrics_functions():
    results = [_result("S001", 130), _result("S002", 130), _result("S003", 130)]
    for result, latency in zip(results, [1.0, 2.0, 3.0]):
        result.latency_metrics.total_latency_seconds = latency
    stats = architecture_descriptive_stats("tool", results)
    assert stats.latency_mean == 2.0
    assert stats.latency_median == 2.0
    assert stats.latency_min == 1.0
    assert stats.latency_max == 3.0


def test_descriptive_stats_summary_is_the_unmodified_metrics_summarize_output():
    results = [_result("S001", 130), _result("S001", 130)]
    stats = architecture_descriptive_stats("tool", results)
    expected_summary = metrics.summarize(results)
    assert stats.summary == expected_summary
