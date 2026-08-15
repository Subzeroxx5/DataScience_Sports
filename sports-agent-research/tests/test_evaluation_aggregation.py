"""Tests for the unified evaluation framework's generic aggregation
(Milestone 11, Step 21) and architecture summary / cross-architecture
comparison structures (Steps 22-23, 31-32).
"""

import pytest

from src.evaluation import metrics
from src.models import ArchitectureType


# ---------------------------------------------------------------------------
# Generic aggregation (Step 21, 25) — hand-checkable
# ---------------------------------------------------------------------------


def test_rate_hand_example_best_line():
    # 4 scenarios: correct, correct, incorrect, correct -> 3/4 = 0.75
    values = [True, True, False, True]
    assert metrics.rate(values) == pytest.approx(0.75)


def test_rate_ev_classification_with_not_applicable():
    # correct, incorrect, N/A, correct -> denominator 3, accuracy 2/3
    values = [True, False, None, True]
    assert metrics.rate(values) == pytest.approx(2 / 3)


def test_mean_completeness_hand_example():
    assert metrics.mean([1.0, 0.75, 0.5]) == pytest.approx(0.75)


def test_rate_ignores_none_not_treats_as_false():
    # All-None input must not silently become 0.0 accuracy.
    assert metrics.rate([None, None]) is None


def test_mean_ignores_none_not_treats_as_zero():
    assert metrics.mean([None, 1.0, None, 3.0]) == pytest.approx(2.0)  # (1+3)/2, not (1+0+0+3)/4


def test_count_counts_everything_including_none():
    assert metrics.count([1, None, 3]) == 3


def test_valid_count_excludes_none():
    assert metrics.valid_count([1, None, 3, None]) == 2


def test_median_basic():
    assert metrics.median([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_median_ignores_none():
    assert metrics.median([1.0, None, 3.0]) == pytest.approx(2.0)


def test_median_empty_is_none():
    assert metrics.median([]) is None


def test_population_stdev_basic():
    # Population stdev of [2, 4, 4, 4, 5, 5, 7, 9] is 2.0 (classic example).
    values = [2, 4, 4, 4, 5, 5, 7, 9]
    assert metrics.population_stdev(values) == pytest.approx(2.0)


def test_population_stdev_single_value_is_zero():
    assert metrics.population_stdev([5.0]) == pytest.approx(0.0)


def test_population_stdev_empty_is_none():
    assert metrics.population_stdev([]) is None


def test_minimum_and_maximum():
    assert metrics.minimum([3.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert metrics.maximum([3.0, 1.0, 2.0]) == pytest.approx(3.0)


def test_minimum_ignores_none():
    assert metrics.minimum([None, 5.0, 2.0]) == pytest.approx(2.0)


def test_minimum_maximum_empty_is_none():
    assert metrics.minimum([]) is None
    assert metrics.maximum([]) is None


# ---------------------------------------------------------------------------
# ArchitectureSummary / summarize() (Steps 21-22)
# ---------------------------------------------------------------------------


def _latency(total: float = 0.01) -> metrics.LatencyMetrics:
    return metrics.LatencyMetrics(total_latency_seconds=total)


def _result(
    scenario_id: str,
    *,
    execution_status: metrics.FailureCategory = metrics.FailureCategory.SUCCESS,
    best_line_correct: bool | None = True,
    best_odds_correct: bool | None = True,
    ev_classification_correct: bool | None = None,
    ev_absolute_error: float | None = None,
    market_reference_absolute_error: float | None = None,
    freshness_correct: bool | None = None,
    completeness: float | None = 1.0,
    unsupported_claim_count: int = 0,
    total_verifiable_claims: int = 1,
    hallucination_detected: bool = False,
    predicted_best_sportsbooks: list[str] | None = None,
    predicted_best_odds: int | None = 125,
    predicted_positive_ev: bool | None = None,
    predicted_market_reference_probability: float | None = None,
    predicted_ev: float | None = None,
) -> metrics.EvaluationResult:
    return metrics.EvaluationResult(
        scenario_id=scenario_id,
        architecture=ArchitectureType.TOOL,
        execution_status=execution_status,
        quant_evaluable=ev_classification_correct is not None,
        predicted_best_sportsbooks=predicted_best_sportsbooks or ["FanDuel"],
        expected_best_sportsbooks=["FanDuel"],
        best_line_correct=best_line_correct,
        predicted_best_odds=predicted_best_odds,
        expected_best_odds=125,
        best_odds_correct=best_odds_correct,
        predicted_positive_ev=predicted_positive_ev,
        expected_positive_ev=None,
        ev_classification_correct=ev_classification_correct,
        predicted_ev=predicted_ev,
        expected_ev=None,
        ev_absolute_error=ev_absolute_error,
        predicted_market_reference_probability=predicted_market_reference_probability,
        expected_market_reference_probability=None,
        market_reference_absolute_error=market_reference_absolute_error,
        freshness_correct=freshness_correct,
        completeness=completeness,
        unsupported_claim_count=unsupported_claim_count,
        total_verifiable_claims=total_verifiable_claims,
        hallucination_detected=hallucination_detected,
        retrieval_metrics=None,
        tool_metrics=None,
        latency_metrics=_latency(),
        errors=[],
    )


def test_summarize_success_rate_and_failure_counts():
    results = [
        _result("S1", execution_status=metrics.FailureCategory.SUCCESS),
        _result("S2", execution_status=metrics.FailureCategory.SUCCESS),
        _result("S3", execution_status=metrics.FailureCategory.TOOL_FAILURE, best_line_correct=None, best_odds_correct=None, completeness=None),
    ]
    summary = metrics.summarize(results)
    assert summary.runs == 3
    assert summary.successes == 2
    assert summary.failures == 1
    assert summary.success_rate == pytest.approx(2 / 3)
    assert summary.failure_counts == {"success": 2, "tool_failure": 1}


def test_summarize_preserves_raw_results():
    results = [_result("S1"), _result("S2")]
    summary = metrics.summarize(results)
    assert summary.raw_results == results
    assert len(summary.raw_results) == 2


def test_summarize_unsupported_claim_rate_hand_example():
    results = [
        _result("S1", unsupported_claim_count=1, total_verifiable_claims=1, hallucination_detected=True),
        _result("S2", unsupported_claim_count=0, total_verifiable_claims=1),
        _result("S3", unsupported_claim_count=0, total_verifiable_claims=1),
        _result("S4", unsupported_claim_count=0, total_verifiable_claims=1),
        _result("S5", unsupported_claim_count=0, total_verifiable_claims=1),
    ]
    summary = metrics.summarize(results)
    assert summary.unsupported_claim_rate == pytest.approx(0.20)


def test_summarize_raises_on_empty_input():
    with pytest.raises(ValueError):
        metrics.summarize([])


def test_summarize_never_averages_across_na_values():
    # If every result is non-quant-evaluable, EV accuracy must be None,
    # never a misleading 0%.
    results = [_result("S1", ev_classification_correct=None), _result("S2", ev_classification_correct=None)]
    summary = metrics.summarize(results)
    assert summary.ev_classification_accuracy is None


# ---------------------------------------------------------------------------
# ArchitectureComparison (Step 23, 31-32) — no automatic winner
# ---------------------------------------------------------------------------


def test_compare_architectures_holds_no_winner_field():
    assert "winner" not in metrics.ArchitectureComparison.model_fields
    assert "best_architecture" not in metrics.ArchitectureComparison.model_fields


def test_compare_architectures_builds_all_three_summaries():
    rag_results = [_result("S1")]
    tool_results = [_result("S1")]
    hybrid_results = [_result("S1")]
    comparison = metrics.compare_architectures(rag_results, tool_results, hybrid_results)
    assert comparison.rag_summary is not None
    assert comparison.tool_summary is not None
    assert comparison.hybrid_summary is not None
    assert comparison.scenario_count == 1
    assert comparison.repetitions == 1


def test_compare_architectures_handles_missing_architecture():
    comparison = metrics.compare_architectures(tool_results=[_result("S1")])
    assert comparison.rag_summary is None
    assert comparison.tool_summary is not None
    assert comparison.hybrid_summary is None


def test_compare_architectures_detects_repetitions():
    repeated = [_result("S1"), _result("S1"), _result("S2")]
    comparison = metrics.compare_architectures(tool_results=repeated)
    assert comparison.scenario_count == 2
    assert comparison.repetitions == 2  # S1 appears twice


# ---------------------------------------------------------------------------
# Serialization / round-trip (Steps 31-32)
# ---------------------------------------------------------------------------


def test_evaluation_result_round_trips_through_json():
    result = _result("S1", predicted_market_reference_probability=0.44, predicted_ev=0.01)
    payload = result.model_dump_json()
    restored = metrics.EvaluationResult.model_validate_json(payload)
    assert restored == result


def test_architecture_summary_round_trips_through_json():
    summary = metrics.summarize([_result("S1"), _result("S2")])
    payload = summary.model_dump_json()
    restored = metrics.ArchitectureSummary.model_validate_json(payload)
    assert restored == summary
    assert len(restored.raw_results) == 2


def test_architecture_comparison_round_trips_through_json():
    comparison = metrics.compare_architectures(tool_results=[_result("S1")])
    payload = comparison.model_dump_json()
    restored = metrics.ArchitectureComparison.model_validate_json(payload)
    assert restored == comparison
