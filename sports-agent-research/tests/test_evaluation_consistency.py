"""Tests for the unified evaluation framework's consistency metric
(Milestone 11, Steps 14, 27): the research-relevant signature comparison
used to measure repeated-run stability. Milestone 11 only defines and
tests this calculation — Milestone 12 is what will actually execute
repeated runs.
"""

import pytest

from src.evaluation import metrics
from src.models import ArchitectureType


def _sig(
    best_sportsbooks=("FanDuel",),
    best_odds=130,
    positive_ev=True,
    market_reference_probability=None,
    expected_value=None,
) -> metrics.ConsistencySignature:
    return metrics.compute_consistency_signature(
        best_sportsbooks=best_sportsbooks,
        best_odds=best_odds,
        positive_ev=positive_ev,
        market_reference_probability=market_reference_probability,
        expected_value=expected_value,
    )


# ---------------------------------------------------------------------------
# Signature construction
# ---------------------------------------------------------------------------


def test_signature_sorts_best_sportsbooks_for_stable_comparison():
    sig_a = _sig(best_sportsbooks=["FanDuel", "DraftKings"])
    sig_b = _sig(best_sportsbooks=["DraftKings", "FanDuel"])
    assert sig_a == sig_b


def test_signature_rounds_floats_to_avoid_noise_false_inconsistency():
    sig_a = metrics.compute_consistency_signature(
        best_sportsbooks=["FanDuel"], best_odds=130, positive_ev=True,
        market_reference_probability=0.440000001, expected_value=0.01,
    )
    sig_b = metrics.compute_consistency_signature(
        best_sportsbooks=["FanDuel"], best_odds=130, positive_ev=True,
        market_reference_probability=0.440000002, expected_value=0.01,
    )
    assert sig_a == sig_b


def test_signature_excludes_latency_and_reasoning_text():
    # ConsistencySignature has no field for latency or free-form text —
    # structurally cannot be compared on them (Step 14).
    fields = set(metrics.ConsistencySignature.model_fields)
    assert not any("latency" in f for f in fields)
    assert not any("reasoning" in f or "summary" in f for f in fields)


# ---------------------------------------------------------------------------
# Consistency score (Step 14, 27 — hand-checkable)
# ---------------------------------------------------------------------------


def test_consistency_identical_runs_is_1():
    signatures = [_sig(), _sig(), _sig()]
    assert metrics.compute_consistency(signatures) == pytest.approx(1.0)


def test_consistency_hand_example_two_of_three_match():
    # Run 1: FanDuel +130, positive_ev=True
    # Run 2: FanDuel +130, positive_ev=True  (identical to Run 1)
    # Run 3: DraftKings +125, positive_ev=True (different)
    run_1 = _sig(best_sportsbooks=["FanDuel"], best_odds=130, positive_ev=True)
    run_2 = _sig(best_sportsbooks=["FanDuel"], best_odds=130, positive_ev=True)
    run_3 = _sig(best_sportsbooks=["DraftKings"], best_odds=125, positive_ev=True)
    assert metrics.compute_consistency([run_1, run_2, run_3]) == pytest.approx(2 / 3)


def test_consistency_all_different_is_1_over_n():
    signatures = [
        _sig(best_odds=100), _sig(best_odds=110), _sig(best_odds=120), _sig(best_odds=130),
    ]
    assert metrics.compute_consistency(signatures) == pytest.approx(0.25)


def test_consistency_differs_on_odds_alone():
    run_1 = _sig(best_odds=130)
    run_2 = _sig(best_odds=125)
    assert metrics.compute_consistency([run_1, run_2]) == pytest.approx(0.5)


def test_consistency_differs_on_positive_ev_alone():
    run_1 = _sig(positive_ev=True)
    run_2 = _sig(positive_ev=False)
    assert metrics.compute_consistency([run_1, run_2]) == pytest.approx(0.5)


def test_consistency_differs_on_market_reference_probability():
    run_1 = _sig(market_reference_probability=0.44)
    run_2 = _sig(market_reference_probability=0.45)
    assert metrics.compute_consistency([run_1, run_2]) == pytest.approx(0.5)


def test_consistency_empty_input_is_none():
    assert metrics.compute_consistency([]) is None


def test_consistency_single_run_is_1():
    assert metrics.compute_consistency([_sig()]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Consistency integrated into ArchitectureSummary (Step 22)
# ---------------------------------------------------------------------------


def _result_with_signature(scenario_id: str, *, best_odds: int, positive_ev: bool) -> metrics.EvaluationResult:
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
        predicted_positive_ev=positive_ev,
        expected_positive_ev=True,
        ev_classification_correct=(positive_ev is True),
        predicted_ev=0.01,
        expected_ev=0.01,
        ev_absolute_error=0.0,
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
        latency_metrics=metrics.LatencyMetrics(total_latency_seconds=0.01),
        errors=[],
    )


def test_summarize_computes_consistency_when_scenario_repeated():
    # Same scenario_id repeated 3 times, 2 identical + 1 different ->
    # modal consistency 2/3 for that scenario; only one scenario group
    # exists, so the summary-level consistency equals it.
    results = [
        _result_with_signature("S1", best_odds=130, positive_ev=True),
        _result_with_signature("S1", best_odds=130, positive_ev=True),
        _result_with_signature("S1", best_odds=125, positive_ev=True),
    ]
    summary = metrics.summarize(results)
    assert summary.consistency == pytest.approx(2 / 3)


def test_summarize_consistency_is_none_without_repetitions():
    # Milestone 11 defines the calculation; without repeated runs of the
    # same scenario there is nothing to measure yet (Milestone 12's job).
    results = [
        _result_with_signature("S1", best_odds=130, positive_ev=True),
        _result_with_signature("S2", best_odds=125, positive_ev=True),
    ]
    summary = metrics.summarize(results)
    assert summary.consistency is None
