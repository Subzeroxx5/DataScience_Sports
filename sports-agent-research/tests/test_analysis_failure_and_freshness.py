"""Tests for src/analysis/failure_analysis.py and
src/analysis/freshness_analysis.py (Milestone 14B, Section 32): failure
denominator handling (percentage relative to that architecture's own
run count, not a global total) and freshness error classification
(known-stale vs. unknown-incorrect). Synthetic data only.
"""

from datetime import datetime, timezone

from src.analysis.failure_analysis import failure_breakdown
from src.analysis.freshness_analysis import freshness_focused_runs, freshness_stats_by_architecture
from src.evaluation import metrics
from src.experiments.config import ExperimentScenario
from src.experiments.runner import RawExperimentRun
from src.models import ArchitectureType, MarketType


def _run(
    architecture: ArchitectureType, scenario_id: str, repetition: int,
    status: metrics.FailureCategory, freshness_correct: bool | None = None,
    predicted_best_odds: int | None = 130,
) -> RawExperimentRun:
    common = metrics.EvaluationResult(
        scenario_id=scenario_id, architecture=architecture, execution_status=status, quant_evaluable=True,
        predicted_best_sportsbooks=["FanDuel"] if predicted_best_odds is not None else [],
        expected_best_sportsbooks=["FanDuel"],
        best_line_correct=None, predicted_best_odds=predicted_best_odds, expected_best_odds=130,
        best_odds_correct=None, predicted_positive_ev=None, expected_positive_ev=None,
        ev_classification_correct=None, predicted_ev=None, expected_ev=None, ev_absolute_error=None,
        predicted_market_reference_probability=None, expected_market_reference_probability=None,
        market_reference_absolute_error=None, freshness_correct=freshness_correct, completeness=None,
        unsupported_claim_count=0, total_verifiable_claims=0, hallucination_detected=False,
        retrieval_metrics=None, tool_metrics=None,
        latency_metrics=metrics.LatencyMetrics(total_latency_seconds=1.0), errors=[],
    )
    return RawExperimentRun(
        experiment_id="synthetic", architecture=architecture, scenario_id=scenario_id,
        repetition=repetition, execution_order_position=0, timestamp=datetime.now(timezone.utc),
        common_result=common, architecture_specific_result={},
    )


# ---------------------------------------------------------------------------
# Failure denominator handling
# ---------------------------------------------------------------------------


def test_failure_percentage_is_relative_to_that_architectures_own_total():
    # RAG: 2 runs, 1 failure -> 50%. TOOL: 4 runs, 1 failure -> 25%.
    # A global-denominator bug would report both as the same percentage
    # (or scale by the wrong total).
    runs = [
        _run(ArchitectureType.RAG, "S001", 1, metrics.FailureCategory.SUCCESS),
        _run(ArchitectureType.RAG, "S001", 2, metrics.FailureCategory.RETRIEVAL_FAILURE),
        _run(ArchitectureType.TOOL, "S001", 1, metrics.FailureCategory.SUCCESS),
        _run(ArchitectureType.TOOL, "S001", 2, metrics.FailureCategory.SUCCESS),
        _run(ArchitectureType.TOOL, "S001", 3, metrics.FailureCategory.SUCCESS),
        _run(ArchitectureType.TOOL, "S001", 4, metrics.FailureCategory.TOOL_FAILURE),
    ]
    breakdown = failure_breakdown(runs)
    rag_failure = next(item for item in breakdown if item.architecture == "rag")
    tool_failure = next(item for item in breakdown if item.architecture == "tool")
    assert rag_failure.percentage_of_observations == 50.0
    assert tool_failure.percentage_of_observations == 25.0


def test_failure_breakdown_excludes_successful_categories():
    runs = [
        _run(ArchitectureType.RAG, "S001", 1, metrics.FailureCategory.SUCCESS),
        _run(ArchitectureType.RAG, "S001", 2, metrics.FailureCategory.QUANT_INSUFFICIENT_DATA),  # also "successful"
    ]
    breakdown = failure_breakdown(runs)
    assert breakdown == []  # both categories are in SUCCESSFUL_CATEGORIES -> no failures recorded


def test_failure_breakdown_lists_scenarios_affected():
    runs = [
        _run(ArchitectureType.RAG, "S001", 1, metrics.FailureCategory.RETRIEVAL_FAILURE),
        _run(ArchitectureType.RAG, "S002", 1, metrics.FailureCategory.RETRIEVAL_FAILURE),
        _run(ArchitectureType.RAG, "S001", 2, metrics.FailureCategory.RETRIEVAL_FAILURE),
    ]
    breakdown = failure_breakdown(runs)
    assert len(breakdown) == 1
    assert breakdown[0].scenarios_affected == ["S001", "S002"]
    assert breakdown[0].count == 3


def test_failure_breakdown_never_drops_failed_runs():
    runs = [_run(ArchitectureType.RAG, "S001", i, metrics.FailureCategory.LLM_OUTPUT_INVALID) for i in range(1, 6)]
    breakdown = failure_breakdown(runs)
    assert breakdown[0].count == 5


# ---------------------------------------------------------------------------
# Freshness-focused subset and error classification
# ---------------------------------------------------------------------------


def test_freshness_focused_runs_uses_the_milestone11_applicability_field():
    runs = [
        _run(ArchitectureType.RAG, "S001", 1, metrics.FailureCategory.SUCCESS, freshness_correct=True),
        _run(ArchitectureType.RAG, "S002", 1, metrics.FailureCategory.SUCCESS, freshness_correct=None),  # not applicable
    ]
    focused = freshness_focused_runs(runs)
    assert len(focused) == 1
    assert focused[0].scenario_id == "S001"


def _manifest_entry(scenario_id: str, game_id: str) -> ExperimentScenario:
    return ExperimentScenario(
        scenario_id=scenario_id, game_id=game_id, market_type=MarketType.MONEYLINE,
        selected_outcome="Test Team", query="test query",
    )


def test_freshness_stats_reports_zero_errors_when_all_correct():
    runs = [_run(ArchitectureType.RAG, "S009", i, metrics.FailureCategory.SUCCESS, freshness_correct=True) for i in range(1, 4)]
    manifest = [_manifest_entry("S009", "G-TEST")]
    stats = freshness_stats_by_architecture("rag", runs, manifest)
    assert stats.cases_evaluated == 3
    assert stats.correct == 3
    assert stats.used_known_stale_value == 0
    assert stats.used_unknown_incorrect_value == 0
    assert stats.used_correct_current_value == 3


def test_freshness_stats_confidence_interval_is_none_for_zero_cases():
    stats = freshness_stats_by_architecture("rag", [], [])
    assert stats.cases_evaluated == 0
    assert stats.confidence_interval is None
    assert stats.accuracy is None
