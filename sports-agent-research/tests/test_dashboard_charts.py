"""Tests for dashboard/charts.py (Milestone 13, Section 14): pure
DataFrame construction from an existing metrics.ArchitectureComparison —
no chart-rendering dependency, no metric computation of its own."""

from dashboard import charts
from src.evaluation import metrics
from src.models import ArchitectureType


def _result(scenario_id: str, best_line_correct: bool | None, latency: float) -> metrics.EvaluationResult:
    return metrics.EvaluationResult(
        scenario_id=scenario_id,
        architecture=ArchitectureType.TOOL,
        execution_status=metrics.FailureCategory.SUCCESS,
        quant_evaluable=True,
        predicted_best_sportsbooks=["FanDuel"],
        expected_best_sportsbooks=["FanDuel"],
        best_line_correct=best_line_correct,
        predicted_best_odds=130,
        expected_best_odds=130,
        best_odds_correct=True,
        predicted_positive_ev=True,
        expected_positive_ev=True,
        ev_classification_correct=True,
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
        latency_metrics=metrics.LatencyMetrics(total_latency_seconds=latency),
        errors=[],
    )


def _comparison() -> metrics.ArchitectureComparison:
    tool_results = [_result("S001", True, 0.1), _result("S002", False, 0.2)]
    return metrics.compare_architectures(tool_results=tool_results)


def test_build_metric_dataframe_has_one_row_per_architecture_that_ran():
    comparison = _comparison()
    df = charts.build_metric_dataframe(comparison, "best_line_accuracy")
    assert list(df.index) == ["TOOL"]
    assert df.loc["TOOL", "best_line_accuracy"] == 0.5


def test_build_metric_dataframe_omits_architectures_that_did_not_run():
    comparison = _comparison()
    df = charts.build_metric_dataframe(comparison, "best_line_accuracy")
    assert "RAG" not in df.index
    assert "HYBRID" not in df.index


def test_build_metric_dataframe_empty_when_metric_is_none_for_every_architecture():
    # freshness_accuracy is None here because no run in the fixture has a
    # freshness_correct value at all.
    comparison = _comparison()
    df = charts.build_metric_dataframe(comparison, "freshness_accuracy")
    assert df.empty


def test_build_metric_dataframe_no_data_at_all_returns_empty_dataframe():
    comparison = metrics.compare_architectures()
    df = charts.build_metric_dataframe(comparison, "best_line_accuracy")
    assert df.empty


def test_comparison_metrics_list_matches_milestone_section_14():
    attrs = {attr for attr, _title in charts.COMPARISON_METRICS}
    assert attrs == {
        "best_line_accuracy",
        "ev_classification_accuracy",
        "freshness_accuracy",
        "mean_completeness",
        "unsupported_claim_rate",
        "consistency",
        "mean_total_latency_seconds",
    }
