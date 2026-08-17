"""Tests for src/analysis/pairing.py (Milestone 14B, Section 32): paired
architecture alignment by (scenario_id, repetition), three-way alignment
for Friedman, N/A filtering, and scenario/architecture grouping — all
against small synthetic RawExperimentRun objects, never the actual
final experiment result.
"""

from datetime import datetime, timezone

from src.analysis.pairing import (
    align_pair,
    align_three_way,
    group_by_architecture,
    group_by_architecture_and_scenario,
    group_by_scenario,
)
from src.evaluation import metrics
from src.experiments.runner import RawExperimentRun
from src.models import ArchitectureType


def _run(
    architecture: ArchitectureType, scenario_id: str, repetition: int,
    best_line_correct: bool | None = True, ev_absolute_error: float | None = None,
) -> RawExperimentRun:
    common = metrics.EvaluationResult(
        scenario_id=scenario_id,
        architecture=architecture,
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
        ev_absolute_error=ev_absolute_error,
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
    return RawExperimentRun(
        experiment_id="synthetic", architecture=architecture, scenario_id=scenario_id,
        repetition=repetition, execution_order_position=0, timestamp=datetime.now(timezone.utc),
        common_result=common, architecture_specific_result={},
    )


def _value_fn_best_line(run: RawExperimentRun):
    return run.common_result.best_line_correct


def _value_fn_ev_error(run: RawExperimentRun):
    return run.common_result.ev_absolute_error


# ---------------------------------------------------------------------------
# align_pair
# ---------------------------------------------------------------------------


def test_align_pair_matches_by_scenario_and_repetition():
    runs = [
        _run(ArchitectureType.RAG, "S001", 1, best_line_correct=True),
        _run(ArchitectureType.TOOL, "S001", 1, best_line_correct=False),
        _run(ArchitectureType.RAG, "S001", 2, best_line_correct=False),
        _run(ArchitectureType.TOOL, "S001", 2, best_line_correct=False),
    ]
    pairs, dropped = align_pair(runs, ArchitectureType.RAG, ArchitectureType.TOOL, _value_fn_best_line)
    assert pairs == [(True, False), (False, False)]
    assert dropped == 0


def test_align_pair_drops_keys_missing_from_one_architecture():
    runs = [
        _run(ArchitectureType.RAG, "S001", 1),
        _run(ArchitectureType.RAG, "S001", 2),
        _run(ArchitectureType.TOOL, "S001", 1),
        # no TOOL run for repetition 2 -> that key must be dropped, not fabricated
    ]
    pairs, dropped = align_pair(runs, ArchitectureType.RAG, ArchitectureType.TOOL, _value_fn_best_line)
    assert len(pairs) == 1


def test_align_pair_drops_pairs_with_na_values_not_zero():
    runs = [
        _run(ArchitectureType.RAG, "S001", 1, ev_absolute_error=0.02),
        _run(ArchitectureType.TOOL, "S001", 1, ev_absolute_error=None),  # N/A
        _run(ArchitectureType.RAG, "S001", 2, ev_absolute_error=0.05),
        _run(ArchitectureType.TOOL, "S001", 2, ev_absolute_error=0.03),
    ]
    pairs, dropped = align_pair(runs, ArchitectureType.RAG, ArchitectureType.TOOL, _value_fn_ev_error)
    assert pairs == [(0.05, 0.03)]
    assert dropped == 1


def test_align_pair_empty_when_no_common_keys():
    runs = [_run(ArchitectureType.RAG, "S001", 1), _run(ArchitectureType.TOOL, "S002", 1)]
    pairs, dropped = align_pair(runs, ArchitectureType.RAG, ArchitectureType.TOOL, _value_fn_best_line)
    assert pairs == []


# ---------------------------------------------------------------------------
# align_three_way
# ---------------------------------------------------------------------------


def test_align_three_way_requires_all_three_architectures_present():
    runs = [
        _run(ArchitectureType.RAG, "S001", 1, ev_absolute_error=0.1),
        _run(ArchitectureType.TOOL, "S001", 1, ev_absolute_error=0.2),
        _run(ArchitectureType.HYBRID, "S001", 1, ev_absolute_error=0.3),
        _run(ArchitectureType.RAG, "S002", 1, ev_absolute_error=0.4),
        _run(ArchitectureType.TOOL, "S002", 1, ev_absolute_error=0.5),
        # missing HYBRID for S002 -> that key must be dropped entirely
    ]
    triples, dropped = align_three_way(runs, _value_fn_ev_error)
    assert triples == [(0.1, 0.2, 0.3)]


def test_align_three_way_drops_na_across_any_architecture():
    runs = [
        _run(ArchitectureType.RAG, "S001", 1, ev_absolute_error=0.1),
        _run(ArchitectureType.TOOL, "S001", 1, ev_absolute_error=None),
        _run(ArchitectureType.HYBRID, "S001", 1, ev_absolute_error=0.3),
    ]
    triples, dropped = align_three_way(runs, _value_fn_ev_error)
    assert triples == []
    assert dropped == 1


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def test_group_by_scenario():
    runs = [_run(ArchitectureType.RAG, "S001", 1), _run(ArchitectureType.RAG, "S002", 1), _run(ArchitectureType.TOOL, "S001", 1)]
    groups = group_by_scenario(runs)
    assert set(groups.keys()) == {"S001", "S002"}
    assert len(groups["S001"]) == 2


def test_group_by_architecture():
    runs = [_run(ArchitectureType.RAG, "S001", 1), _run(ArchitectureType.TOOL, "S001", 1)]
    groups = group_by_architecture(runs)
    assert set(groups.keys()) == {ArchitectureType.RAG, ArchitectureType.TOOL}


def test_group_by_architecture_and_scenario():
    runs = [
        _run(ArchitectureType.RAG, "S001", 1), _run(ArchitectureType.RAG, "S001", 2),
        _run(ArchitectureType.RAG, "S002", 1),
    ]
    groups = group_by_architecture_and_scenario(runs)
    assert len(groups[ArchitectureType.RAG]["S001"]) == 2
    assert len(groups[ArchitectureType.RAG]["S002"]) == 1
