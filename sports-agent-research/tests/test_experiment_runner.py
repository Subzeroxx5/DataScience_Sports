"""Tests for src/experiments/runner.py (Milestone 12): unified-evaluator
reuse, ground-truth isolation, failure isolation, common-query delivery,
consistency aggregation, and expected-run-count validation. See
tests/test_experiment_persistence.py for file-level persistence/
duplicate-protection/reproducibility coverage.
"""

from datetime import datetime, timezone

import pytest

from src.agents.base import AgentRequest
from src.evaluation import hybrid_agent_evaluation as hybrid_eval
from src.evaluation import metrics
from src.evaluation import rag_agent_evaluation as rag_eval
from src.evaluation import tool_agent_evaluation as tool_eval
from src.evaluation.ground_truth import generate_all_ground_truth
from src.experiments import runner
from src.experiments.config import ExperimentConfig
from src.models import ArchitectureType


def _config(tmp_path, **overrides) -> ExperimentConfig:
    defaults = dict(
        experiment_id="run-test",
        experiment_name="run-test",
        scenario_ids=["S001"],
        repetitions=1,
        output_dir=str(tmp_path),
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ---------------------------------------------------------------------------
# Unified evaluator reuse (Step 19) — no runner-local accuracy formulas
# ---------------------------------------------------------------------------


def test_evaluate_scenario_dispatch_is_the_real_evaluator_functions():
    assert runner._EVALUATE_SCENARIO[ArchitectureType.RAG] is rag_eval.evaluate_scenario
    assert runner._EVALUATE_SCENARIO[ArchitectureType.TOOL] is tool_eval.evaluate_scenario
    assert runner._EVALUATE_SCENARIO[ArchitectureType.HYBRID] is hybrid_eval.evaluate_scenario


def test_to_common_result_dispatch_is_the_real_converter_functions():
    assert runner._TO_COMMON_RESULT[ArchitectureType.RAG] is rag_eval.to_common_result
    assert runner._TO_COMMON_RESULT[ArchitectureType.TOOL] is tool_eval.to_common_result
    assert runner._TO_COMMON_RESULT[ArchitectureType.HYBRID] is hybrid_eval.to_common_result


def test_runner_module_defines_no_accuracy_formula_of_its_own():
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "src" / "experiments" / "runner.py").read_text()
    tree = ast.parse(source)
    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    forbidden_patterns = ("best_line_accuracy", "best_odds_accuracy", "ev_classification_accuracy", "calculate_")
    for name in defined:
        for pattern in forbidden_patterns:
            assert pattern not in name.lower(), name


# ---------------------------------------------------------------------------
# Ground-truth isolation (Step 20)
# ---------------------------------------------------------------------------


def test_agent_request_model_has_no_ground_truth_fields():
    forbidden = {"expected_best_sportsbook", "expected_best_odds", "expected_ev", "expected_positive_ev"}
    assert not (set(AgentRequest.model_fields.keys()) & forbidden)


def test_run_experiment_module_only_reads_ground_truth_for_evaluation():
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "src" / "experiments" / "runner.py").read_text()
    tree = ast.parse(source)
    request_construction = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AgentRequest"
    )
    call_source = ast.get_source_segment(source, request_construction)
    assert "ground_truth" not in call_source.lower()


def test_full_small_experiment_agent_request_has_no_ground_truth(tmp_path):
    config = _config(tmp_path, architectures=[ArchitectureType.TOOL], repetitions=1)
    captured_requests: list[AgentRequest] = []
    original = tool_eval.evaluate_scenario

    def capture(agent, tools, request, ground_truth, quant_ground_truth, stale_odds_map=None):
        captured_requests.append(request)
        return original(agent, tools, request, ground_truth, quant_ground_truth, stale_odds_map)

    runner._EVALUATE_SCENARIO[ArchitectureType.TOOL] = capture
    try:
        runner.run_experiment(config)
    finally:
        runner._EVALUATE_SCENARIO[ArchitectureType.TOOL] = original

    assert len(captured_requests) == 1
    request_fields = set(AgentRequest.model_fields.keys())
    assert not (request_fields & {"expected_best_sportsbook", "expected_best_odds", "expected_ev"})


# ---------------------------------------------------------------------------
# Common query reaches every architecture identically (Step 6)
# ---------------------------------------------------------------------------


def test_same_scenario_query_reaches_every_architecture(tmp_path):
    config = _config(
        tmp_path,
        architectures=[ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID],
        scenario_ids=["S001"],
        repetitions=1,
    )
    captured: dict[ArchitectureType, AgentRequest] = {}
    originals = dict(runner._EVALUATE_SCENARIO)

    def make_capture(architecture, original_fn):
        def capture(agent, aux, request, ground_truth, quant_ground_truth, stale_odds_map=None):
            captured[architecture] = request
            return original_fn(agent, aux, request, ground_truth, quant_ground_truth, stale_odds_map)
        return capture

    for architecture, original_fn in originals.items():
        runner._EVALUATE_SCENARIO[architecture] = make_capture(architecture, original_fn)
    try:
        runner.run_experiment(config)
    finally:
        runner._EVALUATE_SCENARIO.update(originals)

    assert len(captured) == 3
    queries = {request.query for request in captured.values()}
    assert len(queries) == 1  # identical query text for all three architectures
    scenario_ids = {request.scenario_id for request in captured.values()}
    assert scenario_ids == {"S001"}


# ---------------------------------------------------------------------------
# Failure isolation (Step 13)
# ---------------------------------------------------------------------------


def test_one_failed_run_does_not_prevent_remaining_runs(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        architectures=[ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID],
        scenario_ids=["S001"],
        repetitions=1,
    )

    real_create_agent = runner.create_agent

    def flaky_create_agent(architecture, cfg, llm_client=None):
        if architecture == ArchitectureType.HYBRID:
            raise RuntimeError("simulated agent construction failure")
        return real_create_agent(architecture, cfg, llm_client)

    monkeypatch.setattr(runner, "create_agent", flaky_create_agent)
    summary = runner.run_experiment(config)

    assert summary.recorded_runs == 3  # all 3 runs recorded, none dropped
    assert summary.failed_runs == 1
    assert summary.successful_runs == 2
    hybrid_result = summary.comparison.hybrid_summary.raw_results[0]
    assert hybrid_result.execution_status == metrics.FailureCategory.UNKNOWN_FAILURE
    assert "simulated agent construction failure" in hybrid_result.errors[0]


# ---------------------------------------------------------------------------
# Consistency aggregation across repetitions (Step 22)
# ---------------------------------------------------------------------------


def _synthetic_run(scenario_id, architecture, repetition, best_odds, position=0) -> runner.RawExperimentRun:
    common = metrics.EvaluationResult(
        scenario_id=scenario_id,
        architecture=architecture,
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
    return runner.RawExperimentRun(
        experiment_id="synthetic",
        architecture=architecture,
        scenario_id=scenario_id,
        repetition=repetition,
        execution_order_position=position,
        timestamp=datetime.now(timezone.utc),
        common_result=common,
        architecture_specific_result={},
    )


def test_summarize_experiment_groups_consistency_by_architecture_and_scenario(tmp_path):
    config = _config(tmp_path, architectures=[ArchitectureType.TOOL], scenario_ids=["S001"], repetitions=3)
    runs = [
        _synthetic_run("S001", ArchitectureType.TOOL, 1, best_odds=130),
        _synthetic_run("S001", ArchitectureType.TOOL, 2, best_odds=130),
        _synthetic_run("S001", ArchitectureType.TOOL, 3, best_odds=125),  # differs
    ]
    summary = runner.summarize_experiment(config, runs)
    assert summary.comparison.tool_summary.consistency == pytest.approx(2 / 3)


def test_summarize_experiment_perfectly_consistent_when_all_repetitions_match(tmp_path):
    config = _config(tmp_path, architectures=[ArchitectureType.TOOL], scenario_ids=["S001"], repetitions=3)
    runs = [_synthetic_run("S001", ArchitectureType.TOOL, r, best_odds=130) for r in (1, 2, 3)]
    summary = runner.summarize_experiment(config, runs)
    assert summary.comparison.tool_summary.consistency == pytest.approx(1.0)


def test_summarize_experiment_no_consistency_value_for_single_repetition(tmp_path):
    config = _config(tmp_path, architectures=[ArchitectureType.TOOL], scenario_ids=["S001"], repetitions=1)
    runs = [_synthetic_run("S001", ArchitectureType.TOOL, 1, best_odds=130)]
    summary = runner.summarize_experiment(config, runs)
    assert summary.comparison.tool_summary.consistency is None


# ---------------------------------------------------------------------------
# Expected run count validation (Step 24)
# ---------------------------------------------------------------------------


def test_summarize_experiment_reports_expected_vs_recorded():
    config = ExperimentConfig(
        experiment_id="e", experiment_name="e",
        architectures=[ArchitectureType.RAG, ArchitectureType.TOOL],
        scenario_ids=["S001", "S002"], repetitions=2,
    )
    runs = [
        _synthetic_run("S001", ArchitectureType.TOOL, 1, best_odds=130),
        _synthetic_run("S002", ArchitectureType.TOOL, 1, best_odds=130),
    ]
    summary = runner.summarize_experiment(config, runs)
    assert summary.expected_runs == 8  # 2 architectures x 2 scenarios x 2 repetitions
    assert summary.recorded_runs == 2
