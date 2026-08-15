"""Tests for cross-architecture parity in the unified evaluation
framework (Milestone 11, Steps 29-30): RAG-only, tool-calling, and
hybrid evaluators must call the SAME shared metric functions rather than
each defining their own architecture-specific reimplementation, and
their results must convert into one common EvaluationResult shape usable
by a single summarize()/compare_architectures() path.
"""

import ast
from pathlib import Path

import pytest

from src.evaluation import hybrid_agent_evaluation, metrics, rag_agent_evaluation, tool_agent_evaluation
from src.models import ArchitectureType

EVALUATION_DIR = Path(__file__).resolve().parent.parent / "src" / "evaluation"
AGENTS_DIR = Path(__file__).resolve().parent.parent / "src" / "agents"

_SMALL_SCENARIO_SET = ["S001", "S007", "S008", "S009"]  # the 4 quant-evaluable scenarios


# ---------------------------------------------------------------------------
# No architecture-specific reimplementations of shared formulas (Step 29)
# ---------------------------------------------------------------------------


def _defined_function_names(filename: str) -> set[str]:
    tree = ast.parse((EVALUATION_DIR / filename).read_text())
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


@pytest.mark.parametrize(
    "filename", ["rag_agent_evaluation.py", "tool_agent_evaluation.py", "hybrid_agent_evaluation.py"]
)
def test_no_architecture_specific_metric_function_names(filename):
    forbidden_patterns = ("best_line_accuracy", "best_odds_accuracy", "ev_classification_accuracy")
    defined = _defined_function_names(filename)
    for name in defined:
        lowered = name.lower()
        for pattern in forbidden_patterns:
            assert pattern not in lowered, f"{filename} defines {name!r} — should reuse src.evaluation.metrics"


def test_all_three_evaluators_import_the_shared_metrics_module():
    for module in ("rag_agent_evaluation.py", "tool_agent_evaluation.py", "hybrid_agent_evaluation.py"):
        source = (EVALUATION_DIR / module).read_text()
        assert "from src.evaluation import metrics" in source, module


def test_best_line_correct_is_one_shared_function_object():
    # Not calculate_rag_best_line_accuracy() / calculate_tool_.../
    # calculate_hybrid_...() — literally the same function object used
    # by every evaluator.
    assert tool_agent_evaluation.metrics.best_line_correct is metrics.best_line_correct
    assert hybrid_agent_evaluation.metrics.best_line_correct is metrics.best_line_correct
    assert rag_agent_evaluation.metrics.best_line_correct is metrics.best_line_correct


# ---------------------------------------------------------------------------
# Identical formula invoked identically regardless of architecture
# ---------------------------------------------------------------------------


def test_same_inputs_produce_same_outputs_across_architectures():
    # Feed equivalent synthetic (predicted, expected) pairs through the
    # shared functions as each evaluator actually calls them.
    predicted_sportsbooks = ["DraftKings", "FanDuel"]
    expected_sportsbooks = ["DraftKings", "FanDuel"]
    predicted_odds = 125
    expected_odds = 125

    for module in (rag_agent_evaluation, tool_agent_evaluation, hybrid_agent_evaluation):
        assert module.metrics.best_line_correct(predicted_sportsbooks, expected_sportsbooks) is True
        assert module.metrics.best_odds_correct(predicted_odds, expected_odds) is True
        assert module.metrics.ev_classification_correct(True, True) is True
        assert module.metrics.ev_absolute_error(0.05, 0.04) == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Ground-truth isolation across all three agents (Step 4, 33)
# ---------------------------------------------------------------------------


def _imported_modules(filename: str) -> set[str]:
    tree = ast.parse((AGENTS_DIR / filename).read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("filename", ["rag_agent.py", "tool_agent.py", "hybrid_agent.py"])
def test_no_agent_imports_evaluation_package(filename):
    modules = _imported_modules(filename)
    assert not any(m.startswith("src.evaluation") for m in modules), filename


def test_evaluators_are_the_only_place_ground_truth_is_read():
    for module in ("rag_agent_evaluation.py", "tool_agent_evaluation.py", "hybrid_agent_evaluation.py"):
        source = (EVALUATION_DIR / module).read_text()
        assert "generate_all_ground_truth" in source or "GroundTruth" in source


# ---------------------------------------------------------------------------
# Real cross-architecture conversion + unified summarize()/compare() (Step 23)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rag_common_results():
    return [rag_agent_evaluation.to_common_result(r) for r in rag_agent_evaluation.evaluate_scenarios(_SMALL_SCENARIO_SET)]


@pytest.fixture(scope="module")
def tool_common_results():
    return [tool_agent_evaluation.to_common_result(r) for r in tool_agent_evaluation.evaluate_scenarios(_SMALL_SCENARIO_SET)]


@pytest.fixture(scope="module")
def hybrid_common_results():
    return [hybrid_agent_evaluation.to_common_result(r) for r in hybrid_agent_evaluation.evaluate_scenarios(_SMALL_SCENARIO_SET)]


def test_to_common_result_produces_evaluation_result_for_all_three(
    rag_common_results, tool_common_results, hybrid_common_results
):
    for results, expected_architecture in (
        (rag_common_results, ArchitectureType.RAG),
        (tool_common_results, ArchitectureType.TOOL),
        (hybrid_common_results, ArchitectureType.HYBRID),
    ):
        assert len(results) == 4
        for result in results:
            assert isinstance(result, metrics.EvaluationResult)
            assert result.architecture == expected_architecture


def test_all_three_architectures_agree_on_best_line_for_shared_scenarios(
    rag_common_results, tool_common_results, hybrid_common_results
):
    # Same controlled benchmark, same shared quant engine -> identical
    # ground-truth-correctness verdicts across architectures (accuracy
    # itself, not necessarily identical predicted values, though for
    # these fully-observable scenarios the predictions match too).
    by_scenario = {}
    for results in (rag_common_results, tool_common_results, hybrid_common_results):
        for result in results:
            by_scenario.setdefault(result.scenario_id, []).append(result)
    for scenario_id, results in by_scenario.items():
        assert all(r.best_line_correct is True for r in results), scenario_id


def test_unified_summarize_works_identically_for_all_three(
    rag_common_results, tool_common_results, hybrid_common_results
):
    rag_summary = metrics.summarize(rag_common_results)
    tool_summary = metrics.summarize(tool_common_results)
    hybrid_summary = metrics.summarize(hybrid_common_results)

    for summary, architecture in (
        (rag_summary, ArchitectureType.RAG), (tool_summary, ArchitectureType.TOOL), (hybrid_summary, ArchitectureType.HYBRID),
    ):
        assert summary.architecture == architecture
        assert summary.runs == 4
        assert summary.best_line_accuracy == pytest.approx(1.0)
        assert summary.ev_classification_accuracy == pytest.approx(1.0)


def test_compare_architectures_end_to_end(rag_common_results, tool_common_results, hybrid_common_results):
    comparison = metrics.compare_architectures(
        rag_results=rag_common_results, tool_results=tool_common_results, hybrid_results=hybrid_common_results,
    )
    assert comparison.rag_summary.best_line_accuracy == pytest.approx(1.0)
    assert comparison.tool_summary.best_line_accuracy == pytest.approx(1.0)
    assert comparison.hybrid_summary.best_line_accuracy == pytest.approx(1.0)
    assert comparison.scenario_count == 4
    # No winner declared anywhere in the structure.
    assert not hasattr(comparison, "winner")


def test_full_comparison_round_trips_through_json(rag_common_results, tool_common_results, hybrid_common_results):
    comparison = metrics.compare_architectures(
        rag_results=rag_common_results, tool_results=tool_common_results, hybrid_results=hybrid_common_results,
    )
    payload = comparison.model_dump_json()
    restored = metrics.ArchitectureComparison.model_validate_json(payload)
    assert restored == comparison
    # Raw per-run results preserved through the round trip (Step 32).
    assert len(restored.rag_summary.raw_results) == 4
    assert len(restored.tool_summary.raw_results) == 4
    assert len(restored.hybrid_summary.raw_results) == 4
