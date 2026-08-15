"""Unit-level tests for src/evaluation/tool_agent_evaluation.py
(Milestone 9B): ground-truth isolation, the individual comparison/
detection functions, and explicit failure classification. See
tests/test_tool_agent_e2e.py for full harness runs across real
controlled scenarios (best line, tie, EV, freshness, completeness,
repeatability, contract parity).
"""

import ast
from pathlib import Path

import pytest

from src.agents.base import AgentRequest
from src.agents.llm_client import ToolCallTurn, ToolUseBlock
from src.agents.tool_agent import ToolAgentTrace, ToolCallingAgent
from src.evaluation.tool_agent_evaluation import (
    DeterministicToolPolicyLLMClient,
    ExecutionStatus,
    _classify_failure,
    _detect_hallucination,
    build_default_tool_agent,
    evaluate_scenario,
    summarize_results,
)
from src.models import AnalysisStatus, ArchitectureType, BettingAnalysis, GroundTruth, MarketType, QuantGroundTruth
from src.providers.controlled import ControlledOddsProvider
from src.tools.sportsbook_tools import SportsbookTools

AGENTS_DIR = Path(__file__).resolve().parent.parent / "src" / "agents"


# ---------------------------------------------------------------------------
# Ground-truth isolation
# ---------------------------------------------------------------------------


def test_evaluator_can_import_ground_truth_generators():
    # The evaluator module itself is explicitly allowed to depend on
    # ground truth (docs/EXPERIMENT_RULES.md: "Evaluator -> Ground Truth").
    from src.evaluation.tool_agent_evaluation import generate_all_ground_truth, generate_all_quant_ground_truth

    assert callable(generate_all_ground_truth)
    assert callable(generate_all_quant_ground_truth)


def test_agent_request_built_by_evaluator_has_no_ground_truth_fields():
    from src.evaluation.tool_agent_evaluation import _build_agent_request

    request = _build_agent_request(
        {
            "scenario_id": "S001",
            "game": {"game_id": "G-2026-001"},
            "market": {"market_type": "moneyline", "selected_outcome": "Los Angeles Lakers"},
        }
    )
    forbidden = {"expected_best_sportsbook", "expected_best_odds", "expected_ev", "expected_positive_ev"}
    assert not (set(AgentRequest.model_fields.keys()) & forbidden)


def test_tool_agent_source_never_imports_evaluation_or_ground_truth():
    # Milestone 9A's own isolation tests already forbid "ground_truth.json"
    # / "quant_ground_truth.json" substrings in tool_agent.py/tool_schemas.py;
    # this additionally forbids importing the evaluation package itself,
    # scoped specifically to this milestone's concern (the agent must never
    # gain a path to GroundTruth/QuantGroundTruth through the evaluator).
    for filename in ("tool_agent.py", "tool_schemas.py"):
        source_path = AGENTS_DIR / filename
        tree = ast.parse(source_path.read_text())
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        assert not any(m.startswith("src.evaluation") for m in imported_modules), (
            f"{filename} must never import src.evaluation"
        )


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def _trace(**overrides) -> ToolAgentTrace:
    defaults = dict(
        model="fake",
        effort="low",
        query="q",
        iterations_used=1,
        tool_calls=[],
        tool_call_order=[],
        redundant_call_count=0,
        validation_status="no_valid_prices",
        quant_status="not_attempted",
        llm_decision_latency_seconds=0.0,
        tool_execution_latency_seconds=0.0,
        quant_latency_seconds=0.0,
        total_latency_seconds=0.0,
        errors=[],
    )
    defaults.update(overrides)
    return ToolAgentTrace(**defaults)


def test_classify_loop_limit():
    trace = _trace(validation_status="loop_limit_exceeded")
    assert _classify_failure(trace) == ExecutionStatus.TOOL_LOOP_LIMIT


def test_classify_llm_output_invalid():
    trace = _trace(errors=["LLM tool-call turn failed: ValueError('boom')"])
    assert _classify_failure(trace) == ExecutionStatus.LLM_OUTPUT_INVALID


def test_classify_tool_data_missing():
    from src.agents.tool_agent import ToolCallRecord

    trace = _trace(
        tool_calls=[
            ToolCallRecord(
                call_sequence=1, tool_name="get_sportsbook_odds",
                arguments={"sportsbook": "FakeBook"}, success=False, is_redundant=False,
                result_summary="error: not found", error="sportsbook 'FakeBook' has no current odds",
                latency_seconds=0.0,
            )
        ]
    )
    assert _classify_failure(trace) == ExecutionStatus.TOOL_DATA_MISSING


def test_classify_tool_argument_error():
    from src.agents.tool_agent import ToolCallRecord

    trace = _trace(
        tool_calls=[
            ToolCallRecord(
                call_sequence=1, tool_name="get_odds",
                arguments={"market_type": "not_a_market"}, success=False, is_redundant=False,
                result_summary="error: 1 validation error for GetOddsInput",
                error="1 validation error for GetOddsInput\nmarket_type\n  Input should be...",
                latency_seconds=0.0,
            )
        ]
    )
    assert _classify_failure(trace) == ExecutionStatus.TOOL_ARGUMENT_ERROR


def test_classify_none_trace_is_llm_output_invalid():
    assert _classify_failure(None) == ExecutionStatus.LLM_OUTPUT_INVALID


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tools():
    return SportsbookTools(ControlledOddsProvider())


def _make_request(**overrides):
    defaults = dict(
        scenario_id="S001", game_id="G-2026-001", market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers", query="q",
    )
    defaults.update(overrides)
    return AgentRequest(**defaults)


def _make_analysis(**overrides):
    defaults = dict(
        scenario_id="S001", game_id="G-2026-001", market=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers", best_sportsbook="FanDuel", best_odds=125,
        best_sportsbooks=["FanDuel"], implied_probability=0.4444,
        status=AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE,
        sportsbooks_considered=["FanDuel"], reasoning_summary="test",
        architecture=ArchitectureType.TOOL,
    )
    defaults.update(overrides)
    return BettingAnalysis(**defaults)


def test_hallucination_not_detected_for_real_matching_claim(tools):
    analysis = _make_analysis(best_sportsbook="FanDuel", best_odds=125)
    assert _detect_hallucination(tools, _make_request(), analysis) is False


def test_hallucination_detected_for_unsupported_sportsbook(tools):
    analysis = _make_analysis(
        best_sportsbook="BetRivers", best_odds=130, best_sportsbooks=["BetRivers"],
        sportsbooks_considered=["BetRivers"],
    )
    assert _detect_hallucination(tools, _make_request(), analysis) is True


def test_hallucination_detected_for_altered_odds(tools):
    # DraftKings is real for this game, but +180 was never actually offered.
    analysis = _make_analysis(best_sportsbook="DraftKings", best_odds=180, best_sportsbooks=["DraftKings"])
    assert _detect_hallucination(tools, _make_request(), analysis) is True


# ---------------------------------------------------------------------------
# evaluate_scenario() — best line / odds / EV correctness and mismatch detection
# ---------------------------------------------------------------------------


def _quant_ground_truth(**overrides) -> QuantGroundTruth:
    defaults = dict(
        scenario_id="S001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        quant_evaluable=False, ineligibility_reason="not evaluated in this test",
    )
    defaults.update(overrides)
    return QuantGroundTruth(**defaults)


class FixedLLMClient:
    model = "fixed"
    effort = "low"

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls_made = 0

    def create_turn(self, *, system_prompt, messages, tools):
        turn = self.turns[self.calls_made]
        self.calls_made += 1
        return turn


def test_evaluate_scenario_detects_correct_best_line(tools):
    request = _make_request()
    ground_truth = GroundTruth(
        scenario_id="S001", expected_best_sportsbook="FanDuel", expected_best_odds=125,
        expected_implied_probability=0.4444, expected_ev=0.0, expected_positive_ev=False,
        expected_sportsbooks=["DraftKings", "FanDuel", "BetMGM", "Caesars"],
        expected_best_sportsbooks=["FanDuel"],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[
            ToolUseBlock(id="t1", name="get_odds", input={
                "game_id": "G-2026-001", "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
            })
        ]),
        ToolCallTurn(stop_reason="end_turn", text="done", tool_uses=[]),
    ]
    agent = ToolCallingAgent(tools, llm_client=FixedLLMClient(turns))
    result = evaluate_scenario(agent, tools, request, ground_truth, _quant_ground_truth())
    assert result.best_line_correct is True
    assert result.best_odds_correct is True
    assert result.execution_status == ExecutionStatus.QUANT_INSUFFICIENT_DATA


def test_evaluate_scenario_detects_incorrect_best_line(tools):
    request = _make_request()
    # Deliberately wrong ground truth to confirm mismatch detection works.
    wrong_ground_truth = GroundTruth(
        scenario_id="S001", expected_best_sportsbook="DraftKings", expected_best_odds=120,
        expected_implied_probability=0.5, expected_ev=0.0, expected_positive_ev=False,
        expected_sportsbooks=["DraftKings", "FanDuel", "BetMGM", "Caesars"],
        expected_best_sportsbooks=["DraftKings"],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[
            ToolUseBlock(id="t1", name="get_odds", input={
                "game_id": "G-2026-001", "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
            })
        ]),
        ToolCallTurn(stop_reason="end_turn", text="done", tool_uses=[]),
    ]
    agent = ToolCallingAgent(tools, llm_client=FixedLLMClient(turns))
    result = evaluate_scenario(agent, tools, request, wrong_ground_truth, _quant_ground_truth())
    assert result.best_line_correct is False
    assert result.best_odds_correct is False


def test_evaluate_scenario_reports_none_ev_fields_when_not_quant_evaluable(tools):
    request = _make_request()
    ground_truth = GroundTruth(
        scenario_id="S001", expected_best_sportsbook="FanDuel", expected_best_odds=125,
        expected_implied_probability=0.4444, expected_ev=0.0, expected_positive_ev=False,
        expected_sportsbooks=["DraftKings", "FanDuel", "BetMGM", "Caesars"],
        expected_best_sportsbooks=["FanDuel"],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[
            ToolUseBlock(id="t1", name="get_odds", input={
                "game_id": "G-2026-001", "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
            })
        ]),
        ToolCallTurn(stop_reason="end_turn", text="done", tool_uses=[]),
    ]
    agent = ToolCallingAgent(tools, llm_client=FixedLLMClient(turns))
    result = evaluate_scenario(agent, tools, request, ground_truth, _quant_ground_truth(quant_evaluable=False))
    assert result.expected_ev is None
    assert result.ev_classification_correct is None
    assert result.ev_absolute_error is None


# ---------------------------------------------------------------------------
# summarize_results()
# ---------------------------------------------------------------------------


def test_summarize_results_excludes_non_evaluable_from_ev_stats(tools):
    request = _make_request()
    ground_truth = GroundTruth(
        scenario_id="S001", expected_best_sportsbook="FanDuel", expected_best_odds=125,
        expected_implied_probability=0.4444, expected_ev=0.0, expected_positive_ev=False,
        expected_sportsbooks=["DraftKings", "FanDuel", "BetMGM", "Caesars"],
        expected_best_sportsbooks=["FanDuel"],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[
            ToolUseBlock(id="t1", name="get_odds", input={
                "game_id": "G-2026-001", "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
            })
        ]),
        ToolCallTurn(stop_reason="end_turn", text="done", tool_uses=[]),
    ]
    agent = ToolCallingAgent(tools, llm_client=FixedLLMClient(turns))
    result = evaluate_scenario(agent, tools, request, ground_truth, _quant_ground_truth(quant_evaluable=False))
    summary = summarize_results([result])
    assert summary["ev_classification_accuracy"] is None  # no denominator, not "0% accuracy"
    assert summary["best_line_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Deterministic policy is a policy, not an answer key
# ---------------------------------------------------------------------------


def test_deterministic_policy_never_imports_ground_truth():
    source_path = (
        Path(__file__).resolve().parent.parent / "src" / "evaluation" / "tool_agent_evaluation.py"
    )
    source = source_path.read_text()
    tree = ast.parse(source)
    policy_class = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "DeterministicToolPolicyLLMClient"
    )
    # Skip the class's own docstring (first statement, if a bare string
    # expression) — it explicitly documents what the class must NOT do,
    # which would otherwise trip a naive substring check.
    body = policy_class.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    for node in body:
        code_source = ast.get_source_segment(source, node) or ""
        assert "GroundTruth" not in code_source
        assert "ground_truth" not in code_source.lower()


def test_build_default_tool_agent_uses_deterministic_policy_by_default(tools):
    agent, agent_tools = build_default_tool_agent()
    assert isinstance(agent.llm_client, DeterministicToolPolicyLLMClient)
    assert isinstance(agent, ToolCallingAgent)
