"""End-to-end tests for the tool-calling agent evaluation harness
(Milestone 9B): full runs across real controlled-benchmark scenarios
using DeterministicToolPolicyLLMClient + the real ControlledOddsProvider.
No real API calls anywhere in this file.
"""

import typing

import pytest

from src.agents.base import Agent, AgentRequest
from src.agents.rag_agent import RagOnlyAgent
from src.agents.tool_agent import MAX_TOOL_ITERATIONS, ToolAnalysisIncomplete, ToolCallingAgent
from src.evaluation.tool_agent_evaluation import (
    DEFAULT_SCENARIO_IDS,
    ExecutionStatus,
    build_default_tool_agent,
    evaluate_scenarios,
    summarize_results,
)
from src.models import BettingAnalysis, MarketType
from src.rag.retriever import Retriever


@pytest.fixture(scope="module")
def results():
    return evaluate_scenarios()


# ---------------------------------------------------------------------------
# Best line: correct single winner / correct tie
# ---------------------------------------------------------------------------


def test_all_default_scenarios_produce_correct_best_line(results):
    for result in results:
        assert result.best_line_correct is True, result.scenario_id
        assert result.best_odds_correct is True, result.scenario_id


def test_tie_scenario_best_line_matches_full_tie_set(results):
    tie_result = next(r for r in results if r.scenario_id == "S007")
    assert set(tie_result.predicted_best_sportsbooks) == {"DraftKings", "FanDuel"}
    assert tie_result.best_line_correct is True


# ---------------------------------------------------------------------------
# EV classification: positive / negative
# ---------------------------------------------------------------------------


def test_positive_ev_scenario_classified_correctly(results):
    # S008 and S009 are positive-EV under the market-consensus methodology
    # (see data/quant_ground_truth.json).
    for scenario_id in ("S008", "S009"):
        result = next(r for r in results if r.scenario_id == scenario_id)
        assert result.predicted_positive_ev is True
        assert result.expected_positive_ev is True
        assert result.ev_classification_correct is True


def test_negative_ev_scenario_classified_correctly(results):
    # S001 and S007 are negative-EV under the market-consensus methodology.
    for scenario_id in ("S001", "S007"):
        result = next(r for r in results if r.scenario_id == scenario_id)
        assert result.predicted_positive_ev is False
        assert result.expected_positive_ev is False
        assert result.ev_classification_correct is True


def test_ev_numerical_error_near_zero_for_quant_evaluable_scenarios(results):
    for result in results:
        if result.quant_evaluable and result.execution_status == ExecutionStatus.SUCCESS:
            assert result.ev_absolute_error == pytest.approx(0.0, abs=1e-9)


def test_non_quant_evaluable_scenarios_never_scored_as_wrong_ev(results):
    for result in results:
        if not result.quant_evaluable:
            assert result.ev_classification_correct is None
            assert result.ev_absolute_error is None


# ---------------------------------------------------------------------------
# Consensus: market-reference probability, leave-one-out exclusion held
# ---------------------------------------------------------------------------


def test_market_reference_probability_matches_ground_truth(results):
    for result in results:
        if result.quant_evaluable and result.execution_status == ExecutionStatus.SUCCESS:
            assert result.market_reference_absolute_error == pytest.approx(0.0, abs=1e-9)
            assert result.predicted_market_reference_probability is not None


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


def test_freshness_scenario_uses_current_not_stale_data(results):
    freshness_result = next(r for r in results if r.scenario_id == "S009")
    assert freshness_result.freshness_correct is True
    assert freshness_result.predicted_best_odds == 140  # current FanDuel line, never a stale snapshot


def test_non_freshness_scenarios_have_no_freshness_verdict(results):
    for result in results:
        if result.scenario_id not in {"S009", "S010", "S011"}:
            assert result.freshness_correct is None


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def test_full_data_scenario_is_fully_complete(results):
    result = next(r for r in results if r.scenario_id == "S001")
    assert result.completeness == pytest.approx(1.0)


def test_missing_sportsbook_scenario_is_complete_relative_to_availability(results):
    # S008 has only 3 sportsbooks quoting (FanDuel doesn't offer this
    # game) — completeness is measured against what's actually
    # available, not against every sportsbook in the dataset globally.
    result = next(r for r in results if r.scenario_id == "S008")
    assert len(result.expected_best_sportsbooks) <= 1
    assert result.completeness == pytest.approx(1.0)
    assert "FanDuel" not in result.predicted_best_sportsbooks


# ---------------------------------------------------------------------------
# Hallucination — none expected on real controlled data
# ---------------------------------------------------------------------------


def test_no_hallucinations_across_default_scenarios(results):
    for result in results:
        assert result.hallucination_detected is False, result.scenario_id


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------


def test_tool_calls_are_counted_and_reasonable(results):
    for result in results:
        assert result.tool_call_count >= 1
        assert result.unique_tool_call_count <= result.tool_call_count
        assert result.redundant_tool_call_count == 0  # deterministic policy never repeats


def test_failed_tool_calls_counted_for_single_sided_scenarios(results):
    # Scenarios with no opposing-side data in current_odds.json (S002-S006)
    # get exactly one failed get_odds call when the policy tries the
    # opposing outcome — traced explicitly, not hidden.
    single_sided = {"S002", "S003", "S004", "S005", "S006"}
    for result in results:
        if result.scenario_id in single_sided:
            assert result.failed_tool_call_count == 1


# ---------------------------------------------------------------------------
# Failure classes via the harness
# ---------------------------------------------------------------------------


class _AlwaysToolUseLLMClient:
    model = "runaway"
    effort = "low"

    def create_turn(self, *, system_prompt, messages, tools):
        from src.agents.llm_client import ToolCallTurn, ToolUseBlock

        return ToolCallTurn(
            stop_reason="tool_use", text=None,
            tool_uses=[ToolUseBlock(id="x", name="get_games", input={})],
        )


def test_loop_limit_failure_class_via_harness():
    from src.providers.controlled import ControlledOddsProvider
    from src.tools.sportsbook_tools import SportsbookTools

    tools = SportsbookTools(ControlledOddsProvider())
    agent = ToolCallingAgent(tools, llm_client=_AlwaysToolUseLLMClient(), max_iterations=MAX_TOOL_ITERATIONS)
    request = AgentRequest(
        scenario_id="S001", game_id="G-2026-001", market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers", query="q",
    )
    with pytest.raises(ToolAnalysisIncomplete) as exc_info:
        agent.analyze(request)
    assert exc_info.value.trace.validation_status == "loop_limit_exceeded"


def test_insufficient_quant_data_failure_class_present(results):
    single_sided_ids = {"S002", "S003", "S004", "S005", "S006", "S012", "S013"}
    for result in results:
        if result.scenario_id in single_sided_ids:
            assert result.execution_status == ExecutionStatus.QUANT_INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_deterministic_evaluation_is_reproducible_ignoring_latency():
    run_1 = evaluate_scenarios(DEFAULT_SCENARIO_IDS)
    run_2 = evaluate_scenarios(DEFAULT_SCENARIO_IDS)

    latency_fields = {
        "llm_decision_latency_seconds", "tool_execution_latency_seconds",
        "quant_latency_seconds", "total_latency_seconds",
    }
    for r1, r2 in zip(run_1, run_2):
        d1 = r1.model_dump(exclude=latency_fields)
        d2 = r2.model_dump(exclude=latency_fields)
        assert d1 == d2, r1.scenario_id


def test_reproducibility_summary_matches():
    run_1 = summarize_results(evaluate_scenarios(DEFAULT_SCENARIO_IDS))
    run_2 = summarize_results(evaluate_scenarios(DEFAULT_SCENARIO_IDS))
    non_latency_1 = {k: v for k, v in run_1.items() if "latency" not in k}
    non_latency_2 = {k: v for k, v in run_2.items() if "latency" not in k}
    assert non_latency_1 == non_latency_2


# ---------------------------------------------------------------------------
# Contract parity (RAG vs. tool architectures)
# ---------------------------------------------------------------------------


class _FakeRagLLMClient:
    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        from src.agents.extraction import ExtractedMarketEvidence

        return ExtractedMarketEvidence(
            game_id="G-2026-001", market_id="G-2026-001-moneyline",
            selected_outcome="Los Angeles Lakers", sportsbook_prices=[],
            missing_evidence_note="contract-parity test, no extraction needed",
        )


@pytest.fixture(scope="module")
def retriever():
    return Retriever.from_directory()


def test_rag_and_tool_agents_both_implement_common_agent_contract(retriever):
    rag_agent = RagOnlyAgent(retriever, llm_client=_FakeRagLLMClient())
    tool_agent, _ = build_default_tool_agent()

    assert isinstance(rag_agent, Agent)
    assert isinstance(tool_agent, Agent)

    for agent_class in (type(rag_agent), type(tool_agent)):
        hints = typing.get_type_hints(agent_class.analyze)
        assert hints["request"] is AgentRequest
        assert hints["return"] is BettingAnalysis


def test_rag_and_tool_agents_both_accept_same_agent_request_and_return_betting_analysis(retriever):
    request = AgentRequest(
        scenario_id="S001", game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="What is the best current price on the Lakers moneyline?",
    )
    tool_agent, _ = build_default_tool_agent()
    tool_result = tool_agent.analyze(request)
    assert isinstance(tool_result, BettingAnalysis)
    assert tool_result.architecture.value == "tool"

    from src.agents.rag_agent import RagAnalysisIncomplete

    rag_agent = RagOnlyAgent(retriever, llm_client=_FakeRagLLMClient())
    with pytest.raises(RagAnalysisIncomplete):
        # Same AgentRequest instance accepted by both architectures —
        # this is the parity check; the RAG agent legitimately finds no
        # extractable evidence from its fake LLM, which is fine here
        # since we're only checking input/output type compatibility.
        rag_agent.analyze(request)
