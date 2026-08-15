"""End-to-end tests for the hybrid agent evaluation harness (Milestone
10B): full runs across real controlled-benchmark scenarios using
DeterministicHybridPolicyLLMClient + the real Retriever/RAG corpus +
ControlledOddsProvider. No real API calls anywhere in this file.
"""

import typing

import pytest

from src.agents.base import Agent, AgentRequest
from src.agents.hybrid_agent import HybridAgent
from src.agents.rag_agent import RagOnlyAgent
from src.agents.tool_agent import ToolCallingAgent
from src.evaluation.hybrid_agent_evaluation import (
    DEFAULT_SCENARIO_IDS,
    build_default_hybrid_agent,
    evaluate_scenarios,
    summarize_results,
)
from src.evaluation.tool_agent_evaluation import build_default_tool_agent
from src.models import BettingAnalysis, MarketType, SourceType
from src.rag.retriever import Retriever


@pytest.fixture(scope="module")
def results():
    return evaluate_scenarios()


# ---------------------------------------------------------------------------
# Best line: normal winner / tie
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
# Quant: no-vig / leave-one-out / EV positive & negative
# ---------------------------------------------------------------------------


def test_positive_ev_scenarios_classified_correctly(results):
    for scenario_id in ("S008", "S009"):
        result = next(r for r in results if r.scenario_id == scenario_id)
        assert result.predicted_positive_ev is True
        assert result.expected_positive_ev is True
        assert result.ev_classification_correct is True


def test_negative_ev_scenarios_classified_correctly(results):
    for scenario_id in ("S001", "S007"):
        result = next(r for r in results if r.scenario_id == scenario_id)
        assert result.predicted_positive_ev is False
        assert result.expected_positive_ev is False
        assert result.ev_classification_correct is True


def test_ev_and_market_reference_error_near_zero_for_quant_evaluable(results):
    for result in results:
        if result.quant_evaluable and result.execution_status.value == "success":
            assert result.ev_absolute_error == pytest.approx(0.0, abs=1e-9)
            assert result.market_reference_absolute_error == pytest.approx(0.0, abs=1e-9)


def test_non_quant_evaluable_scenarios_never_scored_as_wrong_ev(results):
    for result in results:
        if not result.quant_evaluable:
            assert result.ev_classification_correct is None
            assert result.ev_absolute_error is None


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


def test_freshness_scenario_uses_current_not_stale_data(results):
    result = next(r for r in results if r.scenario_id == "S009")
    assert result.freshness_correct is True
    assert result.predicted_best_odds == 140  # current FanDuel line


def test_stale_odds_never_enter_authoritative_state_directly(results):
    result = next(r for r in results if r.scenario_id == "S009")
    # Core safety invariant, regardless of whether this particular run's
    # retrieval happened to also surface a conflicting RAG value: a
    # stale-only RAG observation must never be promoted to authoritative.
    # (The dedicated conflict-detection/resolution mechanism itself —
    # stale RAG vs. current tool — is unit-tested with a deliberately
    # constructed stale-only RAG value in tests/test_hybrid_agent.py's
    # test_case_b_stale_rag_vs_current_tool and
    # tests/test_hybrid_reconciliation.py's Case B; this end-to-end run's
    # real retrieval currently ranks the current DraftKings snapshot
    # above the stale one for this query, so no conflict arises here.)
    assert result.stale_rag_incorrectly_promoted == 0


# ---------------------------------------------------------------------------
# Hallucination
# ---------------------------------------------------------------------------


def test_no_hallucinations_across_default_scenarios(results):
    for result in results:
        assert result.hallucination_detected is False, result.scenario_id


def test_fabricated_sportsbook_detected_by_independent_recheck(tools=None):
    from src.evaluation.hybrid_agent_evaluation import _detect_hallucination
    from src.models import AnalysisStatus, ArchitectureType
    from src.providers.controlled import ControlledOddsProvider
    from src.tools.sportsbook_tools import SportsbookTools

    tools = SportsbookTools(ControlledOddsProvider())
    request = AgentRequest(
        scenario_id="S001", game_id="G-2026-001", market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers", query="q",
    )
    fabricated = BettingAnalysis(
        scenario_id="S001", game_id="G-2026-001", market=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers", best_sportsbook="BetRivers", best_odds=130,
        best_sportsbooks=["BetRivers"], implied_probability=0.4,
        status=AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE, sportsbooks_considered=["BetRivers"],
        reasoning_summary="test", architecture=ArchitectureType.HYBRID,
    )
    assert _detect_hallucination(tools, request, fabricated) is True


def test_altered_odds_detected_by_independent_recheck():
    from src.evaluation.hybrid_agent_evaluation import _detect_hallucination
    from src.models import AnalysisStatus, ArchitectureType
    from src.providers.controlled import ControlledOddsProvider
    from src.tools.sportsbook_tools import SportsbookTools

    tools = SportsbookTools(ControlledOddsProvider())
    request = AgentRequest(
        scenario_id="S001", game_id="G-2026-001", market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers", query="q",
    )
    altered = BettingAnalysis(
        scenario_id="S001", game_id="G-2026-001", market=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers", best_sportsbook="DraftKings", best_odds=999,
        best_sportsbooks=["DraftKings"], implied_probability=0.4,
        status=AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE, sportsbooks_considered=["DraftKings"],
        reasoning_summary="test", architecture=ArchitectureType.HYBRID,
    )
    assert _detect_hallucination(tools, request, altered) is True


# ---------------------------------------------------------------------------
# Reconciliation integrity
# ---------------------------------------------------------------------------


def test_one_authoritative_record_per_sportsbook_outcome(results):
    agent, tools = build_default_hybrid_agent()
    for scenario_id in ("S001", "S007", "S008", "S009"):
        from src.evaluation.dataset import load_scenario_definitions_by_id
        from src.evaluation.hybrid_agent_evaluation import _build_agent_request

        definition = load_scenario_definitions_by_id()[scenario_id]
        request = _build_agent_request(definition)
        agent.analyze(request)
        seen = set()
        for record in agent.last_trace.reconciled_records:
            key = (record.sportsbook, record.selected_outcome)
            assert key not in seen, f"{scenario_id}: duplicate record for {key}"
            seen.add(key)


def test_target_sportsbook_excluded_from_its_own_consensus(results):
    from src.calculations.market import calculate_leave_one_out_consensus, calculate_no_vig_probabilities

    result = next(r for r in results if r.scenario_id == "S001")
    no_vig_fanduel = calculate_no_vig_probabilities([125, -145])[0]
    consensus_excluding_fanduel = calculate_leave_one_out_consensus(
        [
            ("DraftKings", calculate_no_vig_probabilities([120, -140])[0]),
            ("BetMGM", calculate_no_vig_probabilities([115, -135])[0]),
            ("Caesars", calculate_no_vig_probabilities([122, -142])[0]),
            ("FanDuel", no_vig_fanduel),
        ],
        "FanDuel",
    )
    assert result.predicted_market_reference_probability == pytest.approx(consensus_excluding_fanduel)


# ---------------------------------------------------------------------------
# Repeatability
# ---------------------------------------------------------------------------


def test_deterministic_evaluation_is_reproducible_ignoring_latency():
    run_1 = evaluate_scenarios(DEFAULT_SCENARIO_IDS)
    run_2 = evaluate_scenarios(DEFAULT_SCENARIO_IDS)
    latency_fields = {
        "rag_retrieval_latency_seconds", "rag_llm_latency_seconds", "tool_llm_latency_seconds",
        "tool_execution_latency_seconds", "reconciliation_latency_seconds", "quant_latency_seconds",
        "total_latency_seconds",
    }
    for r1, r2 in zip(run_1, run_2):
        assert r1.model_dump(exclude=latency_fields) == r2.model_dump(exclude=latency_fields), r1.scenario_id


def test_reproducibility_summary_matches():
    run_1 = summarize_results(evaluate_scenarios(DEFAULT_SCENARIO_IDS))
    run_2 = summarize_results(evaluate_scenarios(DEFAULT_SCENARIO_IDS))
    non_latency_1 = {k: v for k, v in run_1.items() if "latency" not in k}
    non_latency_2 = {k: v for k, v in run_2.items() if "latency" not in k}
    assert non_latency_1 == non_latency_2


# ---------------------------------------------------------------------------
# Contract parity (RAG / tool / hybrid)
# ---------------------------------------------------------------------------


class _FakeRagLLMClient:
    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        from src.agents.extraction import ExtractedMarketEvidence

        return ExtractedMarketEvidence(
            game_id="G-2026-001", market_id="G-2026-001-moneyline",
            selected_outcome="Los Angeles Lakers", sportsbook_prices=[],
            missing_evidence_note="contract-parity test",
        )


def test_rag_tool_hybrid_share_common_agent_request_and_betting_analysis_contract():
    retriever = Retriever.from_directory()
    rag_agent = RagOnlyAgent(retriever, llm_client=_FakeRagLLMClient())
    tool_agent, _ = build_default_tool_agent()
    hybrid_agent, _ = build_default_hybrid_agent()

    for agent in (rag_agent, tool_agent, hybrid_agent):
        assert isinstance(agent, Agent)
        hints = typing.get_type_hints(type(agent).analyze)
        assert hints["request"] is AgentRequest
        assert hints["return"] is BettingAnalysis


def test_hybrid_output_is_valid_betting_analysis(results):
    for scenario_id in ("S001", "S007", "S008", "S009"):
        agent, tools = build_default_hybrid_agent()
        from src.evaluation.dataset import load_scenario_definitions_by_id
        from src.evaluation.hybrid_agent_evaluation import _build_agent_request

        definition = load_scenario_definitions_by_id()[scenario_id]
        analysis = agent.analyze(_build_agent_request(definition))
        BettingAnalysis.model_validate(analysis.model_dump())


# ---------------------------------------------------------------------------
# Architecture isolation regression (Step 25)
# ---------------------------------------------------------------------------


def _imported_modules(filename: str) -> set[str]:
    import ast
    from pathlib import Path

    tree = ast.parse(
        (Path(__file__).resolve().parent.parent / "src" / "agents" / filename).read_text()
    )
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_rag_only_agent_still_has_no_sportsbook_tool_access():
    modules = _imported_modules("rag_agent.py")
    assert not any(m.startswith("src.tools") or m.startswith("src.providers") for m in modules)


def test_tool_only_agent_still_has_no_rag_access():
    modules = _imported_modules("tool_agent.py")
    assert not any(m.startswith("src.rag") for m in modules)


def test_hybrid_agent_has_both_access_channels():
    modules = _imported_modules("hybrid_agent.py")
    assert any(m.startswith("src.rag") for m in modules)
    assert any(m.startswith("src.tools") for m in modules)


def test_no_agent_module_imports_ground_truth_generators():
    for filename in ("rag_agent.py", "tool_agent.py", "hybrid_agent.py"):
        modules = _imported_modules(filename)
        assert not any(m.startswith("src.evaluation") for m in modules), filename
