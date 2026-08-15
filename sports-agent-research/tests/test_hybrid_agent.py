"""Tests for src/agents/hybrid_agent.py (Milestone 10A) — the hybrid
RAG + tool-calling agent. Uses fake LLM clients throughout (no real API
calls). Every test drives the REAL Retriever/RAG corpus and the REAL
ControlledOddsProvider unless a scenario specifically needs a synthetic
provider/retriever to hit an exact numeric example from the milestone
spec (Section 25's best-line test; graceful-degradation's RAG-failure
case).
"""

import ast
from datetime import datetime
from pathlib import Path

import pytest

from src.agents.base import Agent, AgentRequest
from src.agents.extraction import ExtractedMarketEvidence, ExtractedSportsbookPrice
from src.agents.hybrid_agent import (
    HybridAgent,
    HybridAgentTrace,
    HybridAnalysisIncomplete,
    HybridFailureCategory,
)
from src.agents.llm_client import AnthropicLLMClient, ToolCallTurn, ToolUseBlock
from src.agents.rag_agent import RagOnlyAgent
from src.agents.tool_agent import MAX_TOOL_ITERATIONS, ToolCallingAgent
from src.calculations.market import calculate_leave_one_out_consensus, calculate_no_vig_probabilities
from src.calculations.odds_math import expected_value, implied_probability, is_positive_ev
from src.models import AnalysisStatus, ArchitectureType, BettingAnalysis, Game, MarketType, SourceType, SportsbookOdds
from src.providers.base import OddsProvider
from src.providers.controlled import ControlledOddsProvider
from src.rag.retriever import Retriever
from src.tools.sportsbook_tools import SportsbookTools

AGENTS_DIR = Path(__file__).resolve().parent.parent / "src" / "agents"


class FakeHybridLLMClient:
    """Implements both LLMClient (generate_structured, for RAG
    extraction) and ToolCallingLLMClient (create_turn, for tool
    orchestration) — mirroring how AnthropicLLMClient implements both on
    one class (see src/agents/llm_client.py)."""

    model = "fake-model"
    effort = "low"

    def __init__(self, rag_extraction, tool_turns):
        self.rag_extraction = rag_extraction
        self.tool_turns = list(tool_turns)
        self.tool_calls_made = 0

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        return self.rag_extraction

    def create_turn(self, *, system_prompt, messages, tools):
        turn = self.tool_turns[self.tool_calls_made]
        self.tool_calls_made += 1
        return turn


class RaisingRetriever:
    def retrieve(self, query, k=5):
        raise RuntimeError("simulated RAG retrieval outage")


def tu(id_, name, input_):
    return ToolUseBlock(id=id_, name=name, input=input_)


def rag_price(sportsbook, outcome, odds, is_current, doc_id):
    return ExtractedSportsbookPrice(
        sportsbook=sportsbook, selected_outcome=outcome, american_odds=odds,
        is_current=is_current, source_document_ids=[doc_id],
    )


def empty_rag_extraction(game_id="G-2026-001", market_id="G-2026-001-moneyline", outcome="Los Angeles Lakers"):
    return ExtractedMarketEvidence(
        game_id=game_id, market_id=market_id, selected_outcome=outcome, sportsbook_prices=[]
    )


END_TURN = ToolCallTurn(stop_reason="end_turn", text="done", tool_uses=[])


@pytest.fixture(scope="module")
def retriever():
    return Retriever.from_directory()


@pytest.fixture(scope="module")
def tools():
    return SportsbookTools(ControlledOddsProvider())


def _request(**overrides):
    defaults = dict(
        scenario_id="S001", game_id="G-2026-001", market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="DraftKings FanDuel BetMGM Caesars moneyline price Los Angeles Lakers Boston Celtics",
    )
    defaults.update(overrides)
    return AgentRequest(**defaults)


# ---------------------------------------------------------------------------
# Common agent contract / reuse
# ---------------------------------------------------------------------------


def test_hybrid_agent_implements_common_agent_contract(retriever, tools):
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(empty_rag_extraction(), [END_TURN]))
    assert isinstance(agent, Agent)
    assert agent.architecture == ArchitectureType.HYBRID


def test_hybrid_agent_defaults_to_anthropic_llm_client(retriever, tools):
    try:
        agent = HybridAgent(retriever, tools)
    except Exception:
        pytest.skip("anthropic client construction requires credentials in this environment")
    assert isinstance(agent.llm_client, AnthropicLLMClient)


def test_hybrid_agent_uses_real_rag_evidence_pipeline(retriever, tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-001", "sportsbook": "DraftKings", "market_type": "moneyline",
            "selected_outcome": "Los Angeles Lakers",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(empty_rag_extraction(), turns), top_k=10)
    agent.analyze(_request())
    assert len(agent.last_trace.retrieved_document_ids) == 10
    assert len(agent.last_trace.rag_scores) == 10


def test_hybrid_agent_uses_real_sportsbook_tools(retriever, tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-001", "sportsbook": "DraftKings", "market_type": "moneyline",
            "selected_outcome": "Los Angeles Lakers",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(empty_rag_extraction(), turns), top_k=10)
    analysis = agent.analyze(_request())
    assert analysis.best_odds == 120  # real ControlledOddsProvider value for DraftKings/Lakers


def test_no_duplicated_quant_formulas_in_hybrid_modules():
    forbidden_names = {
        "implied_probability", "expected_value", "is_positive_ev", "best_odds",
        "compare_american_odds", "calculate_no_vig_probabilities",
        "calculate_leave_one_out_consensus", "calculate_probability_edge",
    }
    for filename in ("hybrid_agent.py", "hybrid_reconciliation.py"):
        tree = ast.parse((AGENTS_DIR / filename).read_text())
        defined = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert defined.isdisjoint(forbidden_names), filename


# ---------------------------------------------------------------------------
# Case A — RAG and tool agree
# ---------------------------------------------------------------------------


def test_case_a_agreement(retriever, tools):
    rag = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[rag_price(
            "DraftKings", "Los Angeles Lakers", 120, True,
            "g-2026-001-moneyline-los-angeles-lakers-draftkings-v1",
        )],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-001", "sportsbook": "DraftKings", "market_type": "moneyline",
            "selected_outcome": "Los Angeles Lakers",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(rag, turns), top_k=10)
    analysis = agent.analyze(_request())
    assert analysis.best_odds == 120
    record = agent.last_trace.reconciled_records[0]
    assert record.conflict is False
    assert record.authoritative_source == SourceType.TOOL
    assert agent.last_trace.source_agreements == 1
    assert agent.last_trace.source_conflicts == 0
    assert agent.last_trace.validation_status == HybridFailureCategory.SUCCESS


# ---------------------------------------------------------------------------
# Case B — stale RAG vs. current tool
# ---------------------------------------------------------------------------


def test_case_b_stale_rag_vs_current_tool(retriever, tools):
    request = _request(
        scenario_id="S009", game_id="G-2026-009", selected_outcome="Minnesota Timberwolves",
        query="What price did DraftKings list for the Minnesota Timberwolves moneyline?",
    )
    rag = ExtractedMarketEvidence(
        game_id="G-2026-009", market_id="G-2026-009-moneyline", selected_outcome="Minnesota Timberwolves",
        sportsbook_prices=[rag_price(
            "DraftKings", "Minnesota Timberwolves", 120, False,
            "g-2026-009-moneyline-minnesota-timberwolves-draftkings-v0",
        )],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-009", "sportsbook": "DraftKings", "market_type": "moneyline",
            "selected_outcome": "Minnesota Timberwolves",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(rag, turns), top_k=10)
    analysis = agent.analyze(request)
    assert analysis.best_odds == 135  # real current value, never the stale +120
    record = agent.last_trace.reconciled_records[0]
    assert record.conflict is True
    assert record.conflict_resolution_reason.value == "current_tool_data_precedence"
    assert agent.last_trace.source_conflicts == 1


# ---------------------------------------------------------------------------
# Case D — tool only
# ---------------------------------------------------------------------------


def test_case_d_tool_only(retriever, tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-001", "sportsbook": "FanDuel", "market_type": "moneyline",
            "selected_outcome": "Los Angeles Lakers",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(empty_rag_extraction(), turns), top_k=10)
    analysis = agent.analyze(_request())
    assert analysis.best_odds == 125  # real FanDuel/Lakers current value
    assert agent.last_trace.tool_only_records >= 1
    assert agent.last_trace.rag_only_records == 0


# ---------------------------------------------------------------------------
# Case E — RAG only, tool missing
# ---------------------------------------------------------------------------


def test_case_e_rag_only_stale_tool_missing_not_promoted(retriever, tools):
    rag = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[rag_price(
            "BetMGM", "Los Angeles Lakers", 115, False,
            "g-2026-001-moneyline-los-angeles-lakers-betmgm-v1",
        )],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-001", "sportsbook": "NonexistentBook", "market_type": "moneyline",
            "selected_outcome": "Los Angeles Lakers",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(rag, turns), top_k=10)
    with pytest.raises(HybridAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    trace = exc_info.value.trace
    betmgm_record = next(r for r in trace.reconciled_records if r.sportsbook == "BetMGM")
    assert betmgm_record.authoritative_odds is None  # stale RAG-only never silently promoted
    assert trace.validation_status == HybridFailureCategory.INSUFFICIENT_CURRENT_DATA


def test_case_e_rag_only_current_and_no_tool_coverage_is_used(retriever, tools):
    rag = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[rag_price(
            "BetMGM", "Los Angeles Lakers", 115, True,
            "g-2026-001-moneyline-los-angeles-lakers-betmgm-v1",
        )],
    )
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(rag, [END_TURN]), top_k=10)
    analysis = agent.analyze(_request())
    assert analysis.best_odds == 115
    assert analysis.best_sportsbook == "BetMGM"
    record = agent.last_trace.reconciled_records[0]
    assert record.authoritative_source == SourceType.RAG


# ---------------------------------------------------------------------------
# Case F — hallucinated LLM odds
# ---------------------------------------------------------------------------


def test_case_f_hallucinated_final_prose_rejected(retriever, tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-001", "sportsbook": "DraftKings", "market_type": "moneyline",
            "selected_outcome": "Los Angeles Lakers",
        })]),
        ToolCallTurn(stop_reason="end_turn", text="DraftKings is actually offering +180.", tool_uses=[]),
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(empty_rag_extraction(), turns), top_k=10)
    analysis = agent.analyze(_request())
    assert analysis.best_odds == 120
    assert analysis.best_odds != 180


def test_hallucinated_rag_sportsbook_never_reaches_reconciliation(retriever, tools):
    # extraction.py's validate_extraction_provenance (reused verbatim)
    # rejects a sportsbook that was never actually retrieved — hybrid
    # must never see it at all.
    rag = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="BetRivers", selected_outcome="Los Angeles Lakers", american_odds=130,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"],
            ),
        ],
    )
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(rag, [END_TURN]), top_k=10)
    with pytest.raises(HybridAnalysisIncomplete):
        agent.analyze(_request())
    assert not any(r.sportsbook == "BetRivers" for r in agent.last_trace.reconciled_records)
    assert len(agent.last_trace.rag_rejected_reasons) == 1


# ---------------------------------------------------------------------------
# Two-sided market / best-line / no-vig / consensus / edge / EV
# ---------------------------------------------------------------------------


def test_two_sided_reconciliation_full_quant(retriever, tools):
    rag = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[rag_price(
            "Caesars", "Los Angeles Lakers", 122, True,
            "g-2026-001-moneyline-los-angeles-lakers-caesars-v1",
        )],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_odds", {
            "game_id": "G-2026-001", "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
        })]),
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t2", "get_odds", {
            "game_id": "G-2026-001", "market_type": "moneyline", "selected_outcome": "Boston Celtics",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(rag, turns), top_k=10)
    analysis = agent.analyze(_request())

    assert analysis.status == AnalysisStatus.OK
    assert analysis.best_sportsbook == "FanDuel"
    assert analysis.best_odds == 125

    no_vig = {
        "DraftKings": calculate_no_vig_probabilities([120, -140])[0],
        "FanDuel": calculate_no_vig_probabilities([125, -145])[0],
        "BetMGM": calculate_no_vig_probabilities([115, -135])[0],
        "Caesars": calculate_no_vig_probabilities([122, -142])[0],
    }
    expected_reference = calculate_leave_one_out_consensus(list(no_vig.items()), "FanDuel")
    expected_ev = expected_value(125, expected_reference)

    assert analysis.market_reference_probability == pytest.approx(expected_reference)
    assert analysis.expected_value == pytest.approx(expected_ev)
    assert analysis.positive_ev == is_positive_ev(125, expected_reference)
    assert analysis.implied_probability == pytest.approx(implied_probability(125))


def test_best_line_example_from_spec_section_25():
    # RAG: DraftKings +125 stale, FanDuel +120 (no tool coverage claim
    # here — evidence only). Tools (synthetic, via a FakeProvider so the
    # exact spec numbers can be hit): DraftKings +130 current, FanDuel
    # +135 current, BetMGM +128 current. Expected: FanDuel +135 —
    # stale RAG +125 must not affect the result.
    game = Game(
        game_id="G-FAKE-BESTLINE", home_team="Home", away_team="Away",
        start_time=datetime(2026, 1, 1), sport="basketball",
    )
    current_odds = [
        SportsbookOdds(sportsbook="DraftKings", american_odds=130, is_current=True),
        SportsbookOdds(sportsbook="FanDuel", american_odds=135, is_current=True),
        SportsbookOdds(sportsbook="BetMGM", american_odds=128, is_current=True),
    ]

    class FakeProvider(OddsProvider):
        def get_games(self):
            return [game]

        def get_game(self, game_id):
            if game_id != game.game_id:
                raise LookupError(game_id)
            return game

        def get_odds(self, game_id, market_type, selected_outcome):
            if game_id != game.game_id or selected_outcome != "Home":
                raise LookupError(selected_outcome)
            return list(current_odds)

        def get_sportsbook_odds(self, game_id, sportsbook, market_type, selected_outcome):
            matches = [o for o in self.get_odds(game_id, market_type, selected_outcome) if o.sportsbook == sportsbook]
            if not matches:
                raise LookupError(sportsbook)
            return matches[0]

    fake_tools = SportsbookTools(FakeProvider())
    retriever = Retriever.from_directory()

    rag = ExtractedMarketEvidence(
        game_id=game.game_id, market_id=f"{game.game_id}-moneyline", selected_outcome="Home",
        sportsbook_prices=[
            rag_price("DraftKings", "Home", 125, False, "fake-dk-doc"),
        ],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_odds", {
            "game_id": game.game_id, "market_type": "moneyline", "selected_outcome": "Home",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, fake_tools, llm_client=FakeHybridLLMClient(rag, turns), top_k=1)
    analysis = agent.analyze(AgentRequest(
        scenario_id="S-FAKE", game_id=game.game_id, market_type=MarketType.MONEYLINE,
        selected_outcome="Home", query="best current price on Home moneyline",
    ))
    assert analysis.best_sportsbook == "FanDuel"
    assert analysis.best_odds == 135


# ---------------------------------------------------------------------------
# Bounded tool loop / graceful degradation
# ---------------------------------------------------------------------------


def test_bounded_tool_loop_still_respects_max_iterations(retriever, tools):
    runaway_turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu(f"t{i}", "get_games", {})])
        for i in range(1, 20)
    ]
    agent = HybridAgent(
        retriever, tools, llm_client=FakeHybridLLMClient(empty_rag_extraction(), runaway_turns),
        top_k=10, max_tool_iterations=MAX_TOOL_ITERATIONS,
    )
    with pytest.raises(HybridAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    trace = exc_info.value.trace
    assert trace.tool_iterations_used == MAX_TOOL_ITERATIONS
    assert trace.validation_status == HybridFailureCategory.TOOL_FAILURE


def test_graceful_degradation_rag_unavailable_tools_sufficient(tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-001", "sportsbook": "DraftKings", "market_type": "moneyline",
            "selected_outcome": "Los Angeles Lakers",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(
        RaisingRetriever(), tools, llm_client=FakeHybridLLMClient(empty_rag_extraction(), turns), top_k=10,
    )
    analysis = agent.analyze(_request())
    assert analysis.best_odds == 120
    assert agent.last_trace.validation_status == HybridFailureCategory.SUCCESS
    assert any("RAG retrieval failed" in error for error in agent.last_trace.errors)


def test_graceful_degradation_tools_partial_rag_context_no_fabrication(retriever, tools):
    # Only one side gathered via tools; no opposing-side data anywhere ->
    # best line still honestly reported, EV honestly withheld.
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_odds", {
            "game_id": "G-2026-001", "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(empty_rag_extraction(), turns), top_k=10)
    analysis = agent.analyze(_request())
    assert analysis.status == AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE
    assert analysis.expected_value is None
    assert analysis.positive_ev is None
    assert analysis.best_odds == 125


def test_both_channels_unavailable_explicit_failure(tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-001", "sportsbook": "NonexistentBook", "market_type": "moneyline",
            "selected_outcome": "Los Angeles Lakers",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(
        RaisingRetriever(), tools, llm_client=FakeHybridLLMClient(empty_rag_extraction(), turns), top_k=10,
    )
    with pytest.raises(HybridAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    assert exc_info.value.trace.validation_status == HybridFailureCategory.RAG_RETRIEVAL_FAILURE


# ---------------------------------------------------------------------------
# Provenance / conflict tracing / common output validation
# ---------------------------------------------------------------------------


def test_sources_reflect_only_authoritative_contributions(retriever, tools):
    rag = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[rag_price(
            "BetMGM", "Los Angeles Lakers", 999, False,  # stale, never authoritative
            "g-2026-001-moneyline-los-angeles-lakers-betmgm-v1",
        )],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-001", "sportsbook": "DraftKings", "market_type": "moneyline",
            "selected_outcome": "Los Angeles Lakers",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(rag, turns), top_k=10)
    analysis = agent.analyze(_request())
    assert all(s.source_type == SourceType.TOOL for s in analysis.sources)
    assert "BetMGM" not in {s.sportsbook for s in analysis.sources}


def test_conflict_fully_traced_with_all_required_fields(retriever, tools):
    request = _request(
        scenario_id="S009", game_id="G-2026-009", selected_outcome="Minnesota Timberwolves",
        query="What price did DraftKings list for the Minnesota Timberwolves moneyline?",
    )
    rag = ExtractedMarketEvidence(
        game_id="G-2026-009", market_id="G-2026-009-moneyline", selected_outcome="Minnesota Timberwolves",
        sportsbook_prices=[rag_price(
            "DraftKings", "Minnesota Timberwolves", 120, False,
            "g-2026-009-moneyline-minnesota-timberwolves-draftkings-v0",
        )],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-009", "sportsbook": "DraftKings", "market_type": "moneyline",
            "selected_outcome": "Minnesota Timberwolves",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(rag, turns), top_k=10)
    agent.analyze(request)
    record = agent.last_trace.reconciled_records[0]
    assert record.sportsbook == "DraftKings"
    assert record.selected_outcome == "Minnesota Timberwolves"
    assert record.rag_odds == 120
    assert record.rag_is_current is False
    assert record.tool_odds == 135
    assert record.authoritative_odds == 135
    assert record.authoritative_source == SourceType.TOOL
    assert record.conflict_resolution_reason is not None


def test_final_analysis_round_trips_as_valid_betting_analysis(retriever, tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-001", "sportsbook": "DraftKings", "market_type": "moneyline",
            "selected_outcome": "Los Angeles Lakers",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(empty_rag_extraction(), turns), top_k=10)
    analysis = agent.analyze(_request())
    BettingAnalysis.model_validate(analysis.model_dump())
    assert analysis.architecture == ArchitectureType.HYBRID


def test_trace_is_hybrid_agent_trace_with_expected_architecture_label(retriever, tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_sportsbook_odds", {
            "game_id": "G-2026-001", "sportsbook": "DraftKings", "market_type": "moneyline",
            "selected_outcome": "Los Angeles Lakers",
        })]),
        END_TURN,
    ]
    agent = HybridAgent(retriever, tools, llm_client=FakeHybridLLMClient(empty_rag_extraction(), turns), top_k=10)
    agent.analyze(_request())
    assert isinstance(agent.last_trace, HybridAgentTrace)
    assert agent.last_trace.architecture == "hybrid"


# ---------------------------------------------------------------------------
# Architecture integrity (Step 30) — RAG-only/tool-only boundaries unweakened
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


def test_rag_only_agent_still_has_no_tool_access():
    modules = _imported_modules("rag_agent.py")
    assert not any(m.startswith("src.tools") or m.startswith("src.providers") for m in modules)


def test_tool_only_agent_still_has_no_rag_access():
    modules = _imported_modules("tool_agent.py")
    assert not any(m.startswith("src.rag") for m in modules)


def test_hybrid_agent_has_both_rag_and_tool_access():
    modules = _imported_modules("hybrid_agent.py")
    assert any(m.startswith("src.rag") for m in modules)
    assert any(m.startswith("src.tools") for m in modules)


def test_rag_only_and_tool_only_agents_unmodified_still_single_channel(retriever, tools):
    # Smoke-level confirmation that instantiating RagOnlyAgent/
    # ToolCallingAgent still only needs their own single channel — no
    # new cross-access was introduced by this milestone.
    class DummyRagLLM:
        def generate_structured(self, *, system_prompt, user_prompt, response_model):
            return response_model(
                game_id="G-2026-001", market_id="G-2026-001-moneyline",
                selected_outcome="Los Angeles Lakers", sportsbook_prices=[],
            )

    rag_agent = RagOnlyAgent(retriever, llm_client=DummyRagLLM())
    assert not hasattr(rag_agent, "tools")

    class DummyToolLLM:
        def create_turn(self, *, system_prompt, messages, tools):
            return END_TURN

    tool_agent = ToolCallingAgent(tools, llm_client=DummyToolLLM())
    assert not hasattr(tool_agent, "retriever")
