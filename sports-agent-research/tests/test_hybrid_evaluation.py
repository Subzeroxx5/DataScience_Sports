"""Unit-level tests for src/evaluation/hybrid_agent_evaluation.py
(Milestone 10B): ground-truth isolation, the deterministic fake LLM
policy, and evaluate_scenario()/summarize_results() logic with
constructed fixtures. See tests/test_hybrid_agent_e2e.py for full
harness runs against the real corpus/provider.
"""

import ast
from pathlib import Path

import pytest

from src.agents.base import AgentRequest
from src.agents.hybrid_agent import HybridAgent, HybridFailureCategory
from src.agents.llm_client import ToolCallTurn
from src.evaluation.hybrid_agent_evaluation import (
    DeterministicHybridPolicyLLMClient,
    _build_agent_request,
    _parse_rag_evidence_blocks,
    evaluate_scenario,
    summarize_results,
)
from src.models import GroundTruth, MarketType, QuantGroundTruth
from src.providers.controlled import ControlledOddsProvider
from src.rag.retriever import Retriever
from src.tools.sportsbook_tools import SportsbookTools

AGENTS_DIR = Path(__file__).resolve().parent.parent / "src" / "agents"

RENDERED_EVIDENCE = """Query: q

[DOCUMENT 1]
document_id: g-2026-001-moneyline-los-angeles-lakers-draftkings-v1
rank: 1
similarity_score: 0.7000
source_type: sportsbook_snapshot
game_id: G-2026-001
market_id: G-2026-001-moneyline
market_type: moneyline
outcome: Los Angeles Lakers
sportsbook: DraftKings
american_odds: +120
is_current: True
content: DraftKings lists Los Angeles Lakers at +120 on the moneyline.

[DOCUMENT 2]
document_id: g-2026-001-moneyline-boston-celtics-draftkings-v1
rank: 2
similarity_score: 0.6500
source_type: sportsbook_snapshot
game_id: G-2026-001
market_id: G-2026-001-moneyline
market_type: moneyline
outcome: Boston Celtics
sportsbook: DraftKings
american_odds: -140
is_current: True
content: DraftKings lists Boston Celtics at -140 on the moneyline.
"""


# ---------------------------------------------------------------------------
# Ground-truth isolation
# ---------------------------------------------------------------------------


def test_evaluator_can_import_ground_truth_generators():
    from src.evaluation.hybrid_agent_evaluation import generate_all_ground_truth, generate_all_quant_ground_truth

    assert callable(generate_all_ground_truth)
    assert callable(generate_all_quant_ground_truth)


def test_hybrid_agent_source_never_imports_evaluation():
    for filename in ("hybrid_agent.py", "hybrid_reconciliation.py"):
        tree = ast.parse((AGENTS_DIR / filename).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any(m.startswith("src.evaluation") for m in imported), filename


def test_agent_request_built_by_evaluator_has_no_ground_truth_fields():
    request = _build_agent_request(
        {
            "scenario_id": "S001",
            "game": {"game_id": "G-2026-001", "home_team": "Los Angeles Lakers", "away_team": "Boston Celtics"},
            "market": {"market_type": "moneyline", "selected_outcome": "Los Angeles Lakers"},
        }
    )
    forbidden = {"expected_best_sportsbook", "expected_best_odds", "expected_ev", "expected_positive_ev"}
    assert not (set(AgentRequest.model_fields.keys()) & forbidden)
    assert "Boston Celtics" in request.query  # opposing team correctly included for retrieval


# ---------------------------------------------------------------------------
# RAG evidence parsing (deterministic fake policy)
# ---------------------------------------------------------------------------


def test_parse_rag_evidence_blocks_extracts_known_fields():
    blocks = _parse_rag_evidence_blocks(RENDERED_EVIDENCE)
    assert len(blocks) == 2
    assert blocks[0]["sportsbook"] == "DraftKings"
    assert blocks[0]["outcome"] == "Los Angeles Lakers"
    assert blocks[0]["american_odds"] == "+120"
    assert blocks[0]["is_current"] == "True"


def test_generate_structured_extracts_primary_and_opposing_price():
    header = "Game ID: G-2026-001\nSelected outcome: Los Angeles Lakers\n\n"
    policy = DeterministicHybridPolicyLLMClient()
    from src.agents.extraction import ExtractedMarketEvidence

    extraction = policy.generate_structured(
        system_prompt="sys", user_prompt=header + RENDERED_EVIDENCE, response_model=ExtractedMarketEvidence
    )
    assert len(extraction.sportsbook_prices) == 1
    price = extraction.sportsbook_prices[0]
    assert price.sportsbook == "DraftKings"
    assert price.american_odds == 120
    assert price.opposing_outcome == "Boston Celtics"
    assert price.opposing_american_odds == -140
    assert price.source_document_ids == [
        "g-2026-001-moneyline-los-angeles-lakers-draftkings-v1",
        "g-2026-001-moneyline-boston-celtics-draftkings-v1",
    ]


def test_generate_structured_never_fabricates_missing_sportsbook():
    header = "Game ID: G-2026-001\nSelected outcome: Los Angeles Lakers\n\n"
    policy = DeterministicHybridPolicyLLMClient()
    from src.agents.extraction import ExtractedMarketEvidence

    extraction = policy.generate_structured(
        system_prompt="sys", user_prompt=header, response_model=ExtractedMarketEvidence
    )
    assert extraction.sportsbook_prices == []


def test_policy_never_references_ground_truth():
    source_path = Path(__file__).resolve().parent.parent / "src" / "evaluation" / "hybrid_agent_evaluation.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    policy_class = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "DeterministicHybridPolicyLLMClient"
    )
    body = policy_class.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    for node in body:
        code_source = ast.get_source_segment(source, node) or ""
        assert "GroundTruth" not in code_source
        assert "ground_truth" not in code_source.lower()


def test_tool_side_delegates_to_deterministic_tool_policy():
    policy = DeterministicHybridPolicyLLMClient()
    turn = policy.create_turn(
        system_prompt="sys",
        messages=[{"role": "user", "content": "Game ID: G-2026-001\nMarket: moneyline\nSelected outcome: Los Angeles Lakers\n"}],
        tools=[],
    )
    assert isinstance(turn, ToolCallTurn)
    assert turn.tool_uses[0].name == "get_game"


# ---------------------------------------------------------------------------
# evaluate_scenario() — constructed fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tools():
    return SportsbookTools(ControlledOddsProvider())


@pytest.fixture(scope="module")
def retriever():
    return Retriever.from_directory()


class FixedLLMClient:
    def __init__(self, rag_extraction, tool_turns):
        self.rag_extraction = rag_extraction
        self.tool_turns = list(tool_turns)
        self.calls_made = 0

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        return self.rag_extraction

    def create_turn(self, *, system_prompt, messages, tools):
        turn = self.tool_turns[self.calls_made]
        self.calls_made += 1
        return turn


def _ground_truth(**overrides) -> GroundTruth:
    defaults = dict(
        scenario_id="S001", expected_best_sportsbook="FanDuel", expected_best_odds=125,
        expected_implied_probability=0.4444, expected_ev=0.0, expected_positive_ev=False,
        expected_sportsbooks=["DraftKings", "FanDuel", "BetMGM", "Caesars"],
        expected_best_sportsbooks=["FanDuel"],
    )
    defaults.update(overrides)
    return GroundTruth(**defaults)


def _quant_ground_truth(**overrides) -> QuantGroundTruth:
    defaults = dict(
        scenario_id="S001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        quant_evaluable=False, ineligibility_reason="not evaluated in this test",
    )
    defaults.update(overrides)
    return QuantGroundTruth(**defaults)


def _request(**overrides):
    defaults = dict(
        scenario_id="S001", game_id="G-2026-001", market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers", query="q",
    )
    defaults.update(overrides)
    return AgentRequest(**defaults)


def test_evaluate_scenario_success_case(retriever, tools):
    from src.agents.extraction import ExtractedMarketEvidence
    from src.agents.llm_client import ToolUseBlock

    rag = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[
            ToolUseBlock(
                id="t1", name="get_sportsbook_odds",
                input={"game_id": "G-2026-001", "sportsbook": "DraftKings", "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers"},
            )
        ]),
        ToolCallTurn(stop_reason="end_turn", text="done", tool_uses=[]),
    ]
    agent = HybridAgent(retriever, tools, llm_client=FixedLLMClient(rag, turns), top_k=10)
    result = evaluate_scenario(agent, tools, _request(), _ground_truth(expected_best_sportsbook="DraftKings", expected_best_odds=120, expected_best_sportsbooks=["DraftKings"]), _quant_ground_truth())
    assert result.best_line_correct is True
    assert result.best_odds_correct is True
    assert result.hallucination_detected is False


def test_evaluate_scenario_detects_incorrect_best_line(retriever, tools):
    from src.agents.extraction import ExtractedMarketEvidence
    from src.agents.llm_client import ToolUseBlock

    rag = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[],
    )
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[
            ToolUseBlock(id="t1", name="get_sportsbook_odds", input={
                "game_id": "G-2026-001", "sportsbook": "DraftKings", "market_type": "moneyline",
                "selected_outcome": "Los Angeles Lakers",
            })
        ]),
        ToolCallTurn(stop_reason="end_turn", text="done", tool_uses=[]),
    ]
    agent = HybridAgent(retriever, tools, llm_client=FixedLLMClient(rag, turns), top_k=10)
    # Deliberately wrong ground truth.
    wrong_gt = _ground_truth(expected_best_sportsbook="FanDuel", expected_best_odds=125, expected_best_sportsbooks=["FanDuel"])
    result = evaluate_scenario(agent, tools, _request(), wrong_gt, _quant_ground_truth())
    assert result.best_line_correct is False
    assert result.best_odds_correct is False


def test_summarize_results_conflict_accuracy_none_when_no_conflicts():
    from src.evaluation.hybrid_agent_evaluation import HybridAgentEvaluationResult

    result = HybridAgentEvaluationResult(
        scenario_id="S001", execution_status=HybridFailureCategory.SUCCESS, quant_evaluable=False,
        predicted_best_sportsbooks=["FanDuel"], expected_best_sportsbooks=["FanDuel"], best_line_correct=True,
        predicted_best_odds=125, expected_best_odds=125, best_odds_correct=True,
        predicted_positive_ev=None, expected_positive_ev=None, ev_classification_correct=None,
        predicted_ev=None, expected_ev=None, ev_absolute_error=None,
        predicted_market_reference_probability=None, expected_market_reference_probability=None,
        market_reference_absolute_error=None, freshness_correct=None, completeness=1.0,
        hallucination_detected=False, source_agreements=1, source_conflicts=0,
        correct_conflict_resolutions=0, stale_rag_conflicts=0, stale_rag_incorrectly_promoted=0,
        tool_only_records_used=0, rag_only_records_observed=0, source_reconciliation_failure=False,
        rag_documents_retrieved=5, tool_call_count=1, redundant_tool_call_count=0,
        rag_retrieval_latency_seconds=0.0, rag_llm_latency_seconds=0.0, tool_llm_latency_seconds=0.0,
        tool_execution_latency_seconds=0.0, reconciliation_latency_seconds=0.0, quant_latency_seconds=0.0,
        total_latency_seconds=0.0, errors=[],
    )
    summary = summarize_results([result])
    assert summary["conflict_resolution_accuracy"] is None
    assert summary["source_conflicts"] == 0


def test_summarize_results_does_not_score_non_quant_evaluable_as_wrong_ev():
    from src.evaluation.hybrid_agent_evaluation import evaluate_scenarios

    results = evaluate_scenarios(["S002"])  # single-sided, not quant-evaluable
    summary = summarize_results(results)
    assert results[0].ev_classification_correct is None
    assert summary["ev_classification_accuracy"] is None
