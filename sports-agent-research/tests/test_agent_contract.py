"""Tests for src/agents/base.py (Milestone 8A) — the common agent
contract shared by the future RAG-only, tool-calling-only, and hybrid
architectures. No concrete agent exists yet; these tests validate the
interface and the request/output contracts only."""

import inspect
from abc import ABC

import pytest
from pydantic import ValidationError

from src.agents.base import Agent, AgentRequest
from src.models import ArchitectureType, BettingAnalysis, MarketType


# ---------------------------------------------------------------------------
# Common agent interface
# ---------------------------------------------------------------------------


def test_agent_is_abstract_base_class():
    assert issubclass(Agent, ABC)
    assert inspect.isabstract(Agent)


def test_agent_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Agent()


def test_agent_analyze_is_abstract():
    method = Agent.analyze
    assert getattr(method, "__isabstractmethod__", False) is True


def test_agent_analyze_signature_accepts_agent_request_and_returns_betting_analysis():
    import typing

    hints = typing.get_type_hints(Agent.analyze)
    assert hints["request"] is AgentRequest
    assert hints["return"] is BettingAnalysis


def test_complete_agent_subclass_can_be_instantiated():
    class _StubAgent(Agent):
        architecture = ArchitectureType.RAG

        def analyze(self, request: AgentRequest) -> BettingAnalysis:
            raise NotImplementedError

    agent = _StubAgent()
    assert isinstance(agent, Agent)
    assert agent.architecture == ArchitectureType.RAG


def test_incomplete_agent_subclass_cannot_be_instantiated():
    class _IncompleteAgent(Agent):
        architecture = ArchitectureType.TOOL
        # analyze() intentionally not implemented

    with pytest.raises(TypeError):
        _IncompleteAgent()


@pytest.mark.parametrize("architecture", [ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID])
def test_supported_architecture_identifiers_remain_valid(architecture):
    class _StubAgent(Agent):
        def analyze(self, request: AgentRequest) -> BettingAnalysis:
            raise NotImplementedError

    agent = _StubAgent()
    agent.architecture = architecture
    assert agent.architecture in (ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID)


# ---------------------------------------------------------------------------
# AgentRequest
# ---------------------------------------------------------------------------


def _make_request(**overrides):
    defaults = dict(
        scenario_id="S001",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="What price did DraftKings have on the Lakers moneyline?",
    )
    defaults.update(overrides)
    return AgentRequest(**defaults)


def test_agent_request_valid():
    request = _make_request()
    assert request.scenario_id == "S001"
    assert request.market_type == MarketType.MONEYLINE


def test_agent_request_rejects_blank_scenario_id():
    with pytest.raises(ValidationError):
        _make_request(scenario_id="")


def test_agent_request_rejects_blank_game_id():
    with pytest.raises(ValidationError):
        _make_request(game_id="")


def test_agent_request_rejects_blank_selected_outcome():
    with pytest.raises(ValidationError):
        _make_request(selected_outcome="")


def test_agent_request_rejects_blank_query():
    with pytest.raises(ValidationError):
        _make_request(query="")


def test_agent_request_rejects_invalid_market_type():
    with pytest.raises(ValidationError):
        _make_request(market_type="not_a_real_market")


def test_agent_request_has_no_ground_truth_fields():
    forbidden_field_names = {
        "expected_best_sportsbook",
        "expected_best_odds",
        "expected_ev",
        "expected_positive_ev",
        "expected_implied_probability",
        "estimated_true_probability",
        "market_reference_probability",
        "probability_edge",
        "ground_truth",
        "correct_answer",
    }
    actual_fields = set(AgentRequest.model_fields.keys())
    assert not (actual_fields & forbidden_field_names)


# ---------------------------------------------------------------------------
# Common output contract (BettingAnalysis) remains usable — not populated
# in this milestone, only confirmed suitable
# ---------------------------------------------------------------------------


def _make_analysis(**overrides):
    defaults = dict(
        scenario_id="S001",
        game_id="G-2026-001",
        market=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        best_sportsbook="FanDuel",
        best_odds=125,
        implied_probability=0.4,
        estimated_true_probability=0.45,
        expected_value=0.125,
        positive_ev=True,
        sportsbooks_considered=["DraftKings", "FanDuel"],
        reasoning_summary="Placeholder reasoning for contract testing.",
        architecture=ArchitectureType.RAG,
    )
    defaults.update(overrides)
    return BettingAnalysis(**defaults)


def test_betting_analysis_still_constructs_with_original_required_fields_only():
    analysis = _make_analysis()
    assert analysis.market_reference_probability is None
    assert analysis.probability_edge is None
    assert analysis.best_sportsbooks == ["FanDuel"]


def test_betting_analysis_supports_market_consensus_fields_additively():
    analysis = _make_analysis(
        market_reference_probability=0.4398818182333828,
        probability_edge=-0.004562626211061627,
    )
    assert analysis.market_reference_probability == pytest.approx(0.4398818182333828)
    assert analysis.probability_edge == pytest.approx(-0.004562626211061627)
    # Original controlled-reference fields remain populated alongside.
    assert analysis.estimated_true_probability == 0.45
    assert analysis.expected_value == pytest.approx(0.125)


def test_betting_analysis_supports_tied_best_sportsbooks():
    analysis = _make_analysis(
        best_sportsbook="DraftKings",
        best_sportsbooks=["DraftKings", "FanDuel"],
    )
    assert set(analysis.best_sportsbooks) == {"DraftKings", "FanDuel"}


def test_betting_analysis_rejects_best_sportsbook_not_in_best_sportsbooks():
    with pytest.raises(ValidationError):
        _make_analysis(best_sportsbook="DraftKings", best_sportsbooks=["FanDuel", "BetMGM"])


def test_betting_analysis_rejects_out_of_range_market_reference_probability():
    with pytest.raises(ValidationError):
        _make_analysis(market_reference_probability=1.5)
