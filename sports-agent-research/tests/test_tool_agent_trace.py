"""Milestone 9A, Step 20/21: the required mock-LLM cases (A-F), plus
tool_schemas.py input validation and find_best_line-driven market-state
folding. Complements tests/test_tool_agent.py, which covers the broader
contract/quant/isolation surface. No real API calls anywhere here.
"""

import pytest
from pydantic import ValidationError

from src.agents.base import AgentRequest
from src.agents.llm_client import ToolCallTurn, ToolUseBlock
from src.agents.tool_agent import ToolAnalysisIncomplete, ToolCallingAgent
from src.agents.tool_schemas import (
    GetOddsInput,
    execute_tool,
)
from src.models import MarketType
from src.providers.controlled import ControlledOddsProvider
from src.tools.sportsbook_tools import SportsbookTools


class ScriptedLLMClient:
    model = "fake-model"
    effort = "low"

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls_made = 0

    def create_turn(self, *, system_prompt, messages, tools):
        turn = self.turns[self.calls_made]
        self.calls_made += 1
        return turn


def tu(id_, name, input_):
    return ToolUseBlock(id=id_, name=name, input=input_)


@pytest.fixture(scope="module")
def tools():
    return SportsbookTools(ControlledOddsProvider())


def _request(**overrides):
    defaults = dict(
        scenario_id="S001",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="What is the best current price on the Lakers moneyline?",
    )
    defaults.update(overrides)
    return AgentRequest(**defaults)


END_TURN = ToolCallTurn(stop_reason="end_turn", text="done", tool_uses=[])


# ---------------------------------------------------------------------------
# Case A — Correct Tool Use
# ---------------------------------------------------------------------------


def test_case_a_correct_tool_use(tools):
    turns = [
        ToolCallTurn(
            stop_reason="tool_use", text=None,
            tool_uses=[tu("t1", "get_odds", {
                "game_id": "G-2026-001", "market_type": "moneyline",
                "selected_outcome": "Los Angeles Lakers",
            })],
        ),
        END_TURN,
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    assert agent.last_trace.tool_calls[0].success is True
    assert analysis.best_sportsbook == "FanDuel"
    assert analysis.best_odds == 125


# ---------------------------------------------------------------------------
# Case B — Multiple Tool Calls
# ---------------------------------------------------------------------------


def test_case_b_multiple_tool_calls_correct_order(tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_game", {"game_id": "G-2026-001"})]),
        ToolCallTurn(
            stop_reason="tool_use", text=None,
            tool_uses=[tu("t2", "get_odds", {
                "game_id": "G-2026-001", "market_type": "moneyline",
                "selected_outcome": "Los Angeles Lakers",
            })],
        ),
        END_TURN,
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    assert agent.last_trace.tool_call_order == ["get_game", "get_odds"]
    assert isinstance(analysis.best_odds, int)


# ---------------------------------------------------------------------------
# Case C — Invalid Sportsbook
# ---------------------------------------------------------------------------


def test_case_c_invalid_sportsbook_explicit_failure_no_replacement(tools):
    turns = [
        ToolCallTurn(
            stop_reason="tool_use", text=None,
            tool_uses=[tu("t1", "get_sportsbook_odds", {
                "game_id": "G-2026-001", "sportsbook": "NotARealBook",
                "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
            })],
        ),
        END_TURN,
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    with pytest.raises(ToolAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    assert exc_info.value.trace.tool_calls[0].success is False


# ---------------------------------------------------------------------------
# Case D — Repeated Tool Call
# ---------------------------------------------------------------------------


def test_case_d_repeated_tool_call_traceable(tools):
    call_args = {
        "game_id": "G-2026-001", "market_type": "moneyline",
        "selected_outcome": "Los Angeles Lakers",
    }
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_odds", call_args)]),
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t2", "get_odds", dict(call_args))]),
        END_TURN,
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    agent.analyze(_request())
    trace = agent.last_trace
    assert len(trace.tool_calls) == 2
    assert trace.tool_calls[1].is_redundant is True
    assert trace.redundant_call_count == 1


# ---------------------------------------------------------------------------
# Case E — Tool Loop Limit
# ---------------------------------------------------------------------------


def test_case_e_loop_limit_bounded_explicit_failure(tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu(f"t{i}", "get_games", {})])
        for i in range(1, 20)
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns), max_iterations=6)
    with pytest.raises(ToolAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    assert exc_info.value.trace.validation_status == "loop_limit_exceeded"
    assert exc_info.value.trace.iterations_used == 6
    assert len(exc_info.value.trace.tool_calls) == 6  # bounded, not runaway


# ---------------------------------------------------------------------------
# Case F — LLM Hallucinates Final Odds
# ---------------------------------------------------------------------------


def test_case_f_hallucinated_final_odds_rejected(tools):
    turns = [
        ToolCallTurn(
            stop_reason="tool_use", text=None,
            tool_uses=[tu("t1", "get_sportsbook_odds", {
                "game_id": "G-2026-001", "sportsbook": "DraftKings",
                "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
            })],
        ),
        ToolCallTurn(stop_reason="end_turn", text="DraftKings is offering +180 on the Lakers.", tool_uses=[]),
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    assert analysis.best_odds == 120
    assert analysis.best_odds != 180


# ---------------------------------------------------------------------------
# Tool input validation (Milestone 9A, Step 6)
# ---------------------------------------------------------------------------


def test_get_odds_input_rejects_invalid_market_type():
    with pytest.raises(ValidationError):
        GetOddsInput(game_id="G-2026-001", market_type="not_a_market", selected_outcome="Los Angeles Lakers")


def test_get_odds_input_rejects_missing_required_field():
    with pytest.raises(ValidationError):
        GetOddsInput(game_id="G-2026-001", market_type="moneyline")


def test_execute_tool_rejects_unknown_tool_name(tools):
    with pytest.raises(ValueError):
        execute_tool(tools, "delete_all_odds", {})


def test_execute_tool_rejects_blank_game_id(tools):
    with pytest.raises(ValidationError):
        execute_tool(
            tools, "get_odds",
            {"game_id": "", "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers"},
        )


def test_execute_tool_get_odds_returns_structured_and_raw(tools):
    structured, raw = execute_tool(
        tools, "get_odds",
        {"game_id": "G-2026-001", "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers"},
    )
    assert isinstance(structured, list)
    assert all(isinstance(item, dict) for item in structured)
    assert {item.sportsbook for item in raw} == {"DraftKings", "FanDuel", "BetMGM", "Caesars"}


# ---------------------------------------------------------------------------
# find_best_line-driven market-state folding (Milestone 9A, Step 21 "Best Line")
# ---------------------------------------------------------------------------


def test_find_best_line_tool_call_folds_tied_sportsbooks_into_market_state(tools):
    turns = [
        ToolCallTurn(
            stop_reason="tool_use", text=None,
            tool_uses=[tu("t1", "find_best_line", {
                "game_id": "G-2026-001", "market_type": "moneyline",
                "selected_outcome": "Los Angeles Lakers",
            })],
        ),
        END_TURN,
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    # find_best_line alone only reveals the winning price/sportsbook(s),
    # so no losing side is available for consensus — best line is still
    # correct, but quant must honestly stay insufficient.
    assert analysis.best_odds == 125
    assert analysis.best_sportsbook == "FanDuel"
    assert analysis.status.value == "insufficient_quant_evidence"


def test_get_game_call_does_not_fold_odds_into_market_state(tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu("t1", "get_game", {"game_id": "G-2026-001"})]),
        END_TURN,
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    with pytest.raises(ToolAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    assert exc_info.value.trace.validation_status == "no_valid_prices"
