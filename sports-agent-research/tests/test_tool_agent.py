"""Tests for src/agents/tool_agent.py (Milestone 9A) — the tool-calling
agent. Uses a scripted fake ToolCallingLLMClient throughout; no test in
this file makes a real API call. See experiments/run_tool_agent_smoke_test.py
for the credentialed manual smoke test against the real Anthropic API.

Every test drives the REAL ControlledOddsProvider / data/current_odds.json
(no provider mocking) so tool execution, argument forwarding, and
current-data semantics are verified end-to-end.
"""

import ast
from pathlib import Path

import pytest

from src.agents.base import Agent, AgentRequest
from src.agents.llm_client import ToolCallTurn, ToolUseBlock
from src.agents.tool_agent import (
    MAX_TOOL_ITERATIONS,
    ToolAgentTrace,
    ToolAnalysisIncomplete,
    ToolCallingAgent,
)
from src.calculations.market import calculate_leave_one_out_consensus, calculate_no_vig_probabilities
from src.calculations.odds_math import expected_value, implied_probability, is_positive_ev
from src.models import AnalysisStatus, ArchitectureType, BettingAnalysis, MarketType
from src.providers.controlled import ControlledOddsProvider
from src.tools.sportsbook_tools import SportsbookTools

AGENTS_DIR = Path(__file__).resolve().parent.parent / "src" / "agents"


class ScriptedLLMClient:
    """Fake ToolCallingLLMClient: replays a fixed list of ToolCallTurn
    objects, one per create_turn() call, in order."""

    model = "fake-model"
    effort = "low"

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls_made = 0
        self.received_messages: list[list[dict]] = []

    def create_turn(self, *, system_prompt, messages, tools):
        self.received_messages.append(messages)
        turn = self.turns[self.calls_made]
        self.calls_made += 1
        return turn


class RaisingLLMClient:
    model = "fake-model"
    effort = "low"

    def create_turn(self, *, system_prompt, messages, tools):
        raise ValueError("malformed tool-call turn")


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
        query="What is the best current price on the Lakers moneyline, and is it a positive EV bet?",
    )
    defaults.update(overrides)
    return AgentRequest(**defaults)


LAKERS_ODDS = {"DraftKings": 120, "FanDuel": 125, "BetMGM": 115, "Caesars": 122}
CELTICS_ODDS = {"DraftKings": -140, "FanDuel": -145, "BetMGM": -135, "Caesars": -142}


def _get_odds_turn(call_id, outcome):
    return ToolCallTurn(
        stop_reason="tool_use",
        text=None,
        tool_uses=[
            tu(
                call_id,
                "get_odds",
                {"game_id": "G-2026-001", "market_type": "moneyline", "selected_outcome": outcome},
            )
        ],
    )


END_TURN = ToolCallTurn(stop_reason="end_turn", text="Gathered current odds.", tool_uses=[])


# ---------------------------------------------------------------------------
# Common agent contract / LLM abstraction reuse
# ---------------------------------------------------------------------------


def test_tool_calling_agent_implements_agent_contract(tools):
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient([END_TURN]))
    assert isinstance(agent, Agent)
    assert agent.architecture == ArchitectureType.TOOL


def test_tool_calling_agent_defaults_to_anthropic_llm_client_reusing_shared_config(tools):
    from src.agents.llm_client import DEFAULT_MODEL, DEFAULT_TOOL_MAX_TOKENS, AnthropicLLMClient

    try:
        agent = ToolCallingAgent(tools)
    except Exception:
        pytest.skip("anthropic client construction requires credentials in this environment")
    assert isinstance(agent.llm_client, AnthropicLLMClient)
    assert agent.llm_client.model == DEFAULT_MODEL
    assert agent.llm_client.max_tokens == DEFAULT_TOOL_MAX_TOKENS


# ---------------------------------------------------------------------------
# Valid tool calling / multi-tool / argument forwarding
# ---------------------------------------------------------------------------


def test_single_tool_call_produces_valid_analysis(tools):
    turns = [_get_odds_turn("t1", "Los Angeles Lakers"), END_TURN]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    assert isinstance(analysis, BettingAnalysis)
    assert analysis.architecture == ArchitectureType.TOOL
    assert analysis.best_sportsbook == "FanDuel"
    assert analysis.best_odds == 125


def test_multiple_tool_calls_in_sequence_are_traced_in_order(tools):
    turns = [
        ToolCallTurn(
            stop_reason="tool_use", text=None,
            tool_uses=[tu("t1", "get_game", {"game_id": "G-2026-001"})],
        ),
        _get_odds_turn("t2", "Los Angeles Lakers"),
        _get_odds_turn("t3", "Boston Celtics"),
        END_TURN,
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    assert analysis.status == AnalysisStatus.OK
    assert agent.last_trace.tool_call_order == ["get_game", "get_odds", "get_odds"]
    assert agent.last_trace.iterations_used == 4


def test_argument_forwarding_reaches_the_real_provider(tools):
    # get_sportsbook_odds forwards its exact arguments through
    # SportsbookTools -> ControlledOddsProvider; confirm the real record
    # comes back, not a substitute.
    turns = [
        ToolCallTurn(
            stop_reason="tool_use", text=None,
            tool_uses=[tu("t1", "get_sportsbook_odds", {
                "game_id": "G-2026-001", "sportsbook": "DraftKings",
                "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
            })],
        ),
        END_TURN,
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    assert analysis.best_sportsbook == "DraftKings"
    assert analysis.best_odds == 120
    record = agent.last_trace.tool_calls[0]
    assert record.arguments == {
        "game_id": "G-2026-001", "sportsbook": "DraftKings",
        "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
    }


def test_structured_tool_result_preserved_in_trace(tools):
    turns = [_get_odds_turn("t1", "Los Angeles Lakers"), END_TURN]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    agent.analyze(_request())
    record = agent.last_trace.tool_calls[0]
    assert record.success is True
    assert "4 sportsbook price(s)" in record.result_summary


def test_current_data_semantics_never_expose_stale_odds(tools):
    # G-2026-009 has a stale DraftKings snapshot in the RAG corpus, but
    # the ControlledOddsProvider path (this architecture) must only ever
    # surface the current record.
    request = _request(
        scenario_id="S009",
        game_id="G-2026-009",
        selected_outcome="Minnesota Timberwolves",
        query="What is DraftKings offering on the Timberwolves moneyline right now?",
    )
    turns = [
        ToolCallTurn(
            stop_reason="tool_use", text=None,
            tool_uses=[tu("t1", "get_sportsbook_odds", {
                "game_id": "G-2026-009", "sportsbook": "DraftKings",
                "market_type": "moneyline", "selected_outcome": "Minnesota Timberwolves",
            })],
        ),
        END_TURN,
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(request)
    assert analysis.best_odds == 135  # the current record, never the stale +120 snapshot


# ---------------------------------------------------------------------------
# Best-line / quant integration
# ---------------------------------------------------------------------------


def test_best_line_integration_matches_deterministic_tool(tools):
    turns = [_get_odds_turn("t1", "Los Angeles Lakers"), END_TURN]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    expected = tools.find_best_line("G-2026-001", MarketType.MONEYLINE, "Los Angeles Lakers")
    assert analysis.best_odds == expected.best_odds
    assert analysis.best_sportsbooks == expected.sportsbooks


def test_full_quant_pipeline_matches_shared_calculations_engine(tools):
    turns = [_get_odds_turn("t1", "Los Angeles Lakers"), _get_odds_turn("t2", "Boston Celtics"), END_TURN]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())

    assert analysis.status == AnalysisStatus.OK
    no_vig = {
        book: calculate_no_vig_probabilities([LAKERS_ODDS[book], CELTICS_ODDS[book]])[0]
        for book in LAKERS_ODDS
    }
    expected_reference = calculate_leave_one_out_consensus(list(no_vig.items()), "FanDuel")
    expected_ev = expected_value(125, expected_reference)

    assert analysis.market_reference_probability == pytest.approx(expected_reference)
    assert analysis.expected_value == pytest.approx(expected_ev)
    assert analysis.positive_ev == is_positive_ev(125, expected_reference)
    assert analysis.implied_probability == pytest.approx(implied_probability(125))


def test_insufficient_quant_evidence_when_only_one_side_gathered(tools):
    turns = [_get_odds_turn("t1", "Los Angeles Lakers"), END_TURN]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    assert analysis.status == AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE
    assert analysis.expected_value is None
    assert analysis.positive_ev is None
    assert analysis.market_reference_probability is None
    assert analysis.best_odds == 125  # best line still honestly derivable


def test_leave_one_out_excludes_target_sportsbook(tools):
    turns = [_get_odds_turn("t1", "Los Angeles Lakers"), _get_odds_turn("t2", "Boston Celtics"), END_TURN]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    no_vig_fanduel = calculate_no_vig_probabilities([125, -145])[0]
    # If FanDuel's own fair probability were included in its reference
    # consensus, the reference would be much closer to no_vig_fanduel;
    # confirm it's derived from the OTHER three books instead.
    other_books_only = calculate_leave_one_out_consensus(
        [
            ("DraftKings", calculate_no_vig_probabilities([120, -140])[0]),
            ("BetMGM", calculate_no_vig_probabilities([115, -135])[0]),
            ("Caesars", calculate_no_vig_probabilities([122, -142])[0]),
            ("FanDuel", no_vig_fanduel),
        ],
        "FanDuel",
    )
    assert analysis.market_reference_probability == pytest.approx(other_books_only)


# ---------------------------------------------------------------------------
# Invalid sportsbook / no fabrication
# ---------------------------------------------------------------------------


def test_invalid_sportsbook_raises_explicit_failure_no_fabrication(tools):
    turns = [
        ToolCallTurn(
            stop_reason="tool_use", text=None,
            tool_uses=[tu("t1", "get_sportsbook_odds", {
                "game_id": "G-2026-001", "sportsbook": "FakeBook",
                "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
            })],
        ),
        END_TURN,
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    with pytest.raises(ToolAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    trace = exc_info.value.trace
    assert trace.validation_status == "no_valid_prices"
    assert trace.tool_calls[0].success is False
    assert "FakeBook" in trace.tool_calls[0].error


def test_hallucinated_final_prose_does_not_override_tool_odds(tools):
    turns = [
        ToolCallTurn(
            stop_reason="tool_use", text=None,
            tool_uses=[tu("t1", "get_sportsbook_odds", {
                "game_id": "G-2026-001", "sportsbook": "DraftKings",
                "market_type": "moneyline", "selected_outcome": "Los Angeles Lakers",
            })],
        ),
        ToolCallTurn(stop_reason="end_turn", text="DraftKings is actually offering +180.", tool_uses=[]),
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    assert analysis.best_odds == 120  # real tool value, never the hallucinated +180 prose


# ---------------------------------------------------------------------------
# Loop bound / redundant-call detection
# ---------------------------------------------------------------------------


def test_loop_bound_terminates_and_fails_explicitly(tools):
    turns = [
        ToolCallTurn(stop_reason="tool_use", text=None, tool_uses=[tu(f"t{i}", "get_games", {})])
        for i in range(1, 10)
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns), max_iterations=6)
    with pytest.raises(ToolAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    trace = exc_info.value.trace
    assert trace.validation_status == "loop_limit_exceeded"
    assert trace.iterations_used == 6


def test_default_max_iterations_is_bounded_and_reasonable():
    assert MAX_TOOL_ITERATIONS > 0
    assert MAX_TOOL_ITERATIONS <= 10


def test_redundant_tool_call_is_traceable_and_detected(tools):
    turns = [
        _get_odds_turn("t1", "Los Angeles Lakers"),
        _get_odds_turn("t2", "Los Angeles Lakers"),  # identical to t1
        END_TURN,
    ]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    agent.analyze(_request())
    trace = agent.last_trace
    assert trace.redundant_call_count == 1
    assert [call.is_redundant for call in trace.tool_calls] == [False, True]
    # Both calls remain fully traceable — redundancy is flagged, not hidden.
    assert len(trace.tool_calls) == 2


def test_non_identical_calls_are_not_flagged_redundant(tools):
    turns = [_get_odds_turn("t1", "Los Angeles Lakers"), _get_odds_turn("t2", "Boston Celtics"), END_TURN]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    agent.analyze(_request())
    assert agent.last_trace.redundant_call_count == 0


# ---------------------------------------------------------------------------
# Malformed LLM output / errors
# ---------------------------------------------------------------------------


def test_malformed_llm_turn_handled_gracefully_not_a_crash(tools):
    agent = ToolCallingAgent(tools, llm_client=RaisingLLMClient())
    with pytest.raises(ToolAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    trace = exc_info.value.trace
    assert trace.validation_status == "no_valid_prices"
    assert len(trace.errors) == 1
    assert "malformed tool-call turn" in trace.errors[0]


# ---------------------------------------------------------------------------
# Trace / latency
# ---------------------------------------------------------------------------


def test_trace_records_all_required_fields(tools):
    turns = [_get_odds_turn("t1", "Los Angeles Lakers"), _get_odds_turn("t2", "Boston Celtics"), END_TURN]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    agent.analyze(_request())
    trace = agent.last_trace
    assert isinstance(trace, ToolAgentTrace)
    assert trace.architecture == "tool"
    assert trace.quant_status == "ok"
    for latency in (
        trace.llm_decision_latency_seconds,
        trace.tool_execution_latency_seconds,
        trace.quant_latency_seconds,
        trace.total_latency_seconds,
    ):
        assert latency >= 0.0
    assert trace.total_latency_seconds >= trace.llm_decision_latency_seconds
    assert trace.total_latency_seconds >= trace.tool_execution_latency_seconds


# ---------------------------------------------------------------------------
# Final BettingAnalysis validation / architecture labeling
# ---------------------------------------------------------------------------


def test_final_analysis_is_valid_betting_analysis_labeled_tool(tools):
    turns = [_get_odds_turn("t1", "Los Angeles Lakers"), _get_odds_turn("t2", "Boston Celtics"), END_TURN]
    agent = ToolCallingAgent(tools, llm_client=ScriptedLLMClient(turns))
    analysis = agent.analyze(_request())
    BettingAnalysis.model_validate(analysis.model_dump())  # round-trips cleanly
    assert analysis.architecture == ArchitectureType.TOOL


# ---------------------------------------------------------------------------
# Architecture isolation (docs/EXPERIMENT_RULES.md, "Tool-Calling-Only Boundary")
# ---------------------------------------------------------------------------


FORBIDDEN_MODULE_PREFIXES = ("src.rag",)
FORBIDDEN_SOURCE_SUBSTRINGS = (
    "ground_truth.json",
    "quant_ground_truth.json",
    "corpus.jsonl",
    "vector_index",
    "retriever",
    "RagEvidencePipeline",
)


@pytest.mark.parametrize("module_filename", ["tool_agent.py", "tool_schemas.py"])
def test_no_forbidden_imports(module_filename):
    source_path = AGENTS_DIR / module_filename
    tree = ast.parse(source_path.read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    for module_name in imported_modules:
        for forbidden in FORBIDDEN_MODULE_PREFIXES:
            assert not module_name.startswith(forbidden), (
                f"{module_filename} imports forbidden module {module_name!r}"
            )


def _non_docstring_string_literals(tree: ast.AST) -> list[str]:
    """All string constants in the module except docstrings — so a
    docstring's prose *warning about* a forbidden path/name doesn't
    itself trip the check; only string literals used as actual code
    count."""
    docstring_nodes: set[int] = set()
    doc_owners = [tree] + [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for owner in doc_owners:
        body = getattr(owner, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstring_nodes.add(id(body[0].value))

    literals = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            literals.append(node.value)
    return literals


@pytest.mark.parametrize("module_filename", ["tool_agent.py", "tool_schemas.py"])
def test_no_forbidden_source_references(module_filename):
    source_path = AGENTS_DIR / module_filename
    tree = ast.parse(source_path.read_text())
    literals = _non_docstring_string_literals(tree)
    for literal in literals:
        for forbidden in FORBIDDEN_SOURCE_SUBSTRINGS:
            assert forbidden not in literal, (
                f"{module_filename} references forbidden identifier/path "
                f"{forbidden!r} in code (not just documentation): {literal!r}"
            )


def test_tool_agent_never_reimplements_shared_quant_formulas():
    source_path = AGENTS_DIR / "tool_agent.py"
    tree = ast.parse(source_path.read_text())
    defined_function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden_names = {
        "implied_probability",
        "expected_value",
        "is_positive_ev",
        "best_odds",
        "compare_american_odds",
        "calculate_no_vig_probabilities",
        "calculate_leave_one_out_consensus",
        "calculate_probability_edge",
    }
    assert defined_function_names.isdisjoint(forbidden_names)


def test_tool_agent_never_duplicates_sportsbook_tools_implementation():
    # tool_agent.py must call into SportsbookTools, never reimplement its
    # provider-lookup logic locally.
    source_path = AGENTS_DIR / "tool_agent.py"
    source = source_path.read_text()
    assert "class ControlledOddsProvider" not in source
    assert "json.load" not in source  # no direct dataset file reads
