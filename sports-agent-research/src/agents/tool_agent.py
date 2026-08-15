"""Tool-calling-only agent (Milestone 9A) — the second concrete
architecture.

    AgentRequest
         |
         v
    Bounded multi-turn tool-calling loop     (this module, via
         |                                    src/agents/llm_client.py's
         |                                    ToolCallingLLMClient +
         |                                    src/agents/tool_schemas.py)
         v
    SportsbookTools -> OddsProvider -> ControlledOddsProvider
         |             (src/tools/sportsbook_tools.py — reused verbatim,
         |              never reimplemented; the ONLY path to sportsbook
         |              information for this architecture)
         v
    Structured market state                  (Pydantic objects actually
         |                                    returned by tool calls —
         |                                    never text/prose the LLM
         |                                    wrote about them)
         v
    Shared deterministic quant engine
         |                                    (src/calculations/market.py
         |                                    + odds_math.py — identical
         |                                    to the RAG-only agent's,
         |                                    never reimplemented here)
         v
    BettingAnalysis (architecture=tool)

Architecture isolation (docs/EXPERIMENT_RULES.md, "Tool-Calling-Only
Boundary"): this module must never import src.rag or read
data/ground_truth.json / data/quant_ground_truth.json. The only path to
sportsbook information is SportsbookTools -> OddsProvider, injected by
the caller (see ToolCallingAgent.__init__) — this module never imports a
concrete provider or a raw dataset path itself. Enforced by an AST-based
test in tests/test_tool_agent.py.

Anti-hallucination design: unlike the RAG-only agent (which validates an
LLM's claims against retrieved-document provenance after the fact), this
agent structurally cannot accept an LLM-invented sportsbook or odds
value in its final BettingAnalysis — the quant pipeline below reads
exclusively from `market_state`, which is populated only by
_fold_into_market_state() from the actual Pydantic objects
SportsbookTools methods returned, keyed by the arguments that were
actually validated and used for that call. The LLM's final text (and any
numbers it contains) is never parsed for data — see Step 15/19 in
milestones/current.md.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field

from src.agents.base import Agent, AgentRequest
from src.agents.llm_client import (
    DEFAULT_MODEL,
    DEFAULT_TOOL_MAX_TOKENS,
    AnthropicLLMClient,
    ToolCallingLLMClient,
    ToolCallTurn,
    ToolUseBlock,
)
from src.agents.tool_schemas import TOOL_SCHEMAS, execute_tool
from src.calculations.market import (
    MIN_CONSENSUS_BOOKS,
    calculate_leave_one_out_consensus,
    calculate_no_vig_probabilities,
    calculate_probability_edge,
)
from src.calculations.odds_math import (
    best_odds,
    compare_american_odds,
    expected_value,
    implied_probability,
    is_positive_ev,
)
from src.models import AnalysisStatus, ArchitectureType, BettingAnalysis, MarketType
from src.tools.sportsbook_tools import SportsbookTools

MIN_QUOTING_BOOKS = MIN_CONSENSUS_BOOKS + 1  # target + MIN_CONSENSUS_BOOKS comparison books

# Bounds the multi-turn tool-calling loop (milestones/current.md, Step 13).
# Chosen to comfortably fit get_game + get_odds(primary) + get_odds(opposing)
# + a little headroom, while still failing explicitly on a runaway model.
MAX_TOOL_ITERATIONS = 6

TOOL_AGENT_SYSTEM_PROMPT = """You are a research assistant for a controlled sports-betting study. You have access to tools that return current structured sportsbook odds. You are NOT a betting calculator and you must not act like one.

Rules — follow all of them exactly:
- Use the provided tools to gather sportsbook odds. Never state a sportsbook name or an odds value that a tool did not actually return to you.
- If a tool call fails (e.g. an unknown sportsbook, game, market, or outcome), accept the failure — do not guess a substitute value, and do not silently retry a different, unrequested sportsbook while pretending it is the one that failed.
- To support a full analysis, call get_odds (or get_sportsbook_odds) for BOTH the requested outcome and its opposing outcome in the same market whenever you can determine the opposing outcome's name (call get_game first if you need the team names).
- Do NOT calculate implied probability, no-vig probability, market consensus, expected value, or any other betting math yourself. A separate deterministic system performs every calculation from the structured tool results you gather — your text response is never used for any numeric claim.
- Do NOT decide which sportsbook is "best" and do NOT state a final conclusion, recommendation, or verdict. Once you have gathered enough current odds to answer the research query, stop calling tools.
- Do not call the same tool with the same arguments more than once unless a prior call to it failed.
"""


class ToolCallRecord(BaseModel):
    """One traced tool invocation (milestones/current.md, Step 11)."""

    call_sequence: int
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    is_redundant: bool
    result_summary: str
    error: str | None = None
    latency_seconds: float


class ToolAgentTrace(BaseModel):
    """Deterministic execution record for one tool-calling agent run
    (milestones/current.md, Step 12). Never used as an inference input;
    no secrets (API keys, tokens) are recorded here.

    `effort` is recorded instead of `temperature`: DEFAULT_MODEL rejects
    temperature entirely (see src/agents/llm_client.py's module
    docstring) — effort is the closest determinism lever this model
    family actually accepts.
    """

    architecture: str = "tool"
    model: str
    effort: str
    query: str
    iterations_used: int
    tool_calls: list[ToolCallRecord]
    tool_call_order: list[str]
    redundant_call_count: int
    validation_status: str  # "ok" | "no_valid_prices" | "loop_limit_exceeded"
    quant_status: str  # "ok" | "insufficient_quant_evidence" | "not_attempted"
    llm_decision_latency_seconds: float
    tool_execution_latency_seconds: float
    quant_latency_seconds: float
    total_latency_seconds: float
    errors: list[str]


class ToolAnalysisIncomplete(Exception):
    """Raised instead of returning a BettingAnalysis when no current price
    could be gathered at all, or when the tool-call loop hit
    MAX_TOOL_ITERATIONS without the model naturally finishing.
    BettingAnalysis.best_sportsbook is a required field — there is no
    honest value to put there in either case, so this is a typed,
    catchable failure rather than either a bare crash or a fabricated
    result (mirrors RagAnalysisIncomplete in src/agents/rag_agent.py).

    Carries the full ToolAgentTrace so a caller (a future experiment
    runner) can log exactly which tools were called, in what order, with
    what results, and why no analysis could be produced.
    """

    def __init__(self, trace: ToolAgentTrace) -> None:
        self.trace = trace
        super().__init__(
            f"no sportsbook price could be gathered for query={trace.query!r} "
            f"(validation_status={trace.validation_status!r})"
        )


# One outcome's sportsbook -> american_odds map, keyed by (game_id, market_type).
_MarketState = dict[tuple[str, MarketType], dict[str, dict[str, int]]]


class ToolCallingAgent(Agent):
    """Tool-calling-only architecture: SportsbookTools + shared quant
    engine only. Never imports src.rag and never reads
    data/ground_truth.json / data/quant_ground_truth.json — see module
    docstring.
    """

    architecture = ArchitectureType.TOOL

    def __init__(
        self,
        tools: SportsbookTools,
        llm_client: ToolCallingLLMClient | None = None,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        self.tools = tools
        self.llm_client = llm_client or AnthropicLLMClient(max_tokens=DEFAULT_TOOL_MAX_TOKENS)
        self.max_iterations = max_iterations
        self.last_trace: ToolAgentTrace | None = None

    def analyze(self, request: AgentRequest) -> BettingAnalysis:
        run_start = time.monotonic()
        errors: list[str] = []
        tool_call_records: list[ToolCallRecord] = []
        tool_call_order: list[str] = []
        seen_signatures: set[str] = set()
        market_state: _MarketState = {}

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self._render_user_prompt(request)}
        ]

        llm_latency_total = 0.0
        tool_latency_total = 0.0
        call_sequence = 0
        iterations_used = 0
        loop_limit_exceeded = False

        for iteration in range(1, self.max_iterations + 1):
            iterations_used = iteration
            step_start = time.monotonic()
            try:
                turn = self.llm_client.create_turn(
                    system_prompt=TOOL_AGENT_SYSTEM_PROMPT,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                )
            except Exception as exc:  # malformed/invalid LLM turn, network error, etc.
                errors.append(f"LLM tool-call turn failed: {exc!r}")
                llm_latency_total += time.monotonic() - step_start
                break
            llm_latency_total += time.monotonic() - step_start

            messages.append({"role": "assistant", "content": _turn_to_content_blocks(turn)})

            if not turn.tool_uses:
                break

            tool_result_blocks: list[dict[str, Any]] = []
            for tool_use in turn.tool_uses:
                call_sequence += 1
                record, result_block, latency = self._execute_tool_call(
                    tool_use, call_sequence, seen_signatures, market_state
                )
                tool_call_records.append(record)
                tool_call_order.append(record.tool_name)
                tool_result_blocks.append(result_block)
                tool_latency_total += latency
            messages.append({"role": "user", "content": tool_result_blocks})
        else:
            # The for loop ran every allowed iteration without a natural
            # break (the model still wanted to call tools on the last one).
            loop_limit_exceeded = True

        quant_start = time.monotonic()
        analysis: BettingAnalysis | None = None
        quant_status = "not_attempted"

        primary_key = (request.game_id, request.market_type)
        primary_prices = market_state.get(primary_key, {}).get(request.selected_outcome, {})

        if loop_limit_exceeded:
            validation_status = "loop_limit_exceeded"
        elif not primary_prices:
            validation_status = "no_valid_prices"
        else:
            validation_status = "ok"
            analysis, quant_status = self._run_quant_pipeline(request, market_state)
        quant_latency = time.monotonic() - quant_start

        total_latency = time.monotonic() - run_start
        redundant_call_count = sum(1 for record in tool_call_records if record.is_redundant)

        self.last_trace = ToolAgentTrace(
            model=getattr(self.llm_client, "model", DEFAULT_MODEL),
            effort=getattr(self.llm_client, "effort", "unknown"),
            query=request.query,
            iterations_used=iterations_used,
            tool_calls=tool_call_records,
            tool_call_order=tool_call_order,
            redundant_call_count=redundant_call_count,
            validation_status=validation_status,
            quant_status=quant_status,
            llm_decision_latency_seconds=llm_latency_total,
            tool_execution_latency_seconds=tool_latency_total,
            quant_latency_seconds=quant_latency,
            total_latency_seconds=total_latency,
            errors=errors,
        )

        if analysis is None:
            raise ToolAnalysisIncomplete(self.last_trace)
        return analysis

    @staticmethod
    def _render_user_prompt(request: AgentRequest) -> str:
        return (
            f"Game ID: {request.game_id}\n"
            f"Market: {request.market_type.value}\n"
            f"Selected outcome: {request.selected_outcome}\n"
            f"Research query: {request.query}\n"
        )

    def _execute_tool_call(
        self,
        tool_use: ToolUseBlock,
        call_sequence: int,
        seen_signatures: set[str],
        market_state: _MarketState,
    ) -> tuple[ToolCallRecord, dict[str, Any], float]:
        signature = f"{tool_use.name}:{json.dumps(tool_use.input, sort_keys=True)}"
        is_redundant = signature in seen_signatures
        seen_signatures.add(signature)

        start = time.monotonic()
        try:
            structured_result, raw_result = execute_tool(self.tools, tool_use.name, tool_use.input)
            _fold_into_market_state(market_state, tool_use.name, tool_use.input, raw_result)
            success = True
            error: str | None = None
            result_summary = _summarize_result(tool_use.name, structured_result)
            result_content: Any = structured_result
        except Exception as exc:
            success = False
            error = str(exc)
            result_summary = f"error: {error}"
            result_content = error
        latency = time.monotonic() - start

        record = ToolCallRecord(
            call_sequence=call_sequence,
            tool_name=tool_use.name,
            arguments=tool_use.input,
            success=success,
            is_redundant=is_redundant,
            result_summary=result_summary,
            error=error,
            latency_seconds=latency,
        )
        result_block = {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": json.dumps(result_content) if success else str(result_content),
            "is_error": not success,
        }
        return record, result_block, latency

    def _run_quant_pipeline(
        self, request: AgentRequest, market_state: _MarketState
    ) -> tuple[BettingAnalysis, str]:
        """Determine the best line from gathered current prices (always
        possible once market_state has >=1 price for the requested
        outcome), then attempt the market-consensus EV chain (requires
        >=MIN_QUOTING_BOOKS sportsbooks quoting both this outcome and
        the single opposing outcome). Every calculation here is a direct
        call into src/calculations/ — nothing is recomputed.
        """
        outcomes = market_state[(request.game_id, request.market_type)]
        primary_prices = outcomes[request.selected_outcome]

        sportsbooks_considered = sorted(primary_prices)
        odds_values = [primary_prices[book] for book in sportsbooks_considered]
        winning_odds = best_odds(odds_values)
        best_sportsbooks = sorted(
            book
            for book in sportsbooks_considered
            if compare_american_odds(primary_prices[book], winning_odds) == 0
        )
        winning_implied_probability = implied_probability(winning_odds)

        other_outcomes = [outcome for outcome in outcomes if outcome != request.selected_outcome]

        market_reference_probability: float | None = None
        probability_edge: float | None = None
        computed_expected_value: float | None = None
        computed_positive_ev: bool | None = None
        status = AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE
        quant_status = "insufficient_quant_evidence"

        if len(other_outcomes) == 1:
            opposing_prices = outcomes[other_outcomes[0]]
            complete_books = sorted(book for book in sportsbooks_considered if book in opposing_prices)

            if len(complete_books) >= MIN_QUOTING_BOOKS:
                no_vig_by_book: dict[str, float] = {}
                for book in complete_books:
                    fair = calculate_no_vig_probabilities(
                        [primary_prices[book], opposing_prices[book]]
                    )
                    no_vig_by_book[book] = fair[0]

                # Deterministic (alphabetical) choice among complete-pair
                # books tied for the winning price, matching the
                # tie-break convention used across the project.
                tied_complete_books = sorted(
                    book for book in complete_books if primary_prices[book] == winning_odds
                )
                if tied_complete_books:
                    target = tied_complete_books[0]
                    sportsbook_probabilities = [
                        (book, no_vig_by_book[book]) for book in complete_books
                    ]
                    reference_probability = calculate_leave_one_out_consensus(
                        sportsbook_probabilities, target
                    )
                    book_implied = implied_probability(primary_prices[target])
                    market_reference_probability = reference_probability
                    probability_edge = calculate_probability_edge(reference_probability, book_implied)
                    computed_expected_value = expected_value(primary_prices[target], reference_probability)
                    computed_positive_ev = is_positive_ev(primary_prices[target], reference_probability)
                    status = AnalysisStatus.OK
                    quant_status = "ok"

        reasoning_summary = self._build_reasoning_summary(
            best_sportsbooks=best_sportsbooks,
            winning_odds=winning_odds,
            status=status,
            sportsbooks_considered=sportsbooks_considered,
        )

        analysis = BettingAnalysis(
            scenario_id=request.scenario_id,
            game_id=request.game_id,
            market=request.market_type,
            selected_outcome=request.selected_outcome,
            best_sportsbook=best_sportsbooks[0],
            best_sportsbooks=best_sportsbooks,
            best_odds=winning_odds,
            implied_probability=winning_implied_probability,
            market_reference_probability=market_reference_probability,
            probability_edge=probability_edge,
            expected_value=computed_expected_value,
            positive_ev=computed_positive_ev,
            status=status,
            sportsbooks_considered=sportsbooks_considered,
            reasoning_summary=reasoning_summary,
            architecture=ArchitectureType.TOOL,
        )
        return analysis, quant_status

    @staticmethod
    def _build_reasoning_summary(
        *,
        best_sportsbooks: list[str],
        winning_odds: int,
        status: AnalysisStatus,
        sportsbooks_considered: list[str],
    ) -> str:
        books = ", ".join(best_sportsbooks)
        base = (
            f"From structured sportsbook tool calls, {books} offered the best "
            f"current price ({winning_odds:+d}) among {len(sportsbooks_considered)} "
            f"sportsbook(s) considered."
        )
        if status == AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE:
            base += (
                " Insufficient two-sided current data was gathered to derive a "
                "market reference probability, so no expected-value verdict is reported."
            )
        return base


def _turn_to_content_blocks(turn: ToolCallTurn) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if turn.text:
        blocks.append({"type": "text", "text": turn.text})
    for tool_use in turn.tool_uses:
        blocks.append(
            {"type": "tool_use", "id": tool_use.id, "name": tool_use.name, "input": tool_use.input}
        )
    return blocks


def _fold_into_market_state(
    market_state: _MarketState,
    tool_name: str,
    arguments: dict[str, Any],
    raw_result: Any,
) -> None:
    """Record only what a tool call actually, structurally returned —
    never anything derived from LLM prose. See module docstring
    ("Anti-hallucination design")."""
    if tool_name == "get_odds":
        key = (arguments["game_id"], MarketType(arguments["market_type"]))
        outcome_prices = market_state.setdefault(key, {}).setdefault(arguments["selected_outcome"], {})
        for odds in raw_result:
            outcome_prices[odds.sportsbook] = odds.american_odds
    elif tool_name == "get_sportsbook_odds":
        key = (arguments["game_id"], MarketType(arguments["market_type"]))
        outcome_prices = market_state.setdefault(key, {}).setdefault(arguments["selected_outcome"], {})
        outcome_prices[raw_result.sportsbook] = raw_result.american_odds
    elif tool_name == "find_best_line":
        key = (arguments["game_id"], MarketType(arguments["market_type"]))
        outcome_prices = market_state.setdefault(key, {}).setdefault(arguments["selected_outcome"], {})
        for sportsbook in raw_result.sportsbooks:
            outcome_prices[sportsbook] = raw_result.best_odds
    # get_games / get_game: no odds to fold into market state.


def _summarize_result(tool_name: str, result: Any) -> str:
    if tool_name == "get_games":
        return f"{len(result)} game(s)"
    if tool_name == "get_game":
        return f"game_id={result['game_id']}"
    if tool_name == "get_odds":
        return f"{len(result)} sportsbook price(s)"
    if tool_name == "get_sportsbook_odds":
        return f"{result['sportsbook']} {result['american_odds']:+d}"
    if tool_name == "find_best_line":
        return f"best_odds={result['best_odds']:+d} sportsbooks={result['sportsbooks']}"
    return str(result)
