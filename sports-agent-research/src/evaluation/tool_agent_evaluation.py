"""End-to-end evaluation harness for the tool-calling agent (Milestone
9B): drives ToolCallingAgent (Milestone 9A) across a representative
controlled-benchmark subset and compares its output against deterministic
ground truth, entirely outside the agent:

    AgentRequest
        |
        v
    ToolCallingAgent          (src/agents/tool_agent.py, unmodified)
        |
        v
    BettingAnalysis
        |
        v
    Evaluator (this module)   <-- ground truth enters HERE only
        |
        v
    ToolAgentEvaluationResult

Ground-truth isolation (docs/EXPERIMENT_RULES.md, "Ground Truth"): this
module is the only place in the tool-calling evaluation path that reads
GroundTruth/QuantGroundTruth. Nothing here ever writes an expected value
into an AgentRequest, a tool prompt, a tool result, an LLM message, or a
quant input — every comparison happens strictly after
ToolCallingAgent.analyze() has already returned. See
tests/test_tool_agent_evaluation.py for the automated checks of this
boundary (an AST-based scan confirming src/agents/tool_agent.py and
src/agents/tool_schemas.py never import src.evaluation or read
data/ground_truth.json / data/quant_ground_truth.json).

Run as a script for a deterministic, mock-LLM, CI-free evaluation report:

    python -m src.evaluation.tool_agent_evaluation
    python -m src.evaluation.tool_agent_evaluation --json

DeterministicToolPolicyLLMClient (below) is the fake LLM used by both
this script's default mode and the automated tests
(tests/test_tool_agent_evaluation.py, tests/test_tool_agent_e2e.py). It
is a *policy*, not an answer key: every tool call it issues is derived
from the request's own stated game_id/market_type/selected_outcome (the
same information any real LLM would read from the user prompt) and from
the actual structured content of prior tool_result messages in the
conversation — never from GroundTruth/QuantGroundTruth. The authoritative
numbers in the final BettingAnalysis still come exclusively from
SportsbookTools output + the shared quant engine, exactly as in
Milestone 9A.

Metric formulas (best-line/best-odds/EV classification/EV error/market-
reference error/freshness/completeness) are NOT defined here — this
module calls into src/evaluation/metrics.py (Milestone 11's unified
evaluation framework), the single shared implementation used identically
by the RAG-only, tool-calling, and hybrid evaluators. Only this module's
own architecture-specific concerns remain local: tool-call efficiency
counting and the tool-output re-query hallucination check (`_detect_
hallucination`), since "traceable to actual tool output" is a genuinely
different verification procedure than RAG's or hybrid's.
"""

from __future__ import annotations

import json
import re
import sys
from enum import Enum

from pydantic import BaseModel

from src.agents.base import AgentRequest
from src.agents.llm_client import ToolCallTurn, ToolUseBlock
from src.agents.tool_agent import ToolAnalysisIncomplete, ToolAgentTrace, ToolCallingAgent
from src.evaluation import metrics
from src.evaluation.dataset import load_historical_odds_records, load_scenario_definitions_by_id
from src.evaluation.ground_truth import generate_all_ground_truth
from src.evaluation.quant_ground_truth import generate_all_quant_ground_truth
from src.models import AnalysisStatus, ArchitectureType, BettingAnalysis, GroundTruth, MarketType, QuantGroundTruth
from src.providers.controlled import ControlledOddsProvider
from src.tools.sportsbook_tools import SportsbookTools

# A representative controlled-benchmark subset (milestones/current.md,
# Milestone 9B Step 15): S001 positive odds/negative EV, S002 negative
# odds, S003 mixed-sign odds, S004 large positive odds, S005 large
# negative odds, S006 positive odds, S007 best-line TIE, S008 missing
# sportsbook data (quant-evaluable, positive EV), S009 current/stale
# freshness case (quant-evaluable, positive EV), S012 spread, S013 total.
DEFAULT_SCENARIO_IDS = [
    "S001", "S002", "S003", "S004", "S005",
    "S006", "S007", "S008", "S009", "S012", "S013",
]

# Scenarios with a game_id present in data/historical_odds.json — the
# tool-only path must reflect data/current_odds.json for these, never the
# stale snapshot (see Step 10).
_FRESHNESS_SCENARIO_IDS = frozenset({"S009", "S010", "S011"})


class ExecutionStatus(str, Enum):
    """Explicit failure/outcome classification (Milestone 9B, Step 21) —
    never collapsed into a single generic "error"."""

    SUCCESS = "success"
    QUANT_INSUFFICIENT_DATA = "quant_insufficient_data"
    TOOL_ARGUMENT_ERROR = "tool_argument_error"
    TOOL_DATA_MISSING = "tool_data_missing"
    TOOL_LOOP_LIMIT = "tool_loop_limit"
    LLM_OUTPUT_INVALID = "llm_output_invalid"
    QUANT_VALIDATION_ERROR = "quant_validation_error"
    FINAL_OUTPUT_INVALID = "final_output_invalid"


_SUCCESSFUL_STATUSES = frozenset({ExecutionStatus.SUCCESS, ExecutionStatus.QUANT_INSUFFICIENT_DATA})


class ToolAgentEvaluationResult(BaseModel):
    """One scenario's evaluation outcome (milestones/current.md,
    Milestone 9B Step 4)."""

    scenario_id: str
    execution_status: ExecutionStatus
    quant_evaluable: bool

    predicted_best_sportsbooks: list[str]
    expected_best_sportsbooks: list[str]
    best_line_correct: bool | None

    predicted_best_odds: int | None
    expected_best_odds: int
    best_odds_correct: bool | None

    predicted_positive_ev: bool | None
    expected_positive_ev: bool | None
    ev_classification_correct: bool | None

    predicted_ev: float | None
    expected_ev: float | None
    ev_absolute_error: float | None

    predicted_market_reference_probability: float | None
    expected_market_reference_probability: float | None
    market_reference_absolute_error: float | None

    freshness_correct: bool | None
    completeness: float | None

    hallucination_detected: bool

    tool_call_count: int
    unique_tool_call_count: int
    redundant_tool_call_count: int
    failed_tool_call_count: int

    llm_decision_latency_seconds: float
    tool_execution_latency_seconds: float
    quant_latency_seconds: float
    total_latency_seconds: float

    errors: list[str]


# Maps this evaluator's pre-existing (Milestone 9B) ExecutionStatus onto the
# unified src.evaluation.metrics.FailureCategory (Milestone 11, Step 19) —
# one shared taxonomy, never an architecture-specific synonym for an
# equivalent failure.
_TO_FAILURE_CATEGORY: dict[ExecutionStatus, metrics.FailureCategory] = {
    ExecutionStatus.SUCCESS: metrics.FailureCategory.SUCCESS,
    ExecutionStatus.QUANT_INSUFFICIENT_DATA: metrics.FailureCategory.QUANT_INSUFFICIENT_DATA,
    ExecutionStatus.TOOL_ARGUMENT_ERROR: metrics.FailureCategory.TOOL_ARGUMENT_ERROR,
    ExecutionStatus.TOOL_DATA_MISSING: metrics.FailureCategory.TOOL_FAILURE,
    ExecutionStatus.TOOL_LOOP_LIMIT: metrics.FailureCategory.TOOL_LOOP_LIMIT,
    ExecutionStatus.LLM_OUTPUT_INVALID: metrics.FailureCategory.LLM_OUTPUT_INVALID,
    ExecutionStatus.QUANT_VALIDATION_ERROR: metrics.FailureCategory.QUANT_VALIDATION_ERROR,
    ExecutionStatus.FINAL_OUTPUT_INVALID: metrics.FailureCategory.FINAL_OUTPUT_INVALID,
}


def to_common_result(result: ToolAgentEvaluationResult) -> metrics.EvaluationResult:
    """Convert this evaluator's architecture-specific result into the
    unified src.evaluation.metrics.EvaluationResult shape (Milestone 11,
    Step 2) — for cross-architecture comparison (Step 23). Every field
    here is a straight passthrough/relabel; no metric is recomputed."""
    return metrics.EvaluationResult(
        scenario_id=result.scenario_id,
        architecture=ArchitectureType.TOOL,
        execution_status=_TO_FAILURE_CATEGORY[result.execution_status],
        quant_evaluable=result.quant_evaluable,
        predicted_best_sportsbooks=result.predicted_best_sportsbooks,
        expected_best_sportsbooks=result.expected_best_sportsbooks,
        best_line_correct=result.best_line_correct,
        predicted_best_odds=result.predicted_best_odds,
        expected_best_odds=result.expected_best_odds,
        best_odds_correct=result.best_odds_correct,
        predicted_positive_ev=result.predicted_positive_ev,
        expected_positive_ev=result.expected_positive_ev,
        ev_classification_correct=result.ev_classification_correct,
        predicted_ev=result.predicted_ev,
        expected_ev=result.expected_ev,
        ev_absolute_error=result.ev_absolute_error,
        predicted_market_reference_probability=result.predicted_market_reference_probability,
        expected_market_reference_probability=result.expected_market_reference_probability,
        market_reference_absolute_error=result.market_reference_absolute_error,
        freshness_correct=result.freshness_correct,
        completeness=result.completeness,
        unsupported_claim_count=(1 if result.hallucination_detected else 0),
        total_verifiable_claims=(1 if result.predicted_best_sportsbooks else 0),
        hallucination_detected=result.hallucination_detected,
        retrieval_metrics=None,
        tool_metrics=metrics.ToolMetrics(
            tool_call_count=result.tool_call_count,
            unique_tool_call_count=result.unique_tool_call_count,
            redundant_tool_call_count=result.redundant_tool_call_count,
            failed_tool_call_count=result.failed_tool_call_count,
        ),
        latency_metrics=metrics.LatencyMetrics(
            llm_latency_seconds=result.llm_decision_latency_seconds,
            tool_latency_seconds=result.tool_execution_latency_seconds,
            quant_latency_seconds=result.quant_latency_seconds,
            total_latency_seconds=result.total_latency_seconds,
        ),
        errors=result.errors,
    )


# ---------------------------------------------------------------------------
# Deterministic fake LLM policy (Milestone 9B, Step 16)
# ---------------------------------------------------------------------------

_REQUEST_HEADER_RE = re.compile(
    r"Game ID: (?P<game_id>.+)\nMarket: (?P<market_type>.+)\nSelected outcome: (?P<selected_outcome>.+)\n"
)


class DeterministicToolPolicyLLMClient:
    """Fake ToolCallingLLMClient with a fixed, request-driven policy:
    discover the game, gather the requested outcome's current odds, then
    (for a moneyline market) gather the opposing outcome's odds too, then
    stop. See the module docstring — this never reads GroundTruth/
    QuantGroundTruth; every tool call is derived from the request's own
    stated identity and from actual prior tool_result content.
    """

    model = "deterministic-fake-tool-policy"
    effort = "low"

    def create_turn(self, *, system_prompt, messages, tools):
        header = messages[0]["content"]
        match = _REQUEST_HEADER_RE.search(header)
        if match is None:
            return ToolCallTurn(stop_reason="end_turn", text="Could not parse request.", tool_uses=[])
        game_id = match.group("game_id").strip()
        market_type = match.group("market_type").strip()
        selected_outcome = match.group("selected_outcome").strip()

        turn_index = (len(messages) - 1) // 2

        if turn_index == 0:
            return ToolCallTurn(
                stop_reason="tool_use",
                text=None,
                tool_uses=[ToolUseBlock(id="call-0", name="get_game", input={"game_id": game_id})],
            )

        if turn_index == 1:
            return ToolCallTurn(
                stop_reason="tool_use",
                text=None,
                tool_uses=[
                    ToolUseBlock(
                        id="call-1",
                        name="get_odds",
                        input={
                            "game_id": game_id,
                            "market_type": market_type,
                            "selected_outcome": selected_outcome,
                        },
                    )
                ],
            )

        if turn_index == 2 and market_type == MarketType.MONEYLINE.value:
            opposing_outcome = self._find_opposing_outcome(messages, selected_outcome)
            if opposing_outcome is not None:
                return ToolCallTurn(
                    stop_reason="tool_use",
                    text=None,
                    tool_uses=[
                        ToolUseBlock(
                            id="call-2",
                            name="get_odds",
                            input={
                                "game_id": game_id,
                                "market_type": market_type,
                                "selected_outcome": opposing_outcome,
                            },
                        )
                    ],
                )

        return ToolCallTurn(stop_reason="end_turn", text="Gathered available current odds.", tool_uses=[])

    @staticmethod
    def _find_opposing_outcome(messages: list[dict], selected_outcome: str) -> str | None:
        # messages[2] is the user turn carrying the get_game tool_result
        # (messages[0]=initial user, [1]=assistant get_game call, [2]=its result).
        if len(messages) <= 2:
            return None
        for block in messages[2].get("content", []):
            if block.get("type") != "tool_result" or block.get("is_error"):
                continue
            try:
                payload = json.loads(block["content"])
            except (KeyError, TypeError, ValueError):
                continue
            home, away = payload.get("home_team"), payload.get("away_team")
            if not home or not away:
                continue
            if selected_outcome == home:
                return away
            if selected_outcome == away:
                return home
        return None


# ---------------------------------------------------------------------------
# Harness construction
# ---------------------------------------------------------------------------


def build_default_tool_agent(llm_client=None) -> tuple[ToolCallingAgent, SportsbookTools]:
    tools = SportsbookTools(ControlledOddsProvider())
    agent = ToolCallingAgent(tools, llm_client=llm_client or DeterministicToolPolicyLLMClient())
    return agent, tools


def _build_agent_request(definition: dict) -> AgentRequest:
    game = definition["game"]
    market = definition["market"]
    return AgentRequest(
        scenario_id=definition["scenario_id"],
        game_id=game["game_id"],
        market_type=MarketType(market["market_type"]),
        selected_outcome=market["selected_outcome"],
        query=(
            f"What is the best current price on {market['selected_outcome']} "
            f"({market['market_type']}), and is it a positive EV bet?"
        ),
    )


def _stale_odds_by_scenario_key() -> dict[tuple[str, str, str], dict[str, int]]:
    """(game_id, market_type, selected_outcome) -> {sportsbook: stale odds},
    from data/historical_odds.json (dataset layer, never RAG) — used only
    to confirm the tool agent's result differs from a known-stale value,
    per Step 10."""
    stale_map: dict[tuple[str, str, str], dict[str, int]] = {}
    for record in load_historical_odds_records():
        key = (record["game_id"], record["market_type"], record["selected_outcome"])
        stale_map.setdefault(key, {})[record["sportsbook"]] = record["american_odds"]
    return stale_map


def _detect_hallucination(
    tools: SportsbookTools, request: AgentRequest, analysis: BettingAnalysis
) -> bool:
    """Independently re-query the SAME authoritative tool layer for the
    analysis's headline claim (best_sportsbook/best_odds) — never the
    agent's own trace — so a bug that made market_state internally
    inconsistent would still be caught, not just a bug in trace logging.
    """
    try:
        authoritative = tools.get_sportsbook_odds(
            request.game_id, analysis.best_sportsbook, request.market_type, request.selected_outcome
        )
    except Exception:
        return True  # claimed a sportsbook with no actual current line here
    return authoritative.american_odds != analysis.best_odds


def _classify_failure(trace: ToolAgentTrace | None) -> ExecutionStatus:
    if trace is None:
        return ExecutionStatus.LLM_OUTPUT_INVALID
    if trace.validation_status == "loop_limit_exceeded":
        return ExecutionStatus.TOOL_LOOP_LIMIT
    if trace.errors:
        return ExecutionStatus.LLM_OUTPUT_INVALID
    failed_calls = [call for call in trace.tool_calls if not call.success]
    if any("validation error" in (call.error or "").lower() for call in failed_calls):
        return ExecutionStatus.TOOL_ARGUMENT_ERROR
    return ExecutionStatus.TOOL_DATA_MISSING


def evaluate_scenario(
    agent: ToolCallingAgent,
    tools: SportsbookTools,
    request: AgentRequest,
    ground_truth: GroundTruth,
    quant_ground_truth: QuantGroundTruth,
    stale_odds_map: dict[tuple[str, str, str], dict[str, int]] | None = None,
) -> ToolAgentEvaluationResult:
    stale_odds_map = stale_odds_map or {}
    analysis: BettingAnalysis | None = None
    errors: list[str] = []

    try:
        analysis = agent.analyze(request)
    except ToolAnalysisIncomplete as exc:
        trace = exc.trace
        execution_status = _classify_failure(trace)
        errors = list(trace.errors)
    except Exception as exc:  # defensive: an unexpected internal crash
        trace = agent.last_trace
        execution_status = ExecutionStatus.QUANT_VALIDATION_ERROR
        errors = [repr(exc)]
    else:
        trace = agent.last_trace
        try:
            BettingAnalysis.model_validate(analysis.model_dump())
        except Exception as exc:
            execution_status = ExecutionStatus.FINAL_OUTPUT_INVALID
            errors = [repr(exc)]
        else:
            execution_status = (
                ExecutionStatus.SUCCESS
                if analysis.status == AnalysisStatus.OK
                else ExecutionStatus.QUANT_INSUFFICIENT_DATA
            )

    predicted_best_sportsbooks = analysis.best_sportsbooks if analysis is not None else []
    predicted_best_odds = analysis.best_odds if analysis is not None else None
    best_line_correct = metrics.best_line_correct(
        predicted_best_sportsbooks if analysis is not None else None, ground_truth.expected_best_sportsbooks
    )
    best_odds_correct = metrics.best_odds_correct(predicted_best_odds, ground_truth.expected_best_odds)

    expected_ev = None
    expected_positive_ev = None
    expected_market_reference_probability = None
    if quant_ground_truth.quant_evaluable and predicted_best_sportsbooks:
        # The tool agent's EV/consensus pertains to predicted_best_sportsbooks[0]
        # (alphabetically-first best-odds book with complete two-sided data —
        # see src/agents/tool_agent.py::_run_quant_pipeline's `target`
        # selection). True for every scenario in DEFAULT_SCENARIO_IDS, where
        # every best-odds-tied book also has complete two-sided data.
        target_sportsbook = predicted_best_sportsbooks[0]
        match = next(
            (sa for sa in quant_ground_truth.sportsbook_analyses if sa.sportsbook == target_sportsbook),
            None,
        )
        if match is not None:
            expected_ev = match.expected_value
            expected_positive_ev = match.positive_ev
            expected_market_reference_probability = match.market_reference_probability

    predicted_ev = analysis.expected_value if analysis is not None else None
    predicted_positive_ev = analysis.positive_ev if analysis is not None else None
    predicted_market_reference_probability = (
        analysis.market_reference_probability if analysis is not None else None
    )

    ev_classification_correct = metrics.ev_classification_correct(predicted_positive_ev, expected_positive_ev)
    ev_absolute_error = metrics.ev_absolute_error(predicted_ev, expected_ev)
    market_reference_absolute_error = metrics.market_reference_absolute_error(
        predicted_market_reference_probability, expected_market_reference_probability
    )

    stale_key = (request.game_id, request.market_type.value, request.selected_outcome)
    freshness_correct = None
    if stale_key in stale_odds_map:
        stale_value = stale_odds_map[stale_key].get(analysis.best_sportsbook) if analysis is not None else None
        freshness_correct = metrics.evaluate_freshness(
            predicted_best_odds, ground_truth.expected_best_odds, stale_value
        ).freshness_correct

    completeness = metrics.completeness(
        analysis.sportsbooks_considered if analysis is not None else [], ground_truth.expected_sportsbooks
    )

    hallucination_detected = (
        _detect_hallucination(tools, request, analysis) if analysis is not None else False
    )

    tool_call_count = len(trace.tool_calls) if trace is not None else 0
    redundant_tool_call_count = trace.redundant_call_count if trace is not None else 0
    failed_tool_call_count = (
        sum(1 for call in trace.tool_calls if not call.success) if trace is not None else 0
    )

    return ToolAgentEvaluationResult(
        scenario_id=request.scenario_id,
        execution_status=execution_status,
        quant_evaluable=quant_ground_truth.quant_evaluable,
        predicted_best_sportsbooks=predicted_best_sportsbooks,
        expected_best_sportsbooks=ground_truth.expected_best_sportsbooks,
        best_line_correct=best_line_correct,
        predicted_best_odds=predicted_best_odds,
        expected_best_odds=ground_truth.expected_best_odds,
        best_odds_correct=best_odds_correct,
        predicted_positive_ev=predicted_positive_ev,
        expected_positive_ev=expected_positive_ev,
        ev_classification_correct=ev_classification_correct,
        predicted_ev=predicted_ev,
        expected_ev=expected_ev,
        ev_absolute_error=ev_absolute_error,
        predicted_market_reference_probability=predicted_market_reference_probability,
        expected_market_reference_probability=expected_market_reference_probability,
        market_reference_absolute_error=market_reference_absolute_error,
        freshness_correct=freshness_correct,
        completeness=completeness,
        hallucination_detected=hallucination_detected,
        tool_call_count=tool_call_count,
        unique_tool_call_count=tool_call_count - redundant_tool_call_count,
        redundant_tool_call_count=redundant_tool_call_count,
        failed_tool_call_count=failed_tool_call_count,
        llm_decision_latency_seconds=trace.llm_decision_latency_seconds if trace else 0.0,
        tool_execution_latency_seconds=trace.tool_execution_latency_seconds if trace else 0.0,
        quant_latency_seconds=trace.quant_latency_seconds if trace else 0.0,
        total_latency_seconds=trace.total_latency_seconds if trace else 0.0,
        errors=errors,
    )


def evaluate_scenarios(
    scenario_ids: list[str] | None = None, llm_client=None
) -> list[ToolAgentEvaluationResult]:
    scenario_ids = scenario_ids or DEFAULT_SCENARIO_IDS
    agent, tools = build_default_tool_agent(llm_client)

    ground_truth_by_id = {gt.scenario_id: gt for gt in generate_all_ground_truth()}
    quant_ground_truth_by_id = {qgt.scenario_id: qgt for qgt in generate_all_quant_ground_truth()}
    scenario_definitions_by_id = load_scenario_definitions_by_id()
    stale_odds_map = _stale_odds_by_scenario_key()

    results = []
    for scenario_id in scenario_ids:
        request = _build_agent_request(scenario_definitions_by_id[scenario_id])
        result = evaluate_scenario(
            agent,
            tools,
            request,
            ground_truth_by_id[scenario_id],
            quant_ground_truth_by_id[scenario_id],
            stale_odds_map,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Summary / report (Milestone 9B, Step 22)
# ---------------------------------------------------------------------------


def summarize_results(results: list[ToolAgentEvaluationResult]) -> dict:
    total = len(results)
    successful = [r for r in results if r.execution_status in _SUCCESSFUL_STATUSES]

    return {
        "scenarios_evaluated": total,
        "successful": len(successful),
        "failed": total - len(successful),
        "best_line_accuracy": metrics.rate(r.best_line_correct for r in results),
        "best_odds_accuracy": metrics.rate(r.best_odds_correct for r in results),
        "ev_classification_accuracy": metrics.rate(r.ev_classification_correct for r in results),
        "mean_ev_absolute_error": metrics.mean(r.ev_absolute_error for r in results),
        "market_reference_mae": metrics.mean(r.market_reference_absolute_error for r in results),
        "freshness_accuracy": metrics.rate(r.freshness_correct for r in results),
        "mean_completeness": metrics.mean(r.completeness for r in results),
        "hallucination_rate": metrics.rate(r.hallucination_detected for r in results),
        "total_tool_calls": sum(r.tool_call_count for r in results),
        "average_tool_calls_per_scenario": (sum(r.tool_call_count for r in results) / total) if total else None,
        "redundant_tool_calls": sum(r.redundant_tool_call_count for r in results),
        "failed_tool_calls": sum(r.failed_tool_call_count for r in results),
        "mean_total_latency_seconds": metrics.mean(r.total_latency_seconds for r in results),
    }


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def print_report(results: list[ToolAgentEvaluationResult], summary: dict) -> None:
    print(f"Scenarios evaluated: {summary['scenarios_evaluated']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print()
    print(f"Best-line accuracy: {_fmt_pct(summary['best_line_accuracy'])}")
    print(f"Best-odds accuracy: {_fmt_pct(summary['best_odds_accuracy'])}")
    print()
    print(f"EV classification accuracy: {_fmt_pct(summary['ev_classification_accuracy'])}")
    print(f"Mean EV absolute error: {_fmt_num(summary['mean_ev_absolute_error'])}")
    print(f"Market-reference MAE: {_fmt_num(summary['market_reference_mae'])}")
    print()
    print(f"Freshness accuracy: {_fmt_pct(summary['freshness_accuracy'])}")
    print(f"Completeness (mean): {_fmt_pct(summary['mean_completeness'])}")
    print(f"Unsupported-claim (hallucination) rate: {_fmt_pct(summary['hallucination_rate'])}")
    print()
    print(f"Total tool calls: {summary['total_tool_calls']}")
    avg_calls = summary["average_tool_calls_per_scenario"]
    print(f"Average tool calls/scenario: {avg_calls:.2f}" if avg_calls is not None else "Average tool calls/scenario: n/a")
    print(f"Redundant tool calls: {summary['redundant_tool_calls']}")
    print(f"Failed tool calls: {summary['failed_tool_calls']}")
    print()
    mean_latency = summary["mean_total_latency_seconds"]
    print(f"Mean total latency: {mean_latency:.6f}s" if mean_latency is not None else "Mean total latency: n/a")

    print()
    print(f"{'scenario':<8} {'status':<24} {'best_line':<10} {'best_odds':<10} {'ev_class':<10}")
    for r in results:
        print(
            f"{r.scenario_id:<8} {r.execution_status.value:<24} "
            f"{str(r.best_line_correct):<10} {str(r.best_odds_correct):<10} {str(r.ev_classification_correct):<10}"
        )


def _print_failure_demo(agent: ToolCallingAgent) -> None:
    request = AgentRequest(
        scenario_id="DEMO-FAILURE",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Nonexistent Team",
        query="What is the best current price on the Nonexistent Team moneyline?",
    )
    print("\n--- Missing-data / failure example (synthetic, not part of the accuracy metrics) ---")
    try:
        agent.analyze(request)
        print("ERROR: expected ToolAnalysisIncomplete")
    except ToolAnalysisIncomplete as exc:
        trace = exc.trace
        print(f"validation_status = {trace.validation_status}")
        print(f"execution_status  = {_classify_failure(trace).value}")
        for call in trace.tool_calls:
            print(f"  {call.tool_name}({call.arguments}) -> success={call.success} error={call.error}")


def main() -> None:
    results = evaluate_scenarios()
    summary = summarize_results(results)

    if "--json" in sys.argv:
        print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
        return

    print_report(results, summary)
    agent, _ = build_default_tool_agent()
    _print_failure_demo(agent)


if __name__ == "__main__":
    main()
