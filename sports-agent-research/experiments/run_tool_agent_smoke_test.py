"""Manual smoke test for the tool-calling agent (Milestone 9A) against
the REAL Anthropic API — not a pytest file, not run in CI, and never
part of the automated test suite (see tests/test_tool_agent.py /
test_tool_agent_trace.py for the credential-free unit tests).

Run manually:

    .venv/bin/python -m experiments.run_tool_agent_smoke_test

Requires ANTHROPIC_API_KEY in the environment (see .env.example). If it
is not set, this script prints "REAL LLM SMOKE TEST: NOT RUN" and exits 0
rather than failing — a missing credential must not fail the milestone.

The sportsbook data source remains the controlled provider
(ControlledOddsProvider / data/current_odds.json) throughout — this
script makes zero real sportsbook API requests, only real Claude API
requests for tool orchestration.

Prints, for each of three controlled scenarios, the full pipeline trace:
QUERY -> LLM TOOL DECISIONS -> TOOL-CALL TRACE -> STRUCTURED TOOL RESULTS
-> QUANT OUTPUT -> FINAL BettingAnalysis. No ground truth is read
anywhere in this script.

  1. straightforward best-line case   (G-2026-001, Lakers moneyline)
  2. two-sided quant scenario         (G-2026-001, Lakers moneyline,
                                        query nudges toward gathering
                                        both sides)
  3. missing/unavailable sportsbook   (a sportsbook name that does not
                                        exist in the controlled dataset)
"""

from __future__ import annotations

from src.agents.base import AgentRequest
from src.agents.tool_agent import ToolAnalysisIncomplete, ToolCallingAgent
from src.models import MarketType
from src.providers.controlled import ControlledOddsProvider
from src.tools.sportsbook_tools import SportsbookTools

SCENARIOS = [
    AgentRequest(
        scenario_id="SMOKE-BEST-LINE",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="What is the single best current moneyline price available on the Los Angeles Lakers, and which sportsbook offers it?",
    ),
    AgentRequest(
        scenario_id="SMOKE-TWO-SIDED",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query=(
            "Gather current moneyline prices from every sportsbook for BOTH "
            "the Los Angeles Lakers and the Boston Celtics in this game, so "
            "a full market-consensus and expected-value analysis can be run "
            "on the Lakers side."
        ),
    ),
    AgentRequest(
        scenario_id="SMOKE-MISSING-BOOK",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="What current moneyline price is PhantomBet offering on the Los Angeles Lakers?",
    ),
]


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _print_trace(trace) -> None:
    print(f"\nLLM TOOL DECISIONS / TOOL-CALL TRACE ({trace.iterations_used} iteration(s)):")
    for call in trace.tool_calls:
        marker = "REDUNDANT " if call.is_redundant else ""
        status = "OK" if call.success else "FAILED"
        print(
            f"  [{call.call_sequence}] {marker}{call.tool_name}({call.arguments}) "
            f"-> {status}: {call.result_summary}"
        )
    print(f"\nSTRUCTURED TOOL RESULTS: tool_call_order = {trace.tool_call_order}")
    print(f"  redundant_call_count = {trace.redundant_call_count}")
    print(f"  validation_status = {trace.validation_status}")
    print(
        f"\nLATENCIES: llm_decision={trace.llm_decision_latency_seconds:.3f}s "
        f"tool_execution={trace.tool_execution_latency_seconds:.3f}s "
        f"quant={trace.quant_latency_seconds:.3f}s "
        f"total={trace.total_latency_seconds:.3f}s"
    )
    if trace.errors:
        print(f"  errors = {trace.errors}")


def _run_scenario(agent: ToolCallingAgent, request: AgentRequest) -> None:
    _print_header(f"SCENARIO {request.scenario_id} — {request.game_id}")
    print(f"QUERY: {request.query}")

    try:
        analysis = agent.analyze(request)
    except ToolAnalysisIncomplete as exc:
        _print_trace(exc.trace)
        print("\nFINAL OUTPUT: ToolAnalysisIncomplete raised")
        print(f"  validation_status = {exc.trace.validation_status!r}")
        return

    trace = agent.last_trace
    _print_trace(trace)

    print("\nQUANT OUTPUT:")
    print(f"  status = {analysis.status.value}")
    print(f"  quant_status (trace) = {trace.quant_status}")
    print(f"  market_reference_probability = {analysis.market_reference_probability}")
    print(f"  probability_edge = {analysis.probability_edge}")
    print(f"  expected_value = {analysis.expected_value}")
    print(f"  positive_ev = {analysis.positive_ev}")

    print("\nFINAL OUTPUT (BettingAnalysis):")
    print(analysis.model_dump_json(indent=2))


def main() -> None:
    tools = SportsbookTools(ControlledOddsProvider())
    try:
        agent = ToolCallingAgent(tools)
    except Exception as exc:  # anthropic.Anthropic() construction failure
        print("REAL LLM SMOKE TEST: NOT RUN")
        print(f"Could not construct AnthropicLLMClient — is ANTHROPIC_API_KEY set? ({exc!r})")
        return

    # Cheap connectivity probe: a single real turn with no tools needed.
    # A missing/invalid ANTHROPIC_API_KEY surfaces here (e.g.
    # anthropic.AuthenticationError) rather than mid-scenario — do not
    # let that fail this manual script.
    try:
        agent.llm_client.create_turn(
            system_prompt="Reply with the single word: ready.",
            messages=[{"role": "user", "content": "ready?"}],
            tools=[],
        )
    except Exception as exc:
        print("REAL LLM SMOKE TEST: NOT RUN")
        print(f"Connectivity probe failed — is ANTHROPIC_API_KEY set to a valid key? ({exc!r})")
        return

    for request in SCENARIOS:
        _run_scenario(agent, request)


if __name__ == "__main__":
    main()
