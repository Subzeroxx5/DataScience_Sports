"""Real-LLM evaluation for the tool-calling agent (Milestone 9B, Step
17) — not a pytest file, not run in CI, and never part of the automated
test suite. Its results must never influence pytest pass/fail: this
script reuses evaluate_scenario() (src/evaluation/tool_agent_evaluation.py)
purely for reporting, exactly as the deterministic harness does, but a
low/high accuracy number here does not fail or pass Milestone 9B — only
the deterministic mock-LLM tests do that (see
tests/test_tool_agent_e2e.py).

Run manually:

    .venv/bin/python -m experiments.run_tool_agent_real_llm_evaluation

Requires ANTHROPIC_API_KEY in the environment (see .env.example). If it
is not set, this script prints "REAL LLM EVALUATION: NOT RUN" and exits 0
rather than failing — a missing credential must not block 9B.

Uses the SAME model, effort, prompt, and tool schemas ToolCallingAgent
uses by default (src/agents/tool_agent.py, src/agents/tool_schemas.py) —
nothing is reconfigured for this script, so results are comparable to
what later architecture-comparison work will see. The sportsbook data
source remains the controlled provider throughout; only the LLM calls
are real.
"""

from __future__ import annotations

from src.evaluation.dataset import load_scenario_definitions_by_id
from src.evaluation.ground_truth import generate_all_ground_truth
from src.evaluation.quant_ground_truth import generate_all_quant_ground_truth
from src.evaluation.tool_agent_evaluation import (
    ExecutionStatus,
    _build_agent_request,
    _stale_odds_by_scenario_key,
    evaluate_scenario,
)
from src.providers.controlled import ControlledOddsProvider
from src.tools.sportsbook_tools import SportsbookTools

# A representative 5-scenario subset (Step 17): covers a single winner, a
# tie, positive EV, negative EV, and a missing-sportsbook/quant-evaluable
# case.
REAL_LLM_SCENARIO_IDS = ["S001", "S007", "S008", "S009", "S002"]


def main() -> None:
    tools = SportsbookTools(ControlledOddsProvider())

    try:
        from src.agents.llm_client import AnthropicLLMClient
        from src.agents.tool_agent import DEFAULT_TOOL_MAX_TOKENS, ToolCallingAgent

        llm_client = AnthropicLLMClient(max_tokens=DEFAULT_TOOL_MAX_TOKENS)
        agent = ToolCallingAgent(tools, llm_client=llm_client)
        # Connectivity probe — a missing/invalid ANTHROPIC_API_KEY surfaces
        # here rather than mid-scenario.
        llm_client.create_turn(
            system_prompt="Reply with the single word: ready.",
            messages=[{"role": "user", "content": "ready?"}],
            tools=[],
        )
    except Exception as exc:
        print("REAL LLM EVALUATION: NOT RUN")
        print(f"Could not reach the real Anthropic API — is ANTHROPIC_API_KEY set? ({exc!r})")
        return

    ground_truth_by_id = {gt.scenario_id: gt for gt in generate_all_ground_truth()}
    quant_ground_truth_by_id = {q.scenario_id: q for q in generate_all_quant_ground_truth()}
    scenario_definitions_by_id = load_scenario_definitions_by_id()
    stale_odds_map = _stale_odds_by_scenario_key()

    for scenario_id in REAL_LLM_SCENARIO_IDS:
        request = _build_agent_request(scenario_definitions_by_id[scenario_id])
        print("\n" + "=" * 78)
        print(f"SCENARIO {scenario_id} — {request.game_id} — {request.selected_outcome}")
        print("=" * 78)

        result = evaluate_scenario(
            agent, tools, request,
            ground_truth_by_id[scenario_id], quant_ground_truth_by_id[scenario_id], stale_odds_map,
        )
        trace = agent.last_trace

        print(f"TOOL CALLS ({trace.iterations_used} iteration(s)):")
        for call in trace.tool_calls:
            status = "OK" if call.success else "FAILED"
            print(f"  {call.tool_name}({call.arguments}) -> {status}: {call.result_summary}")

        print(f"\nEXECUTION STATUS: {result.execution_status.value}")
        if result.execution_status != ExecutionStatus.SUCCESS and trace.errors:
            print(f"FAILURE REASON: {trace.errors}")

        print("\nACCURACY:")
        print(f"  best_line_correct = {result.best_line_correct}")
        print(f"  best_odds_correct = {result.best_odds_correct}")
        print(f"  ev_classification_correct = {result.ev_classification_correct}")
        print(f"  ev_absolute_error = {result.ev_absolute_error}")
        print(f"  hallucination_detected = {result.hallucination_detected}")

        print(
            f"\nLATENCY: llm_decision={result.llm_decision_latency_seconds:.3f}s "
            f"tool_execution={result.tool_execution_latency_seconds:.3f}s "
            f"quant={result.quant_latency_seconds:.3f}s "
            f"total={result.total_latency_seconds:.3f}s"
        )


if __name__ == "__main__":
    main()
