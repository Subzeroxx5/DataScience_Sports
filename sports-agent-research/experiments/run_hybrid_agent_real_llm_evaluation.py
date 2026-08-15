"""Real-LLM evaluation for the hybrid agent (Milestone 10B, Step 23) —
not a pytest file, not run in CI, and never part of the automated test
suite. Its results must never influence pytest pass/fail: this script
reuses evaluate_scenario() (src/evaluation/hybrid_agent_evaluation.py)
purely for reporting, exactly as the deterministic harness does, but a
low/high accuracy number here does not fail or pass Milestone 10B — only
the deterministic mock-LLM tests do that (see
tests/test_hybrid_agent_e2e.py).

Run manually:

    .venv/bin/python -m experiments.run_hybrid_agent_real_llm_evaluation

Requires ANTHROPIC_API_KEY in the environment (see .env.example). If it
is not set, this script prints "REAL LLM HYBRID EVALUATION: NOT RUN" and
exits 0 rather than failing — a missing credential must not block 10B.

Uses the SAME model, effort, prompt, RAG pipeline, and tool schemas
HybridAgent uses by default (src/agents/hybrid_agent.py) — the same
configuration used for the RAG-only and tool-calling real-LLM
evaluations, so results stay comparable across architectures. The
sportsbook data source remains the controlled provider and the RAG
evidence source remains the controlled corpus throughout; only the LLM
calls are real.
"""

from __future__ import annotations

from src.evaluation.dataset import load_scenario_definitions_by_id
from src.evaluation.ground_truth import generate_all_ground_truth
from src.evaluation.hybrid_agent_evaluation import _build_agent_request, evaluate_scenario
from src.evaluation.quant_ground_truth import generate_all_quant_ground_truth
from src.evaluation.tool_agent_evaluation import _stale_odds_by_scenario_key
from src.providers.controlled import ControlledOddsProvider
from src.rag.retriever import Retriever
from src.tools.sportsbook_tools import SportsbookTools

# A representative 5-scenario subset (Step 23): covers a single winner, a
# tie, positive EV, negative EV/stale-conflict freshness case, and a
# missing-sportsbook/quant-evaluable case.
REAL_LLM_SCENARIO_IDS = ["S001", "S007", "S008", "S009", "S002"]


def main() -> None:
    tools = SportsbookTools(ControlledOddsProvider())
    retriever = Retriever.from_directory()

    try:
        from src.agents.hybrid_agent import HybridAgent
        from src.agents.llm_client import AnthropicLLMClient, DEFAULT_TOOL_MAX_TOKENS

        llm_client = AnthropicLLMClient(max_tokens=DEFAULT_TOOL_MAX_TOKENS)
        agent = HybridAgent(retriever, tools, llm_client=llm_client)
        # Connectivity probe — a missing/invalid ANTHROPIC_API_KEY surfaces
        # here rather than mid-scenario.
        llm_client.create_turn(
            system_prompt="Reply with the single word: ready.",
            messages=[{"role": "user", "content": "ready?"}],
            tools=[],
        )
    except Exception as exc:
        print("REAL LLM HYBRID EVALUATION: NOT RUN")
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
        print(f"QUERY: {request.query}")

        result = evaluate_scenario(
            agent, tools, request,
            ground_truth_by_id[scenario_id], quant_ground_truth_by_id[scenario_id], stale_odds_map,
        )
        trace = agent.last_trace

        print(f"\nRAG EVIDENCE ({len(trace.retrieved_document_ids)} documents):")
        for doc_id, score in zip(trace.retrieved_document_ids, trace.rag_scores):
            print(f"  {score:.4f}  {doc_id}")
        if trace.rag_rejected_reasons:
            for reason in trace.rag_rejected_reasons:
                print(f"  REJECTED: {reason}")

        print(f"\nTOOL CALLS ({trace.tool_iterations_used} iteration(s)):")
        for call in trace.tool_calls:
            status = "OK" if call.success else "FAILED"
            print(f"  {call.tool_name}({call.arguments}) -> {status}: {call.result_summary}")

        print("\nSOURCE RECONCILIATION / CONFLICTS:")
        for record in trace.reconciled_records:
            marker = " <-- CONFLICT" if record.conflict else ""
            print(
                f"  {record.sportsbook}/{record.selected_outcome}: tool={record.tool_odds} "
                f"rag={record.rag_odds}(current={record.rag_is_current}) "
                f"-> authoritative={record.authoritative_odds} source={record.authoritative_source}{marker}"
            )

        print(f"\nEXECUTION STATUS: {result.execution_status.value}")
        if trace.errors:
            print(f"ERRORS: {trace.errors}")

        print("\nACCURACY:")
        print(f"  best_line_correct = {result.best_line_correct}")
        print(f"  best_odds_correct = {result.best_odds_correct}")
        print(f"  ev_classification_correct = {result.ev_classification_correct}")
        print(f"  ev_absolute_error = {result.ev_absolute_error}")
        print(f"  freshness_correct = {result.freshness_correct}")
        print(f"  hallucination_detected = {result.hallucination_detected}")

        print(
            f"\nLATENCY: rag_retrieval={result.rag_retrieval_latency_seconds:.3f}s "
            f"rag_llm={result.rag_llm_latency_seconds:.3f}s "
            f"tool_llm={result.tool_llm_latency_seconds:.3f}s "
            f"tool_execution={result.tool_execution_latency_seconds:.3f}s "
            f"reconciliation={result.reconciliation_latency_seconds:.3f}s "
            f"quant={result.quant_latency_seconds:.3f}s "
            f"total={result.total_latency_seconds:.3f}s"
        )


if __name__ == "__main__":
    main()
