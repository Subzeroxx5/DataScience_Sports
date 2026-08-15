"""Manual smoke test for the hybrid agent (Milestone 10A) against the
REAL Anthropic API — not a pytest file, not run in CI, and never part of
the automated test suite (see tests/test_hybrid_agent.py /
test_hybrid_reconciliation.py for the credential-free unit tests).

Run manually:

    .venv/bin/python -m experiments.run_hybrid_agent_smoke_test

Requires ANTHROPIC_API_KEY in the environment (see .env.example). If it
is not set, this script prints "REAL LLM HYBRID SMOKE TEST: NOT RUN" and
exits 0 rather than failing — a missing credential must not block the
milestone.

The sportsbook data source remains the controlled provider
(ControlledOddsProvider / data/current_odds.json) and the RAG evidence
source remains the controlled corpus throughout — this script makes zero
real sportsbook API requests, only real Claude API requests for RAG
extraction and tool orchestration.

Prints, for each of three controlled scenarios: QUERY -> RAG EVIDENCE ->
TOOL CALLS -> SOURCE RECONCILIATION -> AUTHORITATIVE MARKET STATE ->
QUANT OUTPUT -> FINAL BettingAnalysis. No ground truth is read anywhere
in this script.

  1. RAG/tool agreement       (G-2026-001, Lakers moneyline — the
                                DraftKings RAG snapshot and the current
                                tool price should agree)
  2. stale-RAG/current-tool   (G-2026-009, Timberwolves moneyline — the
     conflict                  corpus has both a stale and a fresh
                                DraftKings snapshot for this game)
  3. incomplete-source case   (G-2026-002, Warriors moneyline — only one
                                side of the market exists in the corpus
                                and in the controlled dataset, so no
                                complete pair can ever be formed)
"""

from __future__ import annotations

from src.agents.base import AgentRequest
from src.agents.hybrid_agent import HybridAgent, HybridAnalysisIncomplete
from src.models import MarketType
from src.providers.controlled import ControlledOddsProvider
from src.rag.retriever import Retriever
from src.tools.sportsbook_tools import SportsbookTools

SCENARIOS = [
    AgentRequest(
        scenario_id="HYBRID-SMOKE-001",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="DraftKings FanDuel BetMGM Caesars moneyline price Los Angeles Lakers Boston Celtics",
    ),
    AgentRequest(
        scenario_id="HYBRID-SMOKE-009",
        game_id="G-2026-009",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Minnesota Timberwolves",
        query="What price did DraftKings list for the Minnesota Timberwolves moneyline?",
    ),
    AgentRequest(
        scenario_id="HYBRID-SMOKE-002",
        game_id="G-2026-002",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Golden State Warriors",
        query="DraftKings FanDuel BetMGM Caesars moneyline price Golden State Warriors",
    ),
]


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _print_trace(trace) -> None:
    print(f"\nRAG EVIDENCE ({len(trace.retrieved_document_ids)} documents):")
    for doc_id, score in zip(trace.retrieved_document_ids, trace.rag_scores):
        print(f"  {score:.4f}  {doc_id}")
    if trace.rag_rejected_reasons:
        for reason in trace.rag_rejected_reasons:
            print(f"  REJECTED: {reason}")

    print(f"\nTOOL CALLS ({trace.tool_iterations_used} iteration(s)):")
    for call in trace.tool_calls:
        status = "OK" if call.success else "FAILED"
        marker = "REDUNDANT " if call.is_redundant else ""
        print(f"  {marker}{call.tool_name}({call.arguments}) -> {status}: {call.result_summary}")

    print("\nSOURCE RECONCILIATION:")
    print(
        f"  agreements={trace.source_agreements} conflicts={trace.source_conflicts} "
        f"rag_only={trace.rag_only_records} tool_only={trace.tool_only_records}"
    )
    print("\nAUTHORITATIVE MARKET STATE:")
    for record in trace.reconciled_records:
        print(
            f"  {record.sportsbook}/{record.selected_outcome}: tool={record.tool_odds} "
            f"rag={record.rag_odds}(current={record.rag_is_current}) "
            f"-> authoritative={record.authoritative_odds} "
            f"source={record.authoritative_source} reason={record.conflict_resolution_reason}"
        )

    print(
        f"\nLATENCIES: rag_retrieval={trace.rag_retrieval_latency_seconds:.3f}s "
        f"rag_llm={trace.rag_llm_latency_seconds:.3f}s "
        f"tool_llm={trace.tool_llm_latency_seconds:.3f}s "
        f"tool_execution={trace.tool_execution_latency_seconds:.3f}s "
        f"reconciliation={trace.reconciliation_latency_seconds:.3f}s "
        f"quant={trace.quant_latency_seconds:.3f}s "
        f"total={trace.total_latency_seconds:.3f}s"
    )
    if trace.errors:
        print(f"ERRORS: {trace.errors}")


def _run_scenario(agent: HybridAgent, request: AgentRequest) -> None:
    _print_header(f"SCENARIO {request.scenario_id} — {request.game_id}")
    print(f"QUERY: {request.query}")

    try:
        analysis = agent.analyze(request)
    except HybridAnalysisIncomplete as exc:
        _print_trace(exc.trace)
        print(f"\nFINAL OUTPUT: HybridAnalysisIncomplete raised (validation_status={exc.trace.validation_status.value})")
        return

    _print_trace(agent.last_trace)
    print("\nQUANT OUTPUT:")
    print(f"  status = {analysis.status.value}")
    print(f"  market_reference_probability = {analysis.market_reference_probability}")
    print(f"  probability_edge = {analysis.probability_edge}")
    print(f"  expected_value = {analysis.expected_value}")
    print(f"  positive_ev = {analysis.positive_ev}")

    print("\nFINAL OUTPUT (BettingAnalysis):")
    print(analysis.model_dump_json(indent=2))


def main() -> None:
    retriever = Retriever.from_directory()
    tools = SportsbookTools(ControlledOddsProvider())
    try:
        agent = HybridAgent(retriever, tools)
    except Exception as exc:  # anthropic.Anthropic() construction failure
        print("REAL LLM HYBRID SMOKE TEST: NOT RUN")
        print(f"Could not construct AnthropicLLMClient — is ANTHROPIC_API_KEY set? ({exc!r})")
        return

    try:
        agent.llm_client.create_turn(
            system_prompt="Reply with the single word: ready.",
            messages=[{"role": "user", "content": "ready?"}],
            tools=[],
        )
    except Exception as exc:
        print("REAL LLM HYBRID SMOKE TEST: NOT RUN")
        print(f"Connectivity probe failed — is ANTHROPIC_API_KEY set to a valid key? ({exc!r})")
        return

    for request in SCENARIOS:
        _run_scenario(agent, request)


if __name__ == "__main__":
    main()
