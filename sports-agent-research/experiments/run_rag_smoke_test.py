"""Manual smoke test for the RAG-only agent (Milestone 8B) against the
REAL Anthropic API — not a pytest file, not run in CI, and never part of
the automated test suite (see tests/test_rag_agent.py / test_llm_client.py
/ test_rag_extraction.py for the credential-free unit tests).

Run manually:

    .venv/bin/python -m experiments.run_rag_smoke_test

Requires ANTHROPIC_API_KEY in the environment (see .env.example). If it
is not set, this script prints a clear message and exits 0 rather than
failing — a missing credential must not fail the milestone.

Prints, for each of three controlled scenarios, the full pipeline trace:
QUERY -> RETRIEVED EVIDENCE -> LLM EXTRACTION -> QUANT RESULT -> FINAL
BettingAnalysis. The three scenarios are deliberately chosen to exercise
distinct behavior, not to tune the prompt against a known answer (no
ground truth is read anywhere in this script):

  1. normal current evidence   (G-2026-001, Lakers moneyline)
  2. stale-evidence            (G-2026-009, Timberwolves moneyline — the
                                 corpus has both a stale and a fresh
                                 DraftKings snapshot for this game)
  3. incomplete / one-sided    (G-2026-002, Warriors moneyline — only one
                                 side of the market exists in the corpus,
                                 so no complete pair can ever be formed)
"""

from __future__ import annotations

from src.agents.base import AgentRequest
from src.agents.rag_agent import RagAnalysisIncomplete, RagOnlyAgent
from src.models import MarketType
from src.rag.retriever import Retriever

SCENARIOS = [
    AgentRequest(
        scenario_id="SMOKE-001",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="DraftKings FanDuel BetMGM Caesars moneyline price Los Angeles Lakers Boston Celtics",
    ),
    AgentRequest(
        scenario_id="SMOKE-009",
        game_id="G-2026-009",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Minnesota Timberwolves",
        query="What price did DraftKings list for the Minnesota Timberwolves moneyline?",
    ),
    AgentRequest(
        scenario_id="SMOKE-002",
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


def _run_scenario(agent: RagOnlyAgent, request: AgentRequest) -> None:
    _print_header(f"SCENARIO {request.scenario_id} — {request.game_id}")
    print(f"QUERY: {request.query}")

    try:
        analysis = agent.analyze(request)
    except RagAnalysisIncomplete as exc:
        trace = exc.trace
        _print_trace_evidence(trace)
        print("\nFINAL OUTPUT: RagAnalysisIncomplete raised")
        print(f"  validation_status = {trace.validation_status!r}")
        print(f"  rejected_extraction_reasons = {trace.rejected_extraction_reasons}")
        print(f"  errors = {trace.errors}")
        return

    trace = agent.last_trace
    _print_trace_evidence(trace)
    print("\nQUANT RESULT:")
    print(f"  status = {analysis.status.value}")
    print(f"  quant_status (trace) = {trace.quant_status}")
    print(f"  market_reference_probability = {analysis.market_reference_probability}")
    print(f"  probability_edge = {analysis.probability_edge}")
    print(f"  expected_value = {analysis.expected_value}")
    print(f"  positive_ev = {analysis.positive_ev}")

    print("\nFINAL OUTPUT (BettingAnalysis):")
    print(analysis.model_dump_json(indent=2))


def _print_trace_evidence(trace) -> None:
    print(f"\nRETRIEVED EVIDENCE ({len(trace.retrieved_document_ids)} documents, top_k={trace.top_k}):")
    for doc_id, score in zip(trace.retrieved_document_ids, trace.retrieval_scores):
        print(f"  {score:.4f}  {doc_id}")

    print("\nLLM EXTRACTION:")
    if trace.extraction_result is None:
        print("  (extraction failed)")
    else:
        for price in trace.extraction_result.sportsbook_prices:
            print(
                f"  {price.sportsbook}: {price.selected_outcome} "
                f"{price.american_odds:+d} (is_current={price.is_current}) "
                f"sources={price.source_document_ids}"
            )
        if trace.extraction_result.missing_evidence_note:
            print(f"  missing_evidence_note: {trace.extraction_result.missing_evidence_note}")

    print("\nPROVENANCE VALIDATION:")
    print(f"  validation_status = {trace.validation_status}")
    if trace.rejected_extraction_reasons:
        for reason in trace.rejected_extraction_reasons:
            print(f"  REJECTED: {reason}")

    print(
        f"\nLATENCIES: retrieval={trace.retrieval_latency_seconds:.3f}s "
        f"llm={trace.llm_latency_seconds:.3f}s "
        f"quant={trace.quant_latency_seconds:.3f}s "
        f"total={trace.total_latency_seconds:.3f}s"
    )


def main() -> None:
    retriever = Retriever.from_directory()
    try:
        agent = RagOnlyAgent(retriever)
    except Exception as exc:  # anthropic.Anthropic() construction failure
        print("SKIPPED: could not construct AnthropicLLMClient — is ANTHROPIC_API_KEY set?")
        print(f"  ({exc!r})")
        return

    for request in SCENARIOS:
        try:
            _run_scenario(agent, request)
        except Exception as exc:
            # Any credential/network failure surfaces here (e.g.
            # anthropic.AuthenticationError on the first real call) — do
            # not let a missing/invalid key fail this manual script.
            print(f"\nSKIPPED scenario {request.scenario_id}: {exc!r}")
            print("Is ANTHROPIC_API_KEY set to a valid key in your environment?")


if __name__ == "__main__":
    main()
