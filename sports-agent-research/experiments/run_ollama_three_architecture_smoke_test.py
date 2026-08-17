"""Manual smoke test for the local Ollama-backed LLM provider (Pre-
Milestone 14A "Local Real-LLM Backend" checkpoint) — not a pytest file,
never part of the automated suite (see tests/test_ollama_llm_client.py /
test_llm_provider_selection.py for the credential/server-free unit
tests).

Runs ONE real local-model scenario through each architecture (RAG, TOOL,
HYBRID) via the exact same src.experiments.agent_factory.create_agent
factory the experiment runner uses — never a smoke-test-specific agent
construction path. Confirms, per the checkpoint's Step 7/8:

  RAG:    real local LLM inference works, retrieved evidence is
          processed, BettingAnalysis validates.
  TOOL:   real local LLM actually requests/uses a sportsbook tool
          (not merely succeeds without one), structured tool values
          remain authoritative, BettingAnalysis validates.
  HYBRID: real local LLM workflow works with RAG + tools, current-tool
          precedence remains intact, BettingAnalysis validates.

Run manually (requires `ollama serve` / `brew services start ollama`
running locally, and the configured model already pulled):

    .venv/bin/python -m experiments.run_ollama_three_architecture_smoke_test

If the local model cannot support a given architecture (e.g. tool
calling doesn't actually occur, or BettingAnalysis fails to validate),
this script reports the exact incompatibility and exits non-zero —
per the checkpoint's instruction: "If any architecture cannot operate
correctly with the selected local model: STOP. Do not begin Milestone
14A. Report the exact incompatibility."
"""

from __future__ import annotations

import sys

from src.agents.base import AgentRequest
from src.agents.hybrid_agent import HybridAnalysisIncomplete
from src.agents.rag_agent import RagAnalysisIncomplete
from src.agents.tool_agent import ToolAnalysisIncomplete
from src.evaluation.dataset import load_scenario_definitions_by_id
from src.experiments.agent_factory import create_agent
from src.experiments.config import ExecutionMode, ExperimentConfig, build_scenario_manifest
from src.models import ArchitectureType, BettingAnalysis

SMOKE_SCENARIO_ID = "S001"


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _build_request() -> AgentRequest:
    manifest = build_scenario_manifest([SMOKE_SCENARIO_ID])
    scenario = manifest[0]
    definition = load_scenario_definitions_by_id()[SMOKE_SCENARIO_ID]
    return AgentRequest(
        scenario_id=scenario.scenario_id,
        game_id=definition["game"]["game_id"],
        market_type=scenario.market_type,
        selected_outcome=scenario.selected_outcome,
        query=scenario.query,
    )


def _config(model_name: str) -> ExperimentConfig:
    from src.agents.llm_client import DEFAULT_OLLAMA_TEMPERATURE, LLMProviderName

    return ExperimentConfig(
        experiment_id="ollama-3arch-smoke",
        experiment_name="ollama-3arch-smoke",
        repetitions=1,
        execution_mode=ExecutionMode.REAL,
        llm_provider=LLMProviderName.OLLAMA,
        model_name=model_name,
        temperature=DEFAULT_OLLAMA_TEMPERATURE,
    )


def _run_rag(config: ExperimentConfig, request: AgentRequest) -> bool:
    _print_header("RAG — real local inference")
    agent, retriever = create_agent(ArchitectureType.RAG, config)
    try:
        analysis = agent.analyze(request)
    except RagAnalysisIncomplete as exc:
        print(f"FAIL: RagAnalysisIncomplete — {exc}")
        print(f"  retrieved documents: {len(exc.trace.retrieved_document_ids)}")
        print(f"  errors: {exc.trace.errors}")
        return False

    trace = agent.last_trace
    print(f"retrieved documents: {len(trace.retrieved_document_ids)} (top_k={trace.top_k})")
    print(f"extraction produced: {trace.extraction_result is not None}")
    BettingAnalysis.model_validate(analysis.model_dump())
    print(f"BettingAnalysis validates: best_sportsbook={analysis.best_sportsbook} best_odds={analysis.best_odds:+d}")
    print("PASS")
    return True


def _run_tool(config: ExperimentConfig, request: AgentRequest) -> bool:
    _print_header("TOOL — real local inference, must actually call a tool")
    agent, tools = create_agent(ArchitectureType.TOOL, config)
    try:
        analysis = agent.analyze(request)
    except ToolAnalysisIncomplete as exc:
        print(f"FAIL: ToolAnalysisIncomplete — {exc}")
        print(f"  tool_calls made: {len(exc.trace.tool_calls)}")
        print(f"  errors: {exc.trace.errors}")
        return False

    trace = agent.last_trace
    tool_calls_made = len(trace.tool_calls)
    print(f"tool calls made: {tool_calls_made}")
    for call in trace.tool_calls:
        print(f"  #{call.call_sequence} {call.tool_name}({call.arguments}) success={call.success}")

    if tool_calls_made == 0:
        print("FAIL: the local model completed without making a single real tool call — "
              "cannot validate the Tool architecture's tool-calling workflow.")
        return False

    BettingAnalysis.model_validate(analysis.model_dump())
    print(f"BettingAnalysis validates: best_sportsbook={analysis.best_sportsbook} best_odds={analysis.best_odds:+d}")
    print("PASS (real tool call confirmed)")
    return True


def _run_hybrid(config: ExperimentConfig, request: AgentRequest) -> bool:
    _print_header("HYBRID — real local inference, RAG + tools, current-tool precedence")
    agent, tools = create_agent(ArchitectureType.HYBRID, config)
    try:
        analysis = agent.analyze(request)
    except HybridAnalysisIncomplete as exc:
        print(f"FAIL: HybridAnalysisIncomplete — {exc}")
        print(f"  errors: {exc.trace.errors}")
        return False

    trace = agent.last_trace
    print(f"retrieved documents: {len(trace.retrieved_document_ids)}")
    print(f"tool calls made: {len(trace.tool_calls)}")
    print(f"source agreements: {trace.source_agreements}  conflicts: {trace.source_conflicts}")

    from src.agents.hybrid_reconciliation import ConflictResolutionReason

    conflicts_resolved_by_tool_precedence = [
        record for record in trace.reconciled_records
        if record.conflict and record.conflict_resolution_reason == ConflictResolutionReason.CURRENT_TOOL_DATA_PRECEDENCE
    ]
    wrongly_resolved = [
        record for record in trace.reconciled_records
        if record.conflict and record.authoritative_source is not None
        and record.conflict_resolution_reason != ConflictResolutionReason.CURRENT_TOOL_DATA_PRECEDENCE
    ]
    print(f"conflicts resolved via CURRENT_TOOL_DATA_PRECEDENCE: {len(conflicts_resolved_by_tool_precedence)}")
    if wrongly_resolved:
        print(f"FAIL: {len(wrongly_resolved)} conflict(s) NOT resolved via current-tool-data precedence")
        return False

    BettingAnalysis.model_validate(analysis.model_dump())
    print(f"BettingAnalysis validates: best_sportsbook={analysis.best_sportsbook} best_odds={analysis.best_odds:+d}")
    print("PASS (current-tool-data precedence intact)")
    return True


def main() -> int:
    from src.agents.llm_client import DEFAULT_OLLAMA_MODEL

    config = _config(DEFAULT_OLLAMA_MODEL)
    request = _build_request()
    print(f"Model: {config.model_name}  Provider: {config.llm_provider.value}  Scenario: {request.scenario_id}")
    print(f"Query: {request.query}")

    results = {
        "RAG": _run_rag(config, request),
        "TOOL": _run_tool(config, request),
        "HYBRID": _run_hybrid(config, request),
    }

    _print_header("SUMMARY")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not all(results.values()):
        print("\nSTOP: at least one architecture is incompatible with the selected local model. "
              "Do not begin Milestone 14A with this configuration.")
        return 1

    print("\nAll three architectures validated against the local model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
