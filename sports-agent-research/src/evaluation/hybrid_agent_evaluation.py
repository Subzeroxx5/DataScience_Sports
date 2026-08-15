"""End-to-end evaluation harness for the hybrid agent (Milestone 10B):
drives HybridAgent (Milestone 10A) across a representative
controlled-benchmark subset and compares its output against
deterministic ground truth, entirely outside the agent:

    AgentRequest
        |
        v
    HybridAgent                (src/agents/hybrid_agent.py, unmodified)
        |
        v
    BettingAnalysis + HybridAgentTrace
        |
        v
    Evaluator (this module)    <-- ground truth enters HERE only
        |
        v
    HybridAgentEvaluationResult

Ground-truth isolation (docs/EXPERIMENT_RULES.md, "Ground Truth"): this
module is the only place in the hybrid-agent evaluation path that reads
GroundTruth/QuantGroundTruth. Nothing here ever writes an expected value
into an AgentRequest, RAG prompt, tool prompt, tool result, LLM message,
or quant input — every comparison happens strictly after
HybridAgent.analyze() has already returned.

Reuse (milestones/current.md, Milestone 10B Step 3; Milestone 11's
unified framework): this module reuses, rather than redefines, shared
metric formulas from src.evaluation.metrics (best-line/best-odds/EV
classification/EV error/market-reference error/freshness/completeness/
generic aggregation) plus the tool-agent evaluation's own
architecture-specific primitives — `DEFAULT_SCENARIO_IDS`,
`_stale_odds_by_scenario_key`, `_detect_hallucination` — from
src.evaluation.tool_agent_evaluation, so definitions stay identical
across architectures for the later cross-architecture experiment.
`execution_status` reuses HybridAgentTrace's own `HybridFailureCategory`
enum rather than inventing a parallel one; `to_common_result()` maps it
onto the unified `metrics.FailureCategory` taxonomy for cross-
architecture comparison.

Run as a script for a deterministic, mock-LLM, CI-free evaluation report:

    python -m src.evaluation.hybrid_agent_evaluation
    python -m src.evaluation.hybrid_agent_evaluation --json

DeterministicHybridPolicyLLMClient (below) is the fake LLM used by both
this script's default mode and the automated tests
(tests/test_hybrid_evaluation.py, tests/test_hybrid_agent_e2e.py). It is
a *policy*, not an answer key: its tool-calling side delegates verbatim
to DeterministicToolPolicyLLMClient (Milestone 9B — same fixed,
request-driven policy), and its RAG-extraction side deterministically
parses whatever the RAG evidence pipeline actually retrieved (never
GroundTruth/QuantGroundTruth) into a structured, honest extraction — it
never invents a sportsbook, price, or document_id that isn't in the
rendered evidence. The authoritative numbers in the final
BettingAnalysis still come exclusively from reconciled tool/RAG output +
the shared quant engine, exactly as in Milestone 10A.
"""

from __future__ import annotations

import json
import re
import sys

from pydantic import BaseModel

from src.agents.base import AgentRequest
from src.agents.extraction import ExtractedMarketEvidence, ExtractedSportsbookPrice
from src.agents.hybrid_agent import (
    HybridAgent,
    HybridAnalysisIncomplete,
    HybridFailureCategory,
)
from src.evaluation import metrics
from src.evaluation.dataset import load_scenario_definitions_by_id
from src.evaluation.ground_truth import generate_all_ground_truth
from src.evaluation.quant_ground_truth import generate_all_quant_ground_truth
from src.evaluation.tool_agent_evaluation import (
    DEFAULT_SCENARIO_IDS,
    DeterministicToolPolicyLLMClient,
    _detect_hallucination,
    _stale_odds_by_scenario_key,
)
from src.models import (
    AnalysisStatus,
    ArchitectureType,
    BettingAnalysis,
    GroundTruth,
    MarketType,
    QuantGroundTruth,
    SourceType,
)
from src.providers.controlled import ControlledOddsProvider
from src.rag.retriever import Retriever
from src.tools.sportsbook_tools import SportsbookTools

_SUCCESSFUL_STATUSES = frozenset({HybridFailureCategory.SUCCESS, HybridFailureCategory.QUANT_INSUFFICIENT_DATA})


class HybridAgentEvaluationResult(BaseModel):
    """One scenario's evaluation outcome (milestones/current.md,
    Milestone 10B Steps 3/4/26). Field names/semantics for the common
    metrics intentionally match ToolAgentEvaluationResult
    (src/evaluation/tool_agent_evaluation.py) so cross-architecture
    comparison never has to reconcile incompatible definitions."""

    scenario_id: str
    execution_status: HybridFailureCategory
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

    # Hybrid-specific (Step 4)
    source_agreements: int
    source_conflicts: int
    correct_conflict_resolutions: int
    stale_rag_conflicts: int
    stale_rag_incorrectly_promoted: int
    tool_only_records_used: int
    rag_only_records_observed: int
    source_reconciliation_failure: bool

    # Efficiency
    rag_documents_retrieved: int
    tool_call_count: int
    redundant_tool_call_count: int

    # Latency
    rag_retrieval_latency_seconds: float
    rag_llm_latency_seconds: float
    tool_llm_latency_seconds: float
    tool_execution_latency_seconds: float
    reconciliation_latency_seconds: float
    quant_latency_seconds: float
    total_latency_seconds: float

    errors: list[str]


# Maps HybridAgentTrace's own HybridFailureCategory (src/agents/
# hybrid_agent.py, Milestone 10A) onto the unified
# src.evaluation.metrics.FailureCategory (Milestone 11, Step 19) — the
# two vocabularies are already near-identical by design (Milestone 10B
# deliberately reused HybridFailureCategory as this evaluator's own
# execution_status rather than inventing a third taxonomy).
_TO_FAILURE_CATEGORY: dict[HybridFailureCategory, metrics.FailureCategory] = {
    HybridFailureCategory.SUCCESS: metrics.FailureCategory.SUCCESS,
    HybridFailureCategory.RAG_RETRIEVAL_FAILURE: metrics.FailureCategory.RETRIEVAL_FAILURE,
    HybridFailureCategory.TOOL_FAILURE: metrics.FailureCategory.TOOL_FAILURE,
    HybridFailureCategory.SOURCE_RECONCILIATION_FAILURE: metrics.FailureCategory.SOURCE_RECONCILIATION_FAILURE,
    HybridFailureCategory.INSUFFICIENT_CURRENT_DATA: metrics.FailureCategory.INSUFFICIENT_CURRENT_DATA,
    HybridFailureCategory.LLM_OUTPUT_INVALID: metrics.FailureCategory.LLM_OUTPUT_INVALID,
    HybridFailureCategory.QUANT_INSUFFICIENT_DATA: metrics.FailureCategory.QUANT_INSUFFICIENT_DATA,
    HybridFailureCategory.FINAL_OUTPUT_INVALID: metrics.FailureCategory.FINAL_OUTPUT_INVALID,
}


def to_common_result(result: HybridAgentEvaluationResult) -> metrics.EvaluationResult:
    """Convert this evaluator's architecture-specific result into the
    unified src.evaluation.metrics.EvaluationResult shape (Milestone 11,
    Step 2) — for cross-architecture comparison (Step 23). Every field
    here is a straight passthrough/relabel; no metric is recomputed.
    Hybrid-specific fields (source reconciliation, RAG retrieval) live
    only on HybridAgentEvaluationResult — the unified shape captures the
    common surface every architecture shares."""
    return metrics.EvaluationResult(
        scenario_id=result.scenario_id,
        architecture=ArchitectureType.HYBRID,
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
        retrieval_metrics=metrics.RetrievalMetrics(
            retrieved_document_count=result.rag_documents_retrieved,
            # HybridAgentEvaluationResult doesn't track top_k separately,
            # but src/rag/retriever.py always returns exactly k results
            # for this project's corpus size (see
            # test_default_top_k_respected in tests/test_rag_evidence.py),
            # so the retrieved count is the configured top_k here too.
            retrieval_top_k=result.rag_documents_retrieved,
        ),
        tool_metrics=metrics.ToolMetrics(
            tool_call_count=result.tool_call_count,
            unique_tool_call_count=result.tool_call_count - result.redundant_tool_call_count,
            redundant_tool_call_count=result.redundant_tool_call_count,
            failed_tool_call_count=0,
        ),
        latency_metrics=metrics.LatencyMetrics(
            retrieval_latency_seconds=result.rag_retrieval_latency_seconds,
            llm_latency_seconds=result.rag_llm_latency_seconds + result.tool_llm_latency_seconds,
            tool_latency_seconds=result.tool_execution_latency_seconds,
            reconciliation_latency_seconds=result.reconciliation_latency_seconds,
            quant_latency_seconds=result.quant_latency_seconds,
            total_latency_seconds=result.total_latency_seconds,
        ),
        errors=result.errors,
    )


# ---------------------------------------------------------------------------
# Deterministic fake LLM policy (Milestone 10B, Step 21)
# ---------------------------------------------------------------------------

_KNOWN_RAG_FIELDS = {
    "document_id", "source_type", "game_id", "outcome", "sportsbook", "american_odds", "is_current",
}
_HYBRID_RAG_HEADER_RE = re.compile(r"Game ID: (?P<game_id>.+)\nSelected outcome: (?P<selected_outcome>.+)\n")


def _parse_rag_evidence_blocks(text: str) -> list[dict[str, str]]:
    """Parse render_rag_context()'s deterministic per-document text
    (src/agents/rag_evidence.py) into a list of field dicts — one per
    `[DOCUMENT N]` block. Only known field names are captured, and only
    their first occurrence per block, so a `content:`-line that happens
    to contain colon-separated text can never be mistaken for a real
    field (content is always the last line in a block)."""
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in text.splitlines():
        if line.startswith("[DOCUMENT "):
            if current is not None:
                blocks.append(current)
            current = {}
            continue
        if current is None or ": " not in line:
            continue
        key, _, value = line.partition(": ")
        if key in _KNOWN_RAG_FIELDS and key not in current:
            current[key] = value
    if current is not None:
        blocks.append(current)
    return blocks


def extract_honest_rag_evidence(user_prompt: str) -> ExtractedMarketEvidence:
    """Deterministically, honestly parse whatever RAG evidence was
    actually rendered into `user_prompt` (src/agents/rag_evidence.py::
    render_rag_context, via either RagOnlyAgent's or HybridAgent's
    identically-formatted user prompt) into a structured extraction —
    never GroundTruth/QuantGroundTruth, never inventing a sportsbook,
    price, or document_id that isn't in the evidence. Shared by both
    DeterministicHybridPolicyLLMClient (below) and
    src.evaluation.rag_agent_evaluation.DeterministicRagPolicyLLMClient,
    so RAG-extraction fake-LLM behavior is defined exactly once."""
    header_match = _HYBRID_RAG_HEADER_RE.search(user_prompt)
    game_id = header_match.group("game_id").strip() if header_match else "unknown"
    primary_outcome = header_match.group("selected_outcome").strip() if header_match else "unknown"

    by_sportsbook: dict[str, dict[str, dict[str, str]]] = {}
    for block in _parse_rag_evidence_blocks(user_prompt):
        if block.get("source_type") != "sportsbook_snapshot":
            continue
        # Retrieval is similarity-based, not exact-match: a document
        # from an unrelated game can legitimately show up in the
        # evidence (e.g. another game's "DraftKings" snapshot). Never
        # treat it as this game's opposing-outcome price merely
        # because the sportsbook name coincides — that would silently
        # fabricate a cross-game "pair" for the no-vig/consensus math.
        if block.get("game_id") != game_id:
            continue
        sportsbook = block.get("sportsbook")
        outcome = block.get("outcome")
        odds_str = block.get("american_odds")
        document_id = block.get("document_id")
        if not (sportsbook and outcome and odds_str and document_id):
            continue
        # If more than one retrieved document covers the same
        # (sportsbook, outcome) — e.g. a stale and a current snapshot of
        # the same book — keep the FIRST one seen, i.e. the higher
        # retrieval rank (render_rag_context emits documents in rank
        # order), rather than letting whichever happens to be parsed
        # last silently win. This is a deterministic relevance-based
        # tie-break, not a thumb on the scale toward "current": if the
        # only retrieved evidence for a book is stale, that stale record
        # is still what gets used (the freshness behavior under study).
        outcome_map = by_sportsbook.setdefault(sportsbook, {})
        if outcome not in outcome_map:
            outcome_map[outcome] = block

    prices: list[ExtractedSportsbookPrice] = []
    for sportsbook in sorted(by_sportsbook):
        outcomes = by_sportsbook[sportsbook]
        if primary_outcome not in outcomes:
            continue
        primary = outcomes[primary_outcome]
        opposing_outcome = next((o for o in outcomes if o != primary_outcome), None)

        kwargs: dict = dict(
            sportsbook=sportsbook,
            selected_outcome=primary_outcome,
            american_odds=int(primary["american_odds"]),
            is_current=(primary.get("is_current") == "True") if "is_current" in primary else None,
            source_document_ids=[primary["document_id"]],
        )
        if opposing_outcome is not None:
            opposing = outcomes[opposing_outcome]
            kwargs["opposing_outcome"] = opposing_outcome
            kwargs["opposing_american_odds"] = int(opposing["american_odds"])
            kwargs["source_document_ids"] = [primary["document_id"], opposing["document_id"]]
        prices.append(ExtractedSportsbookPrice(**kwargs))

    return ExtractedMarketEvidence(
        game_id=game_id,
        market_id=f"{game_id}-market",
        selected_outcome=primary_outcome,
        sportsbook_prices=prices,
    )


class DeterministicHybridPolicyLLMClient:
    """Fake LLM implementing both LLMClient.generate_structured (RAG
    extraction) and ToolCallingLLMClient.create_turn (tool
    orchestration) — see module docstring."""

    model = "deterministic-fake-hybrid-policy"
    effort = "low"

    def __init__(self) -> None:
        self._tool_policy = DeterministicToolPolicyLLMClient()

    def create_turn(self, *, system_prompt, messages, tools):
        return self._tool_policy.create_turn(system_prompt=system_prompt, messages=messages, tools=tools)

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        return extract_honest_rag_evidence(user_prompt)


# ---------------------------------------------------------------------------
# Harness construction
# ---------------------------------------------------------------------------


def build_default_hybrid_agent(llm_client=None) -> tuple[HybridAgent, SportsbookTools]:
    tools = SportsbookTools(ControlledOddsProvider())
    retriever = Retriever.from_directory()
    agent = HybridAgent(retriever, tools, llm_client=llm_client or DeterministicHybridPolicyLLMClient())
    return agent, tools


def _build_agent_request(definition: dict) -> AgentRequest:
    """Unlike the tool-only evaluator's generic query, the hybrid
    evaluator's query must also drive meaningful RAG retrieval — so it
    names sportsbooks and both teams, matching the query style already
    validated (Milestones 8B/9A/10A) to retrieve the controlled corpus's
    sportsbook_snapshot documents well."""
    game = definition["game"]
    market = definition["market"]
    selected_outcome = market["selected_outcome"]
    home_team, away_team = game.get("home_team"), game.get("away_team")
    opposing = away_team if selected_outcome == home_team else home_team if selected_outcome == away_team else None

    query_parts = ["DraftKings", "FanDuel", "BetMGM", "Caesars", market["market_type"], "price", selected_outcome]
    if opposing:
        query_parts.append(opposing)

    return AgentRequest(
        scenario_id=definition["scenario_id"],
        game_id=game["game_id"],
        market_type=MarketType(market["market_type"]),
        selected_outcome=selected_outcome,
        query=" ".join(query_parts),
    )


def evaluate_scenario(
    agent: HybridAgent,
    tools: SportsbookTools,
    request: AgentRequest,
    ground_truth: GroundTruth,
    quant_ground_truth: QuantGroundTruth,
    stale_odds_map: dict[tuple[str, str, str], dict[str, int]] | None = None,
) -> HybridAgentEvaluationResult:
    stale_odds_map = stale_odds_map or {}
    analysis: BettingAnalysis | None = None
    errors: list[str] = []

    try:
        analysis = agent.analyze(request)
    except HybridAnalysisIncomplete as exc:
        trace = exc.trace
        execution_status = trace.validation_status
        errors = list(trace.errors)
    except Exception as exc:  # defensive: an unexpected internal crash
        trace = agent.last_trace
        execution_status = HybridFailureCategory.FINAL_OUTPUT_INVALID
        errors = [repr(exc)]
    else:
        trace = agent.last_trace
        try:
            BettingAnalysis.model_validate(analysis.model_dump())
        except Exception as exc:
            execution_status = HybridFailureCategory.FINAL_OUTPUT_INVALID
            errors = [repr(exc)]
        else:
            execution_status = (
                HybridFailureCategory.SUCCESS
                if analysis.status == AnalysisStatus.OK
                else HybridFailureCategory.QUANT_INSUFFICIENT_DATA
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
        # Pertains to predicted_best_sportsbooks[0] — see
        # src/agents/hybrid_agent.py::_run_quant_pipeline's `target`
        # selection (identical convention to the tool-only agent).
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

    # Freshness is judged at the final authoritative-market-state level
    # (Step 5) — predicted_best_odds is exactly what fed the quant
    # engine, never merely "was the current value seen somewhere."
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

    reconciled_records = trace.reconciled_records if trace is not None else []
    conflicting_records = [record for record in reconciled_records if record.conflict]
    correct_conflict_resolutions = sum(
        1 for record in conflicting_records if record.authoritative_source == SourceType.TOOL
    )
    stale_rag_conflicts = sum(1 for record in conflicting_records if record.rag_is_current is False)
    stale_rag_incorrectly_promoted = sum(
        1
        for record in reconciled_records
        if record.rag_available and record.rag_is_current is False and record.authoritative_source == SourceType.RAG
    )

    return HybridAgentEvaluationResult(
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
        source_agreements=trace.source_agreements if trace else 0,
        source_conflicts=trace.source_conflicts if trace else 0,
        correct_conflict_resolutions=correct_conflict_resolutions,
        stale_rag_conflicts=stale_rag_conflicts,
        stale_rag_incorrectly_promoted=stale_rag_incorrectly_promoted,
        tool_only_records_used=trace.tool_only_records if trace else 0,
        rag_only_records_observed=trace.rag_only_records if trace else 0,
        source_reconciliation_failure=(execution_status == HybridFailureCategory.SOURCE_RECONCILIATION_FAILURE),
        rag_documents_retrieved=len(trace.retrieved_document_ids) if trace else 0,
        tool_call_count=len(trace.tool_calls) if trace else 0,
        redundant_tool_call_count=trace.redundant_tool_call_count if trace else 0,
        rag_retrieval_latency_seconds=trace.rag_retrieval_latency_seconds if trace else 0.0,
        rag_llm_latency_seconds=trace.rag_llm_latency_seconds if trace else 0.0,
        tool_llm_latency_seconds=trace.tool_llm_latency_seconds if trace else 0.0,
        tool_execution_latency_seconds=trace.tool_execution_latency_seconds if trace else 0.0,
        reconciliation_latency_seconds=trace.reconciliation_latency_seconds if trace else 0.0,
        quant_latency_seconds=trace.quant_latency_seconds if trace else 0.0,
        total_latency_seconds=trace.total_latency_seconds if trace else 0.0,
        errors=errors,
    )


def evaluate_scenarios(
    scenario_ids: list[str] | None = None, llm_client=None
) -> list[HybridAgentEvaluationResult]:
    scenario_ids = scenario_ids or DEFAULT_SCENARIO_IDS
    agent, tools = build_default_hybrid_agent(llm_client)

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
# Summary / report (Milestone 10B, Step 26)
# ---------------------------------------------------------------------------


def summarize_results(results: list[HybridAgentEvaluationResult]) -> dict:
    total = len(results)
    successful = [r for r in results if r.execution_status in _SUCCESSFUL_STATUSES]

    total_conflicts = sum(r.source_conflicts for r in results)
    total_correct_resolutions = sum(r.correct_conflict_resolutions for r in results)
    conflict_resolution_accuracy = (
        (total_correct_resolutions / total_conflicts) if total_conflicts > 0 else None
    )

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
        "source_agreements": sum(r.source_agreements for r in results),
        "source_conflicts": total_conflicts,
        "correct_conflict_resolutions": total_correct_resolutions,
        "conflict_resolution_accuracy": conflict_resolution_accuracy,
        "stale_rag_conflicts": sum(r.stale_rag_conflicts for r in results),
        "stale_rag_incorrectly_promoted": sum(r.stale_rag_incorrectly_promoted for r in results),
        "tool_only_recoveries": sum(r.tool_only_records_used for r in results),
        "rag_only_records_observed": sum(r.rag_only_records_observed for r in results),
        "source_reconciliation_failures": sum(1 for r in results if r.source_reconciliation_failure),
        "total_tool_calls": sum(r.tool_call_count for r in results),
        "redundant_tool_calls": sum(r.redundant_tool_call_count for r in results),
        "mean_total_latency_seconds": metrics.mean(r.total_latency_seconds for r in results),
    }


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def print_report(results: list[HybridAgentEvaluationResult], summary: dict) -> None:
    print(f"Scenarios evaluated: {summary['scenarios_evaluated']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print()
    print(f"Best-line accuracy: {_fmt_pct(summary['best_line_accuracy'])}")
    print(f"Best-odds accuracy: {_fmt_pct(summary['best_odds_accuracy'])}")
    print(f"EV classification accuracy: {_fmt_pct(summary['ev_classification_accuracy'])}")
    print(f"Mean EV absolute error: {_fmt_num(summary['mean_ev_absolute_error'])}")
    print(f"Market-reference MAE: {_fmt_num(summary['market_reference_mae'])}")
    print(f"Freshness accuracy: {_fmt_pct(summary['freshness_accuracy'])}")
    print(f"Completeness (mean): {_fmt_pct(summary['mean_completeness'])}")
    print(f"Unsupported-claim (hallucination) rate: {_fmt_pct(summary['hallucination_rate'])}")
    print()
    print(f"RAG/tool agreements: {summary['source_agreements']}")
    print(f"RAG/tool conflicts: {summary['source_conflicts']}")
    print(f"Correct conflict resolutions: {summary['correct_conflict_resolutions']}")
    print(f"Conflict-resolution accuracy: {_fmt_pct(summary['conflict_resolution_accuracy'])}")
    print(f"Stale-RAG conflicts: {summary['stale_rag_conflicts']}")
    print(f"Stale RAG incorrectly promoted: {summary['stale_rag_incorrectly_promoted']}")
    print(f"Tool-only recoveries: {summary['tool_only_recoveries']}")
    print(f"RAG-only records observed: {summary['rag_only_records_observed']}")
    print(f"Source-reconciliation failures: {summary['source_reconciliation_failures']}")
    print()
    print(f"Total tool calls: {summary['total_tool_calls']}")
    print(f"Redundant tool calls: {summary['redundant_tool_calls']}")
    mean_latency = summary["mean_total_latency_seconds"]
    print(f"Mean total latency: {mean_latency:.6f}s" if mean_latency is not None else "Mean total latency: n/a")

    print()
    print(f"{'scenario':<8} {'status':<26} {'best_line':<10} {'best_odds':<10} {'ev_class':<10} {'conflicts':<10}")
    for r in results:
        print(
            f"{r.scenario_id:<8} {r.execution_status.value:<26} "
            f"{str(r.best_line_correct):<10} {str(r.best_odds_correct):<10} "
            f"{str(r.ev_classification_correct):<10} {r.source_conflicts:<10}"
        )


def _print_failure_demo(agent: HybridAgent) -> None:
    request = AgentRequest(
        scenario_id="DEMO-FAILURE",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Nonexistent Team",
        query="Nonexistent Team moneyline price",
    )
    print("\n--- Explicit failure example (synthetic, not part of the accuracy metrics) ---")
    try:
        agent.analyze(request)
        print("ERROR: expected HybridAnalysisIncomplete")
    except HybridAnalysisIncomplete as exc:
        trace = exc.trace
        print(f"validation_status = {trace.validation_status.value}")
        for call in trace.tool_calls:
            print(f"  {call.tool_name}({call.arguments}) -> success={call.success} error={call.error}")


def main() -> None:
    results = evaluate_scenarios()
    summary = summarize_results(results)

    if "--json" in sys.argv:
        print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
        return

    print_report(results, summary)
    agent, _ = build_default_hybrid_agent()
    _print_failure_demo(agent)


if __name__ == "__main__":
    main()
