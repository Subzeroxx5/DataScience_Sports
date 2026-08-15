"""End-to-end evaluation harness for the RAG-only agent (Milestone 11):
drives RagOnlyAgent (Milestone 8B) across a representative
controlled-benchmark subset and compares its output against
deterministic ground truth, entirely outside the agent:

    AgentRequest
        |
        v
    RagOnlyAgent               (src/agents/rag_agent.py, unmodified)
        |
        v
    BettingAnalysis + RagAgentTrace
        |
        v
    Evaluator (this module)    <-- ground truth enters HERE only
        |
        v
    RagAgentEvaluationResult

This module fills a gap noted in milestones/current.md (Milestone 11's
objective): the RAG-only agent had no dedicated `*_evaluation.py`
harness (it was previously exercised only ad hoc via
experiments/run_rag_smoke_test.py, Milestone 8B). It follows exactly the
same shape as src/evaluation/tool_agent_evaluation.py (Milestone 9B) and
src/evaluation/hybrid_agent_evaluation.py (Milestone 10B/11).

Ground-truth isolation (docs/EXPERIMENT_RULES.md, "Ground Truth"): this
module is the only place in the RAG-only evaluation path that reads
GroundTruth/QuantGroundTruth. Nothing here ever writes an expected value
into an AgentRequest, RAG prompt, or quant input — every comparison
happens strictly after RagOnlyAgent.analyze() has already returned.

Reuse (Milestone 11's unified framework, Step 3/30): metric formulas
(best-line/best-odds/EV classification/EV error/market-reference error/
freshness/completeness/generic aggregation) come from
src.evaluation.metrics — the same shared implementation used identically
by the tool-calling and hybrid evaluators. `execution_status` is
`metrics.FailureCategory` directly (no intermediate architecture-
specific enum, since this is new code with no pre-existing field type to
preserve). The RAG-extraction fake-LLM policy
(`extract_honest_rag_evidence`) is imported from
src.evaluation.hybrid_agent_evaluation rather than duplicated — both
RagOnlyAgent's and HybridAgent's rendered RAG user-prompts use the
identical format (see src/agents/rag_agent.py::_render_user_prompt vs.
src/agents/hybrid_agent.py::_render_rag_user_prompt), so one parser
serves both.

Run as a script for a deterministic, mock-LLM, CI-free evaluation report:

    python -m src.evaluation.rag_agent_evaluation
    python -m src.evaluation.rag_agent_evaluation --json
"""

from __future__ import annotations

import json
import sys

from pydantic import BaseModel

from src.agents.base import AgentRequest
from src.agents.rag_agent import RagAgentTrace, RagAnalysisIncomplete, RagOnlyAgent
from src.agents.rag_evidence import build_rag_evidence_bundle
from src.evaluation import metrics
from src.evaluation.dataset import load_scenario_definitions_by_id
from src.evaluation.ground_truth import generate_all_ground_truth
from src.evaluation.hybrid_agent_evaluation import _build_agent_request, extract_honest_rag_evidence
from src.evaluation.quant_ground_truth import generate_all_quant_ground_truth
from src.evaluation.tool_agent_evaluation import DEFAULT_SCENARIO_IDS, _stale_odds_by_scenario_key
from src.models import AnalysisStatus, ArchitectureType, BettingAnalysis, GroundTruth, MarketType, QuantGroundTruth
from src.rag.documents import RagSourceType
from src.rag.retriever import Retriever


class RagAgentEvaluationResult(BaseModel):
    """One scenario's evaluation outcome. Field names/semantics for the
    common metrics intentionally match ToolAgentEvaluationResult and
    HybridAgentEvaluationResult so cross-architecture comparison never
    has to reconcile incompatible definitions."""

    scenario_id: str
    execution_status: metrics.FailureCategory
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

    retrieved_document_count: int
    retrieval_top_k: int

    retrieval_latency_seconds: float
    llm_latency_seconds: float
    quant_latency_seconds: float
    total_latency_seconds: float

    errors: list[str]


def to_common_result(result: RagAgentEvaluationResult) -> metrics.EvaluationResult:
    """Convert this evaluator's result into the unified
    src.evaluation.metrics.EvaluationResult shape (Milestone 11, Step 2)
    for cross-architecture comparison (Step 23)."""
    return metrics.EvaluationResult(
        scenario_id=result.scenario_id,
        architecture=ArchitectureType.RAG,
        execution_status=result.execution_status,
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
            retrieved_document_count=result.retrieved_document_count,
            retrieval_top_k=result.retrieval_top_k,
        ),
        tool_metrics=None,
        latency_metrics=metrics.LatencyMetrics(
            retrieval_latency_seconds=result.retrieval_latency_seconds,
            llm_latency_seconds=result.llm_latency_seconds,
            quant_latency_seconds=result.quant_latency_seconds,
            total_latency_seconds=result.total_latency_seconds,
        ),
        errors=result.errors,
    )


# ---------------------------------------------------------------------------
# Deterministic fake LLM policy
# ---------------------------------------------------------------------------


class DeterministicRagPolicyLLMClient:
    """Fake LLMClient: honestly, deterministically extracts whatever RAG
    evidence was actually retrieved (never GroundTruth/QuantGroundTruth)
    — delegates entirely to
    src.evaluation.hybrid_agent_evaluation.extract_honest_rag_evidence,
    the exact same parser HybridAgent's evaluator uses, since both
    agents render an identically-formatted RAG user prompt."""

    model = "deterministic-fake-rag-policy"
    effort = "low"

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        return extract_honest_rag_evidence(user_prompt)


# ---------------------------------------------------------------------------
# Harness construction
# ---------------------------------------------------------------------------

# RagOnlyAgent's own class-level default (src/agents/rag_evidence.py::
# DEFAULT_RAG_TOP_K = 5) is unmodified and unrelated to this constant —
# this is an *evaluation-configuration* choice (which top_k this harness
# passes in), not an agent-behavior change. At k=5, none of the 4
# quant-evaluable scenarios in DEFAULT_SCENARIO_IDS retrieve enough of
# the controlled corpus's 4-book two-sided market to reach full quant
# (a genuine, honestly-measured finding about single-pass, fixed-k
# retrieval — not something this evaluator works around). k=10 matches
# the depth already established as necessary for reliable two-sided
# coverage of a 4-book game elsewhere in this project (e.g.
# tests/test_rag_agent.py); using it here keeps EV-classification/
# market-reference comparisons meaningful across architectures rather
# than "not applicable" for nearly every scenario.
RAG_EVALUATION_TOP_K = 10


def build_default_rag_agent(
    llm_client=None, top_k: int = RAG_EVALUATION_TOP_K
) -> tuple[RagOnlyAgent, Retriever]:
    retriever = Retriever.from_directory()
    agent = RagOnlyAgent(retriever, llm_client=llm_client or DeterministicRagPolicyLLMClient(), top_k=top_k)
    return agent, retriever


def _detect_hallucination(
    retriever: Retriever, request: AgentRequest, analysis: BettingAnalysis, top_k: int
) -> bool:
    """Independently re-retrieve (same query, same top_k) and confirm
    the analysis's headline claim (best_sportsbook/best_odds) is
    traceable to an actually-retrieved sportsbook_snapshot document —
    never trusts the agent's own trace (Step 12: "For RAG: must be
    traceable to retrieved RAG evidence")."""
    bundle = build_rag_evidence_bundle(request, retriever, k=top_k)
    for item in bundle.evidence:
        document = item.document
        if (
            document.source_type == RagSourceType.SPORTSBOOK_SNAPSHOT
            and document.sportsbook == analysis.best_sportsbook
            and document.selected_outcome == request.selected_outcome
            and document.american_odds == analysis.best_odds
        ):
            return False
    return True


def _classify_failure(trace: RagAgentTrace | None) -> metrics.FailureCategory:
    if trace is None:
        return metrics.FailureCategory.UNKNOWN_FAILURE
    if trace.validation_status == "extraction_failed":
        return metrics.FailureCategory.LLM_OUTPUT_INVALID
    if trace.validation_status == "no_valid_prices":
        if trace.extraction_result is not None and trace.extraction_result.sportsbook_prices:
            # The LLM claimed prices, but every one was rejected by
            # validate_extraction_provenance (hallucinated sportsbook/
            # odds/source_document_id) — distinct from simply finding
            # nothing to extract.
            return metrics.FailureCategory.PROVENANCE_VALIDATION_FAILURE
        return metrics.FailureCategory.INSUFFICIENT_RETRIEVED_EVIDENCE
    return metrics.FailureCategory.UNKNOWN_FAILURE


def evaluate_scenario(
    agent: RagOnlyAgent,
    retriever: Retriever,
    request: AgentRequest,
    ground_truth: GroundTruth,
    quant_ground_truth: QuantGroundTruth,
    stale_odds_map: dict[tuple[str, str, str], dict[str, int]] | None = None,
) -> RagAgentEvaluationResult:
    stale_odds_map = stale_odds_map or {}
    analysis: BettingAnalysis | None = None
    errors: list[str] = []

    try:
        analysis = agent.analyze(request)
    except RagAnalysisIncomplete as exc:
        trace = exc.trace
        execution_status = _classify_failure(trace)
        errors = list(trace.errors)
    except Exception as exc:  # defensive: an unexpected internal crash
        trace = agent.last_trace
        execution_status = metrics.FailureCategory.UNKNOWN_FAILURE
        errors = [repr(exc)]
    else:
        trace = agent.last_trace
        try:
            BettingAnalysis.model_validate(analysis.model_dump())
        except Exception as exc:
            execution_status = metrics.FailureCategory.FINAL_OUTPUT_INVALID
            errors = [repr(exc)]
        else:
            execution_status = (
                metrics.FailureCategory.SUCCESS
                if analysis.status == AnalysisStatus.OK
                else metrics.FailureCategory.QUANT_INSUFFICIENT_DATA
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
        _detect_hallucination(retriever, request, analysis, agent.top_k) if analysis is not None else False
    )

    return RagAgentEvaluationResult(
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
        retrieved_document_count=len(trace.retrieved_document_ids) if trace is not None else 0,
        retrieval_top_k=trace.top_k if trace is not None else agent.top_k,
        retrieval_latency_seconds=trace.retrieval_latency_seconds if trace is not None else 0.0,
        llm_latency_seconds=trace.llm_latency_seconds if trace is not None else 0.0,
        quant_latency_seconds=trace.quant_latency_seconds if trace is not None else 0.0,
        total_latency_seconds=trace.total_latency_seconds if trace is not None else 0.0,
        errors=errors,
    )


def evaluate_scenarios(
    scenario_ids: list[str] | None = None, llm_client=None
) -> list[RagAgentEvaluationResult]:
    scenario_ids = scenario_ids or DEFAULT_SCENARIO_IDS
    agent, retriever = build_default_rag_agent(llm_client)

    ground_truth_by_id = {gt.scenario_id: gt for gt in generate_all_ground_truth()}
    quant_ground_truth_by_id = {qgt.scenario_id: qgt for qgt in generate_all_quant_ground_truth()}
    scenario_definitions_by_id = load_scenario_definitions_by_id()
    stale_odds_map = _stale_odds_by_scenario_key()

    results = []
    for scenario_id in scenario_ids:
        request = _build_agent_request(scenario_definitions_by_id[scenario_id])
        result = evaluate_scenario(
            agent,
            retriever,
            request,
            ground_truth_by_id[scenario_id],
            quant_ground_truth_by_id[scenario_id],
            stale_odds_map,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Summary / report
# ---------------------------------------------------------------------------


def summarize_results(results: list[RagAgentEvaluationResult]) -> dict:
    total = len(results)
    successful = [r for r in results if r.execution_status in metrics.SUCCESSFUL_CATEGORIES]

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
        "total_documents_retrieved": sum(r.retrieved_document_count for r in results),
        "mean_total_latency_seconds": metrics.mean(r.total_latency_seconds for r in results),
    }


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def print_report(results: list[RagAgentEvaluationResult], summary: dict) -> None:
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
    print(f"Total documents retrieved: {summary['total_documents_retrieved']}")
    mean_latency = summary["mean_total_latency_seconds"]
    print(f"Mean total latency: {mean_latency:.6f}s" if mean_latency is not None else "Mean total latency: n/a")

    print()
    print(f"{'scenario':<8} {'status':<26} {'best_line':<10} {'best_odds':<10} {'ev_class':<10}")
    for r in results:
        print(
            f"{r.scenario_id:<8} {r.execution_status.value:<26} "
            f"{str(r.best_line_correct):<10} {str(r.best_odds_correct):<10} {str(r.ev_classification_correct):<10}"
        )


def _print_failure_demo(agent: RagOnlyAgent) -> None:
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
        print("ERROR: expected RagAnalysisIncomplete")
    except RagAnalysisIncomplete as exc:
        trace = exc.trace
        print(f"validation_status = {trace.validation_status}")
        print(f"execution_status  = {_classify_failure(trace).value}")


def main() -> None:
    results = evaluate_scenarios()
    summary = summarize_results(results)

    if "--json" in sys.argv:
        print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
        return

    print_report(results, summary)
    agent, _ = build_default_rag_agent()
    _print_failure_demo(agent)


if __name__ == "__main__":
    main()
