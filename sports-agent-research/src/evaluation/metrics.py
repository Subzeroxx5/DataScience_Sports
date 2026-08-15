"""Unified evaluation framework (Milestone 11): one shared metric layer
used by every architecture's evaluator (`rag_agent_evaluation.py`,
`tool_agent_evaluation.py`, `hybrid_agent_evaluation.py`) so RAG-only,
tool-calling, and hybrid agents are measured with identical definitions,
never architecture-specific reimplementations of the same formula
(milestones/current.md, Milestone 11, Step 29: "There should not be
`calculate_rag_best_line_accuracy()` / `calculate_tool_best_line_accuracy()`
/ `calculate_hybrid_best_line_accuracy()` when one shared function is
sufficient").

This module contains only pure functions and typed models — no agent, no
LLM, no ground-truth loading, no I/O. It is the single source of truth
for:

  - the canonical architecture identifier (`src.models.ArchitectureType`,
    reused directly rather than re-invented — Step 3)
  - the unified failure taxonomy (`FailureCategory`, Step 19)
  - the common per-run result model (`EvaluationResult`, Step 2)
  - metric formulas: best-line, best-odds, EV classification, EV error,
    market-reference error, freshness (with stale-match diagnostics),
    completeness, unsupported-claim rate (Steps 5-13)
  - the consistency signature/metric for repeated-run comparison
    (Step 14) — defined and tested here; Milestone 12 is what will
    actually run repetitions
  - generic aggregation (`rate`, `mean`, `median`, `population_stdev`,
    `minimum`, `maximum`) that correctly ignores N/A (`None`) values
    rather than treating them as zero (Step 21)
  - `ArchitectureSummary` / `ArchitectureComparison` (Steps 22-23, 31-32)
    — comparison structures that never declare a "winner"; that
    determination belongs to a future statistical-analysis milestone.

Explicitly NOT here (kept distinct per Step 18): retrieval-quality
metrics (Recall@K / Hit@K) — those already exist in
`src/rag/evaluate_retrieval.py` (Milestone 6C) with their own dedicated
query set, and are never recomputed or duplicated in this module.
`RetrievalMetrics` below tracks retrieval *efficiency* only (how many
documents were retrieved, at what top_k), not retrieval *quality*.
"""

from __future__ import annotations

import statistics
from collections import Counter
from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field

from src.models import ArchitectureType

# ---------------------------------------------------------------------------
# Failure taxonomy (Step 19)
# ---------------------------------------------------------------------------


class FailureCategory(str, Enum):
    """One shared failure/outcome vocabulary across all three
    architectures — never a generic "error", and never an
    architecture-specific synonym for an equivalent failure. Each
    architecture's own evaluator maps its (possibly more granular,
    pre-existing) status onto exactly one of these."""

    SUCCESS = "success"

    RETRIEVAL_FAILURE = "retrieval_failure"
    INSUFFICIENT_RETRIEVED_EVIDENCE = "insufficient_retrieved_evidence"

    TOOL_ARGUMENT_ERROR = "tool_argument_error"
    TOOL_FAILURE = "tool_failure"
    TOOL_LOOP_LIMIT = "tool_loop_limit"

    LLM_OUTPUT_INVALID = "llm_output_invalid"
    PROVENANCE_VALIDATION_FAILURE = "provenance_validation_failure"

    SOURCE_RECONCILIATION_FAILURE = "source_reconciliation_failure"

    INSUFFICIENT_CURRENT_DATA = "insufficient_current_data"
    QUANT_INSUFFICIENT_DATA = "quant_insufficient_data"
    QUANT_VALIDATION_ERROR = "quant_validation_error"

    FINAL_OUTPUT_INVALID = "final_output_invalid"

    UNKNOWN_FAILURE = "unknown_failure"


# A "run" counts as successful for success-rate/accuracy purposes if it
# produced a BettingAnalysis at all — full quant (SUCCESS) or an honest
# best-line-only result (QUANT_INSUFFICIENT_DATA), matching the existing
# tool/hybrid evaluators' precedent (Milestones 9B/10B).
SUCCESSFUL_CATEGORIES = frozenset({FailureCategory.SUCCESS, FailureCategory.QUANT_INSUFFICIENT_DATA})


# ---------------------------------------------------------------------------
# Metric applicability (Step 20)
# ---------------------------------------------------------------------------


class MetricApplicability(BaseModel):
    """Which metric categories apply to a given architecture. Common
    correctness/quality metrics (best-line, best-odds, EV classification
    when quant-evaluable, freshness, completeness, hallucination,
    latency) apply to all three architectures identically; only
    channel-specific efficiency/integrity metrics vary."""

    architecture: ArchitectureType
    best_line: bool = True
    best_odds: bool = True
    ev_classification: bool = True  # only when quant_evaluable=True per scenario
    freshness: bool = True
    completeness: bool = True
    hallucination: bool = True
    latency: bool = True
    retrieval_recall: bool
    tool_calls: bool
    conflict_resolution: bool


METRIC_APPLICABILITY: dict[ArchitectureType, MetricApplicability] = {
    ArchitectureType.RAG: MetricApplicability(
        architecture=ArchitectureType.RAG, retrieval_recall=True, tool_calls=False, conflict_resolution=False
    ),
    ArchitectureType.TOOL: MetricApplicability(
        architecture=ArchitectureType.TOOL, retrieval_recall=False, tool_calls=True, conflict_resolution=False
    ),
    ArchitectureType.HYBRID: MetricApplicability(
        architecture=ArchitectureType.HYBRID, retrieval_recall=True, tool_calls=True, conflict_resolution=True
    ),
}


# ---------------------------------------------------------------------------
# Core correctness metrics (Steps 5-9)
# ---------------------------------------------------------------------------


def best_line_correct(predicted: Iterable[str] | None, expected: Iterable[str]) -> bool | None:
    """Step 5: set-based tie semantics. A partial subset of a tied best
    line is NOT correct — the predicted set must equal the expected set
    exactly. Returns None (not applicable) when no prediction exists."""
    if predicted is None:
        return None
    return set(predicted) == set(expected)


def best_odds_correct(predicted: int | None, expected: int) -> bool | None:
    """Step 6: exact integer equality — American odds, no float tolerance."""
    if predicted is None:
        return None
    return predicted == expected


def ev_classification_correct(predicted: bool | None, expected: bool | None) -> bool | None:
    """Step 7: only defined when both a prediction and a quant-evaluable
    expected classification exist; never scored as incorrect for
    non-quant-evaluable scenarios."""
    if predicted is None or expected is None:
        return None
    return predicted == expected


def ev_absolute_error(predicted: float | None, expected: float | None) -> float | None:
    """Step 8: unrounded abs() error; None (not 0) when unavailable."""
    if predicted is None or expected is None:
        return None
    return abs(predicted - expected)


def market_reference_absolute_error(predicted: float | None, expected: float | None) -> float | None:
    """Step 9: same shape as ev_absolute_error, same applicability rule."""
    if predicted is None or expected is None:
        return None
    return abs(predicted - expected)


# ---------------------------------------------------------------------------
# Freshness (Steps 10, 15, 28)
# ---------------------------------------------------------------------------


class FreshnessDiagnostics(BaseModel):
    """Step 15: enough detail to determine *why* a freshness check
    failed — was the predicted value a known stale snapshot, or simply
    wrong? Never assume "wrong" implies "stale" without support."""

    applicable: bool
    freshness_correct: bool | None
    expected_current_value: int | None
    predicted_value: int | None
    matched_known_stale_value: bool
    stale_value_matched: int | None


def evaluate_freshness(
    predicted_value: int | None,
    expected_current_value: int | None,
    known_stale_value: int | None,
) -> FreshnessDiagnostics:
    """Step 10: freshness is judged from the final authoritative value
    that actually fed the analysis — never merely "was current data
    visible somewhere in the trace." A scenario is only applicable when
    a known current expected value exists (i.e. it's a designated
    freshness scenario); otherwise freshness is not_applicable (None),
    not counted as failure."""
    if expected_current_value is None:
        return FreshnessDiagnostics(
            applicable=False, freshness_correct=None, expected_current_value=None,
            predicted_value=predicted_value, matched_known_stale_value=False, stale_value_matched=None,
        )
    if predicted_value is None:
        # No result at all is a freshness failure, not "not applicable" —
        # the scenario demanded a fresh answer and none was produced.
        return FreshnessDiagnostics(
            applicable=True, freshness_correct=False, expected_current_value=expected_current_value,
            predicted_value=None, matched_known_stale_value=False, stale_value_matched=None,
        )

    matches_current = predicted_value == expected_current_value
    matches_known_stale = known_stale_value is not None and predicted_value == known_stale_value
    correct = matches_current and not matches_known_stale

    return FreshnessDiagnostics(
        applicable=True,
        freshness_correct=correct,
        expected_current_value=expected_current_value,
        predicted_value=predicted_value,
        matched_known_stale_value=matches_known_stale,
        stale_value_matched=known_stale_value if matches_known_stale else None,
    )


# ---------------------------------------------------------------------------
# Completeness (Step 11)
# ---------------------------------------------------------------------------


def completeness(acquired: Iterable[str], available: Iterable[str]) -> float | None:
    """Step 11: fraction of the benchmark's required sportsbook records
    that were validly available in the final analysis. Fabricated
    entries must never be passed in `acquired` by the caller — this
    function trusts its input is already provenance-checked. Returns
    None (not a misleading 0.0) when the benchmark itself specifies zero
    required records."""
    available_set = set(available)
    if not available_set:
        return None
    acquired_set = set(acquired)
    return len(acquired_set & available_set) / len(available_set)


# ---------------------------------------------------------------------------
# Unsupported claims / hallucination (Steps 12-13)
# ---------------------------------------------------------------------------


def unsupported_claim_rate(unsupported_count: int, total_verifiable_claims: int) -> float | None:
    """Step 13: never produce a percentage when the denominator is 0 —
    "0 unsupported out of 0 claims" is not evidence of 0% hallucination,
    it's an undefined ratio."""
    if total_verifiable_claims <= 0:
        return None
    return unsupported_count / total_verifiable_claims


# ---------------------------------------------------------------------------
# Consistency (Step 14, 27)
# ---------------------------------------------------------------------------


class ConsistencySignature(BaseModel):
    """The research-relevant subset of a run's output — deliberately
    excludes latency and free-form reasoning-summary text (Step 14).
    Float fields are rounded to a fixed precision so trivial
    floating-point noise doesn't manufacture false inconsistency."""

    best_sportsbooks: tuple[str, ...]
    best_odds: int | None
    positive_ev: bool | None
    market_reference_probability: float | None
    expected_value: float | None


def compute_consistency_signature(
    *,
    best_sportsbooks: Iterable[str] | None,
    best_odds: int | None,
    positive_ev: bool | None,
    market_reference_probability: float | None,
    expected_value: float | None,
    probability_ndigits: int = 6,
) -> ConsistencySignature:
    return ConsistencySignature(
        best_sportsbooks=tuple(sorted(best_sportsbooks)) if best_sportsbooks is not None else (),
        best_odds=best_odds,
        positive_ev=positive_ev,
        market_reference_probability=(
            round(market_reference_probability, probability_ndigits)
            if market_reference_probability is not None
            else None
        ),
        expected_value=(round(expected_value, probability_ndigits) if expected_value is not None else None),
    )


def compute_consistency(signatures: list[ConsistencySignature]) -> float | None:
    """Step 14/27: primary consistency measure — modal_signature_count /
    total_runs. 1.0 means every repeated run produced identical
    research-relevant output; lower values indicate drift. None when no
    runs are given."""
    if not signatures:
        return None
    keys = [
        (sig.best_sportsbooks, sig.best_odds, sig.positive_ev, sig.market_reference_probability, sig.expected_value)
        for sig in signatures
    ]
    counts = Counter(keys)
    modal_count = max(counts.values())
    return modal_count / len(keys)


# ---------------------------------------------------------------------------
# Generic aggregation (Step 21) — all ignore None (N/A) rather than
# treating it as zero.
# ---------------------------------------------------------------------------


def count(values: Iterable) -> int:
    return sum(1 for _ in values)


def valid_count(values: Iterable) -> int:
    return sum(1 for v in values if v is not None)


def rate(values: Iterable[bool | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    return (sum(1 for v in filtered if v) / len(filtered)) if filtered else None


def mean(values: Iterable[float | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    return (sum(filtered) / len(filtered)) if filtered else None


def median(values: Iterable[float | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    return statistics.median(filtered) if filtered else None


def population_stdev(values: Iterable[float | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    if len(filtered) == 1:
        return 0.0
    return statistics.pstdev(filtered)


def minimum(values: Iterable[float | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    return min(filtered) if filtered else None


def maximum(values: Iterable[float | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    return max(filtered) if filtered else None


# ---------------------------------------------------------------------------
# Latency / efficiency (Steps 16-17)
# ---------------------------------------------------------------------------


class LatencyMetrics(BaseModel):
    """Standardized latency fields (Step 16). Only `total_latency_seconds`
    is required — every architecture always has a total. The others are
    None (not a fake 0.0) when that phase does not apply to this
    architecture's pipeline. `reconciliation_latency_seconds` is a
    hybrid-specific addition beyond the milestone's core 5-field list,
    since dropping that real, already-measured phase would lose
    information hybrid's own trace provides."""

    retrieval_latency_seconds: float | None = None
    llm_latency_seconds: float | None = None
    tool_latency_seconds: float | None = None
    reconciliation_latency_seconds: float | None = None
    quant_latency_seconds: float | None = None
    total_latency_seconds: float


class RetrievalMetrics(BaseModel):
    """Retrieval *efficiency* only (Step 17) — document count and
    configured depth. Retrieval *quality* (Recall@K/Hit@K) is
    intentionally kept out of this model; see module docstring."""

    retrieved_document_count: int
    retrieval_top_k: int


class ToolMetrics(BaseModel):
    """Step 17."""

    tool_call_count: int
    unique_tool_call_count: int
    redundant_tool_call_count: int
    failed_tool_call_count: int


# ---------------------------------------------------------------------------
# Common per-run result (Step 2)
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """One scenario/repetition's evaluation outcome, in the SAME shape
    regardless of which architecture produced it. Not every architecture
    populates every architecture-specific field (`retrieval_metrics`,
    `tool_metrics`) — None means genuinely not applicable, never a fake
    zero (Step 2)."""

    scenario_id: str
    architecture: ArchitectureType
    execution_status: FailureCategory
    quant_evaluable: bool

    predicted_best_sportsbooks: list[str] = Field(default_factory=list)
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

    unsupported_claim_count: int
    total_verifiable_claims: int
    hallucination_detected: bool

    retrieval_metrics: RetrievalMetrics | None = None
    tool_metrics: ToolMetrics | None = None
    latency_metrics: LatencyMetrics

    errors: list[str] = Field(default_factory=list)

    def consistency_signature(self) -> ConsistencySignature:
        return compute_consistency_signature(
            best_sportsbooks=self.predicted_best_sportsbooks,
            best_odds=self.predicted_best_odds,
            positive_ev=self.predicted_positive_ev,
            market_reference_probability=self.predicted_market_reference_probability,
            expected_value=self.predicted_ev,
        )


# ---------------------------------------------------------------------------
# Architecture summary / cross-architecture comparison (Steps 22-23, 31-32)
# ---------------------------------------------------------------------------


class ArchitectureSummary(BaseModel):
    """Aggregate statistics for one architecture's evaluation run,
    together with the full raw per-run results it was computed from
    (Step 32: "Do not store only summary averages... retain per-run
    data")."""

    architecture: ArchitectureType
    raw_results: list[EvaluationResult]

    runs: int
    successes: int
    failures: int
    success_rate: float | None

    best_line_accuracy: float | None
    best_odds_accuracy: float | None

    ev_classification_accuracy: float | None
    mean_ev_absolute_error: float | None

    market_reference_mae: float | None

    freshness_accuracy: float | None
    mean_completeness: float | None

    unsupported_claim_rate: float | None

    # None until >=1 scenario has more than one repetition in raw_results
    # (Milestone 11 defines/tests this; Milestone 12 supplies real repeated
    # runs).
    consistency: float | None

    mean_total_latency_seconds: float | None

    failure_counts: dict[str, int]


def summarize(results: list[EvaluationResult]) -> ArchitectureSummary:
    if not results:
        raise ValueError("summarize() requires at least one EvaluationResult")

    architecture = results[0].architecture

    failure_counts: dict[str, int] = {}
    for result in results:
        failure_counts[result.execution_status.value] = failure_counts.get(result.execution_status.value, 0) + 1

    successes = sum(1 for result in results if result.execution_status in SUCCESSFUL_CATEGORIES)

    by_scenario: dict[str, list[EvaluationResult]] = {}
    for result in results:
        by_scenario.setdefault(result.scenario_id, []).append(result)
    per_scenario_consistency = [
        compute_consistency([r.consistency_signature() for r in group])
        for group in by_scenario.values()
        if len(group) > 1
    ]
    consistency = mean(per_scenario_consistency) if per_scenario_consistency else None

    return ArchitectureSummary(
        architecture=architecture,
        raw_results=results,
        runs=len(results),
        successes=successes,
        failures=len(results) - successes,
        success_rate=rate(result.execution_status in SUCCESSFUL_CATEGORIES for result in results),
        best_line_accuracy=rate(result.best_line_correct for result in results),
        best_odds_accuracy=rate(result.best_odds_correct for result in results),
        ev_classification_accuracy=rate(result.ev_classification_correct for result in results),
        mean_ev_absolute_error=mean(result.ev_absolute_error for result in results),
        market_reference_mae=mean(result.market_reference_absolute_error for result in results),
        freshness_accuracy=rate(result.freshness_correct for result in results),
        mean_completeness=mean(result.completeness for result in results),
        unsupported_claim_rate=unsupported_claim_rate(
            sum(result.unsupported_claim_count for result in results),
            sum(result.total_verifiable_claims for result in results),
        ),
        consistency=consistency,
        mean_total_latency_seconds=mean(result.latency_metrics.total_latency_seconds for result in results),
        failure_counts=failure_counts,
    )


class ArchitectureComparison(BaseModel):
    """Cross-architecture comparison container (Step 23). Deliberately
    holds only data — no "winner" field, no encoded expectation about
    which architecture performs best. That determination is reserved for
    a future statistical-analysis milestone."""

    rag_summary: ArchitectureSummary | None = None
    tool_summary: ArchitectureSummary | None = None
    hybrid_summary: ArchitectureSummary | None = None

    scenario_count: int
    repetitions: int


def compare_architectures(
    rag_results: list[EvaluationResult] | None = None,
    tool_results: list[EvaluationResult] | None = None,
    hybrid_results: list[EvaluationResult] | None = None,
) -> ArchitectureComparison:
    scenario_ids: set[str] = set()
    max_repetitions = 0
    for results in (rag_results, tool_results, hybrid_results):
        if not results:
            continue
        scenario_ids.update(result.scenario_id for result in results)
        repetition_counts = Counter(result.scenario_id for result in results)
        max_repetitions = max(max_repetitions, max(repetition_counts.values()))

    return ArchitectureComparison(
        rag_summary=summarize(rag_results) if rag_results else None,
        tool_summary=summarize(tool_results) if tool_results else None,
        hybrid_summary=summarize(hybrid_results) if hybrid_results else None,
        scenario_count=len(scenario_ids),
        repetitions=max_repetitions,
    )
