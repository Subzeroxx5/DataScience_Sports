"""Architecture-level descriptive statistics (Milestone 14B, Section 6).

Every metric FORMULA here is `src.evaluation.metrics` — this module only
adds the extra aggregate views (median, std dev, min, max) Milestone
11's `ArchitectureSummary` doesn't itself expose, computed by feeding
the exact same per-run values / per-scenario consistency signatures
`metrics.summarize()` already derives into the exact same N/A-aware
generic aggregators (`metrics.median`, `metrics.population_stdev`,
`metrics.minimum`, `metrics.maximum`) — never a parallel formula.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation import metrics


@dataclass
class ArchitectureDescriptiveStats:
    architecture: str
    summary: metrics.ArchitectureSummary  # the unmodified Milestone 11 aggregate

    median_ev_absolute_error: float | None
    median_market_reference_absolute_error: float | None
    median_completeness: float | None

    runs_with_unsupported_claim: int

    consistency_mean: float | None
    consistency_median: float | None
    consistency_stdev: float | None
    consistency_min: float | None
    consistency_max: float | None
    consistency_scenario_count: int

    latency_mean: float | None
    latency_median: float | None
    latency_stdev: float | None
    latency_min: float | None
    latency_max: float | None


def _per_scenario_consistency_values(results: list[metrics.EvaluationResult]) -> list[float]:
    """Reproduces exactly the per-scenario consistency list
    metrics.summarize() computes internally (grouping by scenario_id,
    requiring >1 repetition, via the same consistency_signature()/
    compute_consistency() calls) — the only difference is this function
    returns the full list instead of collapsing it to a single mean, so
    std/min/max can be reported too (Section 6)."""
    by_scenario: dict[str, list[metrics.EvaluationResult]] = {}
    for result in results:
        by_scenario.setdefault(result.scenario_id, []).append(result)
    return [
        metrics.compute_consistency([r.consistency_signature() for r in group])
        for group in by_scenario.values()
        if len(group) > 1
    ]


def architecture_descriptive_stats(
    architecture: str, results: list[metrics.EvaluationResult],
) -> ArchitectureDescriptiveStats:
    summary = metrics.summarize(results)

    ev_errors = [r.ev_absolute_error for r in results]
    market_ref_errors = [r.market_reference_absolute_error for r in results]
    completeness_values = [r.completeness for r in results]
    latencies = [r.latency_metrics.total_latency_seconds for r in results]

    consistency_values = _per_scenario_consistency_values(results)

    return ArchitectureDescriptiveStats(
        architecture=architecture,
        summary=summary,
        median_ev_absolute_error=metrics.median(ev_errors),
        median_market_reference_absolute_error=metrics.median(market_ref_errors),
        median_completeness=metrics.median(completeness_values),
        runs_with_unsupported_claim=sum(1 for r in results if r.unsupported_claim_count >= 1),
        consistency_mean=metrics.mean(consistency_values),
        consistency_median=metrics.median(consistency_values),
        consistency_stdev=metrics.population_stdev(consistency_values),
        consistency_min=metrics.minimum(consistency_values),
        consistency_max=metrics.maximum(consistency_values),
        consistency_scenario_count=len(consistency_values),
        latency_mean=metrics.mean(latencies),
        latency_median=metrics.median(latencies),
        latency_stdev=metrics.population_stdev(latencies),
        latency_min=metrics.minimum(latencies),
        latency_max=metrics.maximum(latencies),
    )
