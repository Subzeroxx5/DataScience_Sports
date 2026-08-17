"""Pre-specified pairwise/omnibus architecture comparisons (Milestone
14B, Sections 8-12). This is the one place the metric families, the
three architecture pairs, and the Holm-correction grouping are defined
— frozen BEFORE inspecting which comparisons turn out significant
(Section 9: "Do not choose correction method after inspecting which
makes results significant" — the metric family list itself is likewise
fixed here, not assembled after seeing results).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.analysis.omnibus_tests import FriedmanResult, friedman_test
from src.analysis.pairing import align_pair, align_three_way
from src.analysis.pairwise_tests import McNemarResult, WilcoxonResult, holm_correction, mcnemar_test, wilcoxon_signed_rank
from src.experiments.runner import RawExperimentRun
from src.models import ArchitectureType

ARCHITECTURE_PAIRS: list[tuple[ArchitectureType, ArchitectureType]] = [
    (ArchitectureType.RAG, ArchitectureType.TOOL),
    (ArchitectureType.RAG, ArchitectureType.HYBRID),
    (ArchitectureType.TOOL, ArchitectureType.HYBRID),
]

BINARY_METRICS: dict[str, str] = {
    "best_line_correct": "best_line_correct",
    "best_odds_correct": "best_odds_correct",
    "ev_classification_correct": "ev_classification_correct",
    "freshness_correct": "freshness_correct",
}

CONTINUOUS_METRICS: dict[str, str] = {
    "ev_absolute_error": "ev_absolute_error",
    "market_reference_absolute_error": "market_reference_absolute_error",
    "completeness": "completeness",
    "total_latency_seconds": "total_latency_seconds",
}


def _binary_value_fn(field_name: str):
    def value_fn(run: RawExperimentRun) -> bool | None:
        return getattr(run.common_result, field_name)
    return value_fn


def _continuous_value_fn(field_name: str):
    if field_name == "total_latency_seconds":
        def value_fn(run: RawExperimentRun) -> float | None:
            return run.common_result.latency_metrics.total_latency_seconds
        return value_fn

    def value_fn(run: RawExperimentRun) -> float | None:
        return getattr(run.common_result, field_name)
    return value_fn


@dataclass
class BinaryComparison:
    metric: str
    architecture_a: str
    architecture_b: str
    result: McNemarResult | None
    raw_p: float | None
    holm_adjusted_p: float | None


@dataclass
class ContinuousComparison:
    metric: str
    architecture_a: str
    architecture_b: str
    result: WilcoxonResult | None
    raw_p: float | None
    holm_adjusted_p: float | None


def compute_binary_comparisons(runs: list[RawExperimentRun]) -> list[BinaryComparison]:
    all_comparisons: list[BinaryComparison] = []
    for metric, field_name in BINARY_METRICS.items():
        family_results: list[tuple[tuple[ArchitectureType, ArchitectureType], McNemarResult | None]] = []
        for arch_a, arch_b in ARCHITECTURE_PAIRS:
            pairs, _dropped = align_pair(runs, arch_a, arch_b, _binary_value_fn(field_name))
            family_results.append(((arch_a, arch_b), mcnemar_test(pairs)))

        raw_p_values = [result.p_value if result is not None else None for _pair, result in family_results]
        adjusted = holm_correction(raw_p_values)

        for (arch_a, arch_b), result in family_results:
            index = [pair for pair, _ in family_results].index((arch_a, arch_b))
            all_comparisons.append(
                BinaryComparison(
                    metric=metric, architecture_a=arch_a.value, architecture_b=arch_b.value,
                    result=result,
                    raw_p=result.p_value if result is not None else None,
                    holm_adjusted_p=adjusted[index],
                )
            )
    return all_comparisons


def compute_continuous_comparisons(runs: list[RawExperimentRun]) -> list[ContinuousComparison]:
    all_comparisons: list[ContinuousComparison] = []
    for metric, field_name in CONTINUOUS_METRICS.items():
        family_results: list[tuple[tuple[ArchitectureType, ArchitectureType], WilcoxonResult | None]] = []
        for arch_a, arch_b in ARCHITECTURE_PAIRS:
            pairs, _dropped = align_pair(runs, arch_a, arch_b, _continuous_value_fn(field_name))
            family_results.append(((arch_a, arch_b), wilcoxon_signed_rank(pairs)))

        raw_p_values = [result.p_value if result is not None else None for _pair, result in family_results]
        adjusted = holm_correction(raw_p_values)

        for (arch_a, arch_b), result in family_results:
            index = [pair for pair, _ in family_results].index((arch_a, arch_b))
            all_comparisons.append(
                ContinuousComparison(
                    metric=metric, architecture_a=arch_a.value, architecture_b=arch_b.value,
                    result=result,
                    raw_p=result.p_value if result is not None else None,
                    holm_adjusted_p=adjusted[index],
                )
            )
    return all_comparisons


@dataclass
class OmnibusComparison:
    metric: str
    result: FriedmanResult | None


def compute_omnibus_comparisons(runs: list[RawExperimentRun]) -> list[OmnibusComparison]:
    """Friedman omnibus over matched (scenario_id, repetition) triples
    for completeness/latency, and over matched scenario_id triples for
    consistency (Section 12)."""
    comparisons: list[OmnibusComparison] = []

    for metric, field_name in (("completeness", "completeness"), ("total_latency_seconds", "total_latency_seconds")):
        triples, _dropped = align_three_way(runs, _continuous_value_fn(field_name))
        comparisons.append(OmnibusComparison(metric=metric, result=friedman_test(triples)))

    consistency_triples = _consistency_triples_by_scenario(runs)
    comparisons.append(OmnibusComparison(metric="consistency", result=friedman_test(consistency_triples)))

    return comparisons


def _consistency_triples_by_scenario(runs: list[RawExperimentRun]) -> list[tuple[float, float, float]]:
    from src.analysis.pairing import group_by_architecture_and_scenario
    from src.evaluation import metrics

    grouped = group_by_architecture_and_scenario(runs)
    scenario_ids = (
        set(grouped.get(ArchitectureType.RAG, {}))
        & set(grouped.get(ArchitectureType.TOOL, {}))
        & set(grouped.get(ArchitectureType.HYBRID, {}))
    )

    triples: list[tuple[float, float, float]] = []
    for scenario_id in sorted(scenario_ids):
        values = []
        for architecture in (ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID):
            scenario_runs = grouped[architecture][scenario_id]
            if len(scenario_runs) <= 1:
                values.append(None)
                continue
            signatures = [run.common_result.consistency_signature() for run in scenario_runs]
            values.append(metrics.compute_consistency(signatures))
        if all(v is not None for v in values):
            triples.append(tuple(values))  # type: ignore[arg-type]
    return triples
