"""Assembles every computed analysis artifact into one bundle
(Milestone 14B) so tables/figures/findings.md all consume a single,
already-computed source of truth rather than each recomputing anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.analysis.comparisons import (
    BinaryComparison,
    ContinuousComparison,
    OmnibusComparison,
    compute_binary_comparisons,
    compute_continuous_comparisons,
    compute_omnibus_comparisons,
)
from src.analysis.descriptive import ArchitectureDescriptiveStats, architecture_descriptive_stats
from src.analysis.failure_analysis import FailureCategoryBreakdown, failure_breakdown
from src.analysis.freshness_analysis import FreshnessArchitectureStats, freshness_stats_by_architecture
from src.analysis.hallucination_analysis import HallucinationArchitectureStats, hallucination_stats_by_architecture
from src.analysis.hybrid_conflict_analysis import hybrid_conflict_summary
from src.analysis.latency_analysis import PhaseLatencyStats, architecture_phase_latency
from src.analysis.loading import LoadedFinalDataset
from src.analysis.pairing import group_by_architecture
from src.analysis.scenario_analysis import ScenarioArchitectureRow, scenario_architecture_table
from src.analysis.subgroups import SubgroupRow, subgroup_table
from src.models import ArchitectureType

ARCHITECTURES = [ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID]


@dataclass
class AnalysisBundle:
    dataset: LoadedFinalDataset

    descriptive_stats: dict[str, ArchitectureDescriptiveStats]
    phase_latency: dict[str, list[PhaseLatencyStats]]
    freshness_stats: dict[str, FreshnessArchitectureStats]
    hallucination_stats: dict[str, HallucinationArchitectureStats]

    binary_comparisons: list[BinaryComparison]
    continuous_comparisons: list[ContinuousComparison]
    omnibus_comparisons: list[OmnibusComparison]

    failure_breakdown: list[FailureCategoryBreakdown]
    hybrid_conflict: dict | None

    scenario_rows: list[ScenarioArchitectureRow]
    subgroup_rows: list[SubgroupRow]


def build_analysis_bundle(dataset: LoadedFinalDataset) -> AnalysisBundle:
    runs = dataset.raw_runs
    by_architecture = group_by_architecture(runs)

    descriptive_stats = {
        architecture.value: architecture_descriptive_stats(
            architecture.value, [run.common_result for run in by_architecture.get(architecture, [])]
        )
        for architecture in ARCHITECTURES
    }
    phase_latency = {
        architecture.value: architecture_phase_latency(architecture, by_architecture.get(architecture, []))
        for architecture in ARCHITECTURES
    }
    freshness_stats = {
        architecture.value: freshness_stats_by_architecture(
            architecture.value, by_architecture.get(architecture, []), dataset.manifest
        )
        for architecture in ARCHITECTURES
    }
    hallucination_stats = {
        architecture.value: hallucination_stats_by_architecture(architecture.value, by_architecture.get(architecture, []))
        for architecture in ARCHITECTURES
    }

    quant_evaluable_by_scenario = {scenario.scenario_id: scenario.quant_evaluable for scenario in dataset.manifest}

    return AnalysisBundle(
        dataset=dataset,
        descriptive_stats=descriptive_stats,
        phase_latency=phase_latency,
        freshness_stats=freshness_stats,
        hallucination_stats=hallucination_stats,
        binary_comparisons=compute_binary_comparisons(runs),
        continuous_comparisons=compute_continuous_comparisons(runs),
        omnibus_comparisons=compute_omnibus_comparisons(runs),
        failure_breakdown=failure_breakdown(runs),
        hybrid_conflict=hybrid_conflict_summary(runs),
        scenario_rows=scenario_architecture_table(runs),
        subgroup_rows=subgroup_table(runs, quant_evaluable_by_scenario),
    )
