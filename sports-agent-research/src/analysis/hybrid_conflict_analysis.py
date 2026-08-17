"""Hybrid reconciliation/conflict analysis (Milestone 14B, Section 16).

Mechanism analysis, not a separate primary architecture-comparison
metric (Section 16: "Do not treat it as a separate primary research
outcome"). Reuses src.evaluation.hybrid_agent_evaluation.summarize_results()
— the exact Milestone 10B/11 aggregator — fed from the persisted
architecture_specific_result dicts reconstructed into
HybridAgentEvaluationResult via that model's own validation. No new
conflict-resolution formula.
"""

from __future__ import annotations

from src.evaluation.hybrid_agent_evaluation import HybridAgentEvaluationResult, summarize_results
from src.experiments.runner import RawExperimentRun
from src.models import ArchitectureType


def hybrid_conflict_summary(runs: list[RawExperimentRun]) -> dict | None:
    hybrid_runs = [run for run in runs if run.architecture == ArchitectureType.HYBRID]
    if not hybrid_runs:
        return None
    results = [
        HybridAgentEvaluationResult.model_validate(run.architecture_specific_result) for run in hybrid_runs
    ]
    return summarize_results(results)
