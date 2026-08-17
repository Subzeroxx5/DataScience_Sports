"""Failure analysis (Milestone 14B, Section 15).

Groups already-persisted execution_status values (the Milestone 11
FailureCategory taxonomy) by architecture — a plain tally of an
already-computed categorical field, never a new failure classification
or a redefinition of what counts as a failure (metrics.SUCCESSFUL_CATEGORIES
is reused as-is to determine which categories are "failures").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.evaluation import metrics
from src.experiments.runner import RawExperimentRun


@dataclass
class FailureCategoryBreakdown:
    architecture: str
    category: str
    count: int
    percentage_of_observations: float
    scenarios_affected: list[str] = field(default_factory=list)


def failure_breakdown(runs: list[RawExperimentRun]) -> list[FailureCategoryBreakdown]:
    by_arch_category: dict[tuple[str, str], list[RawExperimentRun]] = {}
    totals_by_arch: dict[str, int] = {}

    for run in runs:
        architecture = run.architecture.value
        totals_by_arch[architecture] = totals_by_arch.get(architecture, 0) + 1
        if run.common_result.execution_status in metrics.SUCCESSFUL_CATEGORIES:
            continue
        category = run.common_result.execution_status.value
        by_arch_category.setdefault((architecture, category), []).append(run)

    breakdown = []
    for (architecture, category), category_runs in sorted(by_arch_category.items()):
        breakdown.append(
            FailureCategoryBreakdown(
                architecture=architecture,
                category=category,
                count=len(category_runs),
                percentage_of_observations=len(category_runs) / totals_by_arch[architecture] * 100,
                scenarios_affected=sorted({run.scenario_id for run in category_runs}),
            )
        )
    return breakdown
