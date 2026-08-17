"""Per-scenario, per-architecture summary (Milestone 14B, Section 20).

Reuses metrics.rate()/compute_consistency() per (architecture, scenario)
group — never a new formula, just a finer aggregation granularity than
the architecture-level ArchitectureSummary.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation import metrics
from src.experiments.runner import RawExperimentRun


@dataclass
class ScenarioArchitectureRow:
    scenario_id: str
    architecture: str
    runs: int
    best_line_accuracy: float | None
    ev_classification_accuracy: float | None
    freshness_accuracy: float | None
    consistency: float | None
    failures: int


def scenario_architecture_table(runs: list[RawExperimentRun]) -> list[ScenarioArchitectureRow]:
    groups: dict[tuple[str, str], list[RawExperimentRun]] = {}
    for run in runs:
        groups.setdefault((run.scenario_id, run.architecture.value), []).append(run)

    rows = []
    for (scenario_id, architecture), group_runs in sorted(groups.items()):
        results = [run.common_result for run in group_runs]
        failures = sum(1 for r in results if r.execution_status not in metrics.SUCCESSFUL_CATEGORIES)
        signatures = [r.consistency_signature() for r in results]
        consistency = metrics.compute_consistency(signatures) if len(results) > 1 else None

        rows.append(
            ScenarioArchitectureRow(
                scenario_id=scenario_id,
                architecture=architecture,
                runs=len(results),
                best_line_accuracy=metrics.rate(r.best_line_correct for r in results),
                ev_classification_accuracy=metrics.rate(r.ev_classification_correct for r in results),
                freshness_accuracy=metrics.rate(r.freshness_correct for r in results),
                consistency=consistency,
                failures=failures,
            )
        )
    return rows
