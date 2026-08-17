"""Exploratory/secondary subgroup analysis (Milestone 14B, Section 21).

Labeled EXPLORATORY / SECONDARY throughout findings.md — never treated
as a primary, pre-specified comparison. Subgroup labels (category tags
like "tie", "missing_data", "freshness", market_type) come from
data/test_scenarios.json's descriptive `category` field — benchmark
metadata about what KIND of test case a scenario is, not a ground-truth
answer (the same category of information src.experiments.config already
reads for `quant_evaluable` manifest metadata). Never fed to an agent;
used only for post-hoc grouping here.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation import metrics
from src.evaluation.dataset import load_scenario_definitions_by_id
from src.experiments.runner import RawExperimentRun


@dataclass
class SubgroupRow:
    subgroup: str
    architecture: str
    runs: int
    best_line_accuracy: float | None
    freshness_accuracy: float | None
    ev_classification_accuracy: float | None


def _scenario_subgroups(scenario_id: str, definitions_by_id: dict) -> set[str]:
    definition = definitions_by_id[scenario_id]
    categories = set(definition.get("category", []))
    market_type = definition["market"]["market_type"]

    subgroups = {f"market:{market_type}"}
    subgroups.add("tie" if "tie" in categories else "non_tie")
    subgroups.add("missing_data" if "missing_data" in categories else "complete_data")
    subgroups.add("freshness_sensitive" if "freshness" in categories else "normal")
    return subgroups


def subgroup_table(runs: list[RawExperimentRun], quant_evaluable_by_scenario: dict[str, bool]) -> list[SubgroupRow]:
    definitions_by_id = load_scenario_definitions_by_id()

    groups: dict[tuple[str, str], list[RawExperimentRun]] = {}
    for run in runs:
        subgroups = _scenario_subgroups(run.scenario_id, definitions_by_id)
        quant_evaluable = quant_evaluable_by_scenario.get(run.scenario_id)
        subgroups.add("quant_evaluable" if quant_evaluable else "not_quant_evaluable")
        for subgroup in subgroups:
            groups.setdefault((subgroup, run.architecture.value), []).append(run)

    rows = []
    for (subgroup, architecture), group_runs in sorted(groups.items()):
        results = [run.common_result for run in group_runs]
        rows.append(
            SubgroupRow(
                subgroup=subgroup,
                architecture=architecture,
                runs=len(results),
                best_line_accuracy=metrics.rate(r.best_line_correct for r in results),
                freshness_accuracy=metrics.rate(r.freshness_correct for r in results),
                ev_classification_accuracy=metrics.rate(r.ev_classification_correct for r in results),
            )
        )
    return rows
