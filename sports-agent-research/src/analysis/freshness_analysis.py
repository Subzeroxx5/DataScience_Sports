"""Freshness-focused analysis (Milestone 14B, Section 13).

The "freshness-focused subset" is defined operationally exactly as
Milestone 11 already defines it: every run where
common_result.freshness_correct is not None (i.e.
metrics.evaluate_freshness() judged the scenario "applicable" —
src/evaluation/metrics.py). No new freshness definition, no hardcoded
scenario ID.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.analysis.confidence_intervals import WilsonInterval, wilson_score_interval
from src.evaluation import metrics
from src.evaluation.tool_agent_evaluation import _stale_odds_by_scenario_key
from src.experiments.config import ExperimentScenario
from src.experiments.runner import RawExperimentRun


@dataclass
class FreshnessArchitectureStats:
    architecture: str
    cases_evaluated: int
    correct: int
    incorrect: int
    accuracy: float | None
    confidence_interval: WilsonInterval | None
    used_known_stale_value: int
    used_unknown_incorrect_value: int
    used_correct_current_value: int


def freshness_focused_runs(runs: list[RawExperimentRun]) -> list[RawExperimentRun]:
    return [run for run in runs if run.common_result.freshness_correct is not None]


def _classify_error(run: RawExperimentRun, stale_odds_for_scenario: dict[str, int] | None) -> str:
    """Section 13: "used known stale value" vs "used unknown incorrect
    value" — determined only from already-persisted fields
    (predicted_best_odds) compared against THIS scenario's known-stale
    sportsbook odds (data/historical_odds.json, via the exact
    tool_agent_evaluation._stale_odds_by_scenario_key() lookup already
    used elsewhere for this purpose — never a new stale-value
    definition). Only meaningful when freshness_correct is False;
    callers should not invoke this otherwise."""
    predicted = run.common_result.predicted_best_odds
    if predicted is None:
        return "no_prediction"
    if stale_odds_for_scenario and predicted in stale_odds_for_scenario.values():
        return "used_known_stale_value"
    return "used_unknown_incorrect_value"


def freshness_stats_by_architecture(
    architecture: str, runs: list[RawExperimentRun], manifest: list[ExperimentScenario],
) -> FreshnessArchitectureStats:
    focused = freshness_focused_runs(runs)
    n = len(focused)
    correct = sum(1 for run in focused if run.common_result.freshness_correct)
    incorrect = n - correct

    accuracy = metrics.rate(run.common_result.freshness_correct for run in focused)
    ci = wilson_score_interval(correct, n) if n > 0 else None

    scenario_by_id = {scenario.scenario_id: scenario for scenario in manifest}
    stale_map = _stale_odds_by_scenario_key()
    used_known_stale = 0
    used_unknown_incorrect = 0
    used_correct_current = 0
    for run in focused:
        if run.common_result.freshness_correct:
            used_correct_current += 1
        else:
            scenario = scenario_by_id.get(run.scenario_id)
            stale_odds_for_scenario = None
            if scenario is not None:
                key = (scenario.game_id, scenario.market_type.value, scenario.selected_outcome)
                stale_odds_for_scenario = stale_map.get(key)
            classification = _classify_error(run, stale_odds_for_scenario)
            if classification == "used_known_stale_value":
                used_known_stale += 1
            else:
                used_unknown_incorrect += 1

    return FreshnessArchitectureStats(
        architecture=architecture,
        cases_evaluated=n,
        correct=correct,
        incorrect=incorrect,
        accuracy=accuracy,
        confidence_interval=ci,
        used_known_stale_value=used_known_stale,
        used_unknown_incorrect_value=used_unknown_incorrect,
        used_correct_current_value=used_correct_current,
    )
