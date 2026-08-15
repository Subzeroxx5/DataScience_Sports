"""Deterministic ground-truth generation for the controlled dataset.

Ground truth is derived exclusively from src/calculations/odds_math.py
applied to src/models.py TestScenario objects loaded from the static
dataset in data/. No LLM, RAG, or external API is involved, and no betting
formula is duplicated here (see docs/research_design.md, "Ground Truth
Methodology").

Run as a script to (re)generate data/ground_truth.json:

    python -m src.evaluation.ground_truth

Running it repeatedly against unchanged source data produces a
byte-identical file: scenario order and dict keys are sorted, and no
wall-clock or other nondeterministic value is written.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.calculations.odds_math import (
    best_odds,
    compare_american_odds,
    expected_value,
    implied_probability,
    is_positive_ev,
)
from src.evaluation.dataset import (
    DATA_DIR,
    build_test_scenario,
    load_current_odds_records,
    load_historical_odds_records,
    load_scenario_definitions,
    load_test_scenarios,
)
from src.models import GroundTruth, MarketType, TestScenario


def generate_ground_truth_for_scenario(
    scenario: TestScenario, current_odds_version: str | None = None
) -> GroundTruth:
    odds_values = [odds.american_odds for odds in scenario.sportsbook_odds]
    best = best_odds(odds_values)

    # A sportsbook is tied for best if its odds are equally favorable to the
    # bettor (equivalent implied probability), not merely numerically equal.
    best_sportsbooks = sorted(
        odds.sportsbook
        for odds in scenario.sportsbook_odds
        if compare_american_odds(odds.american_odds, best) == 0
    )

    implied_prob = implied_probability(best)
    ev = expected_value(best, scenario.estimated_true_probability)
    positive_ev = is_positive_ev(best, scenario.estimated_true_probability)
    available_sportsbooks = sorted(odds.sportsbook for odds in scenario.sportsbook_odds)

    return GroundTruth(
        scenario_id=scenario.scenario_id,
        expected_best_sportsbook=best_sportsbooks[0],
        expected_best_sportsbooks=best_sportsbooks,
        expected_best_odds=best,
        expected_implied_probability=implied_prob,
        expected_ev=ev,
        expected_positive_ev=positive_ev,
        expected_sportsbooks=available_sportsbooks,
        current_odds_version=current_odds_version,
    )


def generate_all_ground_truth() -> list[GroundTruth]:
    definitions = load_scenario_definitions()
    current_odds_records = load_current_odds_records()
    results = []
    for definition in definitions:
        scenario = build_test_scenario(definition, current_odds_records)
        ground_truth = generate_ground_truth_for_scenario(
            scenario, current_odds_version=definition.get("current_odds_version")
        )
        results.append(ground_truth)
    results.sort(key=lambda gt: gt.scenario_id)
    return results


def export_ground_truth(output_path: Path | None = None) -> Path:
    output_path = output_path or (DATA_DIR / "ground_truth.json")
    ground_truth_list = generate_all_ground_truth()
    payload = [gt.model_dump(mode="json") for gt in ground_truth_list]
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return output_path


def summarize_dataset() -> dict:
    """Compute dataset coverage directly from the data, not from hand labels.

    This is what dataset/ground-truth tests assert against, so coverage
    claims are verified rather than merely documented.
    """
    scenarios = {s.scenario_id: s for s in load_test_scenarios()}
    ground_truth = {gt.scenario_id: gt for gt in generate_all_ground_truth()}
    historical = load_historical_odds_records()
    stale_game_ids = {record["game_id"] for record in historical}

    all_sportsbook_names = {
        odds.sportsbook for scenario in scenarios.values() for odds in scenario.sportsbook_odds
    }

    summary = {
        "scenario_count": len(scenarios),
        "positive_odds": 0,
        "negative_odds": 0,
        "mixed_sign_odds": 0,
        "positive_ev": 0,
        "negative_ev": 0,
        "break_even": 0,
        "tie": 0,
        "missing_data": 0,
        "freshness": 0,
        "moneyline": 0,
        "spread": 0,
        "total": 0,
    }

    for scenario_id, scenario in scenarios.items():
        odds_values = [odds.american_odds for odds in scenario.sportsbook_odds]
        has_positive = any(v > 0 for v in odds_values)
        has_negative = any(v < 0 for v in odds_values)
        if has_positive and has_negative:
            summary["mixed_sign_odds"] += 1
        elif has_positive:
            summary["positive_odds"] += 1
        elif has_negative:
            summary["negative_odds"] += 1

        if len(scenario.sportsbook_odds) < len(all_sportsbook_names):
            summary["missing_data"] += 1

        if scenario.game.game_id in stale_game_ids:
            summary["freshness"] += 1

        if scenario.market.market_type == MarketType.MONEYLINE:
            summary["moneyline"] += 1
        elif scenario.market.market_type == MarketType.SPREAD:
            summary["spread"] += 1
        elif scenario.market.market_type == MarketType.TOTAL:
            summary["total"] += 1

        gt = ground_truth[scenario_id]
        if len(gt.expected_best_sportsbooks) > 1:
            summary["tie"] += 1
        if abs(gt.expected_ev) <= 1e-6:
            summary["break_even"] += 1
        elif gt.expected_positive_ev:
            summary["positive_ev"] += 1
        else:
            summary["negative_ev"] += 1

    return summary


if __name__ == "__main__":
    written_path = export_ground_truth()
    count = len(generate_all_ground_truth())
    print(f"Wrote {count} ground truth records to {written_path}")

    summary = summarize_dataset()
    print()
    print(f"Scenario count: {summary['scenario_count']}")
    print()
    print(f"Positive odds only: {summary['positive_odds']}")
    print(f"Negative odds only: {summary['negative_odds']}")
    print(f"Mixed-sign odds: {summary['mixed_sign_odds']}")
    print()
    print(f"Positive EV: {summary['positive_ev']}")
    print(f"Negative EV: {summary['negative_ev']}")
    print(f"Break-even: {summary['break_even']}")
    print()
    print(f"Tie scenarios: {summary['tie']}")
    print(f"Missing-data scenarios: {summary['missing_data']}")
    print(f"Freshness scenarios: {summary['freshness']}")
    print()
    print(f"Moneyline: {summary['moneyline']}")
    print(f"Spread: {summary['spread']}")
    print(f"Total: {summary['total']}")
