"""Deterministic market-consensus ground-truth generation (Milestone 7B).

Extends the controlled benchmark with a market-consensus quantitative
value analysis, kept entirely separate from the Milestone 4
controlled-reference ground truth (src/evaluation/ground_truth.py /
data/ground_truth.json), which this module never reads, writes, or
imports:

    Controlled Reference Mode (unchanged, GroundTruth / ground_truth.json)
        -> uses the scenario's pre-defined estimated_true_probability

    Market Consensus Mode (this module, QuantGroundTruth / quant_ground_truth.json)
        -> derives a reference probability from OTHER sportsbooks'
           no-vig prices via leave-one-sportsbook-out consensus

Data flow:

    data/current_odds.json (current, two-sided where available)
    + data/test_scenarios.json (game/market only, never estimated_true_probability)
              |
              v
    src/calculations/odds_math.py (implied_probability, expected_value, is_positive_ev)
    src/calculations/market.py (no-vig, leave-one-out consensus, edge, dispersion)
              |
              v
    QuantGroundTruth / SportsbookValueGroundTruth / MarketDispersionGroundTruth
    (src/models.py)
              |
              v
    data/quant_ground_truth.json

No LLM, RAG, agent, or external API is involved anywhere in this module.
No betting/market formula is duplicated — every calculation delegates to
src/calculations/odds_math.py or src/calculations/market.py.

Eligibility (see docs/QUANT_STRATEGY.md and QuantGroundTruth's
docstring): a scenario's market is quant-evaluable only if (1) its
market_type is moneyline (a genuine two-outcome market — spread/total
scenarios in this dataset have no opposing-side data and are marked
ineligible), and (2) at least MIN_CONSENSUS_BOOKS + 1 sportsbooks quote
CURRENT prices on both mutually exclusive outcomes (1 target + at least
MIN_CONSENSUS_BOOKS comparison books). Ineligible scenarios are recorded
explicitly with quant_evaluable=False and a reason — never silently
skipped, never fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.calculations.market import (
    MIN_CONSENSUS_BOOKS,
    calculate_leave_one_out_consensus,
    calculate_market_dispersion,
    calculate_no_vig_probabilities,
    calculate_probability_edge,
)
from src.calculations.odds_math import expected_value, implied_probability, is_positive_ev
from src.evaluation.dataset import DATA_DIR, load_current_odds_records, load_scenario_definitions
from src.models import (
    MarketDispersionGroundTruth,
    MarketType,
    QuantGroundTruth,
    SportsbookValueGroundTruth,
)

QUANT_GROUND_TRUTH_PATH = DATA_DIR / "quant_ground_truth.json"

MIN_QUOTING_BOOKS = MIN_CONSENSUS_BOOKS + 1  # target + MIN_CONSENSUS_BOOKS comparison books


def _current_odds_by_outcome(
    game_id: str, market_type: str, current_records: list[dict]
) -> dict[str, dict[str, int]]:
    """outcome -> {sportsbook: american_odds}, current records only."""
    result: dict[str, dict[str, int]] = {}
    for record in current_records:
        if (
            record["game_id"] != game_id
            or record["market_type"] != market_type
            or not record["is_current"]
        ):
            continue
        result.setdefault(record["selected_outcome"], {})[record["sportsbook"]] = record[
            "american_odds"
        ]
    return result


def _ineligible(
    scenario_id: str, market_id: str, selected_outcome: str, reason: str
) -> QuantGroundTruth:
    return QuantGroundTruth(
        scenario_id=scenario_id,
        market_id=market_id,
        selected_outcome=selected_outcome,
        quant_evaluable=False,
        ineligibility_reason=reason,
    )


def generate_quant_ground_truth_for_scenario(
    definition: dict, current_records: list[dict]
) -> QuantGroundTruth:
    scenario_id = definition["scenario_id"]
    game = definition["game"]
    market = definition["market"]
    game_id = game["game_id"]
    market_type = market["market_type"]
    selected_outcome = market["selected_outcome"]
    market_id = f"{game_id}-{market_type}"

    if market_type != MarketType.MONEYLINE.value:
        return _ineligible(
            scenario_id,
            market_id,
            selected_outcome,
            f"market_type={market_type!r} is not a two-outcome moneyline market; "
            "no opposing-side data exists to derive a no-vig probability",
        )

    if selected_outcome == game["home_team"]:
        opposing_outcome = game["away_team"]
    elif selected_outcome == game["away_team"]:
        opposing_outcome = game["home_team"]
    else:
        return _ineligible(
            scenario_id,
            market_id,
            selected_outcome,
            f"selected_outcome {selected_outcome!r} does not match either team in "
            f"game_id={game_id!r}",
        )

    by_outcome = _current_odds_by_outcome(game_id, market_type, current_records)
    target_side = by_outcome.get(selected_outcome, {})
    opposing_side = by_outcome.get(opposing_outcome, {})

    complete_books = sorted(set(target_side) & set(opposing_side))

    if len(complete_books) < MIN_QUOTING_BOOKS:
        return _ineligible(
            scenario_id,
            market_id,
            selected_outcome,
            f"only {len(complete_books)} sportsbook(s) quote current prices on both "
            f"sides of this market ({complete_books}); at least {MIN_QUOTING_BOOKS} "
            f"required (1 target + {MIN_CONSENSUS_BOOKS} comparison books)",
        )

    # No-vig probability of the target outcome, per sportsbook.
    no_vig_target_by_book: dict[str, float] = {}
    for book in complete_books:
        fair = calculate_no_vig_probabilities([target_side[book], opposing_side[book]])
        no_vig_target_by_book[book] = fair[0]

    sportsbook_probabilities = [(book, no_vig_target_by_book[book]) for book in complete_books]

    analyses = []
    for book in complete_books:
        target_odds = target_side[book]
        book_implied = implied_probability(target_odds)
        no_vig = no_vig_target_by_book[book]
        comparison_books = [b for b in complete_books if b != book]
        reference_probability = calculate_leave_one_out_consensus(sportsbook_probabilities, book)
        edge = calculate_probability_edge(reference_probability, book_implied)
        ev = expected_value(target_odds, reference_probability)
        positive = is_positive_ev(target_odds, reference_probability)

        analyses.append(
            SportsbookValueGroundTruth(
                sportsbook=book,
                american_odds=target_odds,
                book_implied_probability=book_implied,
                no_vig_probability=no_vig,
                comparison_sportsbooks=comparison_books,
                market_reference_probability=reference_probability,
                probability_edge=edge,
                expected_value=ev,
                positive_ev=positive,
            )
        )

    dispersion_values = [no_vig_target_by_book[book] for book in complete_books]
    dispersion_stats = calculate_market_dispersion(dispersion_values)
    dispersion = MarketDispersionGroundTruth(
        mean_probability=dispersion_stats.mean_probability,
        median_probability=dispersion_stats.median_probability,
        population_std_dev=dispersion_stats.std_dev,
        probability_range=dispersion_stats.probability_range,
        minimum_probability=min(dispersion_values),
        maximum_probability=max(dispersion_values),
        book_count=dispersion_stats.book_count,
    )

    return QuantGroundTruth(
        scenario_id=scenario_id,
        market_id=market_id,
        selected_outcome=selected_outcome,
        quant_evaluable=True,
        sportsbook_analyses=analyses,
        market_dispersion=dispersion,
    )


def generate_all_quant_ground_truth() -> list[QuantGroundTruth]:
    definitions = load_scenario_definitions()
    current_records = load_current_odds_records()
    results = [
        generate_quant_ground_truth_for_scenario(definition, current_records)
        for definition in definitions
    ]
    results.sort(key=lambda q: q.scenario_id)
    return results


def export_quant_ground_truth(output_path: Path | None = None) -> Path:
    output_path = output_path or QUANT_GROUND_TRUTH_PATH
    items = generate_all_quant_ground_truth()
    payload = [item.model_dump(mode="json") for item in items]
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return output_path


def summarize_quant_coverage() -> dict:
    items = generate_all_quant_ground_truth()
    evaluable = [item for item in items if item.quant_evaluable]
    non_evaluable = [item for item in items if not item.quant_evaluable]
    analyses = [a for item in evaluable for a in item.sportsbook_analyses]
    return {
        "total_scenarios": len(items),
        "quant_evaluable": len(evaluable),
        "non_quant_evaluable": len(non_evaluable),
        "sportsbook_analyses": len(analyses),
        "positive_ev": sum(1 for a in analyses if a.positive_ev),
        "negative_ev": sum(1 for a in analyses if not a.positive_ev and a.expected_value < 0),
        "zero_ev": sum(1 for a in analyses if a.expected_value == 0),
    }


if __name__ == "__main__":
    written_path = export_quant_ground_truth()
    summary = summarize_quant_coverage()
    print(f"Wrote {summary['total_scenarios']} quant ground truth records to {written_path}")
    print()
    print(f"Total scenarios: {summary['total_scenarios']}")
    print(f"Quant-evaluable: {summary['quant_evaluable']}")
    print(f"Non-quant-evaluable: {summary['non_quant_evaluable']}")
    print()
    print(f"Sportsbook target analyses: {summary['sportsbook_analyses']}")
    print(f"Positive EV: {summary['positive_ev']}")
    print(f"Negative EV: {summary['negative_ev']}")
    print(f"Zero EV: {summary['zero_ev']}")
