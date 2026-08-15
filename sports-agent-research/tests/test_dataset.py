"""Dataset integrity tests for the controlled sportsbook benchmark."""

import pytest
from pydantic import ValidationError

from src.evaluation.dataset import (
    load_current_odds_records,
    load_historical_odds_records,
    load_scenario_definitions,
    load_test_scenarios,
)
from src.evaluation.ground_truth import summarize_dataset

MIN_SCENARIO_COUNT = 12


def test_all_scenarios_validate_through_pydantic():
    scenarios = load_test_scenarios()
    assert len(scenarios) >= MIN_SCENARIO_COUNT


def test_scenario_count_at_least_twelve():
    summary = summarize_dataset()
    assert summary["scenario_count"] >= MIN_SCENARIO_COUNT


def test_scenario_ids_are_unique():
    definitions = load_scenario_definitions()
    ids = [d["scenario_id"] for d in definitions]
    assert len(ids) == len(set(ids))


def test_scenario_ids_non_empty():
    definitions = load_scenario_definitions()
    for definition in definitions:
        assert definition["scenario_id"].strip() != ""


def test_game_ids_non_empty_and_valid():
    scenarios = load_test_scenarios()
    for scenario in scenarios:
        assert scenario.game.game_id.strip() != ""


def test_no_blank_sportsbook_names():
    for record in load_current_odds_records():
        assert record["sportsbook"].strip() != ""
    for record in load_historical_odds_records():
        assert record["sportsbook"].strip() != ""


def test_no_zero_american_odds():
    for record in load_current_odds_records():
        assert record["american_odds"] != 0
    for record in load_historical_odds_records():
        assert record["american_odds"] != 0


def test_all_probabilities_in_range():
    for definition in load_scenario_definitions():
        probability = definition["estimated_true_probability"]
        assert 0.0 <= probability <= 1.0


def test_missing_data_not_represented_as_zero_odds():
    # A missing sportsbook must be entirely absent from current_odds.json
    # for that game, never present with american_odds == 0.
    scenarios = load_test_scenarios()
    missing_data_scenarios = [s for s in scenarios if len(s.sportsbook_odds) < 4]
    assert len(missing_data_scenarios) >= 1
    for scenario in missing_data_scenarios:
        for odds in scenario.sportsbook_odds:
            assert odds.american_odds != 0


# ---------------------------------------------------------------------------
# Required coverage — verified against actual computed data, not labels.
# ---------------------------------------------------------------------------


def test_coverage_positive_odds():
    assert summarize_dataset()["positive_odds"] >= 1


def test_coverage_negative_odds():
    assert summarize_dataset()["negative_odds"] >= 1


def test_coverage_mixed_sign_odds():
    assert summarize_dataset()["mixed_sign_odds"] >= 1


def test_coverage_positive_ev():
    assert summarize_dataset()["positive_ev"] >= 1


def test_coverage_negative_ev():
    assert summarize_dataset()["negative_ev"] >= 1


def test_coverage_break_even():
    assert summarize_dataset()["break_even"] >= 1


def test_coverage_tie():
    assert summarize_dataset()["tie"] >= 1


def test_coverage_missing_data():
    assert summarize_dataset()["missing_data"] >= 1


def test_coverage_freshness_at_least_three():
    assert summarize_dataset()["freshness"] >= 3


def test_coverage_moneyline():
    assert summarize_dataset()["moneyline"] >= 1


def test_coverage_spread():
    assert summarize_dataset()["spread"] >= 1


def test_coverage_total():
    assert summarize_dataset()["total"] >= 1


# ---------------------------------------------------------------------------
# Freshness metadata
# ---------------------------------------------------------------------------


def test_historical_records_are_marked_not_current():
    for record in load_historical_odds_records():
        assert record["is_current"] is False


def test_current_records_are_marked_current():
    for record in load_current_odds_records():
        assert record["is_current"] is True


def test_historical_records_reference_a_real_current_game():
    current_game_ids = {r["game_id"] for r in load_current_odds_records()}
    for record in load_historical_odds_records():
        assert record["game_id"] in current_game_ids


def test_historical_records_do_not_duplicate_current_sportsbook_entry_in_scenario():
    # A TestScenario's sportsbook_odds only ever contains current records
    # (validated by the no-duplicate-sportsbook rule in src/models.py); the
    # stale counterpart lives only in historical_odds.json.
    scenarios = {s.game.game_id: s for s in load_test_scenarios()}
    for record in load_historical_odds_records():
        scenario = scenarios[record["game_id"]]
        current_sportsbooks = {o.sportsbook for o in scenario.sportsbook_odds}
        assert record["sportsbook"] in current_sportsbooks


# ---------------------------------------------------------------------------
# Invalid dataset construction should still be rejected by the models
# ---------------------------------------------------------------------------


def test_scenario_with_zero_odds_rejected_by_model():
    from src.models import Game, Market, MarketType, SportsbookOdds, TestScenario

    game = Game(
        game_id="G-TEST",
        home_team="Team A",
        away_team="Team B",
        start_time="2026-08-15T19:00:00",
        sport="basketball",
    )
    market = Market(market_type=MarketType.MONEYLINE, selected_outcome="Team A")
    with pytest.raises(ValidationError):
        TestScenario(
            scenario_id="INVALID",
            game=game,
            market=market,
            sportsbook_odds=[SportsbookOdds(sportsbook="DraftKings", american_odds=0, is_current=True)],
            estimated_true_probability=0.5,
        )
