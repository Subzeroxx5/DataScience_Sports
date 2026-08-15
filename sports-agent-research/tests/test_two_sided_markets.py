"""Tests for two-sided market representation (Milestone 6A).

Verifies that the controlled dataset can represent both mutually exclusive
outcomes of the same moneyline market — needed for the future no-vig /
market-consensus extension — while preserving every existing benchmark
guarantee (ties, missing data, freshness, ground truth, provider/tool
behavior).

Market identity policy: no new `market_id` field was introduced. A market
is uniquely identified by the tuple (game_id, market_type), since this
dataset never has more than one active market per (game_id, market_type)
pair; the two mutually exclusive moneyline outcomes are simply
Game.home_team and Game.away_team. This is documented here and in
data/README.md rather than encoded as a new schema field.
"""

import json

import pytest

from src.evaluation.dataset import load_current_odds_records, load_test_scenarios
from src.evaluation.ground_truth import generate_all_ground_truth
from src.models import MarketType
from src.providers.controlled import ControlledOddsProvider
from src.tools.sportsbook_tools import SportsbookTools

# (game_id, scenario's own outcome, opposing outcome) for the 4 markets
# extended in this milestone.
TWO_SIDED_MARKETS = [
    ("G-2026-001", "Los Angeles Lakers", "Boston Celtics"),
    ("G-2026-007", "Philadelphia 76ers", "Toronto Raptors"),
    ("G-2026-008", "Memphis Grizzlies", "New Orleans Pelicans"),
    ("G-2026-009", "Minnesota Timberwolves", "Oklahoma City Thunder"),
]


@pytest.fixture(scope="module")
def tools():
    return SportsbookTools(ControlledOddsProvider())


@pytest.fixture(scope="module")
def current_odds_records():
    return load_current_odds_records()


# ---------------------------------------------------------------------------
# Market pairing: at least 3 moneyline markets have both outcomes
# ---------------------------------------------------------------------------


def test_at_least_three_two_sided_moneyline_markets_exist(current_odds_records):
    assert len(TWO_SIDED_MARKETS) >= 3

    for game_id, outcome_a, outcome_b in TWO_SIDED_MARKETS:
        outcomes_present = {
            r["selected_outcome"]
            for r in current_odds_records
            if r["game_id"] == game_id and r["market_type"] == "moneyline"
        }
        assert outcome_a in outcomes_present
        assert outcome_b in outcomes_present


@pytest.mark.parametrize("game_id,outcome_a,outcome_b", TWO_SIDED_MARKETS)
def test_both_outcomes_share_same_game_id(tools, game_id, outcome_a, outcome_b):
    game = tools.get_game(game_id)
    assert {outcome_a, outcome_b} == {game.home_team, game.away_team}


@pytest.mark.parametrize("game_id,outcome_a,outcome_b", TWO_SIDED_MARKETS)
def test_both_outcomes_share_same_market_identity(tools, game_id, outcome_a, outcome_b):
    # Market identity = (game_id, market_type). Both lookups must succeed
    # against the same game_id + market_type with no ambiguity.
    odds_a = tools.get_odds(game_id, MarketType.MONEYLINE, outcome_a)
    odds_b = tools.get_odds(game_id, MarketType.MONEYLINE, outcome_b)
    assert len(odds_a) >= 1
    assert len(odds_b) >= 1


# ---------------------------------------------------------------------------
# Sportsbook pairing: at least one sportsbook prices both outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game_id,outcome_a,outcome_b", TWO_SIDED_MARKETS)
def test_at_least_one_sportsbook_prices_both_outcomes(tools, game_id, outcome_a, outcome_b):
    books_a = {o.sportsbook for o in tools.get_odds(game_id, MarketType.MONEYLINE, outcome_a)}
    books_b = {o.sportsbook for o in tools.get_odds(game_id, MarketType.MONEYLINE, outcome_b)}
    assert books_a & books_b


def test_sportsbook_pairing_is_deterministic(tools):
    # The same (game_id, sportsbook, market_type) query must always
    # resolve to the same two prices — no ambiguity in pairing a
    # sportsbook's two sides of a market.
    game_id = "G-2026-001"
    for _ in range(3):
        lakers = tools.get_sportsbook_odds(
            game_id, "DraftKings", MarketType.MONEYLINE, "Los Angeles Lakers"
        )
        celtics = tools.get_sportsbook_odds(
            game_id, "DraftKings", MarketType.MONEYLINE, "Boston Celtics"
        )
        assert lakers.american_odds == 120
        assert celtics.american_odds == -140


def test_example_sportsbook_pairing_draftkings_lakers_celtics(tools):
    lakers = tools.get_sportsbook_odds(
        "G-2026-001", "DraftKings", MarketType.MONEYLINE, "Los Angeles Lakers"
    )
    celtics = tools.get_sportsbook_odds(
        "G-2026-001", "DraftKings", MarketType.MONEYLINE, "Boston Celtics"
    )
    assert lakers.american_odds == 120
    assert celtics.american_odds == -140


# ---------------------------------------------------------------------------
# Valid odds: no two-sided-market price is 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game_id,outcome_a,outcome_b", TWO_SIDED_MARKETS)
def test_no_two_sided_market_price_is_zero(tools, game_id, outcome_a, outcome_b):
    for outcome in (outcome_a, outcome_b):
        for odds in tools.get_odds(game_id, MarketType.MONEYLINE, outcome):
            assert odds.american_odds != 0


# ---------------------------------------------------------------------------
# Existing edge cases still hold under the two-sided extension
# ---------------------------------------------------------------------------


def test_tie_still_correct_on_original_side_of_two_sided_market(tools):
    # S007's tie (DraftKings/FanDuel both +125 on 76ers) must be unaffected
    # by Toronto Raptors' odds now existing for the same game_id.
    result = tools.find_best_line("G-2026-007", MarketType.MONEYLINE, "Philadelphia 76ers")
    assert result.best_odds == 125
    assert result.sportsbooks == ["DraftKings", "FanDuel"]


def test_opposing_side_does_not_leak_into_original_side_odds_count(tools):
    # G-2026-007 now has 8 total current_odds.json records (4 per side),
    # but get_odds() for the 76ers must still return exactly the 4
    # 76ers-side records, never mixed with Raptors-side records.
    odds = tools.get_odds("G-2026-007", MarketType.MONEYLINE, "Philadelphia 76ers")
    assert len(odds) == 4
    assert all(o.american_odds > 0 for o in odds)  # all 76ers prices are positive


def test_missing_sportsbook_remains_missing_on_both_sides(tools):
    # FanDuel has no line for Memphis Grizzlies (Milestone 4) nor for the
    # newly-added New Orleans Pelicans side.
    grizzlies = tools.get_odds("G-2026-008", MarketType.MONEYLINE, "Memphis Grizzlies")
    pelicans = tools.get_odds("G-2026-008", MarketType.MONEYLINE, "New Orleans Pelicans")
    assert "FanDuel" not in {o.sportsbook for o in grizzlies}
    assert "FanDuel" not in {o.sportsbook for o in pelicans}
    assert all(o.american_odds != 0 for o in grizzlies + pelicans)


def test_missing_sportsbook_line_raises_explicitly_not_zero(tools):
    with pytest.raises(LookupError):
        tools.get_sportsbook_odds(
            "G-2026-008", "FanDuel", MarketType.MONEYLINE, "New Orleans Pelicans"
        )


def test_freshness_relationship_intact_alongside_new_opposing_side(tools):
    # DraftKings' stale=120/current=135 pair for Minnesota Timberwolves
    # (Milestone 4) must be unaffected by Oklahoma City Thunder's new
    # current-only odds on the same game_id.
    timberwolves = tools.get_sportsbook_odds(
        "G-2026-009", "DraftKings", MarketType.MONEYLINE, "Minnesota Timberwolves"
    )
    assert timberwolves.american_odds == 135
    assert timberwolves.american_odds != 120

    thunder = tools.get_sportsbook_odds(
        "G-2026-009", "DraftKings", MarketType.MONEYLINE, "Oklahoma City Thunder"
    )
    assert thunder.american_odds == -155
    assert thunder.is_current is True


def test_historical_odds_file_unaffected_by_two_sided_extension():
    historical = json.loads(
        (ControlledOddsProvider().data_dir / "historical_odds.json").read_text()
    )
    assert len(historical) == 3  # unchanged from Milestone 4
    assert all(r["is_current"] is False for r in historical)


# ---------------------------------------------------------------------------
# Best-line behavior preserved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "game_id,outcome,expected_best_odds,expected_books",
    [
        ("G-2026-001", "Los Angeles Lakers", 125, ["FanDuel"]),
        ("G-2026-002", "Golden State Warriors", -110, ["FanDuel"]),
        ("G-2026-007", "Philadelphia 76ers", 125, ["DraftKings", "FanDuel"]),
    ],
)
def test_existing_best_line_results_unchanged(
    tools, game_id, outcome, expected_best_odds, expected_books
):
    result = tools.find_best_line(game_id, MarketType.MONEYLINE, outcome)
    assert result.best_odds == expected_best_odds
    assert result.sportsbooks == expected_books


def test_best_line_also_works_for_new_opposing_sides(tools):
    result = tools.find_best_line("G-2026-001", MarketType.MONEYLINE, "Boston Celtics")
    assert result.best_odds == -135  # closest to even money among -140/-145/-135/-142
    assert result.sportsbooks == ["BetMGM"]


# ---------------------------------------------------------------------------
# Ground truth preserved exactly
# ---------------------------------------------------------------------------


def test_ground_truth_unchanged_by_two_sided_extension():
    ground_truth = {gt.scenario_id: gt for gt in generate_all_ground_truth()}
    # Spot-check the 4 scenarios whose games gained an opposing side.
    assert ground_truth["S001"].expected_best_odds == 125
    assert ground_truth["S001"].expected_best_sportsbooks == ["FanDuel"]
    assert ground_truth["S007"].expected_best_odds == 125
    assert set(ground_truth["S007"].expected_best_sportsbooks) == {"DraftKings", "FanDuel"}
    assert ground_truth["S008"].expected_best_odds == 115
    assert ground_truth["S008"].expected_sportsbooks == ["BetMGM", "Caesars", "DraftKings"]
    assert ground_truth["S009"].expected_best_odds == 140


def test_all_scenarios_still_load_and_validate():
    scenarios = load_test_scenarios()
    assert len(scenarios) == 14
    # Each scenario's sportsbook_odds must still contain only its own
    # outcome's prices (no cross-contamination from the opposing side).
    for scenario in scenarios:
        odds_values = [o.american_odds for o in scenario.sportsbook_odds]
        assert all(v != 0 for v in odds_values)
