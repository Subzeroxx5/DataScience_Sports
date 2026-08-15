"""Integration tests for the full SportsbookTools -> ControlledOddsProvider
-> controlled dataset path. No provider mocking here — every test drives
the real dataset in data/.
"""

import pytest

from src.evaluation.dataset import load_scenario_definitions
from src.evaluation.ground_truth import generate_all_ground_truth
from src.models import MarketType
from src.providers.controlled import ControlledOddsProvider
from src.tools.sportsbook_tools import SportsbookTools


@pytest.fixture(scope="module")
def tools():
    return SportsbookTools(ControlledOddsProvider())


# ---------------------------------------------------------------------------
# get_games / get_game
# ---------------------------------------------------------------------------


def test_get_games_returns_known_controlled_games(tools):
    games = tools.get_games()
    game_ids = {g.game_id for g in games}
    assert "G-2026-001" in game_ids
    assert "G-2026-014" in game_ids
    assert len(games) >= 12


def test_get_game_returns_known_controlled_game(tools):
    game = tools.get_game("G-2026-007")
    assert game.home_team == "Philadelphia 76ers"
    assert game.away_team == "Toronto Raptors"


# ---------------------------------------------------------------------------
# get_odds / get_sportsbook_odds
# ---------------------------------------------------------------------------


def test_get_odds_returns_expected_current_records(tools):
    odds = tools.get_odds("G-2026-002", MarketType.MONEYLINE, "Golden State Warriors")
    by_book = {o.sportsbook: o.american_odds for o in odds}
    assert by_book == {"DraftKings": -120, "FanDuel": -110, "BetMGM": -130, "Caesars": -115}


def test_get_sportsbook_odds_returns_expected_current_record(tools):
    odds = tools.get_sportsbook_odds(
        "G-2026-002", "FanDuel", MarketType.MONEYLINE, "Golden State Warriors"
    )
    assert odds.american_odds == -110
    assert odds.is_current is True


# ---------------------------------------------------------------------------
# find_best_line — one scenario per required category (Part 10)
# ---------------------------------------------------------------------------


def test_find_best_line_positive_odds_scenario_S001(tools):
    result = tools.find_best_line("G-2026-001", MarketType.MONEYLINE, "Los Angeles Lakers")
    assert result.best_odds == 125
    assert result.sportsbooks == ["FanDuel"]


def test_find_best_line_negative_odds_scenario_S002(tools):
    result = tools.find_best_line("G-2026-002", MarketType.MONEYLINE, "Golden State Warriors")
    assert result.best_odds == -110
    assert result.sportsbooks == ["FanDuel"]


def test_find_best_line_mixed_sign_scenario_S003(tools):
    result = tools.find_best_line("G-2026-003", MarketType.MONEYLINE, "Milwaukee Bucks")
    assert result.best_odds == 105
    assert result.sportsbooks == ["FanDuel"]


def test_find_best_line_tie_scenario_S007(tools):
    result = tools.find_best_line(
        "G-2026-007", MarketType.MONEYLINE, "Philadelphia 76ers"
    )
    assert result.best_odds == 125
    assert result.sportsbooks == ["DraftKings", "FanDuel"]


def test_find_best_line_stale_current_scenario_S009(tools):
    # DraftKings stale=120, current=135; FanDuel current=140 is best.
    result = tools.find_best_line(
        "G-2026-009", MarketType.MONEYLINE, "Minnesota Timberwolves"
    )
    assert result.best_odds == 140
    assert result.sportsbooks == ["FanDuel"]
    assert result.best_odds != 120


def test_find_best_line_missing_data_scenario_S008(tools):
    # FanDuel has no line at all for this scenario.
    result = tools.find_best_line("G-2026-008", MarketType.MONEYLINE, "Memphis Grizzlies")
    assert "FanDuel" not in result.sportsbooks
    assert result.best_odds == 115
    assert result.sportsbooks == ["BetMGM"]


# ---------------------------------------------------------------------------
# Missing-data / unknown-input error behavior through the full stack
# ---------------------------------------------------------------------------


def test_find_best_line_unknown_game_fails_explicitly(tools):
    with pytest.raises(LookupError):
        tools.find_best_line("G-DOES-NOT-EXIST", MarketType.MONEYLINE, "Anyone")


def test_find_best_line_unknown_outcome_fails_explicitly(tools):
    with pytest.raises(LookupError):
        tools.find_best_line("G-2026-001", MarketType.MONEYLINE, "Golden State Warriors")


def test_get_sportsbook_odds_unavailable_sportsbook_fails_explicitly(tools):
    with pytest.raises(LookupError):
        tools.get_sportsbook_odds(
            "G-2026-008", "FanDuel", MarketType.MONEYLINE, "Memphis Grizzlies"
        )


# ---------------------------------------------------------------------------
# Ground-truth cross-check (Part 11) — no second definition of correctness
# ---------------------------------------------------------------------------


def _scenario_lookup_params():
    definitions = {d["scenario_id"]: d for d in load_scenario_definitions()}
    ground_truth = {gt.scenario_id: gt for gt in generate_all_ground_truth()}
    params = []
    for scenario_id, definition in definitions.items():
        market = definition["market"]
        if market["market_type"] != "moneyline":
            # find_best_line's core EV-analysis scope is moneyline; spread/
            # total are schema-validation-only per data/README.md.
            continue
        params.append(
            (
                scenario_id,
                definition["game"]["game_id"],
                market["selected_outcome"],
                ground_truth[scenario_id],
            )
        )
    return params


@pytest.mark.parametrize(
    "scenario_id,game_id,selected_outcome,ground_truth",
    _scenario_lookup_params(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_find_best_line_matches_ground_truth(tools, scenario_id, game_id, selected_outcome, ground_truth):
    result = tools.find_best_line(game_id, MarketType.MONEYLINE, selected_outcome)
    assert result.best_odds == ground_truth.expected_best_odds
    assert set(result.sportsbooks) == set(ground_truth.expected_best_sportsbooks)
