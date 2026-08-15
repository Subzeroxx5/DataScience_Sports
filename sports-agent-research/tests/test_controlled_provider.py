"""Tests for src/providers/controlled.py — ControlledOddsProvider."""

import json

import pytest
from pydantic import ValidationError

from src.models import Game, MarketType, SportsbookOdds
from src.providers.base import OddsProvider
from src.providers.controlled import (
    ControlledOddsProvider,
    GameNotFoundError,
    OddsNotFoundError,
)


@pytest.fixture(scope="module")
def provider():
    return ControlledOddsProvider()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_valid_dataset_loads_successfully():
    provider = ControlledOddsProvider()
    assert isinstance(provider, OddsProvider)


def test_default_constructor_uses_project_data_dir():
    provider = ControlledOddsProvider()
    assert len(provider.get_games()) >= 12


def test_explicit_data_dir_accepted(tmp_path):
    # Copy the real dataset files into a temp dir to prove data_dir is
    # actually used (dependency injection), not a hard-coded global path.
    import shutil

    real_dir = ControlledOddsProvider().data_dir
    shutil.copy(real_dir / "current_odds.json", tmp_path / "current_odds.json")
    shutil.copy(real_dir / "test_scenarios.json", tmp_path / "test_scenarios.json")

    provider = ControlledOddsProvider(data_dir=tmp_path)
    assert len(provider.get_games()) >= 12


def test_nonexistent_data_dir_fails_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError):
        ControlledOddsProvider(data_dir=tmp_path / "does_not_exist")


def test_missing_current_odds_file_fails_explicitly(tmp_path):
    (tmp_path / "test_scenarios.json").write_text("[]")
    with pytest.raises(FileNotFoundError):
        ControlledOddsProvider(data_dir=tmp_path)


def test_malformed_json_fails_explicitly(tmp_path):
    (tmp_path / "test_scenarios.json").write_text("[]")
    (tmp_path / "current_odds.json").write_text("{not valid json")
    with pytest.raises(ValueError):
        ControlledOddsProvider(data_dir=tmp_path)


def test_non_array_json_fails_explicitly(tmp_path):
    (tmp_path / "test_scenarios.json").write_text("[]")
    (tmp_path / "current_odds.json").write_text('{"not": "an array"}')
    with pytest.raises(ValueError):
        ControlledOddsProvider(data_dir=tmp_path)


def test_schema_invalid_scenario_game_fails_explicitly(tmp_path):
    # Game requires home_team != away_team (see src/models.py).
    bad_scenarios = [
        {
            "scenario_id": "BAD",
            "game": {
                "game_id": "G-BAD",
                "home_team": "Same Team",
                "away_team": "Same Team",
                "start_time": "2026-08-15T19:00:00",
                "sport": "basketball",
            },
            "market": {"market_type": "moneyline", "selected_outcome": "Same Team", "line": None},
            "estimated_true_probability": 0.5,
        }
    ]
    (tmp_path / "test_scenarios.json").write_text(json.dumps(bad_scenarios))
    (tmp_path / "current_odds.json").write_text("[]")
    with pytest.raises(ValidationError):
        ControlledOddsProvider(data_dir=tmp_path)


def test_schema_invalid_odds_record_fails_explicitly(tmp_path):
    scenarios = [
        {
            "scenario_id": "OK",
            "game": {
                "game_id": "G-OK",
                "home_team": "Team A",
                "away_team": "Team B",
                "start_time": "2026-08-15T19:00:00",
                "sport": "basketball",
            },
            "market": {"market_type": "moneyline", "selected_outcome": "Team A", "line": None},
            "estimated_true_probability": 0.5,
        }
    ]
    bad_odds = [
        {
            "game_id": "G-OK",
            "sportsbook": "DraftKings",
            "market_type": "moneyline",
            "selected_outcome": "Team A",
            "american_odds": 0,  # invalid: zero odds
            "is_current": True,
            "timestamp": "2026-08-10T12:00:00",
        }
    ]
    (tmp_path / "test_scenarios.json").write_text(json.dumps(scenarios))
    (tmp_path / "current_odds.json").write_text(json.dumps(bad_odds))
    with pytest.raises(ValidationError):
        ControlledOddsProvider(data_dir=tmp_path)


def test_odds_record_missing_field_fails_explicitly(tmp_path):
    scenarios = [
        {
            "scenario_id": "OK",
            "game": {
                "game_id": "G-OK",
                "home_team": "Team A",
                "away_team": "Team B",
                "start_time": "2026-08-15T19:00:00",
                "sport": "basketball",
            },
            "market": {"market_type": "moneyline", "selected_outcome": "Team A", "line": None},
            "estimated_true_probability": 0.5,
        }
    ]
    incomplete_odds = [
        {
            "game_id": "G-OK",
            "sportsbook": "DraftKings",
            "market_type": "moneyline",
            "selected_outcome": "Team A",
            # american_odds missing entirely
            "is_current": True,
        }
    ]
    (tmp_path / "test_scenarios.json").write_text(json.dumps(scenarios))
    (tmp_path / "current_odds.json").write_text(json.dumps(incomplete_odds))
    with pytest.raises(ValueError):
        ControlledOddsProvider(data_dir=tmp_path)


def test_odds_record_for_unknown_game_fails_explicitly(tmp_path):
    (tmp_path / "test_scenarios.json").write_text("[]")
    orphan_odds = [
        {
            "game_id": "G-NOT-IN-SCENARIOS",
            "sportsbook": "DraftKings",
            "market_type": "moneyline",
            "selected_outcome": "Team A",
            "american_odds": 120,
            "is_current": True,
        }
    ]
    (tmp_path / "current_odds.json").write_text(json.dumps(orphan_odds))
    with pytest.raises(ValueError):
        ControlledOddsProvider(data_dir=tmp_path)


# ---------------------------------------------------------------------------
# get_games
# ---------------------------------------------------------------------------


def test_get_games_returns_games(provider):
    games = provider.get_games()
    assert len(games) >= 12
    assert all(isinstance(g, Game) for g in games)


def test_get_games_ids_are_unique(provider):
    games = provider.get_games()
    ids = [g.game_id for g in games]
    assert len(ids) == len(set(ids))


def test_get_games_ordering_is_deterministic(provider):
    first_call = [g.game_id for g in provider.get_games()]
    second_call = [g.game_id for g in provider.get_games()]
    assert first_call == second_call
    assert first_call == sorted(first_call)


# ---------------------------------------------------------------------------
# get_game
# ---------------------------------------------------------------------------


def test_get_game_known_id_returns_correct_game(provider):
    game = provider.get_game("G-2026-001")
    assert game.game_id == "G-2026-001"
    assert game.home_team == "Los Angeles Lakers"
    assert game.away_team == "Boston Celtics"


def test_get_game_unknown_id_raises_explicitly(provider):
    with pytest.raises(GameNotFoundError):
        provider.get_game("G-DOES-NOT-EXIST")


def test_get_game_does_not_fuzzy_match(provider):
    with pytest.raises(GameNotFoundError):
        provider.get_game("g-2026-001")  # wrong case, exact match required


# ---------------------------------------------------------------------------
# get_odds
# ---------------------------------------------------------------------------


def test_get_odds_known_combination_returns_current_odds(provider):
    odds = provider.get_odds("G-2026-001", MarketType.MONEYLINE, "Los Angeles Lakers")
    assert len(odds) == 4
    assert all(isinstance(o, SportsbookOdds) for o in odds)
    assert all(o.is_current for o in odds)


def test_get_odds_returns_multiple_sportsbooks(provider):
    odds = provider.get_odds("G-2026-001", MarketType.MONEYLINE, "Los Angeles Lakers")
    names = {o.sportsbook for o in odds}
    assert names == {"DraftKings", "FanDuel", "BetMGM", "Caesars"}


def test_get_odds_unknown_game_raises_game_not_found(provider):
    with pytest.raises(GameNotFoundError):
        provider.get_odds("G-DOES-NOT-EXIST", MarketType.MONEYLINE, "Anyone")


def test_get_odds_unknown_outcome_raises_odds_not_found(provider):
    with pytest.raises(OddsNotFoundError):
        provider.get_odds("G-2026-001", MarketType.MONEYLINE, "Chicago Bulls")


def test_get_odds_wrong_market_raises_odds_not_found(provider):
    # G-2026-001 only has moneyline odds, not spread.
    with pytest.raises(OddsNotFoundError):
        provider.get_odds("G-2026-001", MarketType.SPREAD, "Los Angeles Lakers")


def test_get_odds_missing_sportsbook_scenario_excludes_absent_book(provider):
    # S008 / G-2026-008: FanDuel has no line (missing-data scenario).
    odds = provider.get_odds("G-2026-008", MarketType.MONEYLINE, "Memphis Grizzlies")
    names = {o.sportsbook for o in odds}
    assert "FanDuel" not in names
    assert names == {"DraftKings", "BetMGM", "Caesars"}
    assert all(o.american_odds != 0 for o in odds)


def test_get_odds_ordering_is_deterministic(provider):
    first_call = [o.sportsbook for o in provider.get_odds(
        "G-2026-001", MarketType.MONEYLINE, "Los Angeles Lakers"
    )]
    second_call = [o.sportsbook for o in provider.get_odds(
        "G-2026-001", MarketType.MONEYLINE, "Los Angeles Lakers"
    )]
    assert first_call == second_call
    assert first_call == sorted(first_call)


# ---------------------------------------------------------------------------
# get_sportsbook_odds
# ---------------------------------------------------------------------------


def test_get_sportsbook_odds_known_combination_succeeds(provider):
    odds = provider.get_sportsbook_odds(
        "G-2026-001", "FanDuel", MarketType.MONEYLINE, "Los Angeles Lakers"
    )
    assert odds.sportsbook == "FanDuel"
    assert odds.american_odds == 125
    assert odds.is_current is True


def test_get_sportsbook_odds_unknown_sportsbook_raises_explicitly(provider):
    with pytest.raises(OddsNotFoundError):
        provider.get_sportsbook_odds(
            "G-2026-001", "NotARealSportsbook", MarketType.MONEYLINE, "Los Angeles Lakers"
        )


def test_get_sportsbook_odds_unavailable_line_raises_explicitly(provider):
    # S008 / G-2026-008: FanDuel offers no line at all.
    with pytest.raises(OddsNotFoundError):
        provider.get_sportsbook_odds(
            "G-2026-008", "FanDuel", MarketType.MONEYLINE, "Memphis Grizzlies"
        )


def test_get_sportsbook_odds_unknown_game_raises_game_not_found(provider):
    with pytest.raises(GameNotFoundError):
        provider.get_sportsbook_odds(
            "G-DOES-NOT-EXIST", "DraftKings", MarketType.MONEYLINE, "Anyone"
        )


def test_get_sportsbook_odds_does_not_substitute_another_sportsbook(provider):
    with pytest.raises(OddsNotFoundError):
        provider.get_sportsbook_odds(
            "G-2026-001", "PointsBet", MarketType.MONEYLINE, "Los Angeles Lakers"
        )


# ---------------------------------------------------------------------------
# Freshness — the most important behavior in this milestone
# ---------------------------------------------------------------------------


def test_freshness_get_odds_returns_only_current_value(provider):
    # S009 / G-2026-009: DraftKings stale=120, current=135 (see
    # data/historical_odds.json vs data/current_odds.json).
    odds = provider.get_odds(
        "G-2026-009", MarketType.MONEYLINE, "Minnesota Timberwolves"
    )
    draftkings = next(o for o in odds if o.sportsbook == "DraftKings")
    assert draftkings.american_odds == 135
    assert draftkings.american_odds != 120
    assert draftkings.is_current is True


def test_freshness_get_sportsbook_odds_returns_only_current_value(provider):
    odds = provider.get_sportsbook_odds(
        "G-2026-009", "DraftKings", MarketType.MONEYLINE, "Minnesota Timberwolves"
    )
    assert odds.american_odds == 135
    assert odds.american_odds != 120


@pytest.mark.parametrize(
    "game_id,sportsbook,outcome,stale_odds,current_odds",
    [
        ("G-2026-009", "DraftKings", "Minnesota Timberwolves", 120, 135),
        ("G-2026-010", "FanDuel", "Sacramento Kings", 130, 150),
        ("G-2026-011", "BetMGM", "Atlanta Hawks", -130, -110),
    ],
)
def test_freshness_all_three_stale_scenarios(
    provider, game_id, sportsbook, outcome, stale_odds, current_odds
):
    odds = provider.get_sportsbook_odds(game_id, sportsbook, MarketType.MONEYLINE, outcome)
    assert odds.american_odds == current_odds
    assert odds.american_odds != stale_odds
    assert odds.is_current is True


def test_provider_never_reads_historical_odds_file(provider):
    # historical_odds.json is out of scope for this provider entirely —
    # confirm none of its stale values leak into get_odds() results.
    stale_records = json.loads((provider.data_dir / "historical_odds.json").read_text())
    for stale in stale_records:
        odds = provider.get_odds(
            stale["game_id"], MarketType(stale["market_type"]), stale["selected_outcome"]
        )
        matching = [o for o in odds if o.sportsbook == stale["sportsbook"]]
        assert len(matching) == 1
        assert matching[0].american_odds != stale["american_odds"]


# ---------------------------------------------------------------------------
# No business logic in the provider
# ---------------------------------------------------------------------------


def test_provider_has_no_find_best_line_method(provider):
    assert not hasattr(provider, "find_best_line")
