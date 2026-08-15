"""Unit tests for the core Pydantic data models in src/models.py."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models import (
    ArchitectureType,
    BettingAnalysis,
    Game,
    GroundTruth,
    Market,
    MarketType,
    SourceReference,
    SourceType,
    SportsbookOdds,
    TestScenario,
)


# ---------------------------------------------------------------------------
# SportsbookOdds
# ---------------------------------------------------------------------------


def test_sportsbook_odds_valid_positive():
    odds = SportsbookOdds(sportsbook="DraftKings", american_odds=150, is_current=True)
    assert odds.sportsbook == "DraftKings"
    assert odds.american_odds == 150


def test_sportsbook_odds_valid_negative():
    odds = SportsbookOdds(sportsbook="FanDuel", american_odds=-200, is_current=True)
    assert odds.american_odds == -200


def test_sportsbook_odds_rejects_zero():
    with pytest.raises(ValidationError):
        SportsbookOdds(sportsbook="DraftKings", american_odds=0, is_current=True)


def test_sportsbook_odds_rejects_blank_name():
    with pytest.raises(ValidationError):
        SportsbookOdds(sportsbook="", american_odds=150, is_current=True)


def test_sportsbook_odds_rejects_whitespace_name():
    with pytest.raises(ValidationError):
        SportsbookOdds(sportsbook="   ", american_odds=150, is_current=True)


def test_sportsbook_odds_rejects_unrealistic_value():
    with pytest.raises(ValidationError):
        SportsbookOdds(sportsbook="DraftKings", american_odds=10_000_000, is_current=True)


def test_sportsbook_odds_timestamp_optional():
    odds = SportsbookOdds(sportsbook="DraftKings", american_odds=150, is_current=True)
    assert odds.timestamp is None


def test_sportsbook_odds_accepts_datetime_timestamp():
    ts = datetime(2026, 8, 14, 12, 0, 0)
    odds = SportsbookOdds(
        sportsbook="DraftKings", american_odds=150, is_current=True, timestamp=ts
    )
    assert odds.timestamp == ts


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


def test_market_moneyline_no_line_required():
    market = Market(market_type=MarketType.MONEYLINE, selected_outcome="Los Angeles Lakers")
    assert market.line is None


def test_market_spread_requires_line():
    with pytest.raises(ValidationError):
        Market(market_type=MarketType.SPREAD, selected_outcome="Los Angeles Lakers")


def test_market_spread_with_line_valid():
    market = Market(
        market_type=MarketType.SPREAD, selected_outcome="Los Angeles Lakers", line=-4.5
    )
    assert market.line == -4.5


def test_market_rejects_blank_outcome():
    with pytest.raises(ValidationError):
        Market(market_type=MarketType.MONEYLINE, selected_outcome="")


def test_market_rejects_invalid_market_type():
    with pytest.raises(ValidationError):
        Market(market_type="parlay", selected_outcome="Los Angeles Lakers")


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------


def _make_game(**overrides):
    defaults = dict(
        game_id="game-1",
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        start_time=datetime(2026, 8, 14, 19, 0, 0),
        sport="basketball",
    )
    defaults.update(overrides)
    return Game(**defaults)


def test_game_valid():
    game = _make_game()
    assert game.home_team == "Los Angeles Lakers"


def test_game_rejects_identical_teams():
    with pytest.raises(ValidationError):
        _make_game(home_team="Los Angeles Lakers", away_team="Los Angeles Lakers")


def test_game_rejects_identical_teams_case_insensitive():
    with pytest.raises(ValidationError):
        _make_game(home_team="lakers", away_team="Lakers")


def test_game_rejects_blank_game_id():
    with pytest.raises(ValidationError):
        _make_game(game_id="")


def test_game_rejects_blank_sport():
    with pytest.raises(ValidationError):
        _make_game(sport="")


# ---------------------------------------------------------------------------
# TestScenario
# ---------------------------------------------------------------------------


def _make_scenario(sportsbook_odds=None, probability=0.45, **overrides):
    game = _make_game()
    market = Market(market_type=MarketType.MONEYLINE, selected_outcome="Los Angeles Lakers")
    if sportsbook_odds is None:
        sportsbook_odds = [
            SportsbookOdds(sportsbook="DraftKings", american_odds=150, is_current=True),
            SportsbookOdds(sportsbook="FanDuel", american_odds=-200, is_current=True),
        ]
    defaults = dict(
        scenario_id="scenario-1",
        game=game,
        market=market,
        sportsbook_odds=sportsbook_odds,
        estimated_true_probability=probability,
    )
    defaults.update(overrides)
    return TestScenario(**defaults)


def test_scenario_valid():
    scenario = _make_scenario()
    assert scenario.scenario_id == "scenario-1"


def test_scenario_valid_probability():
    scenario = _make_scenario(probability=0.45)
    assert scenario.estimated_true_probability == 0.45


def test_scenario_rejects_negative_probability():
    with pytest.raises(ValidationError):
        _make_scenario(probability=-0.10)


def test_scenario_rejects_probability_above_one():
    with pytest.raises(ValidationError):
        _make_scenario(probability=1.10)


def test_scenario_rejects_blank_id():
    with pytest.raises(ValidationError):
        _make_scenario(scenario_id="")


def test_scenario_rejects_empty_sportsbook_list():
    with pytest.raises(ValidationError):
        _make_scenario(sportsbook_odds=[])


def test_scenario_rejects_duplicate_sportsbooks():
    duplicate_odds = [
        SportsbookOdds(sportsbook="DraftKings", american_odds=150, is_current=True),
        SportsbookOdds(sportsbook="DraftKings", american_odds=-110, is_current=True),
    ]
    with pytest.raises(ValidationError):
        _make_scenario(sportsbook_odds=duplicate_odds)


def test_scenario_rejects_duplicate_sportsbooks_case_insensitive():
    duplicate_odds = [
        SportsbookOdds(sportsbook="draftkings", american_odds=150, is_current=True),
        SportsbookOdds(sportsbook="DraftKings", american_odds=-110, is_current=True),
    ]
    with pytest.raises(ValidationError):
        _make_scenario(sportsbook_odds=duplicate_odds)


def test_scenario_rejects_outcome_not_in_game():
    game = _make_game()
    market = Market(market_type=MarketType.MONEYLINE, selected_outcome="Golden State Warriors")
    odds = [SportsbookOdds(sportsbook="DraftKings", american_odds=150, is_current=True)]
    with pytest.raises(ValidationError):
        TestScenario(
            scenario_id="scenario-1",
            game=game,
            market=market,
            sportsbook_odds=odds,
            estimated_true_probability=0.45,
        )


# ---------------------------------------------------------------------------
# GroundTruth
# ---------------------------------------------------------------------------


def _make_ground_truth(**overrides):
    defaults = dict(
        scenario_id="scenario-1",
        expected_best_sportsbook="DraftKings",
        expected_best_odds=150,
        expected_implied_probability=0.40,
        expected_ev=0.125,
        expected_positive_ev=True,
    )
    defaults.update(overrides)
    return GroundTruth(**defaults)


def test_ground_truth_valid():
    gt = _make_ground_truth()
    assert gt.expected_positive_ev is True


def test_ground_truth_rejects_implied_probability_out_of_range():
    with pytest.raises(ValidationError):
        _make_ground_truth(expected_implied_probability=1.5)


def test_ground_truth_rejects_blank_sportsbook():
    with pytest.raises(ValidationError):
        _make_ground_truth(expected_best_sportsbook="")


def test_ground_truth_rejects_zero_odds():
    with pytest.raises(ValidationError):
        _make_ground_truth(expected_best_odds=0)


def test_ground_truth_rejects_blank_expected_sportsbook_entry():
    with pytest.raises(ValidationError):
        _make_ground_truth(expected_sportsbooks=["DraftKings", ""])


# ---------------------------------------------------------------------------
# BettingAnalysis / ArchitectureType
# ---------------------------------------------------------------------------


def _make_analysis(**overrides):
    defaults = dict(
        scenario_id="scenario-1",
        game_id="game-1",
        market=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        best_sportsbook="DraftKings",
        best_odds=150,
        implied_probability=0.40,
        estimated_true_probability=0.45,
        expected_value=0.125,
        positive_ev=True,
        sportsbooks_considered=["DraftKings", "FanDuel"],
        reasoning_summary="DraftKings offers the best odds among considered sportsbooks.",
        architecture=ArchitectureType.TOOL,
    )
    defaults.update(overrides)
    return BettingAnalysis(**defaults)


def test_betting_analysis_valid():
    analysis = _make_analysis()
    assert analysis.architecture == ArchitectureType.TOOL


@pytest.mark.parametrize("architecture", ["rag", "tool", "hybrid"])
def test_betting_analysis_accepts_supported_architectures(architecture):
    analysis = _make_analysis(architecture=architecture)
    assert analysis.architecture.value == architecture


def test_betting_analysis_rejects_unsupported_architecture():
    with pytest.raises(ValidationError):
        _make_analysis(architecture="random_agent")


def test_betting_analysis_rejects_probability_out_of_range():
    with pytest.raises(ValidationError):
        _make_analysis(implied_probability=1.5)


def test_betting_analysis_rejects_zero_odds():
    with pytest.raises(ValidationError):
        _make_analysis(best_odds=0)


def test_betting_analysis_rejects_blank_reasoning_summary():
    with pytest.raises(ValidationError):
        _make_analysis(reasoning_summary="")


def test_betting_analysis_rejects_empty_sportsbooks_considered():
    with pytest.raises(ValidationError):
        _make_analysis(sportsbooks_considered=[])


def test_betting_analysis_rejects_blank_entry_in_sportsbooks_considered():
    with pytest.raises(ValidationError):
        _make_analysis(sportsbooks_considered=["DraftKings", ""])


def test_betting_analysis_rejects_blank_identifiers():
    with pytest.raises(ValidationError):
        _make_analysis(scenario_id="")
    with pytest.raises(ValidationError):
        _make_analysis(game_id="")


# ---------------------------------------------------------------------------
# SourceReference
# ---------------------------------------------------------------------------


def test_source_reference_valid():
    ref = SourceReference(source_type=SourceType.TOOL, source_id="draftkings-lookup-1")
    assert ref.sportsbook is None


def test_source_reference_rejects_blank_source_id():
    with pytest.raises(ValidationError):
        SourceReference(source_type=SourceType.RAG, source_id="")
