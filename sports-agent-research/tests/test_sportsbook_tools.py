"""Wiring tests for SportsbookTools <-> OddsProvider (Milestone 5D).

Uses a lightweight fake provider (not the real JSON-backed
ControlledOddsProvider) to verify delegation, argument forwarding, return
value passthrough, error propagation, and freshness preservation in
isolation from any actual dataset. A separate smoke test at the bottom
exercises the real ControlledOddsProvider end-to-end.
"""

from datetime import datetime

import pytest

from src.models import Game, MarketType, SportsbookOdds
from src.providers.base import OddsProvider
from src.providers.controlled import ControlledOddsProvider
from src.tools.sportsbook_tools import SportsbookTools

_GAME = Game(
    game_id="G-FAKE-001",
    home_team="Fake Home",
    away_team="Fake Away",
    start_time=datetime(2026, 8, 15, 19, 0, 0),
    sport="basketball",
)

_CURRENT_ODDS = [
    SportsbookOdds(sportsbook="DraftKings", american_odds=120, is_current=True),
    SportsbookOdds(sportsbook="FanDuel", american_odds=125, is_current=True),
]

_STALE_ODDS = SportsbookOdds(sportsbook="DraftKings", american_odds=100, is_current=False)


class GameNotFound(LookupError):
    pass


class OddsNotFound(LookupError):
    pass


class FakeProvider(OddsProvider):
    """Records every call it receives and returns canned, pre-validated
    structured results — a stand-in for any real OddsProvider
    implementation (JSON-backed or, eventually, API-backed)."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    def get_games(self) -> list[Game]:
        self.calls.append(("get_games", (), {}))
        return [_GAME]

    def get_game(self, game_id: str) -> Game:
        self.calls.append(("get_game", (game_id,), {}))
        if game_id != _GAME.game_id:
            raise GameNotFound(game_id)
        return _GAME

    def get_odds(
        self, game_id: str, market_type: MarketType, selected_outcome: str
    ) -> list[SportsbookOdds]:
        self.calls.append(("get_odds", (game_id, market_type, selected_outcome), {}))
        if game_id != _GAME.game_id:
            raise GameNotFound(game_id)
        if selected_outcome != "Fake Home":
            raise OddsNotFound(selected_outcome)
        # Freshness is the provider's job: this fake, like the real
        # ControlledOddsProvider, only ever returns current records.
        return list(_CURRENT_ODDS)

    def get_sportsbook_odds(
        self,
        game_id: str,
        sportsbook: str,
        market_type: MarketType,
        selected_outcome: str,
    ) -> SportsbookOdds:
        self.calls.append(
            ("get_sportsbook_odds", (game_id, sportsbook, market_type, selected_outcome), {})
        )
        if game_id != _GAME.game_id:
            raise GameNotFound(game_id)
        matches = [o for o in _CURRENT_ODDS if o.sportsbook == sportsbook]
        if not matches:
            raise OddsNotFound(sportsbook)
        return matches[0]


@pytest.fixture
def fake_provider():
    return FakeProvider()


@pytest.fixture
def tools(fake_provider):
    return SportsbookTools(fake_provider)


class ConfigurableOddsProvider(OddsProvider):
    """Fake provider whose get_odds() result is set directly per test, for
    exercising find_best_line()'s odds-comparison logic against exact,
    hand-picked inputs without going through the real JSON dataset."""

    def __init__(self, odds: list[SportsbookOdds]):
        self._odds = odds

    def get_games(self) -> list[Game]:
        return [_GAME]

    def get_game(self, game_id: str) -> Game:
        return _GAME

    def get_odds(
        self, game_id: str, market_type: MarketType, selected_outcome: str
    ) -> list[SportsbookOdds]:
        if not self._odds:
            raise OddsNotFound("no current odds available")
        return list(self._odds)

    def get_sportsbook_odds(
        self,
        game_id: str,
        sportsbook: str,
        market_type: MarketType,
        selected_outcome: str,
    ) -> SportsbookOdds:
        for odds in self._odds:
            if odds.sportsbook == sportsbook:
                return odds
        raise OddsNotFound(sportsbook)


def _odds(**by_sportsbook: int) -> list[SportsbookOdds]:
    return [
        SportsbookOdds(sportsbook=name, american_odds=value, is_current=True)
        for name, value in by_sportsbook.items()
    ]


def _find_best_line(odds: list[SportsbookOdds]):
    tools = SportsbookTools(ConfigurableOddsProvider(odds))
    return tools.find_best_line("G-FAKE-001", MarketType.MONEYLINE, "Fake Home")


# ---------------------------------------------------------------------------
# find_best_line — odds comparison and tie handling (Milestone 5E)
# ---------------------------------------------------------------------------


def test_find_best_line_positive_odds():
    result = _find_best_line(_odds(DraftKings=120, FanDuel=125, BetMGM=115))
    assert result.best_odds == 125
    assert result.sportsbooks == ["FanDuel"]


def test_find_best_line_negative_odds():
    result = _find_best_line(_odds(DraftKings=-120, FanDuel=-110, BetMGM=-125))
    assert result.best_odds == -110
    assert result.sportsbooks == ["FanDuel"]


def test_find_best_line_mixed_sign_odds():
    result = _find_best_line(_odds(DraftKings=-105, FanDuel=105, BetMGM=-110, Caesars=100))
    assert result.best_odds == 105
    assert result.sportsbooks == ["FanDuel"]


def test_find_best_line_tie():
    result = _find_best_line(_odds(DraftKings=125, FanDuel=125, BetMGM=120))
    assert result.best_odds == 125
    assert result.sportsbooks == ["DraftKings", "FanDuel"]


def test_find_best_line_tie_ordering_is_alphabetical():
    # Insertion order is FanDuel, DraftKings — output must still be
    # alphabetical (DraftKings, FanDuel), proving the tie-break policy is
    # not "first in the input" but a fixed deterministic ordering.
    result = _find_best_line(_odds(FanDuel=125, DraftKings=125, BetMGM=120))
    assert result.sportsbooks == ["DraftKings", "FanDuel"]


def test_find_best_line_single_sportsbook():
    result = _find_best_line(_odds(DraftKings=-115))
    assert result.best_odds == -115
    assert result.sportsbooks == ["DraftKings"]


def test_find_best_line_no_odds_raises_explicitly():
    with pytest.raises(OddsNotFound):
        _find_best_line([])


def test_find_best_line_never_compares_by_absolute_value():
    # If compared by abs(), -120 (abs 120) would appear to "tie" or beat
    # +110 (abs 110) — the correct favorability winner is +110.
    result = _find_best_line(_odds(DraftKings=-120, FanDuel=110))
    assert result.best_odds == 110
    assert result.sportsbooks == ["FanDuel"]


@pytest.mark.parametrize(
    "odds_a,odds_b,winner",
    [
        (120, 125, 125),
        (-120, -110, -110),
        (-105, 105, 105),
        (100, -105, 100),
        (-105, -110, -105),
        (-110, -120, -110),
        (-120, -150, -120),
    ],
)
def test_find_best_line_required_pairwise_ordering(odds_a, odds_b, winner):
    result = _find_best_line(_odds(BookA=odds_a, BookB=odds_b))
    assert result.best_odds == winner


def test_find_best_line_reuses_calculations_layer():
    # find_best_line's result must agree with calling best_odds()/
    # compare_american_odds() directly on the same inputs — i.e. it is
    # not silently reimplementing its own ordering logic.
    from src.calculations.odds_math import best_odds

    values = [120, 125, 115, -300]
    result = _find_best_line(_odds(DraftKings=120, FanDuel=125, BetMGM=115, Caesars=-300))
    assert result.best_odds == best_odds(values)


# ---------------------------------------------------------------------------
# Provider delegation
# ---------------------------------------------------------------------------


def test_get_games_delegates_to_provider(tools, fake_provider):
    tools.get_games()
    assert fake_provider.calls == [("get_games", (), {})]


def test_get_game_delegates_to_provider(tools, fake_provider):
    tools.get_game("G-FAKE-001")
    assert fake_provider.calls == [("get_game", ("G-FAKE-001",), {})]


def test_get_odds_delegates_to_provider(tools, fake_provider):
    tools.get_odds("G-FAKE-001", MarketType.MONEYLINE, "Fake Home")
    assert fake_provider.calls == [
        ("get_odds", ("G-FAKE-001", MarketType.MONEYLINE, "Fake Home"), {})
    ]


def test_get_sportsbook_odds_delegates_to_provider(tools, fake_provider):
    tools.get_sportsbook_odds("G-FAKE-001", "DraftKings", MarketType.MONEYLINE, "Fake Home")
    assert fake_provider.calls == [
        (
            "get_sportsbook_odds",
            ("G-FAKE-001", "DraftKings", MarketType.MONEYLINE, "Fake Home"),
            {},
        )
    ]


# ---------------------------------------------------------------------------
# Argument forwarding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "game_id,market_type,selected_outcome",
    [
        ("G-FAKE-001", MarketType.MONEYLINE, "Fake Home"),
        ("G-OTHER", MarketType.SPREAD, "Some Outcome"),
    ],
)
def test_get_odds_forwards_arguments_unchanged(fake_provider, game_id, market_type, selected_outcome):
    tools = SportsbookTools(fake_provider)
    try:
        tools.get_odds(game_id, market_type, selected_outcome)
    except (GameNotFound, OddsNotFound):
        pass
    name, args, kwargs = fake_provider.calls[-1]
    assert name == "get_odds"
    assert args == (game_id, market_type, selected_outcome)
    assert kwargs == {}


def test_get_sportsbook_odds_forwards_all_four_arguments(fake_provider):
    tools = SportsbookTools(fake_provider)
    tools.get_sportsbook_odds("G-FAKE-001", "FanDuel", MarketType.MONEYLINE, "Fake Home")
    name, args, kwargs = fake_provider.calls[-1]
    assert args == ("G-FAKE-001", "FanDuel", MarketType.MONEYLINE, "Fake Home")


# ---------------------------------------------------------------------------
# Return values
# ---------------------------------------------------------------------------


def test_get_games_returns_provider_result(tools):
    result = tools.get_games()
    assert result == [_GAME]
    assert isinstance(result[0], Game)


def test_get_game_returns_provider_result(tools):
    result = tools.get_game("G-FAKE-001")
    assert result is _GAME


def test_get_odds_returns_provider_result(tools):
    result = tools.get_odds("G-FAKE-001", MarketType.MONEYLINE, "Fake Home")
    assert result == _CURRENT_ODDS
    assert all(isinstance(o, SportsbookOdds) for o in result)


def test_get_sportsbook_odds_returns_provider_result(tools):
    result = tools.get_sportsbook_odds(
        "G-FAKE-001", "DraftKings", MarketType.MONEYLINE, "Fake Home"
    )
    assert result.sportsbook == "DraftKings"
    assert result.american_odds == 120


# ---------------------------------------------------------------------------
# Errors propagate, are not swallowed or converted into fake data
# ---------------------------------------------------------------------------


def test_get_game_unknown_id_error_propagates(tools):
    with pytest.raises(GameNotFound):
        tools.get_game("G-DOES-NOT-EXIST")


def test_get_odds_unknown_game_error_propagates(tools):
    with pytest.raises(GameNotFound):
        tools.get_odds("G-DOES-NOT-EXIST", MarketType.MONEYLINE, "Fake Home")


def test_get_odds_unknown_outcome_error_propagates(tools):
    with pytest.raises(OddsNotFound):
        tools.get_odds("G-FAKE-001", MarketType.MONEYLINE, "Nonexistent Outcome")


def test_get_sportsbook_odds_unknown_sportsbook_error_propagates(tools):
    with pytest.raises(OddsNotFound):
        tools.get_sportsbook_odds(
            "G-FAKE-001", "NotARealBook", MarketType.MONEYLINE, "Fake Home"
        )


def test_tool_layer_does_not_substitute_data_on_error(tools):
    # A failed lookup must raise, never return a Game/SportsbookOdds-shaped
    # placeholder as if it were a real result.
    with pytest.raises(GameNotFound):
        result = tools.get_game("G-DOES-NOT-EXIST")
        assert result is None  # unreachable if the exception is raised correctly


# ---------------------------------------------------------------------------
# Freshness preservation
# ---------------------------------------------------------------------------


def test_tool_layer_never_reintroduces_stale_odds(tools):
    # The fake provider's get_odds() never returns _STALE_ODDS; confirm the
    # tool layer doesn't add it back in or otherwise alter the result.
    result = tools.get_odds("G-FAKE-001", MarketType.MONEYLINE, "Fake Home")
    assert _STALE_ODDS not in result
    assert all(o.is_current for o in result)


def test_tool_layer_preserves_is_current_flag_from_provider(tools):
    result = tools.get_sportsbook_odds(
        "G-FAKE-001", "DraftKings", MarketType.MONEYLINE, "Fake Home"
    )
    assert result.is_current is True


# ---------------------------------------------------------------------------
# Dependency boundary: no direct JSON/dataset access in the tool layer
# ---------------------------------------------------------------------------


def test_tool_module_source_has_no_json_or_file_access():
    # AST-based, not substring-based: docstrings legitimately mention
    # "data/current_odds.json" when explaining the architecture, but the
    # actual code must never call open()/json.load/json.loads or import
    # json/pathlib for its own use.
    import ast
    from pathlib import Path

    source_path = (
        Path(__file__).resolve().parent.parent / "src" / "tools" / "sportsbook_tools.py"
    )
    tree = ast.parse(source_path.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in ("json", "pathlib"), (
                    f"sportsbook_tools.py must not import {alias.name!r}"
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open", "sportsbook_tools.py must not call open()"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in ("load", "loads"), (
                "sportsbook_tools.py must not call json.load/json.loads"
            )


def test_sportsbook_tools_only_imports_models_provider_and_calculations():
    import ast
    from pathlib import Path

    source_path = (
        Path(__file__).resolve().parent.parent / "src" / "tools" / "sportsbook_tools.py"
    )
    tree = ast.parse(source_path.read_text())

    allowed = {"__future__", "src.models", "src.providers.base", "src.calculations.odds_math"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in allowed, f"unexpected import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module in allowed, f"unexpected import: {node.module}"


# ---------------------------------------------------------------------------
# End-to-end smoke test with the real ControlledOddsProvider
# ---------------------------------------------------------------------------


def test_end_to_end_with_real_controlled_provider():
    tools = SportsbookTools(ControlledOddsProvider())

    games = tools.get_games()
    assert len(games) >= 12

    game = tools.get_game("G-2026-001")
    assert game.home_team == "Los Angeles Lakers"

    odds = tools.get_odds("G-2026-001", MarketType.MONEYLINE, "Los Angeles Lakers")
    assert {o.sportsbook for o in odds} == {"DraftKings", "FanDuel", "BetMGM", "Caesars"}
    assert all(o.is_current for o in odds)

    fanduel_odds = tools.get_sportsbook_odds(
        "G-2026-001", "FanDuel", MarketType.MONEYLINE, "Los Angeles Lakers"
    )
    assert fanduel_odds.american_odds == 125

    best_line = tools.find_best_line("G-2026-001", MarketType.MONEYLINE, "Los Angeles Lakers")
    assert best_line.best_odds == 125
    assert best_line.sportsbooks == ["FanDuel"]


def test_end_to_end_freshness_through_full_stack():
    # S009 / G-2026-009: DraftKings stale=120 (historical_odds.json),
    # current=135 (current_odds.json). The full path Tool -> Provider ->
    # Dataset must surface only the current value.
    tools = SportsbookTools(ControlledOddsProvider())
    odds = tools.get_sportsbook_odds(
        "G-2026-009", "DraftKings", MarketType.MONEYLINE, "Minnesota Timberwolves"
    )
    assert odds.american_odds == 135
    assert odds.american_odds != 120


def test_end_to_end_find_best_line_excludes_stale_odds():
    # DraftKings' stale +120 must never win best-line selection over its
    # own current +135, or over another book's current price.
    tools = SportsbookTools(ControlledOddsProvider())
    result = tools.find_best_line(
        "G-2026-009", MarketType.MONEYLINE, "Minnesota Timberwolves"
    )
    assert result.best_odds != 120
    draftkings_current = tools.get_sportsbook_odds(
        "G-2026-009", "DraftKings", MarketType.MONEYLINE, "Minnesota Timberwolves"
    )
    assert draftkings_current.american_odds == 135
