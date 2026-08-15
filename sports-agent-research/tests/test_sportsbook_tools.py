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


def test_sportsbook_tools_only_imports_models_and_provider_abstraction():
    import ast
    from pathlib import Path

    source_path = (
        Path(__file__).resolve().parent.parent / "src" / "tools" / "sportsbook_tools.py"
    )
    tree = ast.parse(source_path.read_text())

    allowed = {"__future__", "src.models", "src.providers.base"}
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
