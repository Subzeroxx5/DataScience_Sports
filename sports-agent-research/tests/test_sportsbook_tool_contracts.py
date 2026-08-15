"""Contract tests for src/tools/sportsbook_tools.py.

These tests validate the public interface: signatures, type hints,
docstrings, and structured response/error models — the parts of the
Milestone 5A contract that remain stable across Milestone 5D's provider
wiring. find_best_line() is still unimplemented (deferred to Milestone
5E) and is tested accordingly. get_games/get_game/get_odds/
get_sportsbook_odds are now wired to an injected provider; their
delegation behavior (argument forwarding, return values, error
propagation, freshness) is covered by tests/test_sportsbook_tools.py using
a fake provider, not here.

Milestone 5A originally exposed get_games/get_game/get_odds/
get_sportsbook_odds/find_best_line as free module-level functions, each
permanently raising NotImplementedError, explicitly as placeholders
("may temporarily raise NotImplementedError where necessary"). Milestone
5D requires these to receive an injected OddsProvider; a free function
cannot hold injected state without resorting to module-global mutable
state, so they were converted into methods on a SportsbookTools class
constructed with a provider (`SportsbookTools(provider)`), per Milestone
5D's own guidance to prefer explicit constructor-based dependency
injection. This is the one deliberate, documented break from the
Milestone 5A test file's exact shape; the parameter/return/error
contracts themselves (game_id, market_type, selected_outcome, sportsbook,
and the Game/SportsbookOdds/BestLineResult/exception types) are
unchanged.
"""

import inspect
import typing

import pytest
from pydantic import ValidationError

from src.models import BestLineResult, Game, MarketType, SportsbookOdds
from src.providers.base import OddsProvider
from src.tools.sportsbook_tools import (
    GameNotFoundError,
    MarketNotFoundError,
    NoOddsAvailableError,
    OutcomeNotFoundError,
    SportsbookNotFoundError,
    SportsbookTools,
    SportsbookToolError,
)

PUBLIC_METHODS = {
    "get_games": SportsbookTools.get_games,
    "get_game": SportsbookTools.get_game,
    "get_odds": SportsbookTools.get_odds,
    "get_sportsbook_odds": SportsbookTools.get_sportsbook_odds,
    "find_best_line": SportsbookTools.find_best_line,
}


class _StubProvider(OddsProvider):
    """Minimal OddsProvider stub, used only to instantiate SportsbookTools
    for contract-level checks (e.g. find_best_line's NotImplementedError).
    Delegation behavior is tested separately in test_sportsbook_tools.py.
    """

    def get_games(self) -> list[Game]:
        return []

    def get_game(self, game_id: str) -> Game:
        raise NotImplementedError

    def get_odds(
        self, game_id: str, market_type: MarketType, selected_outcome: str
    ) -> list[SportsbookOdds]:
        raise NotImplementedError

    def get_sportsbook_odds(
        self,
        game_id: str,
        sportsbook: str,
        market_type: MarketType,
        selected_outcome: str,
    ) -> SportsbookOdds:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PUBLIC_METHODS.keys())
def test_public_method_exists(name):
    assert hasattr(SportsbookTools, name)
    assert callable(getattr(SportsbookTools, name))


def test_sportsbook_tools_requires_provider():
    with pytest.raises(TypeError):
        SportsbookTools()  # missing required provider argument


# ---------------------------------------------------------------------------
# Type annotations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,method", PUBLIC_METHODS.items())
def test_public_method_fully_type_annotated(name, method):
    signature = inspect.signature(method)
    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue
        assert param.annotation is not inspect.Parameter.empty, (
            f"{name} parameter {param_name!r} is missing a type annotation"
        )
    assert signature.return_annotation is not inspect.Signature.empty, (
        f"{name} is missing a return type annotation"
    )


@pytest.mark.parametrize("name,method", PUBLIC_METHODS.items())
def test_public_method_has_docstring(name, method):
    assert method.__doc__ is not None and method.__doc__.strip() != "", (
        f"{name} is missing a docstring"
    )


# ---------------------------------------------------------------------------
# Expected parameters
# ---------------------------------------------------------------------------


def test_get_games_has_no_required_parameters_besides_self():
    signature = inspect.signature(SportsbookTools.get_games)
    assert list(signature.parameters.keys()) == ["self"]


def test_get_game_parameters():
    signature = inspect.signature(SportsbookTools.get_game)
    assert list(signature.parameters.keys()) == ["self", "game_id"]
    assert typing.get_type_hints(SportsbookTools.get_game)["game_id"] is str


def test_get_odds_parameters():
    signature = inspect.signature(SportsbookTools.get_odds)
    assert list(signature.parameters.keys()) == [
        "self",
        "game_id",
        "market_type",
        "selected_outcome",
    ]
    assert typing.get_type_hints(SportsbookTools.get_odds)["market_type"] is MarketType


def test_get_sportsbook_odds_parameters():
    signature = inspect.signature(SportsbookTools.get_sportsbook_odds)
    assert list(signature.parameters.keys()) == [
        "self",
        "game_id",
        "sportsbook",
        "market_type",
        "selected_outcome",
    ]
    assert (
        typing.get_type_hints(SportsbookTools.get_sportsbook_odds)["market_type"] is MarketType
    )


def test_find_best_line_parameters():
    signature = inspect.signature(SportsbookTools.find_best_line)
    assert list(signature.parameters.keys()) == [
        "self",
        "game_id",
        "market_type",
        "selected_outcome",
    ]
    assert typing.get_type_hints(SportsbookTools.find_best_line)["market_type"] is MarketType


# ---------------------------------------------------------------------------
# Return type annotations reuse existing domain models
# ---------------------------------------------------------------------------


def test_get_games_returns_list_of_game():
    assert typing.get_type_hints(SportsbookTools.get_games)["return"] == list[Game]


def test_get_game_returns_game():
    assert typing.get_type_hints(SportsbookTools.get_game)["return"] is Game


def test_get_odds_returns_list_of_sportsbook_odds():
    assert typing.get_type_hints(SportsbookTools.get_odds)["return"] == list[SportsbookOdds]


def test_get_sportsbook_odds_returns_sportsbook_odds():
    assert typing.get_type_hints(SportsbookTools.get_sportsbook_odds)["return"] is SportsbookOdds


def test_find_best_line_returns_best_line_result():
    assert typing.get_type_hints(SportsbookTools.find_best_line)["return"] is BestLineResult


# ---------------------------------------------------------------------------
# find_best_line remains unimplemented (deferred to Milestone 5E)
# ---------------------------------------------------------------------------


def test_find_best_line_not_implemented():
    tools = SportsbookTools(_StubProvider())
    with pytest.raises(NotImplementedError):
        tools.find_best_line("G-2026-001", MarketType.MONEYLINE, "Los Angeles Lakers")


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exception_class",
    [
        GameNotFoundError,
        MarketNotFoundError,
        OutcomeNotFoundError,
        SportsbookNotFoundError,
        NoOddsAvailableError,
    ],
)
def test_error_types_are_sportsbook_tool_errors(exception_class):
    assert issubclass(exception_class, SportsbookToolError)
    assert issubclass(exception_class, Exception)


def test_error_types_are_distinct():
    error_classes = {
        GameNotFoundError,
        MarketNotFoundError,
        OutcomeNotFoundError,
        SportsbookNotFoundError,
        NoOddsAvailableError,
    }
    assert len(error_classes) == 5


# ---------------------------------------------------------------------------
# BestLineResult — tie-capable structured response model
# ---------------------------------------------------------------------------


def test_best_line_result_valid_single_sportsbook():
    result = BestLineResult(
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        best_odds=150,
        sportsbooks=["FanDuel"],
    )
    assert result.sportsbooks == ["FanDuel"]


def test_best_line_result_supports_tie_with_multiple_sportsbooks():
    result = BestLineResult(
        game_id="G-2026-007",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Philadelphia 76ers",
        best_odds=125,
        sportsbooks=["DraftKings", "FanDuel"],
    )
    assert set(result.sportsbooks) == {"DraftKings", "FanDuel"}
    assert result.best_odds == 125


def test_best_line_result_rejects_zero_odds():
    with pytest.raises(ValidationError):
        BestLineResult(
            game_id="G-2026-001",
            market_type=MarketType.MONEYLINE,
            selected_outcome="Los Angeles Lakers",
            best_odds=0,
            sportsbooks=["FanDuel"],
        )


def test_best_line_result_rejects_empty_sportsbooks_list():
    with pytest.raises(ValidationError):
        BestLineResult(
            game_id="G-2026-001",
            market_type=MarketType.MONEYLINE,
            selected_outcome="Los Angeles Lakers",
            best_odds=150,
            sportsbooks=[],
        )


def test_best_line_result_rejects_blank_sportsbook_entry():
    with pytest.raises(ValidationError):
        BestLineResult(
            game_id="G-2026-001",
            market_type=MarketType.MONEYLINE,
            selected_outcome="Los Angeles Lakers",
            best_odds=150,
            sportsbooks=["FanDuel", ""],
        )


def test_best_line_result_rejects_duplicate_sportsbooks():
    with pytest.raises(ValidationError):
        BestLineResult(
            game_id="G-2026-001",
            market_type=MarketType.MONEYLINE,
            selected_outcome="Los Angeles Lakers",
            best_odds=150,
            sportsbooks=["FanDuel", "FanDuel"],
        )


def test_best_line_result_rejects_blank_game_id():
    with pytest.raises(ValidationError):
        BestLineResult(
            game_id="",
            market_type=MarketType.MONEYLINE,
            selected_outcome="Los Angeles Lakers",
            best_odds=150,
            sportsbooks=["FanDuel"],
        )
