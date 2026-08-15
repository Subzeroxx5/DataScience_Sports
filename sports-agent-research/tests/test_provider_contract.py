"""Contract tests for src/providers/base.py.

These tests validate only the abstraction: that OddsProvider is a proper
ABC exposing the required methods, that it cannot be instantiated
directly, that a complete subclass can be, and that the module has no
forbidden dependencies. No concrete data-loading provider exists yet.
"""

import ast
import inspect
import typing
from abc import ABC
from pathlib import Path

import pytest

from src.models import Game, MarketType, SportsbookOdds
from src.providers.base import OddsProvider

REQUIRED_METHODS = ["get_games", "get_game", "get_odds", "get_sportsbook_odds"]


# ---------------------------------------------------------------------------
# Existence and abstractness
# ---------------------------------------------------------------------------


def test_odds_provider_exists():
    assert OddsProvider is not None


def test_odds_provider_is_abstract_base_class():
    assert issubclass(OddsProvider, ABC)
    assert inspect.isabstract(OddsProvider)


def test_odds_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        OddsProvider()


@pytest.mark.parametrize("method_name", REQUIRED_METHODS)
def test_required_method_exists(method_name):
    assert hasattr(OddsProvider, method_name)


@pytest.mark.parametrize("method_name", REQUIRED_METHODS)
def test_required_method_is_abstract(method_name):
    method = getattr(OddsProvider, method_name)
    assert getattr(method, "__isabstractmethod__", False) is True


def test_find_best_line_not_defined_on_provider():
    assert not hasattr(OddsProvider, "find_best_line")


# ---------------------------------------------------------------------------
# Type annotations and docstrings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_name", REQUIRED_METHODS)
def test_required_method_fully_type_annotated(method_name):
    method = getattr(OddsProvider, method_name)
    signature = inspect.signature(method)
    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue
        assert param.annotation is not inspect.Parameter.empty, (
            f"{method_name} parameter {param_name!r} is missing a type annotation"
        )
    assert signature.return_annotation is not inspect.Signature.empty, (
        f"{method_name} is missing a return type annotation"
    )


@pytest.mark.parametrize("method_name", REQUIRED_METHODS)
def test_required_method_has_docstring(method_name):
    method = getattr(OddsProvider, method_name)
    assert method.__doc__ is not None and method.__doc__.strip() != ""


def test_get_game_returns_game():
    hints = typing.get_type_hints(OddsProvider.get_game)
    assert hints["return"] is Game
    assert hints["game_id"] is str


def test_get_games_returns_list_of_game():
    hints = typing.get_type_hints(OddsProvider.get_games)
    assert hints["return"] == list[Game]


def test_get_odds_returns_list_of_sportsbook_odds():
    hints = typing.get_type_hints(OddsProvider.get_odds)
    assert hints["return"] == list[SportsbookOdds]
    assert hints["market_type"] is MarketType


def test_get_sportsbook_odds_returns_sportsbook_odds():
    hints = typing.get_type_hints(OddsProvider.get_sportsbook_odds)
    assert hints["return"] is SportsbookOdds
    assert hints["market_type"] is MarketType


# ---------------------------------------------------------------------------
# Complete vs. incomplete subclasses
# ---------------------------------------------------------------------------


class _CompleteProvider(OddsProvider):
    def get_games(self) -> list[Game]:
        return []

    def get_game(self, game_id: str) -> Game:
        raise NotImplementedError

    def get_odds(
        self, game_id: str, market_type: MarketType, selected_outcome: str
    ) -> list[SportsbookOdds]:
        return []

    def get_sportsbook_odds(
        self,
        game_id: str,
        sportsbook: str,
        market_type: MarketType,
        selected_outcome: str,
    ) -> SportsbookOdds:
        raise NotImplementedError


class _IncompleteProvider(OddsProvider):
    def get_games(self) -> list[Game]:
        return []

    # get_game, get_odds, get_sportsbook_odds intentionally omitted


def test_complete_subclass_can_be_instantiated():
    provider = _CompleteProvider()
    assert isinstance(provider, OddsProvider)
    assert provider.get_games() == []


def test_incomplete_subclass_cannot_be_instantiated():
    with pytest.raises(TypeError):
        _IncompleteProvider()


# ---------------------------------------------------------------------------
# Dependency direction: providers -> models + stdlib only
# ---------------------------------------------------------------------------


def test_provider_module_has_no_forbidden_imports():
    source_path = Path(__file__).resolve().parent.parent / "src" / "providers" / "base.py"
    tree = ast.parse(source_path.read_text())

    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_prefixes = (
        "src.tools",
        "src.agents",
        "src.rag",
        "src.evaluation",
        "anthropic",
        "openai",
        "requests",
        "httpx",
        "langchain",
    )
    for module_name in imported_modules:
        assert not module_name.startswith(forbidden_prefixes), (
            f"src/providers/base.py must not import {module_name!r}"
        )


def test_provider_module_only_imports_models_and_stdlib():
    source_path = Path(__file__).resolve().parent.parent / "src" / "providers" / "base.py"
    tree = ast.parse(source_path.read_text())

    allowed = {"__future__", "abc", "src.models"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in allowed, f"unexpected import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module in allowed, f"unexpected import: {node.module}"
