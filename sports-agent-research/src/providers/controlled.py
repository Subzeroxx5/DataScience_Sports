"""Concrete OddsProvider backed by the project's controlled JSON dataset.

Dataset determination: the "authoritative current sportsbook data" for this
provider is the combination of two files (see data/README.md for the
Milestone 4 dataset design):

    data/current_odds.json    the odds ledger — one record per
                               (game, sportsbook, market, outcome), always
                               is_current=True. This is the sole source of
                               SportsbookOdds data.
    data/test_scenarios.json  scenario definitions, each embedding a full
                               "game" object (game_id, home_team, away_team,
                               sport, start_time). This is the sole source
                               of Game data, since current_odds.json
                               deliberately does not duplicate it (it only
                               references games by game_id).

data/historical_odds.json is intentionally never read by this provider:
it holds stale (is_current=False) snapshots reserved for future RAG
document ingestion (Milestone 6), and the freshness contract in
src/providers/base.py requires that normal provider methods never expose
stale data as current.

Deviation from the suggested single `data_path` constructor: because Game
data and SportsbookOdds data live in two separate controlled files (by
Milestone 4's own design, to avoid duplicating odds inside every scenario
definition), the constructor takes a `data_dir` directory containing both
expected filenames, rather than a single file path. This mirrors the
DATA_DIR pattern already used by src/evaluation/dataset.py. No dataset
redesign was needed — this is the existing Milestone 4 file layout as-is.

This provider does not depend on src.tools, src.agents, src.rag, or
src.evaluation — only on domain models (src/models.py) and the standard
library, preserving the dependency direction defined in
src/providers/base.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models import Game, MarketType, SportsbookOdds
from src.providers.base import OddsProvider

_PROJECT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CURRENT_ODDS_FILENAME = "current_odds.json"
_SCENARIOS_FILENAME = "test_scenarios.json"


class GameNotFoundError(LookupError):
    """Raised when a requested game_id has no corresponding game in the
    controlled dataset. Never fabricate a placeholder Game instead."""


class OddsNotFoundError(LookupError):
    """Raised when no current odds match the requested combination of
    game, market, outcome, and (for single-sportsbook lookups) sportsbook.

    Covers every "unknown" case uniformly rather than a separate exception
    per case (unknown market, unknown outcome, unknown sportsbook, or a
    game/market/outcome with no current lines at all) — the provider layer
    deliberately keeps its error hierarchy small; the messages describe
    exactly which part of the request could not be satisfied. Never
    substitutes another sportsbook's or outcome's odds instead of raising.
    """


class ControlledOddsProvider(OddsProvider):
    """OddsProvider backed by the controlled JSON dataset in data/.

    Loads and validates data/current_odds.json and data/test_scenarios.json
    once, at construction time, using the existing Pydantic domain models
    (Game, SportsbookOdds, MarketType) for validation. Malformed JSON,
    missing files, and schema-invalid records all fail immediately and
    explicitly in the constructor — nothing invalid is silently discarded
    and no missing field is silently defaulted.

    Contains no betting logic: no best-line selection, no implied
    probability, no expected value. It only retrieves validated structured
    data, per the responsibility split documented in
    src/providers/base.py.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else _PROJECT_DATA_DIR
        self._games: dict[str, Game] = {}
        self._odds_index: dict[tuple[str, MarketType, str], list[SportsbookOdds]] = {}
        self._load()

    # -- loading ------------------------------------------------------

    def _load(self) -> None:
        scenario_definitions = self._read_json_array(self.data_dir / _SCENARIOS_FILENAME)
        current_odds_records = self._read_json_array(self.data_dir / _CURRENT_ODDS_FILENAME)

        self._games = self._build_games(scenario_definitions)
        self._odds_index = self._build_odds_index(current_odds_records)

    @staticmethod
    def _read_json_array(path: Path) -> list[dict]:
        if not path.is_file():
            raise FileNotFoundError(f"controlled dataset file not found: {path}")
        with path.open() as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON in {path}: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError(
                f"expected a JSON array of records in {path}, got {type(data).__name__}"
            )
        return data

    @staticmethod
    def _build_games(scenario_definitions: list[dict]) -> dict[str, Game]:
        games: dict[str, Game] = {}
        for definition in scenario_definitions:
            if "game" not in definition:
                raise ValueError(f"scenario definition missing 'game' field: {definition!r}")
            game = Game(**definition["game"])  # raises pydantic.ValidationError if invalid
            existing = games.get(game.game_id)
            if existing is not None and existing != game:
                raise ValueError(
                    f"conflicting Game data for game_id={game.game_id!r}: "
                    f"{existing!r} vs {game!r}"
                )
            games[game.game_id] = game
        return games

    def _build_odds_index(
        self, records: list[dict]
    ) -> dict[tuple[str, MarketType, str], list[SportsbookOdds]]:
        index: dict[tuple[str, MarketType, str], list[SportsbookOdds]] = {}
        for record in records:
            try:
                game_id = record["game_id"]
                market_type = MarketType(record["market_type"])  # raises ValueError if invalid
                selected_outcome = record["selected_outcome"]
                odds = SportsbookOdds(
                    sportsbook=record["sportsbook"],
                    american_odds=record["american_odds"],
                    is_current=record["is_current"],
                    timestamp=record.get("timestamp"),
                )
            except KeyError as exc:
                raise ValueError(f"odds record missing required field {exc}: {record!r}") from exc

            if not isinstance(game_id, str) or not game_id.strip():
                raise ValueError(f"odds record has invalid game_id: {record!r}")
            if game_id not in self._games:
                raise ValueError(
                    f"odds record references unknown game_id={game_id!r} "
                    f"(not present in {_SCENARIOS_FILENAME}): {record!r}"
                )
            if not isinstance(selected_outcome, str) or not selected_outcome.strip():
                raise ValueError(f"odds record has invalid selected_outcome: {record!r}")

            if not odds.is_current:
                # Freshness contract: normal lookups never expose stale data.
                # current_odds.json is defined to hold only current records
                # (stale snapshots live only in historical_odds.json, which
                # this provider never reads) — a non-current record here is
                # simply excluded rather than surfaced.
                continue

            key = (game_id, market_type, selected_outcome.strip())
            index.setdefault(key, []).append(odds)
        return index

    # -- OddsProvider interface ----------------------------------------

    def get_games(self) -> list[Game]:
        """Return all games in the controlled dataset, sorted by game_id
        for deterministic ordering."""
        return sorted(self._games.values(), key=lambda game: game.game_id)

    def get_game(self, game_id: str) -> Game:
        """Return the game matching game_id exactly.

        Raises:
            GameNotFoundError: game_id is not present in the controlled
                dataset. No fuzzy matching or ID guessing is performed.
        """
        try:
            return self._games[game_id]
        except KeyError:
            raise GameNotFoundError(f"unknown game_id: {game_id!r}") from None

    def get_odds(
        self,
        game_id: str,
        market_type: MarketType,
        selected_outcome: str,
    ) -> list[SportsbookOdds]:
        """Return all current sportsbook odds for this game/market/outcome,
        sorted by sportsbook name for deterministic ordering.

        Raises:
            GameNotFoundError: game_id is not present in the dataset.
            OddsNotFoundError: game_id is known, but no current odds exist
                for this exact market_type/selected_outcome combination
                (unknown market, unknown outcome, or simply no current
                lines). Never guesses at a different market or outcome.
        """
        if game_id not in self._games:
            raise GameNotFoundError(f"unknown game_id: {game_id!r}")

        key = (game_id, market_type, selected_outcome.strip())
        odds = self._odds_index.get(key)
        if not odds:
            raise OddsNotFoundError(
                f"no current odds found for game_id={game_id!r}, "
                f"market_type={market_type.value!r}, selected_outcome={selected_outcome!r}"
            )
        return sorted(odds, key=lambda o: o.sportsbook)

    def get_sportsbook_odds(
        self,
        game_id: str,
        sportsbook: str,
        market_type: MarketType,
        selected_outcome: str,
    ) -> SportsbookOdds:
        """Return the current odds from exactly one sportsbook.

        Matching is an exact (whitespace-trimmed) string comparison — no
        normalization or fuzzy matching beyond that already performed by
        the SportsbookOdds model.

        Raises:
            GameNotFoundError: game_id is not present in the dataset.
            OddsNotFoundError: the game/market/outcome is known but this
                sportsbook has no current line for it (unrecognized
                sportsbook name, or a recognized sportsbook that simply
                did not offer a line). Never substitutes another
                sportsbook's odds instead.
        """
        odds_list = self.get_odds(game_id, market_type, selected_outcome)
        target = sportsbook.strip()
        for odds in odds_list:
            if odds.sportsbook == target:
                return odds
        raise OddsNotFoundError(
            f"sportsbook {sportsbook!r} has no current odds for game_id={game_id!r}, "
            f"market_type={market_type.value!r}, selected_outcome={selected_outcome!r}"
        )
