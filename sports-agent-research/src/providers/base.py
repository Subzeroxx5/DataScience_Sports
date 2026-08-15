"""Odds provider abstraction (interface only — no data-source implementation yet).

This module defines the boundary between the sportsbook tool layer
(src/tools/sportsbook_tools.py) and wherever sportsbook data actually
lives. The tool layer, and everything above it (agents, evaluation),
depends only on this abstract interface — never on a concrete data
source. That is what lets a controlled JSON provider be swapped for a
live API provider later without touching tools, agents, or evaluation
code:

    Agent
      -> Sportsbook Tools
      -> OddsProvider (this module)      <- concrete providers plug in here
           - ControlledOddsProvider (future: Milestone 5C, reads data/*.json)
           - ApiOddsProvider (future, reads a live sportsbook API)

Dependency direction (must not be violated):

    models  <-  providers  <-  tools  <-  agents

This module may depend on domain models (src/models.py) and the standard
library only. It must never import src/tools, src/agents, src/rag, an LLM
SDK, or an HTTP/API client library — those are exactly the concerns a
concrete provider implementation is allowed to have that this interface
is not.

Responsibility split (do not blur this line):

    OddsProvider  -> retrieves structured data, as-is, from a data source.
    Sportsbook tools -> interpret that data (e.g. determine the best odds
                         across sportsbooks via find_best_line()).

Best-line selection is therefore intentionally NOT part of this
interface: it is business/interpretation logic, and duplicating it across
every future provider implementation (JSON, live API, ...) would risk
each one computing "best odds" slightly differently. There is exactly one
place that logic will live: the tool layer, reusing
src/calculations/odds_math.py.

Freshness contract (binding on every concrete implementation):

    Provider methods that return current sportsbook odds must return only
    records designated as current (is_current=True) by the underlying
    data source. The controlled research dataset may contain both current
    and stale records (see data/current_odds.json vs.
    data/historical_odds.json) — stale records exist for later RAG
    experiments and must never be surfaced by these methods as current
    data.

Missing-data contract (binding on every concrete implementation):

    - Unknown game_id: must not fabricate a Game. Fail explicitly rather
      than returning a placeholder.
    - Unknown sportsbook: must not silently substitute another
      sportsbook's odds.
    - Unknown market_type for a game: must not guess or fall back to a
      different market.
    - Unknown selected_outcome: must not automatically select the home
      team, away team, or any other outcome.
    - No current odds for an otherwise-valid game/market/outcome: must be
      explicitly and unambiguously distinguishable from a real odds
      value. American odds of 0 is invalid (see
      src/models.py::_validate_american_odds) and must never be used to
      mean "no odds available."

    This interface intentionally does not prescribe a specific exception
    hierarchy yet (kept deliberately simple at this stage — see
    src/tools/sportsbook_tools.py for the richer error contract the tool
    layer already defines above this provider). A concrete implementation
    must raise or return some explicit, documented failure for each case
    above rather than fabricating data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import Game, MarketType, SportsbookOdds


class OddsProvider(ABC):
    """Abstract interface for a source of structured sportsbook data.

    Concrete subclasses (e.g. a future ControlledOddsProvider backed by
    the JSON dataset, or a future ApiOddsProvider backed by a live
    sportsbook API) supply the actual data retrieval. This class defines
    only the contract every data source must satisfy.
    """

    @abstractmethod
    def get_games(self) -> list[Game]:
        """Return all games available from the underlying data source.

        Returns:
            A list of Game objects. An empty list is a valid result if no
            games are currently available; this is not an error.

        Freshness:
            Reflects only the current-data view of the underlying source.
        """
        raise NotImplementedError

    @abstractmethod
    def get_game(self, game_id: str) -> Game:
        """Return one specific game.

        Args:
            game_id: Identifier of the game to look up.

        Returns:
            The Game matching game_id.

        Missing-data contract:
            If game_id does not correspond to a known game, the
            implementation must fail explicitly rather than fabricating a
            placeholder Game.
        """
        raise NotImplementedError

    @abstractmethod
    def get_odds(
        self,
        game_id: str,
        market_type: MarketType,
        selected_outcome: str,
    ) -> list[SportsbookOdds]:
        """Return currently available sportsbook odds for a game/market/outcome.

        Args:
            game_id: Identifier of the game.
            market_type: Market to query (moneyline, spread, or total).
            selected_outcome: The specific outcome within that market.

        Returns:
            A list of SportsbookOdds — one per sportsbook currently
            offering a line for this exact game/market/outcome. A
            sportsbook with no current line is simply absent from the
            list; it is never represented with american_odds=0.

        Freshness:
            Only current (is_current=True) records may be returned; stale
            records must never be included.

        Missing-data contract:
            Unknown game, unknown market, or unknown outcome must each
            fail explicitly rather than guessing or substituting a
            different game/market/outcome. If the game/market/outcome is
            valid but no sportsbook currently has a line, that must also
            be explicit and unambiguous — never an empty list disguised
            as "no lines exist" when the real cause is an invalid
            game/market/outcome, and never american_odds=0.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sportsbook_odds(
        self,
        game_id: str,
        sportsbook: str,
        market_type: MarketType,
        selected_outcome: str,
    ) -> SportsbookOdds:
        """Return current odds from one specific sportsbook.

        Args:
            game_id: Identifier of the game.
            sportsbook: Name of the sportsbook to query.
            market_type: Market to query (moneyline, spread, or total).
            selected_outcome: The specific outcome within that market.

        Returns:
            The single current SportsbookOdds record for this sportsbook
            on this exact game/market/outcome.

        Freshness:
            Only a current (is_current=True) record may be returned; a
            stale record must never be returned as if it were current.

        Missing-data contract:
            Unknown game, unknown sportsbook, unknown market, or unknown
            outcome must each fail explicitly. A sportsbook that simply
            did not offer a line for this game/market/outcome must not be
            silently substituted with another sportsbook's odds.
        """
        raise NotImplementedError
