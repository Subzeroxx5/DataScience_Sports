"""Sportsbook tools: the ONLY boundary through which future tool-calling
and hybrid agents may access sportsbook information.

Per the architecture boundaries in docs/research_design.md, the
tool-calling-only and hybrid architectures must reach the controlled
dataset exclusively through this layer:

    Agent
      -> SportsbookTools    (this module)
      -> OddsProvider       (src/providers/base.py — Milestone 5B)
      -> ControlledOddsProvider (src/providers/controlled.py — Milestone 5C)
      -> Controlled Dataset (data/current_odds.json, etc. — Milestone 4)

Agents must never read data/current_odds.json, data/test_scenarios.json,
or data/ground_truth.json directly. Enforcing that boundary at the tool
layer is what makes "tool-calling-only" vs. "RAG-only" a meaningful,
measurable architectural difference for this study.

Provider injection (Milestone 5D): SportsbookTools is constructed with an
OddsProvider and delegates get_games/get_game/get_odds/get_sportsbook_odds
directly to it. This module depends only on the abstract OddsProvider type
(src/providers/base.py), never on a concrete provider such as
ControlledOddsProvider — so the same SportsbookTools code works unchanged
against a future ApiOddsProvider. It performs no JSON parsing, holds no
dataset file path, and does no re-validation or re-transformation of
provider results: whatever structured model the provider returns is
returned as-is.

Error propagation: provider exceptions are NOT caught, wrapped, or
translated here. If provider.get_game(...) raises because game_id is
unknown, that exception propagates to the caller unchanged. This is a
deliberate simplicity choice for this milestone (see module-level
docstring notes on SportsbookToolError below) — translating would require
this module to know the concrete exception types of whichever provider is
injected, which would defeat provider swappability.

find_best_line() remains intentionally unimplemented in this milestone
(still raises NotImplementedError) — best-line selection is business
logic that reuses src/calculations/odds_math.compare_american_odds, and
is deferred to Milestone 5E.

Freshness rule (binding on every method below):

    Sportsbook tools represent the current structured-data source. Stale
    records must never be returned as current odds. The controlled
    dataset may contain stale records (data/historical_odds.json) for
    later RAG experiments, but these tools only ever expose current,
    is_current=True data — enforced by the injected OddsProvider, not by
    this module (see src/providers/controlled.py).

Missing-data rule: American odds of 0 are invalid (see
src/models.py::_validate_american_odds) and must never be used to signal
"no odds available." Absence is always represented by the provider raising
an explicit exception, never by a sentinel odds value.

No betting math is implemented or duplicated in this module. Implied
probability, decimal odds, and expected value belong exclusively to
src/calculations/odds_math.py.

SportsbookToolError and its subclasses below are the tool layer's own
documented exception vocabulary, established in Milestone 5A. As of this
milestone, provider exceptions are not yet translated into this
hierarchy (see "Error propagation" above); it remains defined for forward
compatibility and as the target contract for a future translation layer,
should one prove necessary once more than one provider implementation
exists.
"""

from __future__ import annotations

from src.models import BestLineResult, Game, MarketType, SportsbookOdds
from src.providers.base import OddsProvider


class SportsbookToolError(Exception):
    """Base class for all sportsbook tool contract errors."""


class GameNotFoundError(SportsbookToolError):
    """Raised when game_id does not correspond to a known game.

    The future implementation must raise this rather than fabricating a
    Game record for an unrecognized game_id.
    """


class MarketNotFoundError(SportsbookToolError):
    """Raised when no odds exist for the requested market_type on this game.

    The future implementation must raise this rather than guessing at or
    substituting a different market type.
    """


class OutcomeNotFoundError(SportsbookToolError):
    """Raised when selected_outcome is not valid for this game/market.

    The future implementation must raise this rather than automatically
    falling back to the home team, away team, or any other outcome.
    """


class SportsbookNotFoundError(SportsbookToolError):
    """Raised when the requested sportsbook has no record for this game/market/outcome.

    This covers both an unrecognized sportsbook name and a recognized
    sportsbook that simply did not offer a line (the controlled dataset's
    missing-data scenarios omit the record entirely rather than using
    american_odds=0). The future implementation must raise this rather
    than silently substituting another sportsbook.
    """


class NoOddsAvailableError(SportsbookToolError):
    """Raised when a game/market/outcome is otherwise valid but no current
    sportsbook odds exist for it (e.g. all books have pulled their line).

    Must never be represented by returning american_odds=0 or an empty
    successful result that looks like valid data.
    """


class SportsbookTools:
    """Sportsbook tool layer, wired to an injected OddsProvider.

    Construct with any OddsProvider implementation:

        tools = SportsbookTools(ControlledOddsProvider())

    and later, without any change to this class:

        tools = SportsbookTools(ApiOddsProvider(...))

    Every method below is a thin delegation to the provider: it forwards
    arguments unchanged, returns the provider's structured result
    unchanged, and lets provider exceptions propagate unchanged. No JSON
    parsing, dataset file paths, or betting math live in this class.
    """

    def __init__(self, provider: OddsProvider) -> None:
        self.provider = provider

    def get_games(self) -> list[Game]:
        """Return every sporting event available from the current structured-data source.

        Returns:
            A list of Game objects, exactly as returned by the injected
            provider's get_games(). An empty list is a valid (non-error)
            result if no games are currently available.

        Freshness:
            Reflects the current-data source only (see module docstring).
        """
        return self.provider.get_games()

    def get_game(self, game_id: str) -> Game:
        """Return structured information for one specific game.

        Args:
            game_id: Identifier of the game to look up.

        Returns:
            The Game matching game_id, exactly as returned by the
            provider.

        Freshness:
            Reflects the current-data source only (see module docstring).

        Raises:
            Whatever exception the injected provider raises for an
            unknown game_id (see src/providers/base.py — providers must
            fail explicitly rather than fabricating a placeholder Game).
            Not translated or caught here.
        """
        return self.provider.get_game(game_id)

    def get_odds(
        self,
        game_id: str,
        market_type: MarketType,
        selected_outcome: str,
    ) -> list[SportsbookOdds]:
        """Return all currently available sportsbook odds for a game/market/outcome.

        Args:
            game_id: Identifier of the game.
            market_type: Market to query (moneyline, spread, or total).
            selected_outcome: The specific outcome within that market (e.g.
                a team name for moneyline/spread, or "Over"/"Under" for
                total).

        Returns:
            A list of SportsbookOdds exactly as returned by the provider —
            one per sportsbook currently offering a line for this exact
            game/market/outcome. Never includes stale (is_current=False)
            records; a sportsbook with no current line is simply absent,
            never represented with american_odds=0.

        Freshness:
            Current records only — enforced by the provider, not this
            method (see module docstring).

        Raises:
            Whatever exception the injected provider raises for an
            unknown game, market, or outcome, or for a valid
            game/market/outcome with no current lines at all. Not
            translated or caught here — never falls back to a different
            outcome or fabricates data.
        """
        return self.provider.get_odds(game_id, market_type, selected_outcome)

    def get_sportsbook_odds(
        self,
        game_id: str,
        sportsbook: str,
        market_type: MarketType,
        selected_outcome: str,
    ) -> SportsbookOdds:
        """Return the current odds from one specific sportsbook.

        Args:
            game_id: Identifier of the game.
            sportsbook: Name of the sportsbook to query (e.g. "DraftKings").
            market_type: Market to query (moneyline, spread, or total).
            selected_outcome: The specific outcome within that market.

        Returns:
            The single current SportsbookOdds record exactly as returned
            by the provider.

        Freshness:
            Current records only — enforced by the provider, not this
            method (see module docstring).

        Raises:
            Whatever exception the injected provider raises for an
            unknown game, sportsbook, market, or outcome. Not translated
            or caught here — never substitutes another sportsbook's odds.
        """
        return self.provider.get_sportsbook_odds(
            game_id, sportsbook, market_type, selected_outcome
        )

    def find_best_line(
        self,
        game_id: str,
        market_type: MarketType,
        selected_outcome: str,
    ) -> BestLineResult:
        """Return the mathematically best currently available American odds,
        and every sportsbook tied at that price.

        Args:
            game_id: Identifier of the game.
            market_type: Market to query (moneyline, spread, or total).
            selected_outcome: The specific outcome within that market.

        Returns:
            A BestLineResult whose `best_odds` is the most favorable
            current American odds for this game/market/outcome (per the
            favorability ordering defined in
            src/calculations/odds_math.compare_american_odds — never
            recomputed here) and whose `sportsbooks` lists every
            sportsbook offering odds equally favorable to that best
            price. If multiple sportsbooks are tied, all of them are
            included; the future implementation must never arbitrarily
            narrow a tie down to one sportsbook.

            Example: DraftKings +125, FanDuel +125, BetMGM +120,
            Caesars +118 must yield best_odds=125,
            sportsbooks=["DraftKings", "FanDuel"].

        Freshness:
            Computed only over current records — never considers stale
            odds as candidates for best line (see module docstring).

        Raises:
            GameNotFoundError: game_id does not correspond to a known game.
            MarketNotFoundError: no odds exist for market_type on this game.
            OutcomeNotFoundError: selected_outcome is not valid for this
                game/market.
            NoOddsAvailableError: the game/market/outcome is valid but no
                sportsbook currently offers a line for it, so no best
                line can be determined.

        Not implemented in this milestone (Milestone 5D). The comparison
        and tie-detection logic will reuse
        src/calculations/odds_math.compare_american_odds rather than
        reimplementing odds-favorability math here — deferred to
        Milestone 5E.
        """
        raise NotImplementedError
