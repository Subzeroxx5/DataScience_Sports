"""Core Pydantic data models for the sports agent research project.

These models are the shared data contract for controlled sportsbook data,
test scenarios, ground truth, and agent outputs. They are strictly
declarative: validation and representation only. Betting math (implied
probability, expected value, best-line selection, etc.) belongs in future
deterministic calculation modules, not here.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

# American odds outside this range are not realistic for the controlled
# scenarios in this study and likely indicate a data-entry error.
MIN_AMERICAN_ODDS = -100_000
MAX_AMERICAN_ODDS = 100_000


class MarketType(str, Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"


class ArchitectureType(str, Enum):
    RAG = "rag"
    TOOL = "tool"
    HYBRID = "hybrid"


class SourceType(str, Enum):
    RAG = "rag"
    TOOL = "tool"
    DATASET = "dataset"


def _validate_american_odds(value: int) -> int:
    if value == 0:
        raise ValueError("american_odds cannot be 0")
    if not (MIN_AMERICAN_ODDS <= value <= MAX_AMERICAN_ODDS):
        raise ValueError(
            f"american_odds must be between {MIN_AMERICAN_ODDS} and "
            f"{MAX_AMERICAN_ODDS}, got {value}"
        )
    return value


class SportsbookOdds(BaseModel):
    """One sportsbook's odds for one outcome."""

    sportsbook: str = Field(..., min_length=1)
    american_odds: int
    is_current: bool
    timestamp: datetime | None = None

    @field_validator("sportsbook")
    @classmethod
    def sportsbook_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("sportsbook cannot be empty or whitespace")
        return stripped

    @field_validator("american_odds")
    @classmethod
    def odds_in_valid_range(cls, value: int) -> int:
        return _validate_american_odds(value)


class Market(BaseModel):
    """What is being evaluated: a market type, outcome, and optional line."""

    market_type: MarketType
    selected_outcome: str = Field(..., min_length=1)
    line: float | None = None

    @field_validator("selected_outcome")
    @classmethod
    def outcome_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("selected_outcome cannot be empty or whitespace")
        return stripped

    @model_validator(mode="after")
    def line_required_for_spread_and_total(self) -> "Market":
        if self.market_type in (MarketType.SPREAD, MarketType.TOTAL) and self.line is None:
            raise ValueError(f"line is required for market_type={self.market_type.value}")
        return self


class Game(BaseModel):
    """A sporting event."""

    game_id: str = Field(..., min_length=1)
    home_team: str = Field(..., min_length=1)
    away_team: str = Field(..., min_length=1)
    start_time: datetime
    sport: str = Field(..., min_length=1)

    @field_validator("game_id", "home_team", "away_team", "sport")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty or whitespace")
        return stripped

    @model_validator(mode="after")
    def teams_must_differ(self) -> "Game":
        if self.home_team.strip().lower() == self.away_team.strip().lower():
            raise ValueError("home_team and away_team cannot be identical")
        return self


class SourceReference(BaseModel):
    """Provenance metadata for a piece of information used by an agent."""

    source_type: SourceType
    source_id: str = Field(..., min_length=1)
    sportsbook: str | None = None
    timestamp: datetime | None = None

    @field_validator("source_id")
    @classmethod
    def source_id_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("source_id cannot be empty or whitespace")
        return stripped


class TestScenario(BaseModel):
    """One controlled research test scenario."""

    scenario_id: str = Field(..., min_length=1)
    game: Game
    market: Market
    sportsbook_odds: list[SportsbookOdds]
    estimated_true_probability: float = Field(..., ge=0.0, le=1.0)

    @field_validator("scenario_id")
    @classmethod
    def scenario_id_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("scenario_id cannot be empty or whitespace")
        return stripped

    @field_validator("sportsbook_odds")
    @classmethod
    def at_least_one_sportsbook(cls, value: list[SportsbookOdds]) -> list[SportsbookOdds]:
        if len(value) == 0:
            raise ValueError("sportsbook_odds must contain at least one entry")
        return value

    @model_validator(mode="after")
    def no_duplicate_sportsbooks(self) -> "TestScenario":
        seen: set[str] = set()
        for odds in self.sportsbook_odds:
            key = odds.sportsbook.strip().lower()
            if key in seen:
                raise ValueError(
                    f"duplicate sportsbook entry for the same outcome: {odds.sportsbook!r}"
                )
            seen.add(key)
        return self

    @model_validator(mode="after")
    def outcome_matches_game(self) -> "TestScenario":
        if self.market.market_type == MarketType.MONEYLINE:
            valid_outcomes = {
                self.game.home_team.strip().lower(),
                self.game.away_team.strip().lower(),
            }
            if self.market.selected_outcome.strip().lower() not in valid_outcomes:
                raise ValueError(
                    f"selected_outcome {self.market.selected_outcome!r} does not "
                    f"correspond to either team in game {self.game.game_id!r}"
                )
        return self


class GroundTruth(BaseModel):
    """Deterministically-computed ground truth for a test scenario."""

    scenario_id: str = Field(..., min_length=1)
    expected_best_sportsbook: str = Field(..., min_length=1)
    expected_best_odds: int
    expected_implied_probability: float = Field(..., ge=0.0, le=1.0)
    expected_ev: float
    expected_positive_ev: bool
    expected_sportsbooks: list[str] = Field(default_factory=list)
    current_odds_version: str | None = None

    # All sportsbooks tied for the best available price, e.g. ["DraftKings",
    # "FanDuel"] when both offer the same best odds. If omitted, defaults to
    # a single-element list containing expected_best_sportsbook, so existing
    # single-sportsbook ground truth (Milestone 2) remains valid unchanged.
    expected_best_sportsbooks: list[str] = Field(default_factory=list)

    @field_validator("scenario_id", "expected_best_sportsbook")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty or whitespace")
        return stripped

    @field_validator("expected_best_odds")
    @classmethod
    def odds_in_valid_range(cls, value: int) -> int:
        return _validate_american_odds(value)

    @field_validator("expected_sportsbooks", "expected_best_sportsbooks")
    @classmethod
    def sportsbooks_not_blank(cls, value: list[str]) -> list[str]:
        for sportsbook in value:
            if not sportsbook.strip():
                raise ValueError("sportsbook list entries cannot be blank")
        return value

    @model_validator(mode="after")
    def default_and_validate_best_sportsbooks(self) -> "GroundTruth":
        if not self.expected_best_sportsbooks:
            self.expected_best_sportsbooks = [self.expected_best_sportsbook]
        if len(set(self.expected_best_sportsbooks)) != len(self.expected_best_sportsbooks):
            raise ValueError("expected_best_sportsbooks cannot contain duplicates")
        if self.expected_best_sportsbook not in self.expected_best_sportsbooks:
            raise ValueError(
                "expected_best_sportsbook must be a member of expected_best_sportsbooks"
            )
        return self


class BestLineResult(BaseModel):
    """The mathematically best currently available odds for one outcome,
    and every sportsbook tied at that price.

    Returned by the sportsbook tool contract find_best_line() (see
    src/tools/sportsbook_tools.py). Ties are preserved explicitly: if two
    or more sportsbooks offer equally favorable odds, all of them appear
    in `sportsbooks` rather than an arbitrary single choice.
    """

    game_id: str = Field(..., min_length=1)
    market_type: MarketType
    selected_outcome: str = Field(..., min_length=1)
    best_odds: int
    sportsbooks: list[str] = Field(..., min_length=1)

    @field_validator("game_id", "selected_outcome")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty or whitespace")
        return stripped

    @field_validator("best_odds")
    @classmethod
    def odds_in_valid_range(cls, value: int) -> int:
        return _validate_american_odds(value)

    @field_validator("sportsbooks")
    @classmethod
    def sportsbooks_not_blank_and_unique(cls, value: list[str]) -> list[str]:
        for sportsbook in value:
            if not sportsbook.strip():
                raise ValueError("sportsbooks entries cannot be blank")
        if len(set(value)) != len(value):
            raise ValueError("sportsbooks cannot contain duplicates")
        return value


class BettingAnalysis(BaseModel):
    """Common structured output all agent architectures must produce."""

    scenario_id: str = Field(..., min_length=1)
    game_id: str = Field(..., min_length=1)
    market: MarketType
    selected_outcome: str = Field(..., min_length=1)

    best_sportsbook: str = Field(..., min_length=1)
    best_odds: int

    implied_probability: float = Field(..., ge=0.0, le=1.0)
    estimated_true_probability: float = Field(..., ge=0.0, le=1.0)
    expected_value: float
    positive_ev: bool

    sportsbooks_considered: list[str]

    reasoning_summary: str = Field(..., min_length=1)
    architecture: ArchitectureType

    sources: list[SourceReference] = Field(default_factory=list)

    @field_validator(
        "scenario_id", "game_id", "selected_outcome", "best_sportsbook", "reasoning_summary"
    )
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty or whitespace")
        return stripped

    @field_validator("best_odds")
    @classmethod
    def odds_in_valid_range(cls, value: int) -> int:
        return _validate_american_odds(value)

    @field_validator("sportsbooks_considered")
    @classmethod
    def sportsbooks_not_blank(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("sportsbooks_considered must contain at least one entry")
        for sportsbook in value:
            if not sportsbook.strip():
                raise ValueError("sportsbooks_considered entries cannot be blank")
        return value
