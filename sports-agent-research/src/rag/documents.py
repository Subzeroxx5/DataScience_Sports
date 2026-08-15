"""RAG document schema (Milestone 6B).

A RagDocument represents one piece of text a future RAG-only or hybrid
architecture could retrieve. This module defines the schema only —
generation lives in src/rag/build_corpus.py, and there is intentionally
no embedding, vector store, or retriever code here (Milestone 6C).

RagDocument is deliberately NOT an embedding/retrieval object: it has no
`embedding`, `vector`, or `similarity_score` field. Those concerns belong
entirely to the future retrieval layer, kept separate from what a document
*is* versus how it gets *found*.

Ground-truth leakage is a hard constraint: no RagDocument may express or
imply expected_best_sportsbook, expected_best_odds, expected_ev,
expected_positive_ev, a reference/consensus probability, or any other
value derived from src/evaluation/ground_truth.py. This module cannot
enforce that at the type level (content is free text), so
tests/test_rag_corpus.py enforces it with an automated leakage scan
instead.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from src.models import MAX_AMERICAN_ODDS, MIN_AMERICAN_ODDS, MarketType


class RagSourceType(str, Enum):
    """Classifies what kind of information a RagDocument carries.

    sportsbook_snapshot -- one sportsbook's price for one outcome at one
        point in time, expressed as text (current or intentionally stale).
    game_context -- general, non-numeric context about a game (a
        controlled "preview" style note). Carries no odds.
    market_context -- general, non-numeric context about a specific
        market on a game (e.g. that multiple sportsbooks track it).
        May reference market_id/market_type but never a specific
        sportsbook's price.
    """

    SPORTSBOOK_SNAPSHOT = "sportsbook_snapshot"
    GAME_CONTEXT = "game_context"
    MARKET_CONTEXT = "market_context"


def _non_blank(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} cannot be empty or whitespace")
    return stripped


class RagDocument(BaseModel):
    """One retrievable document in the controlled RAG corpus.

    Not every field is required for every document: contextual documents
    (game_context, market_context) carry no sportsbook or odds
    information at all. sportsbook_snapshot documents must carry a full,
    consistent set of sportsbook/odds/market fields (enforced below).
    """

    document_id: str = Field(..., min_length=1)
    source_type: RagSourceType
    content: str = Field(..., min_length=1)

    game_id: str = Field(..., min_length=1)
    market_id: str | None = None
    market_type: MarketType | None = None
    selected_outcome: str | None = None

    sportsbook: str | None = None
    american_odds: int | None = None
    is_current: bool | None = None
    timestamp: datetime | None = None
    version: str | None = None

    @field_validator("document_id", "content", "game_id")
    @classmethod
    def required_fields_not_blank(cls, value: str, info) -> str:
        return _non_blank(value, info.field_name)

    @field_validator("market_id", "selected_outcome", "sportsbook", "version")
    @classmethod
    def optional_fields_not_blank_if_present(cls, value: str | None, info) -> str | None:
        if value is None:
            return value
        return _non_blank(value, info.field_name)

    @field_validator("american_odds")
    @classmethod
    def odds_in_valid_range_if_present(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value == 0:
            raise ValueError("american_odds cannot be 0")
        if not (MIN_AMERICAN_ODDS <= value <= MAX_AMERICAN_ODDS):
            raise ValueError(
                f"american_odds must be between {MIN_AMERICAN_ODDS} and "
                f"{MAX_AMERICAN_ODDS}, got {value}"
            )
        return value

    @model_validator(mode="after")
    def sportsbook_snapshot_fields_complete(self) -> "RagDocument":
        if self.source_type != RagSourceType.SPORTSBOOK_SNAPSHOT:
            return self
        missing = [
            name
            for name, value in (
                ("sportsbook", self.sportsbook),
                ("american_odds", self.american_odds),
                ("market_id", self.market_id),
                ("market_type", self.market_type),
                ("selected_outcome", self.selected_outcome),
                ("is_current", self.is_current),
                ("timestamp", self.timestamp),
                ("version", self.version),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"source_type=sportsbook_snapshot requires {missing} to be set"
            )
        return self
