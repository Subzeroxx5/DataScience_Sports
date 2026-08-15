"""Deterministic source reconciliation for the hybrid agent (Milestone
10A).

This module contains the ONE place that decides, per (sportsbook,
outcome), which of a RAG-derived price and a tool-derived price is
authoritative for current-line analysis. It is pure Python — no LLM
involvement of any kind (milestones/current.md, Milestone 10A Step 16:
"The final decision about which numeric sportsbook odds enter the quant
engine must be deterministic... Instead implement explicit Python
rules").

Authoritative-source policy (docs/EXPERIMENT_RULES.md, "Hybrid Boundary"
+ milestones/current.md Step 12):

    1. current structured sportsbook tool data — always wins when present
    2. current RAG evidence — only when no tool result exists for that
       (sportsbook, outcome) AND the RAG evidence's own is_current flag
       is True (never promote stale-RAG-only or unknown-freshness data)
    3. stale (or freshness-unknown) RAG-only evidence — contextual/
       history only, never used as a current authoritative price

Do NOT average the two prices. Do NOT let an LLM choose between them.
Every conflict (both sources present with different values) is recorded
with an explicit resolution reason, never silently dropped.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from src.models import SourceType


class ConflictResolutionReason(str, Enum):
    """Why authoritative_odds/authoritative_source were chosen the way
    they were — recorded for every record, not only conflicts, so the
    freshness research metric (docs/PROJECT_STATE.md) can be computed
    directly from the trace."""

    SOURCES_AGREE = "sources_agree"
    CURRENT_TOOL_DATA_PRECEDENCE = "current_tool_data_precedence"
    TOOL_ONLY = "tool_only"
    RAG_CURRENT_NO_TOOL_COVERAGE = "rag_current_no_tool_coverage"
    STALE_RAG_ONLY_NOT_PROMOTED = "stale_rag_only_not_promoted"


class RagPriceObservation(BaseModel):
    """One RAG-extracted, provenance-validated price for a single
    sportsbook/outcome (built from an accepted
    src.agents.extraction.ExtractedSportsbookPrice — never from raw,
    unvalidated LLM output)."""

    american_odds: int
    is_current: bool | None
    timestamp: datetime | None
    document_id: str = Field(..., min_length=1)


class HybridMarketRecord(BaseModel):
    """One sportsbook's reconciled record for one outcome — the ONLY
    structure the quant engine and the final BettingAnalysis may read
    numeric current odds from (milestones/current.md, Step 19/20)."""

    sportsbook: str = Field(..., min_length=1)
    selected_outcome: str = Field(..., min_length=1)

    tool_odds: int | None = None
    rag_odds: int | None = None
    tool_available: bool
    rag_available: bool
    rag_is_current: bool | None = None
    rag_timestamp: datetime | None = None
    rag_document_id: str | None = None

    conflict: bool
    authoritative_odds: int | None = None
    authoritative_source: SourceType | None = None
    conflict_resolution_reason: ConflictResolutionReason | None = None


def reconcile_outcome(
    selected_outcome: str,
    tool_prices: dict[str, int],
    rag_prices: dict[str, RagPriceObservation],
) -> list[HybridMarketRecord]:
    """Reconcile every sportsbook seen from either channel for one
    outcome, applying the authoritative-source policy above. Each
    sportsbook contributes exactly one HybridMarketRecord (Step 20: "Each
    sportsbook/outcome should contribute at most one authoritative
    value").
    """
    sportsbooks = sorted(set(tool_prices) | set(rag_prices))
    records: list[HybridMarketRecord] = []

    for sportsbook in sportsbooks:
        tool_odds = tool_prices.get(sportsbook)
        rag_observation = rag_prices.get(sportsbook)
        rag_odds = rag_observation.american_odds if rag_observation is not None else None

        tool_available = tool_odds is not None
        rag_available = rag_observation is not None
        rag_is_current = rag_observation.is_current if rag_observation is not None else None
        conflict = tool_available and rag_available and tool_odds != rag_odds

        authoritative_odds: int | None
        authoritative_source: SourceType | None
        reason: ConflictResolutionReason | None

        if tool_available:
            authoritative_odds = tool_odds
            authoritative_source = SourceType.TOOL
            if not rag_available:
                reason = ConflictResolutionReason.TOOL_ONLY
            elif conflict:
                reason = ConflictResolutionReason.CURRENT_TOOL_DATA_PRECEDENCE
            else:
                reason = ConflictResolutionReason.SOURCES_AGREE
        elif rag_available and rag_is_current:
            authoritative_odds = rag_odds
            authoritative_source = SourceType.RAG
            reason = ConflictResolutionReason.RAG_CURRENT_NO_TOOL_COVERAGE
        elif rag_available:
            # Stale, or freshness-unknown, RAG-only evidence: contextual/
            # history only — never promoted to a current authoritative
            # price (docs/EXPERIMENT_RULES.md default policy).
            authoritative_odds = None
            authoritative_source = None
            reason = ConflictResolutionReason.STALE_RAG_ONLY_NOT_PROMOTED
        else:
            authoritative_odds = None
            authoritative_source = None
            reason = None

        records.append(
            HybridMarketRecord(
                sportsbook=sportsbook,
                selected_outcome=selected_outcome,
                tool_odds=tool_odds,
                rag_odds=rag_odds,
                tool_available=tool_available,
                rag_available=rag_available,
                rag_is_current=rag_is_current,
                rag_timestamp=rag_observation.timestamp if rag_observation is not None else None,
                rag_document_id=rag_observation.document_id if rag_observation is not None else None,
                conflict=conflict,
                authoritative_odds=authoritative_odds,
                authoritative_source=authoritative_source,
                conflict_resolution_reason=reason,
            )
        )

    return records
