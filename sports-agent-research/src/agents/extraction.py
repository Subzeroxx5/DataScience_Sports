"""Structured extraction schema, prompt, and provenance validation for the
RAG-only agent (Milestone 8B).

The LLM's job is evidence interpretation, not calculation: read the
rendered RAG context (src/agents/rag_evidence.py::render_rag_context) and
extract whatever sportsbook prices are explicitly present, as structured
data with provenance. It must never compute implied probability, no-vig
probability, market consensus, or expected value — those stay exclusively
in src/calculations/ (see RAG_EXTRACTION_SYSTEM_PROMPT below) and are
applied afterward, in src/agents/rag_agent.py, to whatever this module
validates.

validate_extraction_provenance() is the hallucination gate: an extracted
price is accepted only if every source_document_id it claims was actually
retrieved AND the retrieved document's own sportsbook/odds match what the
LLM reported. A real document_id paired with a fabricated sportsbook name
or altered odds value is rejected exactly like a fully invented
document_id — matching on the ID alone is not enough.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from src.agents.rag_evidence import RagEvidenceBundle
from src.models import MAX_AMERICAN_ODDS, MIN_AMERICAN_ODDS
from src.rag.documents import RagSourceType


RAG_EXTRACTION_SYSTEM_PROMPT = """You are an evidence interpreter for a controlled sports-betting research study. You are NOT a betting calculator and you must not act like one.

You will be given a set of retrieved documents (RAG evidence) about sportsbook prices for one betting market. Extract only the sportsbook price information that is EXPLICITLY present in the supplied evidence, as structured data.

Rules — follow all of them exactly:
- Use only the supplied evidence. Do not use outside knowledge about real sportsbooks, teams, leagues, or odds.
- Do not guess a missing sportsbook's odds. If a sportsbook does not appear in the evidence, do not include it.
- Do not fabricate a sportsbook name that does not appear in the evidence.
- Do not fabricate a document_id. Every value you place in source_document_ids must be copied exactly, character for character, from a document_id shown in the evidence.
- Extract numeric odds exactly as shown in the evidence — the same sign and the same magnitude. Do not round, adjust, average, or "correct" them.
- Preserve each document's is_current and timestamp exactly as shown. Do not infer freshness, and do not assume a document is current just because it appears first.
- If the evidence is insufficient to answer confidently, say so explicitly by filling in missing_evidence_note rather than inventing data or leaving the question unanswered with no explanation.
- Do NOT calculate implied probability, no-vig probability, market consensus, expected value, or any other betting math. A separate deterministic system performs every calculation from the data you extract.
- Do NOT decide which sportsbook is "best" and do NOT state a final conclusion, recommendation, or verdict. Extraction only — no analysis, no opinion.
"""


class ExtractedSportsbookPrice(BaseModel):
    """One sportsbook price the LLM claims to have found in the supplied
    evidence. Not trusted until validate_extraction_provenance() confirms
    every source_document_ids entry actually appeared in that evidence and
    matches its content."""

    sportsbook: str = Field(..., min_length=1)
    selected_outcome: str = Field(..., min_length=1)
    american_odds: int
    opposing_outcome: str | None = None
    opposing_american_odds: int | None = None
    source_document_ids: list[str] = Field(..., min_length=1)
    timestamp: datetime | None = None
    is_current: bool | None = None

    @field_validator("sportsbook", "selected_outcome")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty or whitespace")
        return stripped

    @field_validator("american_odds", "opposing_american_odds")
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

    @field_validator("source_document_ids")
    @classmethod
    def source_document_ids_not_blank(cls, value: list[str]) -> list[str]:
        for document_id in value:
            if not document_id.strip():
                raise ValueError("source_document_ids entries cannot be blank")
        return value


class ExtractedMarketEvidence(BaseModel):
    """Everything the LLM extracted for one AgentRequest's market."""

    game_id: str = Field(..., min_length=1)
    market_id: str = Field(..., min_length=1)
    selected_outcome: str = Field(..., min_length=1)
    sportsbook_prices: list[ExtractedSportsbookPrice] = Field(default_factory=list)
    missing_evidence_note: str | None = None


def validate_extraction_provenance(
    extraction: ExtractedMarketEvidence,
    evidence_bundle: RagEvidenceBundle,
) -> tuple[list[ExtractedSportsbookPrice], list[str]]:
    """Split extracted prices into (accepted, rejection_reasons).

    A price is rejected — never silently repaired — if:
    - any source_document_ids entry was not actually retrieved
      (hallucinated document_id), or
    - a claimed source document exists but its own sportsbook or
      american_odds field does not match what the LLM reported
      (hallucinated sportsbook or hallucinated odds, even against a real
      document_id), or
    - it duplicates/conflicts with an already-accepted
      (sportsbook, selected_outcome) pair.

    An accepted price's opposing_outcome/opposing_american_odds claim is
    additionally checked against retrieved evidence for that same
    sportsbook; if unverifiable, only those two fields are cleared — the
    primary (already-verified) price itself is still accepted.
    """
    retrieved_by_id = {item.document_id: item.document for item in evidence_bundle.evidence}
    accepted: list[ExtractedSportsbookPrice] = []
    reasons: list[str] = []
    seen_sportsbook_outcome: set[tuple[str, str]] = set()

    for price in extraction.sportsbook_prices:
        unknown_ids = [
            document_id
            for document_id in price.source_document_ids
            if document_id not in retrieved_by_id
        ]
        if unknown_ids:
            reasons.append(
                f"{price.sportsbook}/{price.selected_outcome}: hallucinated "
                f"source_document_ids not present in retrieved evidence: {unknown_ids}"
            )
            continue

        # A price's source_document_ids may legitimately include documents
        # for BOTH the primary outcome and the opposing outcome (see the
        # opposing-side check below) — so the primary claim only needs to
        # be corroborated by at least one cited document, not all of them.
        # A source document that belongs to a different (real) outcome or
        # sportsbook is not itself a hallucination; a primary claim with
        # zero corroborating documents is.
        primary_confirmed = any(
            retrieved_by_id[document_id].source_type == RagSourceType.SPORTSBOOK_SNAPSHOT
            and retrieved_by_id[document_id].sportsbook == price.sportsbook
            and retrieved_by_id[document_id].selected_outcome == price.selected_outcome
            and retrieved_by_id[document_id].american_odds == price.american_odds
            for document_id in price.source_document_ids
        )
        if not primary_confirmed:
            reasons.append(
                f"{price.sportsbook}/{price.selected_outcome} at {price.american_odds:+d}: "
                "no cited source_document_ids entry actually confirms this sportsbook, "
                "outcome, and odds combination (hallucinated sportsbook and/or odds)"
            )
            continue

        key = (price.sportsbook, price.selected_outcome)
        if key in seen_sportsbook_outcome:
            reasons.append(
                f"{price.sportsbook}/{price.selected_outcome}: duplicate or conflicting "
                "record for the same sportsbook and outcome"
            )
            continue
        seen_sportsbook_outcome.add(key)

        if price.opposing_outcome is not None or price.opposing_american_odds is not None:
            opposing_document = next(
                (
                    document
                    for document in retrieved_by_id.values()
                    if document.source_type == RagSourceType.SPORTSBOOK_SNAPSHOT
                    and document.sportsbook == price.sportsbook
                    and document.selected_outcome == price.opposing_outcome
                ),
                None,
            )
            if (
                opposing_document is None
                or opposing_document.american_odds != price.opposing_american_odds
            ):
                price = price.model_copy(
                    update={"opposing_outcome": None, "opposing_american_odds": None}
                )

        accepted.append(price)

    return accepted, reasons
