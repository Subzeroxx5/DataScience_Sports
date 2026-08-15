"""Deterministic RAG evidence pipeline (Milestone 8A).

The boundary between the verified retriever (src/rag/retriever.py,
Milestone 6C) and the future RAG-only LLM agent (Milestone 8B). No LLM
call happens anywhere in this module.

    AgentRequest (src/agents/base.py)
              |
              v
    Retriever.retrieve()  (src/rag/retriever.py — the ONE retrieval
                            implementation; never reimplemented here)
              |
              v
    Top-K RetrievalResults
              |
              v
    RagEvidenceBundle (this module)

Architecture isolation (see docs/EXPERIMENT_RULES.md, "RAG-Only
Boundary"): this module must never import src.tools, src.providers, or
read data/current_odds.json / data/ground_truth.json /
data/quant_ground_truth.json directly. The only path to sportsbook
information here is RAG corpus -> vector index -> retriever -> this
evidence pipeline. Enforced by an AST-based test in
tests/test_rag_evidence.py.

Ordering (see docs/EXPERIMENT_RULES.md, "RAG-Only Boundary" and
milestones/current.md, Step 7): evidence items preserve the retriever's
rank and similarity_score exactly as returned. Nothing here reorders,
filters, or removes documents based on freshness (is_current), odds
favorability, or any other criterion — a stale document retrieved at
rank 1 stays at rank 1. This is essential to measuring RAG freshness
behavior honestly instead of silently repairing it at the evidence layer.

No quantitative calculation (no-vig, consensus, edge, EV) happens here —
this milestone establishes what evidence RAG receives, not what it
concludes from that evidence. That extraction-and-calculation step is
Milestone 8B's responsibility, reusing src/calculations/ exactly as every
other layer does.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents.base import AgentRequest
from src.models import MarketType
from src.rag.documents import RagDocument, RagSourceType
from src.rag.retriever import Retriever, RetrievalResult

DEFAULT_RAG_TOP_K = 5


class RagEvidenceItem(BaseModel):
    """One retrieved document, reusing RagDocument wholesale (not
    duplicating its fields) alongside the retrieval rank/score that made
    it part of this evidence bundle."""

    document_id: str = Field(..., min_length=1)
    rank: int = Field(..., ge=1)
    similarity_score: float
    document: RagDocument

    @classmethod
    def from_retrieval_result(cls, result: RetrievalResult) -> "RagEvidenceItem":
        return cls(
            document_id=result.document_id,
            rank=result.rank,
            similarity_score=result.score,
            document=result.document,
        )


class RagEvidenceBundle(BaseModel):
    """Everything retrieved for one AgentRequest, in retrieval order.

    Contains no ground truth, expected answer, or quant benchmark label
    of any kind — only the request identity, the query used, and the
    evidence actually retrieved.
    """

    scenario_id: str = Field(..., min_length=1)
    game_id: str = Field(..., min_length=1)
    market_type: MarketType
    selected_outcome: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(..., ge=1)
    evidence: list[RagEvidenceItem] = Field(default_factory=list)


class EvidenceDiagnostics(BaseModel):
    """Debugging/evaluation diagnostics derived from retrieved evidence
    only — never consults ground truth or any other out-of-band source."""

    sportsbooks_found: list[str]
    outcomes_found: list[str]
    contains_both_market_sides: bool
    contains_stale_documents: bool
    contains_current_documents: bool


def build_rag_evidence_bundle(
    request: AgentRequest,
    retriever: Retriever,
    k: int = DEFAULT_RAG_TOP_K,
) -> RagEvidenceBundle:
    """Retrieve evidence for request.query and assemble it into a bundle.

    Args:
        request: identifies the scenario/game/market/outcome and carries
            the natural-language query to retrieve evidence for. Never
            contains ground truth (see AgentRequest).
        retriever: a Retriever already loaded from a persisted vector
            index (Retriever.from_directory()) — the sole retrieval path.
        k: retrieval depth. Defaults to DEFAULT_RAG_TOP_K; override only
            for testing, not to tune per-scenario results.
    """
    results = retriever.retrieve(request.query, k=k)
    evidence = [RagEvidenceItem.from_retrieval_result(result) for result in results]
    return RagEvidenceBundle(
        scenario_id=request.scenario_id,
        game_id=request.game_id,
        market_type=request.market_type,
        selected_outcome=request.selected_outcome,
        query=request.query,
        top_k=k,
        evidence=evidence,
    )


def render_rag_context(bundle: RagEvidenceBundle) -> str:
    """Deterministically render a bundle's evidence as plain text, in
    retrieval order, with clear per-document boundaries and full
    provenance/freshness metadata. No LLM involved; no summarization or
    alteration of document content.
    """
    lines = [f"Query: {bundle.query}", ""]
    for item in bundle.evidence:
        document = item.document
        lines.append(f"[DOCUMENT {item.rank}]")
        lines.append(f"document_id: {item.document_id}")
        lines.append(f"rank: {item.rank}")
        lines.append(f"similarity_score: {item.similarity_score:.4f}")
        lines.append(f"source_type: {document.source_type.value}")
        lines.append(f"game_id: {document.game_id}")
        if document.market_id is not None:
            lines.append(f"market_id: {document.market_id}")
        if document.market_type is not None:
            lines.append(f"market_type: {document.market_type.value}")
        if document.selected_outcome is not None:
            lines.append(f"outcome: {document.selected_outcome}")
        if document.sportsbook is not None:
            lines.append(f"sportsbook: {document.sportsbook}")
        if document.american_odds is not None:
            lines.append(f"american_odds: {document.american_odds:+d}")
        if document.timestamp is not None:
            lines.append(f"timestamp: {document.timestamp.isoformat()}")
        if document.is_current is not None:
            lines.append(f"is_current: {document.is_current}")
        lines.append(f"content: {document.content}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def compute_evidence_diagnostics(bundle: RagEvidenceBundle) -> EvidenceDiagnostics:
    """Derived entirely from bundle.evidence — never consults ground
    truth. contains_both_market_sides is inferred from how many distinct
    outcome names appear among retrieved sportsbook_snapshot documents
    (>=2 means both sides of a two-sided market were retrieved), not from
    any external knowledge of which two outcomes a market "should" have.
    """
    sportsbook_docs = [
        item.document
        for item in bundle.evidence
        if item.document.source_type == RagSourceType.SPORTSBOOK_SNAPSHOT
    ]
    sportsbooks_found = sorted(
        {doc.sportsbook for doc in sportsbook_docs if doc.sportsbook is not None}
    )
    outcomes_found = sorted(
        {doc.selected_outcome for doc in sportsbook_docs if doc.selected_outcome is not None}
    )
    return EvidenceDiagnostics(
        sportsbooks_found=sportsbooks_found,
        outcomes_found=outcomes_found,
        contains_both_market_sides=len(outcomes_found) >= 2,
        contains_stale_documents=any(doc.is_current is False for doc in sportsbook_docs),
        contains_current_documents=any(doc.is_current is True for doc in sportsbook_docs),
    )
