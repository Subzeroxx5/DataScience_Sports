"""Semantic retriever over the persisted vector index (Milestone 6C).

This module performs retrieval only — no LLM call, no answer generation,
no reasoning. It is deliberately kept independent of any agent so
retrieval quality can be evaluated on its own (see
src/rag/evaluate_retrieval.py and milestones/current.md's "Required
Gate").

Retrieval must NOT filter out stale (is_current=False) documents. A
RAG-only architecture needs to be able to retrieve stale snapshots so
freshness failures can be measured later (see docs/EXPERIMENT_RULES.md).
Metadata filtering may be added in the future, but baseline retrieve()
always searches the full corpus as indexed.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from src.rag.build_index import RAG_INDEX_DIR
from src.rag.documents import RagDocument
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_index import VectorIndex


class RetrievalResult(BaseModel):
    """One ranked search result, with enough information to inspect the
    game/market/outcome/sportsbook/odds/timestamp/freshness/content behind
    it without a second lookup."""

    document_id: str = Field(..., min_length=1)
    score: float
    rank: int = Field(..., ge=1)
    document: RagDocument


class Retriever:
    """Loads a persisted VectorIndex and an EmbeddingModel, and retrieves
    the k most semantically similar documents for a query."""

    def __init__(
        self,
        index: VectorIndex,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.index = index
        self.embedding_model = embedding_model or EmbeddingModel()

    @classmethod
    def from_directory(cls, directory: Path | str = RAG_INDEX_DIR) -> "Retriever":
        index = VectorIndex.load_index(directory)
        return cls(index)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Return the top-k documents most semantically similar to query.

        No filtering by is_current, source_type, or any other metadata —
        baseline retrieval searches the full indexed corpus. Results are
        ordered by descending similarity score (rank 1 = most similar).
        """
        query_embedding = self.embedding_model.encode_query(query)
        raw_results = self.index.search(query_embedding, k=k)

        results = []
        for rank, (document_id, score) in enumerate(raw_results, start=1):
            document = RagDocument(**self.index.metadata[document_id])
            results.append(
                RetrievalResult(
                    document_id=document_id,
                    score=score,
                    rank=rank,
                    document=document,
                )
            )
        return results
