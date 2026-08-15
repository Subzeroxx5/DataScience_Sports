"""Exact vector index over the RAG corpus, backed by FAISS (Milestone 6C).

Index type: `faiss.IndexFlatIP` — an exact, brute-force inner-product
index. No IVF/HNSW/PQ or other approximate-nearest-neighbor structure:
the controlled corpus (under 100 documents as of Milestone 6B) is small
enough that exact search is cheap, and retrieval *correctness* matters
far more than speed at this stage (see milestones/current.md).

Similarity metric: inner product over L2-normalized vectors, which is
mathematically equivalent to cosine similarity. Every vector that enters
this index (via build_index or search) is expected to already be
normalized by src/rag/embeddings.py — this module does not re-normalize,
so mixing normalized and unnormalized vectors would silently break the
cosine-similarity interpretation. See test_vector_index.py for a check
that query/index vectors share the same dimension and are unit-length.

Document ID mapping: FAISS only stores raw vectors and returns integer
row positions from search(). This module keeps a parallel
`document_ids: list[str]` so every row position maps back to a
document_id, and a `metadata: dict[str, dict]` so every document_id maps
back to its full RagDocument payload — preserving the chain

    vector index position -> document_id -> RagDocument metadata

required by Milestone 6C.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


class VectorIndex:
    """Exact inner-product FAISS index with a document_id <-> row mapping."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self.document_ids: list[str] = []
        self.metadata: dict[str, dict] = {}

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def build_index(self, vectors: np.ndarray, document_ids: list[str], metadata: dict[str, dict]) -> None:
        """Build the index from scratch. Replaces any existing contents.

        Args:
            vectors: shape (n, dimension), float32, L2-normalized.
            document_ids: length n, document_ids[i] corresponds to vectors[i].
            metadata: document_id -> full document payload (e.g. a
                RagDocument's model_dump()), used to resolve search
                results back to their source document.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"expected vectors of shape (n, {self.dimension}), got {vectors.shape}"
            )
        if len(document_ids) != vectors.shape[0]:
            raise ValueError(
                f"document_ids length ({len(document_ids)}) must match "
                f"vectors row count ({vectors.shape[0]})"
            )
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document_ids must be unique")
        missing = set(document_ids) - set(metadata.keys())
        if missing:
            raise ValueError(f"metadata missing entries for document_ids: {sorted(missing)}")

        self._index = faiss.IndexFlatIP(self.dimension)
        self._index.add(vectors.astype(np.float32))
        self.document_ids = list(document_ids)
        self.metadata = dict(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Search for the k nearest documents to query_embedding.

        Returns a list of (document_id, score) tuples, ordered by
        descending score (most similar first). `score` is the raw
        inner-product / cosine similarity from FAISS (higher is better).
        """
        if query_embedding.shape != (self.dimension,):
            raise ValueError(
                f"expected a query vector of shape ({self.dimension},), "
                f"got {query_embedding.shape}"
            )
        if self.ntotal == 0:
            return []
        k = min(k, self.ntotal)
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query, k)
        results = []
        for score, row in zip(scores[0], indices[0]):
            if row == -1:
                continue
            results.append((self.document_ids[row], float(score)))
        return results

    def save_index(self, directory: Path | str) -> None:
        """Persist the FAISS index, document_id mapping, and metadata."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILENAME))
        payload = {
            "dimension": self.dimension,
            "document_ids": self.document_ids,
            "metadata": self.metadata,
        }
        with (directory / METADATA_FILENAME).open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def load_index(cls, directory: Path | str) -> "VectorIndex":
        """Reload a previously saved index (see save_index)."""
        directory = Path(directory)
        with (directory / METADATA_FILENAME).open() as f:
            payload = json.load(f)

        instance = cls(dimension=payload["dimension"])
        instance._index = faiss.read_index(str(directory / INDEX_FILENAME))
        instance.document_ids = payload["document_ids"]
        instance.metadata = payload["metadata"]

        if instance._index.ntotal != len(instance.document_ids):
            raise ValueError(
                f"loaded index has {instance._index.ntotal} vectors but "
                f"{len(instance.document_ids)} document_ids — index/metadata mismatch"
            )
        return instance
