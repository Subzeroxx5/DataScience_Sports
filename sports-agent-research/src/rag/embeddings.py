"""Embedding configuration and a thin wrapper around SentenceTransformers
(Milestone 6C).

Model selection:

- Name: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Normalized: yes — embeddings are unit-length (L2 norm == 1.0), both by
  this model's own default behavior and by explicitly passing
  `normalize_embeddings=True` below (belt-and-suspenders; documented so
  the guarantee doesn't silently depend on a model-specific default).
- Why this model: it is one of the most widely used general-purpose
  sentence-embedding models, small enough to run quickly on CPU (~80MB,
  6-layer), and well suited to short factual sentences like this
  project's controlled RAG corpus content — a good fit for "small
  general-purpose text retrieval model appropriate for local CPU
  execution" rather than a large model this controlled benchmark does
  not need.

The model name is centralized here (`DEFAULT_EMBEDDING_MODEL`) rather
than repeated elsewhere; every other module that needs embeddings
imports `EmbeddingModel` from this file instead of instantiating
SentenceTransformer directly.

Document and query embeddings use the identical model. `encode_document`
and `encode_query` are kept as separate methods (using this model's
`encode_document`/`encode_query` calls, which are the retrieval-oriented
entry points SentenceTransformers exposes) so a future swap to an
asymmetric retrieval model that *does* distinguish query vs. document
encoding requires no caller changes.

This module only ever embeds `RagDocument.content` (or raw query text
passed in by the retriever). It has no knowledge of ground truth, ORM/DB
concerns, or any prohibited field — see src/rag/documents.py and
docs/EXPERIMENT_RULES.md for what a RagDocument may/may not contain.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDINGS_NORMALIZED = True


class EmbeddingModel:
    """Loads one SentenceTransformers model and encodes text to vectors.

    All vectors returned by this class are L2-normalized float32 arrays
    of shape (EMBEDDING_DIMENSION,) or (n, EMBEDDING_DIMENSION), suitable
    for inner-product similarity search (see src/rag/vector_index.py).
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        actual_dimension = self._model.get_embedding_dimension()
        if actual_dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"model {model_name!r} produces {actual_dimension}-dimensional "
                f"embeddings, but EMBEDDING_DIMENSION is {EMBEDDING_DIMENSION}. "
                "Update EMBEDDING_DIMENSION if you intentionally changed the model."
            )

    def encode_document(self, text: str) -> np.ndarray:
        """Encode one document's content to a single normalized vector."""
        return self.encode_documents([text])[0]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode multiple documents' content to normalized vectors, shape (n, dim)."""
        if not texts:
            raise ValueError("texts cannot be empty")
        for text in texts:
            if not text or not text.strip():
                raise ValueError("cannot embed empty or whitespace-only text")
        vectors = self._model.encode_document(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode one query string to a single normalized vector."""
        if not text or not text.strip():
            raise ValueError("cannot embed empty or whitespace-only text")
        vector = self._model.encode_query(
            text, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vector, dtype=np.float32)
