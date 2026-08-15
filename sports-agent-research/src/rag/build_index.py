"""Deterministic vector index build command (Milestone 6C).

    data/rag_documents/corpus.jsonl
              |
              v
    load + validate each line as a RagDocument (src/rag/documents.py)
              |
              v
    embed RagDocument.content only (src/rag/embeddings.py) --
    never ground_truth, expected_best_sportsbook, expected_ev, or any
    other prohibited field (see docs/EXPERIMENT_RULES.md)
              |
              v
    build an exact FAISS index (src/rag/vector_index.py)
              |
              v
    data/rag_index/index.faiss, metadata.json, config.json

Run:

    python -m src.rag.build_index

`config.json` records enough information to reproduce the index
(embedding model, dimension, similarity metric, corpus path, document
count) without any nondeterministic runtime metadata (no wall-clock
timestamp, no hostname, no random seed) that would prevent two builds
from producing equivalent output.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rag.build_corpus import CORPUS_PATH
from src.rag.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDINGS_NORMALIZED,
    EmbeddingModel,
)
from src.rag.documents import RagDocument
from src.rag.vector_index import VectorIndex

RAG_INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "rag_index"
CONFIG_FILENAME = "config.json"
SIMILARITY_METRIC = "inner_product_normalized (cosine equivalent)"


def load_corpus_documents(corpus_path: Path = CORPUS_PATH) -> list[RagDocument]:
    documents = []
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            documents.append(RagDocument(**json.loads(line)))
    return documents


def build_index(
    corpus_path: Path = CORPUS_PATH,
    embedding_model: EmbeddingModel | None = None,
) -> tuple[VectorIndex, list[RagDocument]]:
    documents = load_corpus_documents(corpus_path)
    if not documents:
        raise ValueError(f"no documents found in {corpus_path}")

    model = embedding_model or EmbeddingModel()
    contents = [doc.content for doc in documents]
    vectors = model.encode_documents(contents)

    document_ids = [doc.document_id for doc in documents]
    metadata = {doc.document_id: doc.model_dump(mode="json") for doc in documents}

    index = VectorIndex(dimension=vectors.shape[1])
    index.build_index(vectors, document_ids, metadata)
    return index, documents


def export_index(
    output_dir: Path = RAG_INDEX_DIR,
    corpus_path: Path = CORPUS_PATH,
) -> Path:
    index, documents = build_index(corpus_path)
    index.save_index(output_dir)

    config = {
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embeddings_normalized": EMBEDDINGS_NORMALIZED,
        "similarity_metric": SIMILARITY_METRIC,
        "corpus_path": str(corpus_path.relative_to(corpus_path.parent.parent.parent)),
        "document_count": len(documents),
    }
    with (output_dir / CONFIG_FILENAME).open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    return output_dir


if __name__ == "__main__":
    written_dir = export_index()
    index, documents = build_index()
    print(f"Corpus documents: {len(documents)}")
    print(f"Indexed vectors: {index.ntotal}")
    print(f"Embedding model: {DEFAULT_EMBEDDING_MODEL}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
    print(f"Similarity metric: {SIMILARITY_METRIC}")
    print(f"Wrote index to {written_dir}")
