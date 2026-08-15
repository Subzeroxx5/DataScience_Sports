"""Deterministic retrieval evaluation (Milestone 6C).

Measures retrieval quality only — Recall@K and Hit@K over the retriever
(src/rag/retriever.py) against the predefined query set
(data/retrieval_queries.json). This is NOT agent accuracy: no LLM
reasoning is involved anywhere in this module.

Run:

    python -m src.rag.evaluate_retrieval

Metric definitions, for a single query with a known relevant-document set:

    Recall@K = (relevant documents retrieved in top K) / (total relevant documents)
    Hit@K    = 1 if at least one relevant document appears in top K, else 0

Both are averaged (unweighted mean) across all queries to produce the
aggregate Recall@K / Hit@K reported below.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rag.retriever import Retriever, RetrievalResult

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RETRIEVAL_QUERIES_PATH = DATA_DIR / "retrieval_queries.json"

K_VALUES = (1, 3, 5)


def load_queries(path: Path = RETRIEVAL_QUERIES_PATH) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> int:
    top_k = set(retrieved_ids[:k])
    return 1 if (top_k & relevant_ids) else 0


def evaluate_query(
    retriever: Retriever, query: dict, k_values: tuple[int, ...] = K_VALUES
) -> dict:
    relevant_ids = set(query["relevant_document_ids"])
    max_k = max(k_values)
    results: list[RetrievalResult] = retriever.retrieve(query["query_text"], k=max_k)
    retrieved_ids = [r.document_id for r in results]

    metrics = {
        f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values
    }
    metrics.update({f"hit@{k}": hit_at_k(retrieved_ids, relevant_ids, k) for k in k_values})

    return {
        "query_id": query["query_id"],
        "query_text": query["query_text"],
        "relevant_document_ids": sorted(relevant_ids),
        "retrieved_document_ids": retrieved_ids,
        "results": results,
        "metrics": metrics,
    }


def evaluate_all(
    retriever: Retriever | None = None,
    queries_path: Path = RETRIEVAL_QUERIES_PATH,
    k_values: tuple[int, ...] = K_VALUES,
) -> list[dict]:
    retriever = retriever or Retriever.from_directory()
    queries = load_queries(queries_path)
    return [evaluate_query(retriever, q, k_values) for q in queries]


def aggregate_metrics(evaluations: list[dict], k_values: tuple[int, ...] = K_VALUES) -> dict:
    n = len(evaluations)
    if n == 0:
        raise ValueError("evaluations cannot be empty")
    aggregate = {}
    for k in k_values:
        aggregate[f"recall@{k}"] = sum(e["metrics"][f"recall@{k}"] for e in evaluations) / n
        aggregate[f"hit@{k}"] = sum(e["metrics"][f"hit@{k}"] for e in evaluations) / n
    return aggregate


def failed_queries(evaluations: list[dict], k: int = 5) -> list[dict]:
    """Queries where no relevant document appears in the top k."""
    return [e for e in evaluations if e["metrics"][f"hit@{k}"] == 0]


if __name__ == "__main__":
    retriever = Retriever.from_directory()
    evaluations = evaluate_all(retriever)
    aggregate = aggregate_metrics(evaluations)

    print(f"Queries evaluated: {len(evaluations)}")
    print()
    for k in K_VALUES:
        print(f"Recall@{k}: {aggregate[f'recall@{k}']:.4f}")
    print()
    for k in K_VALUES:
        print(f"Hit@{k}: {aggregate[f'hit@{k}']:.4f}")

    failures = failed_queries(evaluations, k=5)
    print()
    print(f"Failed queries (no relevant document in top 5): {len(failures)}")
    for failure in failures:
        print()
        print(f"  query_id: {failure['query_id']}")
        print(f"  query: {failure['query_text']}")
        print(f"  expected: {failure['relevant_document_ids']}")
        print(f"  retrieved top 5: {failure['retrieved_document_ids']}")
