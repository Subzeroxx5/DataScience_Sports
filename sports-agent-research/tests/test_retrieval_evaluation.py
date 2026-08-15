"""Tests for src/rag/evaluate_retrieval.py (Milestone 6C).

Covers the Recall@K / Hit@K math with hand-verifiable synthetic examples,
plus structural checks on data/retrieval_queries.json, before running the
real evaluation end-to-end against the persisted corpus index.
"""

import json
from pathlib import Path

import pytest

from src.rag.build_index import export_index
from src.rag.evaluate_retrieval import (
    RETRIEVAL_QUERIES_PATH,
    aggregate_metrics,
    evaluate_all,
    failed_queries,
    hit_at_k,
    load_queries,
    recall_at_k,
)
from src.rag.retriever import Retriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Recall@K / Hit@K math — synthetic, hand-verifiable examples
# ---------------------------------------------------------------------------


def test_recall_at_k_all_relevant_found():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_recall_at_k_partial():
    retrieved = ["a", "x", "y"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.5)


def test_recall_at_k_zero():
    retrieved = ["x", "y", "z"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)


def test_recall_at_k_respects_k_cutoff():
    # relevant document is at position 3, so recall@2 must be 0.
    retrieved = ["x", "y", "a"]
    relevant = {"a"}
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_recall_at_k_rejects_empty_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)


def test_hit_at_k_single_relevant_document_present():
    assert hit_at_k(["a", "b", "c"], {"a"}, k=3) == 1


def test_hit_at_k_single_relevant_document_absent():
    assert hit_at_k(["x", "y", "z"], {"a"}, k=3) == 0


def test_hit_at_k_respects_k_cutoff():
    retrieved = ["x", "y", "a"]
    assert hit_at_k(retrieved, {"a"}, k=2) == 0
    assert hit_at_k(retrieved, {"a"}, k=3) == 1


def test_aggregate_metrics_known_synthetic_example():
    # Two queries, hand-computed expected aggregate.
    evaluations = [
        {
            "metrics": {
                "recall@1": 1.0,
                "recall@3": 1.0,
                "hit@1": 1,
                "hit@3": 1,
            }
        },
        {
            "metrics": {
                "recall@1": 0.0,
                "recall@3": 0.5,
                "hit@1": 0,
                "hit@3": 1,
            }
        },
    ]
    aggregate = aggregate_metrics(evaluations, k_values=(1, 3))
    assert aggregate["recall@1"] == pytest.approx(0.5)
    assert aggregate["recall@3"] == pytest.approx(0.75)
    assert aggregate["hit@1"] == pytest.approx(0.5)
    assert aggregate["hit@3"] == pytest.approx(1.0)


def test_aggregate_metrics_rejects_empty_evaluations():
    with pytest.raises(ValueError):
        aggregate_metrics([])


# ---------------------------------------------------------------------------
# data/retrieval_queries.json structure
# ---------------------------------------------------------------------------


def test_retrieval_queries_file_exists():
    assert RETRIEVAL_QUERIES_PATH.is_file()


def test_at_least_ten_queries():
    queries = load_queries()
    assert len(queries) >= 10


def test_query_ids_are_unique():
    queries = load_queries()
    ids = [q["query_id"] for q in queries]
    assert len(ids) == len(set(ids))


def test_every_query_has_required_fields():
    for query in load_queries():
        assert query["query_id"].strip()
        assert query["query_text"].strip()
        assert len(query["relevant_document_ids"]) >= 1


def test_relevant_document_ids_exist_in_corpus():
    corpus_path = DATA_DIR / "rag_documents" / "corpus.jsonl"
    corpus_ids = set()
    with corpus_path.open() as f:
        for line in f:
            if line.strip():
                corpus_ids.add(json.loads(line)["document_id"])

    for query in load_queries():
        for doc_id in query["relevant_document_ids"]:
            assert doc_id in corpus_ids, f"{doc_id!r} (from {query['query_id']}) not in corpus"


def test_current_and_stale_cases_represented():
    queries = load_queries()
    query_types = {q["query_type"] for q in queries}
    assert "current_snapshot_target" in query_types
    assert "stale_snapshot_target" in query_types


def test_two_sided_market_query_present():
    queries = load_queries()
    two_sided = [q for q in queries if q["query_type"] == "two_sided_market"]
    assert len(two_sided) >= 1
    assert len(two_sided[0]["relevant_document_ids"]) == 2


def test_freshness_query_present():
    queries = load_queries()
    freshness = [q for q in queries if q["query_type"] == "freshness_stale_vs_current"]
    assert len(freshness) >= 1


# ---------------------------------------------------------------------------
# End-to-end evaluation against the real corpus index
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluations(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("rag_index")
    export_index(output_dir=index_dir)
    retriever = Retriever.from_directory(index_dir)
    return evaluate_all(retriever)


def test_evaluate_all_covers_every_query(evaluations):
    assert len(evaluations) == len(load_queries())


def test_aggregate_metrics_are_between_zero_and_one(evaluations):
    aggregate = aggregate_metrics(evaluations)
    for value in aggregate.values():
        assert 0.0 <= value <= 1.0


def test_no_failed_queries_at_k5(evaluations):
    # Documents actual retrieval quality: with this corpus and model, no
    # query should fail to surface any relevant document in the top 5.
    failures = failed_queries(evaluations, k=5)
    assert failures == [], f"unexpected retrieval failures: {[f['query_id'] for f in failures]}"


def test_module_docstring_clarifies_retrieval_only_not_agent_accuracy():
    from src.rag import evaluate_retrieval

    docstring = (evaluate_retrieval.__doc__ or "").lower()
    assert "not agent accuracy" in docstring or "not\nagent accuracy" in docstring
