"""Tests for src/rag/vector_index.py (Milestone 6C). Uses small synthetic
vectors, not the real embedding model, to keep these tests fast and
isolated to indexing/persistence logic."""

import numpy as np
import pytest

from src.rag.vector_index import VectorIndex

DIMENSION = 8


def _unit_vectors(n: int, dimension: int = DIMENSION, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.random((n, dimension)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def _sample_index(n: int = 5) -> tuple[VectorIndex, np.ndarray, list[str], dict]:
    vectors = _unit_vectors(n)
    document_ids = [f"doc-{i}" for i in range(n)]
    metadata = {doc_id: {"content": f"content {i}"} for i, doc_id in enumerate(document_ids)}
    index = VectorIndex(dimension=DIMENSION)
    index.build_index(vectors, document_ids, metadata)
    return index, vectors, document_ids, metadata


def test_index_builds():
    index, vectors, document_ids, metadata = _sample_index()
    assert index.ntotal == 5


def test_correct_vector_count_is_stored():
    index, vectors, document_ids, metadata = _sample_index(n=7)
    assert index.ntotal == 7
    assert len(index.document_ids) == 7


def test_search_returns_self_as_top_result():
    index, vectors, document_ids, metadata = _sample_index()
    results = index.search(vectors[0], k=1)
    assert results[0][0] == document_ids[0]
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_search_respects_k():
    index, vectors, document_ids, metadata = _sample_index(n=5)
    results = index.search(vectors[0], k=3)
    assert len(results) == 3


def test_search_results_are_rank_ordered_by_descending_score():
    index, vectors, document_ids, metadata = _sample_index()
    results = index.search(vectors[0], k=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_k_larger_than_corpus_is_clamped():
    index, vectors, document_ids, metadata = _sample_index(n=3)
    results = index.search(vectors[0], k=100)
    assert len(results) == 3


def test_search_on_empty_index_returns_empty_list():
    index = VectorIndex(dimension=DIMENSION)
    query = _unit_vectors(1)[0]
    assert index.search(query, k=5) == []


def test_search_rejects_wrong_dimension_query():
    index, vectors, document_ids, metadata = _sample_index()
    bad_query = np.zeros(DIMENSION + 1, dtype=np.float32)
    with pytest.raises(ValueError):
        index.search(bad_query, k=1)


def test_build_index_rejects_mismatched_lengths():
    index = VectorIndex(dimension=DIMENSION)
    vectors = _unit_vectors(3)
    with pytest.raises(ValueError):
        index.build_index(vectors, ["a", "b"], {"a": {}, "b": {}})


def test_build_index_rejects_duplicate_document_ids():
    index = VectorIndex(dimension=DIMENSION)
    vectors = _unit_vectors(2)
    with pytest.raises(ValueError):
        index.build_index(vectors, ["a", "a"], {"a": {}})


def test_build_index_rejects_missing_metadata():
    index = VectorIndex(dimension=DIMENSION)
    vectors = _unit_vectors(2)
    with pytest.raises(ValueError):
        index.build_index(vectors, ["a", "b"], {"a": {}})


def test_save_and_load_preserves_searchable_index(tmp_path):
    index, vectors, document_ids, metadata = _sample_index()
    index.save_index(tmp_path)

    reloaded = VectorIndex.load_index(tmp_path)
    assert reloaded.ntotal == index.ntotal
    original_results = index.search(vectors[0], k=3)
    reloaded_results = reloaded.search(vectors[0], k=3)
    assert original_results == reloaded_results


def test_save_and_load_preserves_document_id_mapping(tmp_path):
    index, vectors, document_ids, metadata = _sample_index()
    index.save_index(tmp_path)
    reloaded = VectorIndex.load_index(tmp_path)
    assert reloaded.document_ids == document_ids


def test_save_and_load_preserves_metadata(tmp_path):
    index, vectors, document_ids, metadata = _sample_index()
    index.save_index(tmp_path)
    reloaded = VectorIndex.load_index(tmp_path)
    assert reloaded.metadata == metadata


def test_load_detects_index_metadata_mismatch(tmp_path):
    index, vectors, document_ids, metadata = _sample_index()
    index.save_index(tmp_path)

    import json

    metadata_path = tmp_path / "metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["document_ids"].append("extra-doc-not-in-index")
    metadata_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        VectorIndex.load_index(tmp_path)
