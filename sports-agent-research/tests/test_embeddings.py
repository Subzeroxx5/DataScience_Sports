"""Tests for src/rag/embeddings.py (Milestone 6C)."""

import numpy as np
import pytest

from src.rag.embeddings import EMBEDDING_DIMENSION, EmbeddingModel


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel()


def test_document_embeddings_are_generated(model):
    vectors = model.encode_documents(["DraftKings lists Lakers at +120 on the moneyline."])
    assert vectors.shape == (1, EMBEDDING_DIMENSION)


def test_query_embeddings_are_generated(model):
    vector = model.encode_query("What price did DraftKings have on the Lakers?")
    assert vector.shape == (EMBEDDING_DIMENSION,)


def test_single_document_encode_matches_batch(model):
    text = "FanDuel lists Boston Celtics at -145 on the moneyline."
    single = model.encode_document(text)
    batch = model.encode_documents([text])
    assert single.shape == (EMBEDDING_DIMENSION,)
    assert np.allclose(single, batch[0])


def test_output_dimensions_match_between_query_and_document(model):
    doc_vec = model.encode_document("Some sportsbook content.")
    query_vec = model.encode_query("Some query.")
    assert doc_vec.shape == query_vec.shape == (EMBEDDING_DIMENSION,)


def test_vectors_contain_finite_values(model):
    vectors = model.encode_documents(
        ["DraftKings lists Lakers at +120.", "FanDuel lists Celtics at -145."]
    )
    assert np.all(np.isfinite(vectors))
    query = model.encode_query("Lakers odds")
    assert np.all(np.isfinite(query))


def test_vectors_are_normalized(model):
    vectors = model.encode_documents(["A short document.", "A longer document with more words."])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    query_norm = np.linalg.norm(model.encode_query("A query"))
    assert query_norm == pytest.approx(1.0, abs=1e-5)


def test_empty_document_list_rejected(model):
    with pytest.raises(ValueError):
        model.encode_documents([])


def test_blank_document_text_rejected(model):
    with pytest.raises(ValueError):
        model.encode_documents(["   "])


def test_blank_query_text_rejected(model):
    with pytest.raises(ValueError):
        model.encode_query("")


def test_model_name_is_centralized():
    from src.rag.embeddings import DEFAULT_EMBEDDING_MODEL

    assert DEFAULT_EMBEDDING_MODEL
    default_model = EmbeddingModel()
    assert default_model.model_name == DEFAULT_EMBEDDING_MODEL


def test_no_llm_sdk_imported():
    import ast
    from pathlib import Path

    source_path = Path(__file__).resolve().parent.parent / "src" / "rag" / "embeddings.py"
    tree = ast.parse(source_path.read_text())
    forbidden = {"openai", "anthropic", "langchain", "llama_index"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden
