"""Tests for src/rag/retriever.py (Milestone 6C). Uses the real, persisted
corpus index (built once via build_index.py) so retrieval is exercised
end-to-end against actual corpus content."""

import ast
from pathlib import Path

import pytest

from src.rag.build_index import export_index
from src.rag.documents import RagDocument
from src.rag.retriever import Retriever, RetrievalResult


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("rag_index")
    export_index(output_dir=index_dir)
    return Retriever.from_directory(index_dir)


def test_known_query_returns_results(retriever):
    results = retriever.retrieve("What price did DraftKings have on the Lakers moneyline?", k=5)
    assert len(results) == 5


def test_top_result_is_the_expected_document(retriever):
    results = retriever.retrieve("What price did DraftKings have on the Lakers moneyline?", k=1)
    assert results[0].document_id == "g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"


def test_k_is_respected(retriever):
    for k in (1, 3, 5, 10):
        results = retriever.retrieve("Lakers moneyline odds", k=k)
        assert len(results) == k


def test_results_are_rank_ordered(retriever):
    results = retriever.retrieve("Celtics moneyline odds", k=5)
    ranks = [r.rank for r in results]
    assert ranks == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_scores_are_present_and_finite(retriever):
    results = retriever.retrieve("Timberwolves moneyline odds", k=3)
    for r in results:
        assert isinstance(r.score, float)
        assert r.score == r.score  # not NaN


def test_returned_documents_are_structured_rag_documents(retriever):
    results = retriever.retrieve("Grizzlies moneyline odds", k=3)
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert isinstance(r.document, RagDocument)
        assert r.document.document_id == r.document_id


def test_stale_documents_are_not_silently_filtered(retriever):
    # DraftKings' stale Timberwolves document (is_current=False) must be
    # retrievable, not excluded by default.
    results = retriever.retrieve(
        "What price did DraftKings list for the Minnesota Timberwolves moneyline?", k=5
    )
    document_ids = {r.document_id for r in results}
    assert "g-2026-009-moneyline-minnesota-timberwolves-draftkings-v0" in document_ids
    stale_present = any(
        r.document_id == "g-2026-009-moneyline-minnesota-timberwolves-draftkings-v0"
        and r.document.is_current is False
        for r in results
    )
    assert stale_present


def test_retrieval_result_preserves_full_document_metadata(retriever):
    results = retriever.retrieve("DraftKings Lakers moneyline price", k=1)
    doc = results[0].document
    assert doc.game_id is not None
    assert doc.market_id is not None
    assert doc.market_type is not None
    assert doc.selected_outcome is not None
    assert doc.sportsbook is not None
    assert doc.american_odds is not None
    assert doc.timestamp is not None
    assert doc.is_current is not None
    assert doc.content


def test_retriever_module_never_imports_llm_sdk():
    source_path = Path(__file__).resolve().parent.parent / "src" / "rag" / "retriever.py"
    tree = ast.parse(source_path.read_text())
    forbidden = {"openai", "anthropic", "langchain", "llama_index"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden
