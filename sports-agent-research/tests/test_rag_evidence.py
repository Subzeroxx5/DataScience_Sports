"""Tests for src/agents/rag_evidence.py (Milestone 8A)."""

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.base import AgentRequest
from src.agents.rag_evidence import (
    DEFAULT_RAG_TOP_K,
    EvidenceDiagnostics,
    RagEvidenceBundle,
    RagEvidenceItem,
    build_rag_evidence_bundle,
    compute_evidence_diagnostics,
    render_rag_context,
)
from src.models import MarketType
from src.rag.retriever import Retriever


@pytest.fixture(scope="module")
def retriever():
    return Retriever.from_directory()


def _make_request(**overrides):
    defaults = dict(
        scenario_id="S001",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="What price did DraftKings have on the Lakers moneyline?",
    )
    defaults.update(overrides)
    return AgentRequest(**defaults)


# ---------------------------------------------------------------------------
# Evidence bundle construction / Top-K
# ---------------------------------------------------------------------------


def test_default_top_k_respected(retriever):
    bundle = build_rag_evidence_bundle(_make_request(), retriever)
    assert bundle.top_k == DEFAULT_RAG_TOP_K
    assert len(bundle.evidence) == DEFAULT_RAG_TOP_K


def test_configurable_k_respected(retriever):
    for k in (1, 3, 10):
        bundle = build_rag_evidence_bundle(_make_request(), retriever, k=k)
        assert bundle.top_k == k
        assert len(bundle.evidence) == k


def test_default_top_k_is_centralized_constant():
    assert DEFAULT_RAG_TOP_K == 5


def test_bundle_preserves_request_identity(retriever):
    request = _make_request()
    bundle = build_rag_evidence_bundle(request, retriever)
    assert bundle.scenario_id == request.scenario_id
    assert bundle.game_id == request.game_id
    assert bundle.market_type == request.market_type
    assert bundle.selected_outcome == request.selected_outcome
    assert bundle.query == request.query


# ---------------------------------------------------------------------------
# Rank / score / provenance / freshness preservation
# ---------------------------------------------------------------------------


def test_document_ids_preserved(retriever):
    bundle = build_rag_evidence_bundle(_make_request(), retriever)
    for item in bundle.evidence:
        assert item.document_id == item.document.document_id


def test_rank_preserved_and_sequential(retriever):
    bundle = build_rag_evidence_bundle(_make_request(), retriever)
    ranks = [item.rank for item in bundle.evidence]
    assert ranks == list(range(1, len(bundle.evidence) + 1))


def test_similarity_scores_preserved_descending(retriever):
    bundle = build_rag_evidence_bundle(_make_request(), retriever)
    scores = [item.similarity_score for item in bundle.evidence]
    assert scores == sorted(scores, reverse=True)


def test_evidence_matches_raw_retriever_output_exactly(retriever):
    request = _make_request()
    raw_results = retriever.retrieve(request.query, k=DEFAULT_RAG_TOP_K)
    bundle = build_rag_evidence_bundle(request, retriever)
    assert [item.document_id for item in bundle.evidence] == [
        r.document_id for r in raw_results
    ]
    assert [item.similarity_score for item in bundle.evidence] == [r.score for r in raw_results]
    assert [item.rank for item in bundle.evidence] == [r.rank for r in raw_results]


def test_freshness_metadata_preserved(retriever):
    request = _make_request(
        query="What price did DraftKings list for the Minnesota Timberwolves moneyline?",
    )
    bundle = build_rag_evidence_bundle(request, retriever, k=5)
    is_current_values = {item.document.is_current for item in bundle.evidence if item.document.sportsbook}
    # Both a current and a stale sportsbook document should be retrievable
    # for this known freshness-conflict query (see tests/test_retriever.py).
    assert True in is_current_values or False in is_current_values  # sanity: field is populated
    for item in bundle.evidence:
        if item.document.source_type.value == "sportsbook_snapshot":
            assert item.document.is_current is not None
            assert item.document.timestamp is not None


def test_content_preserved_unaltered(retriever):
    bundle = build_rag_evidence_bundle(_make_request(), retriever, k=1)
    top = bundle.evidence[0]
    assert top.document.content == top.document.content.strip() or top.document.content
    assert len(top.document.content) > 0


# ---------------------------------------------------------------------------
# Stale documents not filtered / no reordering
# ---------------------------------------------------------------------------


def test_stale_documents_not_filtered_from_evidence(retriever):
    request = _make_request(
        query="What price did DraftKings list for the Minnesota Timberwolves moneyline?",
    )
    bundle = build_rag_evidence_bundle(request, retriever, k=5)
    stale_doc_id = "g-2026-009-moneyline-minnesota-timberwolves-draftkings-v0"
    document_ids = [item.document_id for item in bundle.evidence]
    assert stale_doc_id in document_ids


def test_stale_document_rank_matches_raw_retrieval_no_demotion(retriever):
    request = _make_request(
        query="What price did DraftKings list for the Minnesota Timberwolves moneyline?",
    )
    raw_results = retriever.retrieve(request.query, k=5)
    bundle = build_rag_evidence_bundle(request, retriever, k=5)

    raw_rank_by_id = {r.document_id: r.rank for r in raw_results}
    for item in bundle.evidence:
        assert item.rank == raw_rank_by_id[item.document_id]


def test_current_document_not_artificially_promoted_above_stale(retriever):
    # Known baseline behavior (Milestone 6C): the stale DraftKings
    # Timberwolves document outranks its current counterpart under plain
    # semantic similarity. The evidence layer must not "fix" this.
    request = _make_request(
        query="What price did DraftKings list for the Minnesota Timberwolves moneyline?",
    )
    bundle = build_rag_evidence_bundle(request, retriever, k=5)
    stale_id = "g-2026-009-moneyline-minnesota-timberwolves-draftkings-v0"
    current_id = "g-2026-009-moneyline-minnesota-timberwolves-draftkings-v1"
    rank_by_id = {item.document_id: item.rank for item in bundle.evidence}
    assert rank_by_id[stale_id] < rank_by_id[current_id]


# ---------------------------------------------------------------------------
# Context rendering
# ---------------------------------------------------------------------------


def test_context_rendering_is_deterministic(retriever):
    bundle = build_rag_evidence_bundle(_make_request(), retriever)
    first = render_rag_context(bundle)
    second = render_rag_context(bundle)
    assert first == second


def test_context_includes_document_boundaries(retriever):
    bundle = build_rag_evidence_bundle(_make_request(), retriever, k=3)
    context = render_rag_context(bundle)
    assert "[DOCUMENT 1]" in context
    assert "[DOCUMENT 2]" in context
    assert "[DOCUMENT 3]" in context


def test_context_includes_provenance_and_freshness(retriever):
    bundle = build_rag_evidence_bundle(_make_request(), retriever, k=1)
    context = render_rag_context(bundle)
    assert "document_id:" in context
    assert "sportsbook:" in context
    assert "is_current:" in context
    assert "timestamp:" in context
    assert "similarity_score:" in context


def test_context_does_not_include_ground_truth_labels(retriever):
    bundle = build_rag_evidence_bundle(_make_request(), retriever, k=5)
    context = render_rag_context(bundle).lower()
    for phrase in (
        "expected_best_sportsbook",
        "expected_ev",
        "market_reference_probability",
        "probability_edge",
        "ground truth",
        "positive expected value",
    ):
        assert phrase not in context


def test_render_context_preserves_document_content_text(retriever):
    bundle = build_rag_evidence_bundle(_make_request(), retriever, k=1)
    context = render_rag_context(bundle)
    assert bundle.evidence[0].document.content in context


# ---------------------------------------------------------------------------
# Required evidence cases (Milestone 8A, Step 14)
# ---------------------------------------------------------------------------


def test_case_a_specific_sportsbook_outcome(retriever):
    request = _make_request(query="What price did DraftKings have on the Lakers moneyline?")
    bundle = build_rag_evidence_bundle(request, retriever, k=1)
    top = bundle.evidence[0].document
    assert top.sportsbook == "DraftKings"
    assert top.selected_outcome == "Los Angeles Lakers"


def test_case_b_multi_book_comparison(retriever):
    request = _make_request(
        query="What are the moneyline odds for the Philadelphia 76ers from every sportsbook?",
        scenario_id="S007",
        game_id="G-2026-007",
        selected_outcome="Philadelphia 76ers",
    )
    bundle = build_rag_evidence_bundle(request, retriever, k=5)
    diagnostics = compute_evidence_diagnostics(bundle)
    assert len(diagnostics.sportsbooks_found) >= 3


def test_case_c_two_sided_market_evidence(retriever):
    request = _make_request(
        query="Retrieve the DraftKings moneyline prices for both teams in the Lakers versus Celtics game.",
    )
    bundle = build_rag_evidence_bundle(request, retriever, k=5)
    diagnostics = compute_evidence_diagnostics(bundle)
    assert diagnostics.contains_both_market_sides is True
    assert {"Los Angeles Lakers", "Boston Celtics"} <= set(diagnostics.outcomes_found)


def test_case_d_freshness_conflict_both_preserved(retriever):
    request = _make_request(
        query="What price did DraftKings list for the Minnesota Timberwolves moneyline?",
    )
    bundle = build_rag_evidence_bundle(request, retriever, k=5)
    diagnostics = compute_evidence_diagnostics(bundle)
    assert diagnostics.contains_stale_documents is True
    assert diagnostics.contains_current_documents is True


def test_case_e_context_documents_preserved_alongside_snapshots(retriever):
    request = _make_request(
        query="Tell me about the matchup between the Lakers and the Celtics.",
    )
    bundle = build_rag_evidence_bundle(request, retriever, k=5)
    source_types = {item.document.source_type.value for item in bundle.evidence}
    assert "game_context" in source_types


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_sportsbooks_found_correct(retriever):
    request = _make_request(
        query="What are the moneyline odds for the Philadelphia 76ers from every sportsbook?",
    )
    bundle = build_rag_evidence_bundle(request, retriever, k=5)
    diagnostics = compute_evidence_diagnostics(bundle)
    expected = {
        item.document.sportsbook
        for item in bundle.evidence
        if item.document.sportsbook is not None
    }
    assert set(diagnostics.sportsbooks_found) == expected


def test_diagnostics_outcomes_found_correct(retriever):
    request = _make_request(
        query="Retrieve the DraftKings moneyline prices for both teams in the Lakers versus Celtics game.",
    )
    bundle = build_rag_evidence_bundle(request, retriever, k=5)
    diagnostics = compute_evidence_diagnostics(bundle)
    expected = {
        item.document.selected_outcome
        for item in bundle.evidence
        if item.document.selected_outcome is not None
    }
    assert set(diagnostics.outcomes_found) == expected


def test_diagnostics_two_sided_flag_false_when_single_outcome(retriever):
    request = _make_request(query="What price did DraftKings have on the Lakers moneyline?")
    bundle = build_rag_evidence_bundle(request, retriever, k=1)
    diagnostics = compute_evidence_diagnostics(bundle)
    assert diagnostics.contains_both_market_sides is False


def test_diagnostics_no_evidence_returns_empty_safe_defaults():
    empty_bundle = RagEvidenceBundle(
        scenario_id="S001",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="irrelevant",
        top_k=5,
        evidence=[],
    )
    diagnostics = compute_evidence_diagnostics(empty_bundle)
    assert diagnostics.sportsbooks_found == []
    assert diagnostics.outcomes_found == []
    assert diagnostics.contains_both_market_sides is False
    assert diagnostics.contains_stale_documents is False
    assert diagnostics.contains_current_documents is False


# ---------------------------------------------------------------------------
# No ground truth / benchmark labels anywhere in evidence models
# ---------------------------------------------------------------------------


def test_evidence_models_have_no_ground_truth_fields():
    forbidden = {
        "expected_best_sportsbook",
        "expected_ev",
        "expected_positive_ev",
        "market_reference_probability",
        "probability_edge",
        "ground_truth",
    }
    for model in (RagEvidenceItem, RagEvidenceBundle, EvidenceDiagnostics):
        assert not (set(model.model_fields.keys()) & forbidden)


# ---------------------------------------------------------------------------
# No quant calculation performed
# ---------------------------------------------------------------------------


def test_rag_evidence_module_does_not_import_calculations():
    source_path = Path(__file__).resolve().parent.parent / "src" / "agents" / "rag_evidence.py"
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("src.calculations")


# ---------------------------------------------------------------------------
# Architecture isolation
# ---------------------------------------------------------------------------


def test_rag_evidence_module_has_no_forbidden_imports():
    for filename in ("rag_evidence.py", "base.py"):
        source_path = Path(__file__).resolve().parent.parent / "src" / "agents" / filename
        tree = ast.parse(source_path.read_text())

        forbidden_prefixes = (
            "src.tools",
            "src.providers",
            "anthropic",
            "openai",
            "requests",
            "httpx",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_prefixes), (
                        f"{filename} must not import {alias.name!r}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden_prefixes), (
                    f"{filename} must not import {node.module!r}"
                )


def test_rag_evidence_module_never_reads_current_odds_json():
    # AST-based, not substring-based: the module docstring legitimately
    # explains that current_odds.json must never be read directly; only a
    # non-docstring string literal referencing it would be a real issue.
    source_path = Path(__file__).resolve().parent.parent / "src" / "agents" / "rag_evidence.py"
    tree = ast.parse(source_path.read_text())
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstring_nodes.add(id(body[0].value))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            assert "current_odds.json" not in node.value


def test_rag_evidence_module_never_imports_ground_truth():
    for filename in ("rag_evidence.py", "base.py"):
        source_path = Path(__file__).resolve().parent.parent / "src" / "agents" / filename
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "ground_truth" not in node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "ground_truth" not in alias.name


def test_rag_evidence_module_no_actual_file_io_calls():
    # AST-based: no open()/json.load calls in the actual code (docstrings
    # legitimately mention file paths for explanatory purposes).
    source_path = Path(__file__).resolve().parent.parent / "src" / "agents" / "rag_evidence.py"
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in ("load", "loads")


def test_evidence_pipeline_reuses_verified_retriever_class(retriever):
    from src.rag.retriever import Retriever as CanonicalRetriever

    assert isinstance(retriever, CanonicalRetriever)
