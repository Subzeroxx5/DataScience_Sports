"""Tests for the controlled RAG corpus (Milestone 6B).

Covers schema validation, corpus coverage requirements, determinism,
benchmark preservation, and — most importantly — an automated scan
proving no ground-truth information leaks into any document.
"""

import ast
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.rag.build_corpus import export_corpus, generate_documents
from src.rag.documents import RagDocument, RagSourceType

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "rag_documents" / "corpus.jsonl"

MIN_STALE_DOCUMENTS = 3


@pytest.fixture(scope="module")
def documents():
    return generate_documents()


@pytest.fixture(scope="module")
def corpus_lines():
    with CORPUS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_valid_sportsbook_snapshot_document():
    from datetime import datetime

    doc = RagDocument(
        document_id="g-2026-001-moneyline-lakers-draftkings-v1",
        source_type=RagSourceType.SPORTSBOOK_SNAPSHOT,
        content="DraftKings lists Lakers at +120 on the moneyline.",
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        market_type="moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook="DraftKings",
        american_odds=120,
        is_current=True,
        timestamp=datetime(2026, 8, 10, 12, 0, 0),
        version="v1",
    )
    assert doc.american_odds == 120


def test_valid_context_document_without_odds():
    doc = RagDocument(
        document_id="g-2026-001-game-context",
        source_type=RagSourceType.GAME_CONTEXT,
        content="A synthetic game preview.",
        game_id="G-2026-001",
    )
    assert doc.sportsbook is None
    assert doc.american_odds is None


def test_rejects_blank_document_id():
    with pytest.raises(ValidationError):
        RagDocument(
            document_id="",
            source_type=RagSourceType.GAME_CONTEXT,
            content="text",
            game_id="G-2026-001",
        )


def test_rejects_blank_content():
    with pytest.raises(ValidationError):
        RagDocument(
            document_id="doc-1",
            source_type=RagSourceType.GAME_CONTEXT,
            content="",
            game_id="G-2026-001",
        )


def test_rejects_zero_american_odds():
    with pytest.raises(ValidationError):
        RagDocument(
            document_id="doc-1",
            source_type=RagSourceType.SPORTSBOOK_SNAPSHOT,
            content="text",
            game_id="G-2026-001",
            market_id="G-2026-001-moneyline",
            market_type="moneyline",
            selected_outcome="Lakers",
            sportsbook="DraftKings",
            american_odds=0,
            is_current=True,
            version="v1",
        )


def test_rejects_blank_sportsbook_when_provided():
    with pytest.raises(ValidationError):
        RagDocument(
            document_id="doc-1",
            source_type=RagSourceType.GAME_CONTEXT,
            content="text",
            game_id="G-2026-001",
            sportsbook="",
        )


def test_rejects_blank_version_when_provided():
    with pytest.raises(ValidationError):
        RagDocument(
            document_id="doc-1",
            source_type=RagSourceType.SPORTSBOOK_SNAPSHOT,
            content="text",
            game_id="G-2026-001",
            market_id="G-2026-001-moneyline",
            market_type="moneyline",
            selected_outcome="Lakers",
            sportsbook="DraftKings",
            american_odds=120,
            is_current=True,
            version="",
        )


def test_rejects_malformed_timestamp():
    with pytest.raises(ValidationError):
        RagDocument(
            document_id="doc-1",
            source_type=RagSourceType.GAME_CONTEXT,
            content="text",
            game_id="G-2026-001",
            timestamp="not-a-timestamp",
        )


def test_rejects_incomplete_sportsbook_snapshot():
    with pytest.raises(ValidationError):
        RagDocument(
            document_id="doc-1",
            source_type=RagSourceType.SPORTSBOOK_SNAPSHOT,
            content="text",
            game_id="G-2026-001",
            sportsbook="DraftKings",
            american_odds=120,
            # market_id, market_type, selected_outcome, is_current, version missing
        )


def test_rejects_invalid_source_type():
    with pytest.raises(ValidationError):
        RagDocument(
            document_id="doc-1",
            source_type="not_a_real_source_type",
            content="text",
            game_id="G-2026-001",
        )


def test_no_embedding_style_fields_exist():
    fields = set(RagDocument.model_fields.keys())
    assert "embedding" not in fields
    assert "vector" not in fields
    assert "similarity_score" not in fields


# ---------------------------------------------------------------------------
# Generated corpus: schema + uniqueness + coverage
# ---------------------------------------------------------------------------


def test_all_generated_documents_are_valid_rag_documents(documents):
    assert len(documents) > 0
    assert all(isinstance(d, RagDocument) for d in documents)


def test_document_ids_are_unique(documents):
    ids = [d.document_id for d in documents]
    assert len(ids) == len(set(ids))


def test_no_document_has_blank_content(documents):
    for doc in documents:
        assert doc.content.strip() != ""


def test_multiple_sportsbooks_represented(documents):
    sportsbooks = {d.sportsbook for d in documents if d.sportsbook is not None}
    assert sportsbooks == {"DraftKings", "FanDuel", "BetMGM", "Caesars"}


def test_context_documents_exist_for_multiple_games(documents):
    game_context_ids = {d.game_id for d in documents if d.source_type == RagSourceType.GAME_CONTEXT}
    assert len(game_context_ids) >= 3


# ---------------------------------------------------------------------------
# Two-sided market coverage
# ---------------------------------------------------------------------------


TWO_SIDED_MARKET_GAME_IDS = ["G-2026-001", "G-2026-007", "G-2026-008", "G-2026-009"]


def test_two_sided_markets_have_both_outcomes_represented(documents):
    for game_id in TWO_SIDED_MARKET_GAME_IDS:
        outcomes = {
            d.selected_outcome
            for d in documents
            if d.source_type == RagSourceType.SPORTSBOOK_SNAPSHOT and d.game_id == game_id
        }
        assert len(outcomes) == 2, f"{game_id} does not have both outcomes: {outcomes}"


def test_opposing_outcomes_share_market_id(documents):
    for game_id in TWO_SIDED_MARKET_GAME_IDS:
        market_ids = {
            d.market_id
            for d in documents
            if d.source_type == RagSourceType.SPORTSBOOK_SNAPSHOT and d.game_id == game_id
        }
        assert market_ids == {f"{game_id}-moneyline"}


def test_shared_market_identity_pairs_a_sportsbook_across_outcomes(documents):
    # DraftKings must be resolvable on both sides of G-2026-001 via the
    # same market_id.
    draftkings_docs = [
        d
        for d in documents
        if d.source_type == RagSourceType.SPORTSBOOK_SNAPSHOT
        and d.game_id == "G-2026-001"
        and d.sportsbook == "DraftKings"
        and d.is_current
    ]
    assert len(draftkings_docs) == 2
    assert {d.market_id for d in draftkings_docs} == {"G-2026-001-moneyline"}
    assert {d.selected_outcome for d in draftkings_docs} == {
        "Los Angeles Lakers",
        "Boston Celtics",
    }


# ---------------------------------------------------------------------------
# Stale / current coverage
# ---------------------------------------------------------------------------


def test_at_least_three_stale_sportsbook_documents(documents):
    stale = [
        d
        for d in documents
        if d.source_type == RagSourceType.SPORTSBOOK_SNAPSHOT and d.is_current is False
    ]
    assert len(stale) >= MIN_STALE_DOCUMENTS


def test_at_least_one_current_sportsbook_document(documents):
    current = [
        d
        for d in documents
        if d.source_type == RagSourceType.SPORTSBOOK_SNAPSHOT and d.is_current is True
    ]
    assert len(current) >= 1


def test_stale_documents_preserve_older_timestamp_and_version(documents):
    stale = [
        d
        for d in documents
        if d.source_type == RagSourceType.SPORTSBOOK_SNAPSHOT and d.is_current is False
    ]
    for doc in stale:
        assert doc.version == "v0"
        assert doc.timestamp is not None


def test_stale_and_current_documents_coexist_for_freshness_scenarios(documents):
    # G-2026-009's DraftKings has both a current and a stale document.
    dk_g009 = [
        d
        for d in documents
        if d.source_type == RagSourceType.SPORTSBOOK_SNAPSHOT
        and d.game_id == "G-2026-009"
        and d.sportsbook == "DraftKings"
        and d.selected_outcome == "Minnesota Timberwolves"
    ]
    assert {d.is_current for d in dk_g009} == {True, False}
    stale_doc = next(d for d in dk_g009 if not d.is_current)
    current_doc = next(d for d in dk_g009 if d.is_current)
    assert stale_doc.american_odds == 120
    assert current_doc.american_odds == 135


# ---------------------------------------------------------------------------
# Ground-truth leakage prevention (hard requirement)
# ---------------------------------------------------------------------------


FORBIDDEN_SUBSTRINGS = [
    "expected_best_sportsbook",
    "expected_best_odds",
    "expected_ev",
    "expected_positive_ev",
    "expected_implied_probability",
    "reference_probability",
    "estimated_true_probability",
    "market_consensus_probability",
    "no_vig_probability",
    "no-vig",
    "correct_answer",
    "ground_truth",
    "ground truth",
    "best sportsbook",
    "best odds",
    "best line",
    "positive expected value",
    "positive ev",
    "negative expected value",
    "negative ev",
    "the correct answer",
]


def test_no_forbidden_fields_in_document_json(documents):
    forbidden_field_names = {
        "expected_best_sportsbook",
        "expected_best_odds",
        "expected_ev",
        "expected_positive_ev",
        "expected_implied_probability",
        "reference_probability",
        "market_consensus_probability",
        "no_vig_probability",
        "correct_answer",
        "ground_truth",
    }
    for doc in documents:
        payload_keys = set(doc.model_dump().keys())
        assert not (payload_keys & forbidden_field_names)


def test_no_forbidden_phrases_in_content(documents):
    for doc in documents:
        content_lower = doc.content.lower()
        for phrase in FORBIDDEN_SUBSTRINGS:
            assert phrase.lower() not in content_lower, (
                f"leakage phrase {phrase!r} found in document {doc.document_id!r}: "
                f"{doc.content!r}"
            )


def test_no_forbidden_phrases_anywhere_in_generated_corpus_json(documents):
    # Belt-and-suspenders: scan the full serialized JSON of every
    # document, not just `content`, in case a future field addition
    # accidentally carries a leaking value.
    for doc in documents:
        full_json = json.dumps(doc.model_dump(mode="json")).lower()
        for phrase in FORBIDDEN_SUBSTRINGS:
            assert phrase.lower() not in full_json, (
                f"leakage phrase {phrase!r} found in serialized document "
                f"{doc.document_id!r}"
            )


def test_build_corpus_module_never_imports_ground_truth():
    source_path = (
        Path(__file__).resolve().parent.parent / "src" / "rag" / "build_corpus.py"
    )
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "ground_truth" not in alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert "ground_truth" not in node.module


def test_build_corpus_module_never_references_ground_truth_file_in_code():
    # AST-based, not substring-based: the module docstring legitimately
    # explains that ground_truth.json is never read; only an actual string
    # literal used in code (e.g. passed to open()/Path()) would matter.
    source_path = (
        Path(__file__).resolve().parent.parent / "src" / "rag" / "build_corpus.py"
    )
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
            assert "ground_truth" not in node.value, (
                f"unexpected non-docstring reference to ground_truth: {node.value!r}"
            )


# ---------------------------------------------------------------------------
# Deterministic generation
# ---------------------------------------------------------------------------


def test_generate_documents_is_deterministic():
    first = [d.model_dump(mode="json") for d in generate_documents()]
    second = [d.model_dump(mode="json") for d in generate_documents()]
    assert first == second


def test_export_corpus_is_byte_identical_across_runs(tmp_path):
    path_a = tmp_path / "corpus_a.jsonl"
    path_b = tmp_path / "corpus_b.jsonl"
    export_corpus(path_a)
    export_corpus(path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_committed_corpus_matches_fresh_generation(tmp_path):
    fresh_path = tmp_path / "fresh_corpus.jsonl"
    export_corpus(fresh_path)
    committed_hash = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    fresh_hash = hashlib.sha256(fresh_path.read_bytes()).hexdigest()
    assert committed_hash == fresh_hash


def test_corpus_uses_deterministic_sort_order(corpus_lines):
    keys = [
        (
            line["game_id"],
            line.get("market_id") or "",
            line.get("selected_outcome") or "",
            line.get("sportsbook") or "",
            line.get("version") or "",
            line["document_id"],
        )
        for line in corpus_lines
    ]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Benchmark preservation
# ---------------------------------------------------------------------------


def test_corpus_generation_does_not_mutate_benchmark_files(tmp_path):
    benchmark_files = [
        DATA_DIR / "current_odds.json",
        DATA_DIR / "test_scenarios.json",
        DATA_DIR / "ground_truth.json",
        DATA_DIR / "historical_odds.json",
    ]
    before_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in benchmark_files}

    export_corpus(tmp_path / "corpus.jsonl")

    after_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in benchmark_files}
    assert before_hashes == after_hashes


# ---------------------------------------------------------------------------
# JSONL file itself is well-formed
# ---------------------------------------------------------------------------


def test_corpus_file_each_line_is_valid_json_and_rag_document(corpus_lines):
    for line in corpus_lines:
        RagDocument(**line)


def test_corpus_line_count_matches_generated_document_count(documents, corpus_lines):
    assert len(corpus_lines) == len(documents)
