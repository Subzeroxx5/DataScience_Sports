"""Tests for src/agents/hybrid_reconciliation.py (Milestone 10A) — the
deterministic source-reconciliation policy at the heart of the hybrid
agent. Pure unit tests: no LLM, no retriever, no provider involved.
"""

from datetime import datetime, timezone

import pytest

from src.agents.hybrid_reconciliation import (
    ConflictResolutionReason,
    HybridMarketRecord,
    RagPriceObservation,
    reconcile_outcome,
)
from src.models import SourceType


def _obs(odds: int, is_current: bool | None, doc_id: str = "doc-1") -> RagPriceObservation:
    return RagPriceObservation(
        american_odds=odds, is_current=is_current, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        document_id=doc_id,
    )


# ---------------------------------------------------------------------------
# Case A — RAG and tool agree
# ---------------------------------------------------------------------------


def test_agreement_recorded_and_tool_is_authoritative():
    records = reconcile_outcome("Los Angeles Lakers", {"DraftKings": 120}, {"DraftKings": _obs(120, True)})
    record = records[0]
    assert record.authoritative_odds == 120
    assert record.authoritative_source == SourceType.TOOL
    assert record.conflict is False
    assert record.conflict_resolution_reason == ConflictResolutionReason.SOURCES_AGREE


def test_agreement_does_not_duplicate_the_record():
    # One authoritative record enters the quant engine per sportsbook,
    # even though both sources reported it (Step 7).
    records = reconcile_outcome("X", {"FanDuel": 125}, {"FanDuel": _obs(125, True)})
    assert len(records) == 1


# ---------------------------------------------------------------------------
# Case B — stale RAG vs. current tool
# ---------------------------------------------------------------------------


def test_stale_rag_vs_current_tool_tool_wins():
    records = reconcile_outcome("Minnesota Timberwolves", {"DraftKings": 135}, {"DraftKings": _obs(120, False)})
    record = records[0]
    assert record.authoritative_odds == 135
    assert record.authoritative_source == SourceType.TOOL
    assert record.conflict is True
    assert record.conflict_resolution_reason == ConflictResolutionReason.CURRENT_TOOL_DATA_PRECEDENCE


def test_stale_rag_value_never_used_even_as_fallback():
    records = reconcile_outcome("X", {"DraftKings": 135}, {"DraftKings": _obs(120, False)})
    assert records[0].authoritative_odds != 120


def test_no_averaging_of_conflicting_values():
    records = reconcile_outcome("X", {"DraftKings": 100}, {"DraftKings": _obs(200, False)})
    record = records[0]
    assert record.authoritative_odds == 100  # never (100+200)/2 = 150 or similar


# ---------------------------------------------------------------------------
# Case C — current RAG vs. current tool conflict
# ---------------------------------------------------------------------------


def test_current_rag_vs_current_tool_tool_still_wins():
    records = reconcile_outcome("X", {"DraftKings": 130}, {"DraftKings": _obs(125, True)})
    record = records[0]
    assert record.authoritative_odds == 130
    assert record.authoritative_source == SourceType.TOOL
    assert record.conflict is True
    assert record.conflict_resolution_reason == ConflictResolutionReason.CURRENT_TOOL_DATA_PRECEDENCE


# ---------------------------------------------------------------------------
# Case D — tool only
# ---------------------------------------------------------------------------


def test_tool_only_record_available_for_current_analysis():
    records = reconcile_outcome("X", {"FanDuel": 140}, {})
    record = records[0]
    assert record.authoritative_odds == 140
    assert record.authoritative_source == SourceType.TOOL
    assert record.rag_available is False
    assert record.conflict is False
    assert record.conflict_resolution_reason == ConflictResolutionReason.TOOL_ONLY


# ---------------------------------------------------------------------------
# Case E — RAG-only, tool missing
# ---------------------------------------------------------------------------


def test_rag_only_stale_not_promoted_to_current():
    records = reconcile_outcome("X", {}, {"BetMGM": _obs(115, False)})
    record = records[0]
    assert record.authoritative_odds is None
    assert record.authoritative_source is None
    assert record.conflict_resolution_reason == ConflictResolutionReason.STALE_RAG_ONLY_NOT_PROMOTED


def test_rag_only_current_is_promoted_when_policy_allows():
    # Default policy (Step 12, point 2): current RAG evidence may serve
    # as authoritative ONLY when no tool coverage exists at all.
    records = reconcile_outcome("X", {}, {"BetMGM": _obs(115, True)})
    record = records[0]
    assert record.authoritative_odds == 115
    assert record.authoritative_source == SourceType.RAG
    assert record.conflict_resolution_reason == ConflictResolutionReason.RAG_CURRENT_NO_TOOL_COVERAGE


def test_rag_only_unknown_freshness_not_promoted():
    # is_current=None (unknown) must be treated as conservatively as stale.
    records = reconcile_outcome("X", {}, {"BetMGM": _obs(115, None)})
    record = records[0]
    assert record.authoritative_odds is None
    assert record.conflict_resolution_reason == ConflictResolutionReason.STALE_RAG_ONLY_NOT_PROMOTED


def test_neither_source_available_no_record_fabricated():
    records = reconcile_outcome("X", {}, {})
    assert records == []


# ---------------------------------------------------------------------------
# Multi-sportsbook / ordering / provenance
# ---------------------------------------------------------------------------


def test_multiple_sportsbooks_each_get_their_own_record():
    records = reconcile_outcome(
        "X",
        {"DraftKings": 120, "FanDuel": 125},
        {"BetMGM": _obs(115, True), "Caesars": _obs(118, False)},
    )
    assert {r.sportsbook for r in records} == {"DraftKings", "FanDuel", "BetMGM", "Caesars"}
    assert [r.sportsbook for r in records] == sorted(r.sportsbook for r in records)  # deterministic order


def test_provenance_preserved_for_rag_sourced_record():
    records = reconcile_outcome("X", {}, {"BetMGM": _obs(115, True, doc_id="g-001-betmgm-v1")})
    record = records[0]
    assert record.rag_document_id == "g-001-betmgm-v1"
    assert record.rag_timestamp is not None


def test_every_sportsbook_contributes_at_most_one_authoritative_value():
    records = reconcile_outcome("X", {"DraftKings": 120}, {"DraftKings": _obs(125, True)})
    draftkings_records = [r for r in records if r.sportsbook == "DraftKings"]
    assert len(draftkings_records) == 1


def test_reconciliation_is_pure_python_no_llm_dependency():
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "src" / "agents" / "hybrid_reconciliation.py"
    ).read_text()
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    assert not any("llm" in module.lower() for module in imported_modules)
    assert "anthropic" not in source.lower()


def test_reconcile_outcome_returns_typed_model_not_dict():
    records = reconcile_outcome("X", {"DraftKings": 120}, {})
    assert isinstance(records[0], HybridMarketRecord)
