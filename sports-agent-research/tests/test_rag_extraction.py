"""Tests for src/agents/extraction.py (Milestone 8B): schema validation and
the provenance/hallucination gate. Uses the real, persisted corpus index
so provenance checks run against actual retrieved evidence — no fake LLM
output is trusted just because it "looks" plausible."""

import pytest
from pydantic import ValidationError

from src.agents.base import AgentRequest
from src.agents.extraction import (
    ExtractedMarketEvidence,
    ExtractedSportsbookPrice,
    validate_extraction_provenance,
)
from src.agents.rag_evidence import build_rag_evidence_bundle
from src.models import MarketType
from src.rag.retriever import Retriever


@pytest.fixture(scope="module")
def retriever():
    return Retriever.from_directory()


def _request(**overrides):
    defaults = dict(
        scenario_id="S001",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="Retrieve the DraftKings moneyline prices for both teams in the Lakers versus Celtics game.",
    )
    defaults.update(overrides)
    return AgentRequest(**defaults)


@pytest.fixture(scope="module")
def two_sided_bundle(retriever):
    return build_rag_evidence_bundle(_request(), retriever, k=5)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_valid_extracted_price():
    price = ExtractedSportsbookPrice(
        sportsbook="DraftKings",
        selected_outcome="Los Angeles Lakers",
        american_odds=120,
        source_document_ids=["doc-1"],
    )
    assert price.american_odds == 120


def test_extracted_price_rejects_zero_odds():
    with pytest.raises(ValidationError):
        ExtractedSportsbookPrice(
            sportsbook="DraftKings",
            selected_outcome="Los Angeles Lakers",
            american_odds=0,
            source_document_ids=["doc-1"],
        )


def test_extracted_price_rejects_blank_sportsbook():
    with pytest.raises(ValidationError):
        ExtractedSportsbookPrice(
            sportsbook="",
            selected_outcome="Los Angeles Lakers",
            american_odds=120,
            source_document_ids=["doc-1"],
        )


def test_extracted_price_rejects_empty_source_document_ids():
    with pytest.raises(ValidationError):
        ExtractedSportsbookPrice(
            sportsbook="DraftKings",
            selected_outcome="Los Angeles Lakers",
            american_odds=120,
            source_document_ids=[],
        )


def test_extracted_price_rejects_blank_source_document_id():
    with pytest.raises(ValidationError):
        ExtractedSportsbookPrice(
            sportsbook="DraftKings",
            selected_outcome="Los Angeles Lakers",
            american_odds=120,
            source_document_ids=[""],
        )


def test_extraction_allows_empty_prices_with_missing_evidence_note():
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[],
        missing_evidence_note="No sportsbook snapshot documents were retrieved.",
    )
    assert extraction.sportsbook_prices == []


# ---------------------------------------------------------------------------
# Provenance validation — normal extraction
# ---------------------------------------------------------------------------


def test_correct_prices_extracted_and_accepted(two_sided_bundle):
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="DraftKings",
                selected_outcome="Los Angeles Lakers",
                american_odds=120,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"],
            ),
        ],
    )
    accepted, reasons = validate_extraction_provenance(extraction, two_sided_bundle)
    assert reasons == []
    assert len(accepted) == 1
    assert accepted[0].sportsbook == "DraftKings"
    assert accepted[0].american_odds == 120


def test_source_ids_validated_against_retrieved_evidence(two_sided_bundle):
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="DraftKings",
                selected_outcome="Los Angeles Lakers",
                american_odds=120,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"],
            ),
        ],
    )
    accepted, _ = validate_extraction_provenance(extraction, two_sided_bundle)
    assert accepted[0].source_document_ids == [
        "g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"
    ]


def test_multiple_sportsbooks_preserved(two_sided_bundle):
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="DraftKings",
                selected_outcome="Los Angeles Lakers",
                american_odds=120,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"],
            ),
            ExtractedSportsbookPrice(
                sportsbook="FanDuel",
                selected_outcome="Los Angeles Lakers",
                american_odds=125,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-fanduel-v1"],
            ),
        ],
    )
    accepted, reasons = validate_extraction_provenance(extraction, two_sided_bundle)
    assert reasons == []
    assert {price.sportsbook for price in accepted} == {"DraftKings", "FanDuel"}


def test_opposing_outcome_extracted_when_present_and_provenanced(two_sided_bundle):
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="DraftKings",
                selected_outcome="Los Angeles Lakers",
                american_odds=120,
                opposing_outcome="Boston Celtics",
                opposing_american_odds=-140,
                source_document_ids=[
                    "g-2026-001-moneyline-los-angeles-lakers-draftkings-v1",
                    "g-2026-001-moneyline-boston-celtics-draftkings-v1",
                ],
            ),
        ],
    )
    accepted, reasons = validate_extraction_provenance(extraction, two_sided_bundle)
    assert reasons == []
    assert accepted[0].opposing_outcome == "Boston Celtics"
    assert accepted[0].opposing_american_odds == -140


def test_unverifiable_opposing_claim_is_stripped_not_rejected(two_sided_bundle):
    # DraftKings' primary Lakers price is real and retrieved; the claimed
    # opposing Celtics price (-999, never actually offered/retrieved) is
    # not evidenced. Only the opposing fields should be cleared — the
    # primary, verified price is still accepted.
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="DraftKings",
                selected_outcome="Los Angeles Lakers",
                american_odds=120,
                opposing_outcome="Boston Celtics",
                opposing_american_odds=-999,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"],
            ),
        ],
    )
    accepted, reasons = validate_extraction_provenance(extraction, two_sided_bundle)
    assert reasons == []
    assert len(accepted) == 1
    assert accepted[0].opposing_outcome is None
    assert accepted[0].opposing_american_odds is None


# ---------------------------------------------------------------------------
# Hallucination rejection — the critical cases
# ---------------------------------------------------------------------------


def test_hallucinated_sportsbook_rejected(two_sided_bundle):
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="BetRivers",  # never retrieved
                selected_outcome="Los Angeles Lakers",
                american_odds=130,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"],
            ),
        ],
    )
    accepted, reasons = validate_extraction_provenance(extraction, two_sided_bundle)
    assert accepted == []
    assert len(reasons) == 1
    assert "BetRivers" in reasons[0]


def test_hallucinated_odds_rejected(two_sided_bundle):
    # Real document_id, but the claimed odds don't match what that
    # document actually says (+180 claimed vs +120 actual).
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="DraftKings",
                selected_outcome="Los Angeles Lakers",
                american_odds=180,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"],
            ),
        ],
    )
    accepted, reasons = validate_extraction_provenance(extraction, two_sided_bundle)
    assert accepted == []
    assert len(reasons) == 1


def test_hallucinated_source_document_id_rejected(two_sided_bundle):
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="DraftKings",
                selected_outcome="Los Angeles Lakers",
                american_odds=120,
                source_document_ids=["fake-doc-123"],
            ),
        ],
    )
    accepted, reasons = validate_extraction_provenance(extraction, two_sided_bundle)
    assert accepted == []
    assert "fake-doc-123" in reasons[0]


def test_real_but_unretrieved_document_id_rejected(retriever):
    # A document that genuinely exists in the corpus but was not part of
    # THIS bundle's retrieved evidence must be treated the same as a
    # fully fabricated ID — the LLM could not have legitimately seen it.
    bundle = build_rag_evidence_bundle(_request(), retriever, k=1)
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="FanDuel",
                selected_outcome="Los Angeles Lakers",
                american_odds=125,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-fanduel-v1"],
            ),
        ],
    )
    accepted, reasons = validate_extraction_provenance(extraction, bundle)
    assert accepted == []
    assert len(reasons) == 1


def test_context_document_cannot_be_cited_as_sportsbook_price(retriever):
    bundle = build_rag_evidence_bundle(
        _request(query="Tell me about the matchup between the Lakers and the Celtics."),
        retriever,
        k=1,
    )
    # rank-1 result for this query is the game_context document.
    context_doc_id = bundle.evidence[0].document_id
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="DraftKings",
                selected_outcome="Los Angeles Lakers",
                american_odds=120,
                source_document_ids=[context_doc_id],
            ),
        ],
    )
    accepted, reasons = validate_extraction_provenance(extraction, bundle)
    assert accepted == []
    assert len(reasons) == 1


def test_duplicate_sportsbook_outcome_rejected(two_sided_bundle):
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="DraftKings",
                selected_outcome="Los Angeles Lakers",
                american_odds=120,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"],
            ),
            ExtractedSportsbookPrice(
                sportsbook="DraftKings",
                selected_outcome="Los Angeles Lakers",
                american_odds=120,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"],
            ),
        ],
    )
    accepted, reasons = validate_extraction_provenance(extraction, two_sided_bundle)
    assert len(accepted) == 1
    assert any("duplicate" in reason for reason in reasons)


def test_system_prompt_instructs_extraction_only_no_calculation():
    from src.agents.extraction import RAG_EXTRACTION_SYSTEM_PROMPT

    lowered = RAG_EXTRACTION_SYSTEM_PROMPT.lower()
    assert "do not" in lowered
    assert "expected value" in lowered or "expected_value" in lowered
    assert "fabricate" in lowered
