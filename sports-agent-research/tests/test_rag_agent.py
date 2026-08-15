"""Tests for src/agents/rag_agent.py (Milestone 8B) — the RAG-only agent.

Uses FakeLLMClient stand-ins throughout; no test in this file makes a real
API call. See experiments/run_rag_smoke_test.py for the credentialed
manual smoke test against the real Anthropic API.
"""

import ast
from pathlib import Path

import pytest

from src.agents.base import AgentRequest
from src.agents.extraction import ExtractedMarketEvidence, ExtractedSportsbookPrice
from src.agents.rag_agent import RagAgentTrace, RagAnalysisIncomplete, RagOnlyAgent
from src.calculations.market import (
    calculate_leave_one_out_consensus,
    calculate_no_vig_probabilities,
    calculate_probability_edge,
)
from src.calculations.odds_math import best_odds, expected_value, implied_probability
from src.models import AnalysisStatus, ArchitectureType, BettingAnalysis, MarketType
from src.rag.retriever import Retriever

AGENTS_DIR = Path(__file__).resolve().parent.parent / "src" / "agents"


class FakeLLMClient:
    model = "fake-model"

    def __init__(self, canned):
        self.canned = canned

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        return self.canned


class RaisingLLMClient:
    model = "fake-model"

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        raise ValueError("malformed structured output")


@pytest.fixture(scope="module")
def retriever():
    return Retriever.from_directory()


def _request(**overrides):
    defaults = dict(
        scenario_id="S001",
        game_id="G-2026-001",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="DraftKings FanDuel BetMGM Caesars moneyline price Los Angeles Lakers Boston Celtics",
    )
    defaults.update(overrides)
    return AgentRequest(**defaults)


def _price(sportsbook, odds, opposing_odds, primary_id, opposing_id):
    return ExtractedSportsbookPrice(
        sportsbook=sportsbook,
        selected_outcome="Los Angeles Lakers",
        american_odds=odds,
        opposing_outcome="Boston Celtics",
        opposing_american_odds=opposing_odds,
        source_document_ids=[primary_id, opposing_id],
    )


THREE_BOOK_EXTRACTION = ExtractedMarketEvidence(
    game_id="G-2026-001",
    market_id="G-2026-001-moneyline",
    selected_outcome="Los Angeles Lakers",
    sportsbook_prices=[
        _price(
            "DraftKings",
            120,
            -140,
            "g-2026-001-moneyline-los-angeles-lakers-draftkings-v1",
            "g-2026-001-moneyline-boston-celtics-draftkings-v1",
        ),
        _price(
            "FanDuel",
            125,
            -145,
            "g-2026-001-moneyline-los-angeles-lakers-fanduel-v1",
            "g-2026-001-moneyline-boston-celtics-fanduel-v1",
        ),
        _price(
            "BetMGM",
            115,
            -135,
            "g-2026-001-moneyline-los-angeles-lakers-betmgm-v1",
            "g-2026-001-moneyline-boston-celtics-betmgm-v1",
        ),
    ],
)

TWO_BOOK_EXTRACTION = ExtractedMarketEvidence(
    game_id="G-2026-001",
    market_id="G-2026-001-moneyline",
    selected_outcome="Los Angeles Lakers",
    sportsbook_prices=[
        _price(
            "DraftKings",
            120,
            -140,
            "g-2026-001-moneyline-los-angeles-lakers-draftkings-v1",
            "g-2026-001-moneyline-boston-celtics-draftkings-v1",
        ),
        _price(
            "FanDuel",
            125,
            -145,
            "g-2026-001-moneyline-los-angeles-lakers-fanduel-v1",
            "g-2026-001-moneyline-boston-celtics-fanduel-v1",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Happy path — full quant (>= MIN_QUOTING_BOOKS complete pairs)
# ---------------------------------------------------------------------------


def test_full_quant_pipeline_matches_shared_calculations_engine(retriever):
    agent = RagOnlyAgent(retriever, llm_client=FakeLLMClient(THREE_BOOK_EXTRACTION), top_k=10)
    analysis = agent.analyze(_request())

    assert isinstance(analysis, BettingAnalysis)
    assert analysis.architecture == ArchitectureType.RAG
    assert analysis.status == AnalysisStatus.OK
    assert analysis.best_odds == best_odds([120, 125, 115])
    assert analysis.best_sportsbook == "FanDuel"
    assert analysis.implied_probability == pytest.approx(implied_probability(125))

    # Independently recompute expected reference probability/EV via the
    # same shared engine calls, with the same inputs, to cross-check the
    # agent didn't reimplement or drift from the deterministic formulas.
    no_vig = {
        "DraftKings": calculate_no_vig_probabilities([120, -140])[0],
        "FanDuel": calculate_no_vig_probabilities([125, -145])[0],
        "BetMGM": calculate_no_vig_probabilities([115, -135])[0],
    }
    expected_reference = calculate_leave_one_out_consensus(
        list(no_vig.items()), "FanDuel"
    )
    expected_edge = calculate_probability_edge(expected_reference, implied_probability(125))
    expected_ev = expected_value(125, expected_reference)

    assert analysis.market_reference_probability == pytest.approx(expected_reference)
    assert analysis.probability_edge == pytest.approx(expected_edge)
    assert analysis.expected_value == pytest.approx(expected_ev)
    assert analysis.positive_ev == (expected_ev > 0)


def test_multiple_sportsbooks_considered_preserved(retriever):
    agent = RagOnlyAgent(retriever, llm_client=FakeLLMClient(THREE_BOOK_EXTRACTION), top_k=10)
    analysis = agent.analyze(_request())
    assert analysis.sportsbooks_considered == ["BetMGM", "DraftKings", "FanDuel"]


def test_two_sided_extraction_used_when_present(retriever):
    agent = RagOnlyAgent(retriever, llm_client=FakeLLMClient(THREE_BOOK_EXTRACTION), top_k=10)
    agent.analyze(_request())
    assert agent.last_trace.extraction_result.sportsbook_prices[0].opposing_outcome == "Boston Celtics"


def test_trace_records_retrieval_llm_quant_details(retriever):
    agent = RagOnlyAgent(retriever, llm_client=FakeLLMClient(THREE_BOOK_EXTRACTION), top_k=10)
    agent.analyze(_request())
    trace = agent.last_trace
    assert isinstance(trace, RagAgentTrace)
    assert trace.architecture == "rag"
    assert len(trace.retrieved_document_ids) == 10
    assert len(trace.retrieval_scores) == 10
    assert trace.validation_status == "ok"
    assert trace.quant_status == "ok"
    assert trace.rejected_extraction_reasons == []
    assert trace.errors == []
    for latency in (
        trace.retrieval_latency_seconds,
        trace.llm_latency_seconds,
        trace.quant_latency_seconds,
        trace.total_latency_seconds,
    ):
        assert latency >= 0.0
    assert trace.total_latency_seconds >= trace.retrieval_latency_seconds


# ---------------------------------------------------------------------------
# Incomplete evidence — best line still derivable, EV honestly withheld
# ---------------------------------------------------------------------------


def test_insufficient_quant_evidence_status_and_null_ev(retriever):
    agent = RagOnlyAgent(retriever, llm_client=FakeLLMClient(TWO_BOOK_EXTRACTION), top_k=10)
    analysis = agent.analyze(_request())

    assert analysis.status == AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE
    assert analysis.expected_value is None
    assert analysis.positive_ev is None
    assert analysis.market_reference_probability is None
    # best line is still honestly derivable from 2 validated prices.
    assert analysis.best_sportsbook == "FanDuel"
    assert analysis.best_odds == 125
    assert agent.last_trace.quant_status == "insufficient_quant_evidence"


def test_insufficient_quant_evidence_rejects_fabricated_ev_at_model_level():
    with pytest.raises(Exception):
        BettingAnalysis(
            scenario_id="S001",
            game_id="G-2026-001",
            market=MarketType.MONEYLINE,
            selected_outcome="Los Angeles Lakers",
            best_sportsbook="FanDuel",
            best_odds=125,
            implied_probability=implied_probability(125),
            expected_value=0.05,
            positive_ev=True,
            status=AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE,
            sportsbooks_considered=["FanDuel"],
            reasoning_summary="test",
            architecture=ArchitectureType.RAG,
        )


# ---------------------------------------------------------------------------
# Hallucination rejection -> RagAnalysisIncomplete
# ---------------------------------------------------------------------------


def test_hallucinated_sportsbook_raises_and_traces_reason(retriever):
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001",
        market_id="G-2026-001-moneyline",
        selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="BetRivers",
                selected_outcome="Los Angeles Lakers",
                american_odds=130,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-draftkings-v1"],
            ),
        ],
    )
    agent = RagOnlyAgent(retriever, llm_client=FakeLLMClient(extraction), top_k=10)
    with pytest.raises(RagAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    assert exc_info.value.trace.validation_status == "no_valid_prices"
    assert "BetRivers" in exc_info.value.trace.rejected_extraction_reasons[0]


def test_hallucinated_odds_raises_and_traces_reason(retriever):
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
    agent = RagOnlyAgent(retriever, llm_client=FakeLLMClient(extraction), top_k=10)
    with pytest.raises(RagAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    assert exc_info.value.trace.validation_status == "no_valid_prices"


def test_hallucinated_source_document_id_raises_and_traces_reason(retriever):
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
    agent = RagOnlyAgent(retriever, llm_client=FakeLLMClient(extraction), top_k=10)
    with pytest.raises(RagAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    assert "fake-doc-123" in exc_info.value.trace.rejected_extraction_reasons[0]


def test_malformed_llm_output_handled_gracefully_not_a_crash(retriever):
    agent = RagOnlyAgent(retriever, llm_client=RaisingLLMClient(), top_k=10)
    with pytest.raises(RagAnalysisIncomplete) as exc_info:
        agent.analyze(_request())
    trace = exc_info.value.trace
    assert trace.validation_status == "extraction_failed"
    assert trace.quant_status == "not_attempted"
    assert len(trace.errors) == 1
    assert "malformed structured output" in trace.errors[0]


# ---------------------------------------------------------------------------
# Freshness preservation — stale evidence must not be silently "corrected"
# ---------------------------------------------------------------------------


def test_stale_evidence_preserved_not_corrected(retriever):
    # DraftKings has both a stale (v0, is_current=False, 120) and a fresh
    # (v1, is_current=True, 135) snapshot for this game. The LLM here only
    # extracted the stale one — the agent has no tool/provider access to
    # "fix" this, and must not silently substitute the fresher value.
    request = AgentRequest(
        scenario_id="S009",
        game_id="G-2026-009",
        market_type=MarketType.MONEYLINE,
        selected_outcome="Minnesota Timberwolves",
        query="What price did DraftKings list for the Minnesota Timberwolves moneyline?",
    )
    extraction = ExtractedMarketEvidence(
        game_id="G-2026-009",
        market_id="G-2026-009-moneyline",
        selected_outcome="Minnesota Timberwolves",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="DraftKings",
                selected_outcome="Minnesota Timberwolves",
                american_odds=120,
                is_current=False,
                source_document_ids=[
                    "g-2026-009-moneyline-minnesota-timberwolves-draftkings-v0"
                ],
            ),
        ],
    )
    agent = RagOnlyAgent(retriever, llm_client=FakeLLMClient(extraction), top_k=10)
    analysis = agent.analyze(request)

    assert analysis.best_odds == 120  # stale value, honestly reported
    assert analysis.best_odds != 135  # not silently corrected to the fresh value
    assert agent.last_trace.validation_status == "ok"
    extracted_price = agent.last_trace.extraction_result.sportsbook_prices[0]
    assert extracted_price.is_current is False


# ---------------------------------------------------------------------------
# Architecture isolation (docs/EXPERIMENT_RULES.md, "RAG-Only Boundary")
# ---------------------------------------------------------------------------


FORBIDDEN_MODULE_PREFIXES = ("src.tools", "src.providers")
FORBIDDEN_SOURCE_SUBSTRINGS = (
    "current_odds.json",
    "ground_truth.json",
    "quant_ground_truth.json",
    "OddsProvider",
    "ControlledOddsProvider",
)


@pytest.mark.parametrize(
    "module_filename", ["rag_agent.py", "extraction.py", "llm_client.py"]
)
def test_no_forbidden_imports(module_filename):
    source_path = AGENTS_DIR / module_filename
    tree = ast.parse(source_path.read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    for module_name in imported_modules:
        for forbidden in FORBIDDEN_MODULE_PREFIXES:
            assert not module_name.startswith(forbidden), (
                f"{module_filename} imports forbidden module {module_name!r}"
            )


def _non_docstring_string_literals(tree: ast.AST) -> list[str]:
    """All string constants in the module except docstrings (module,
    class, function/async-function leading string-literal statements) —
    so a docstring's prose *warning about* a forbidden path/name doesn't
    itself trip the check; only string literals used as actual code
    (import targets, file paths, identifiers) count."""
    docstring_nodes: set[int] = set()
    doc_owners = [tree] + [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for owner in doc_owners:
        body = getattr(owner, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstring_nodes.add(id(body[0].value))

    literals = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            literals.append(node.value)
    return literals


@pytest.mark.parametrize(
    "module_filename", ["rag_agent.py", "extraction.py", "llm_client.py"]
)
def test_no_forbidden_source_references(module_filename):
    source_path = AGENTS_DIR / module_filename
    tree = ast.parse(source_path.read_text())
    literals = _non_docstring_string_literals(tree)
    for literal in literals:
        for forbidden in FORBIDDEN_SOURCE_SUBSTRINGS:
            assert forbidden not in literal, (
                f"{module_filename} references forbidden identifier/path "
                f"{forbidden!r} in code (not just documentation): {literal!r}"
            )


def test_rag_agent_never_reimplements_shared_quant_formulas():
    # rag_agent.py must call into src/calculations/ for every calculation,
    # never redefine implied_probability / no_vig / consensus / EV itself.
    source_path = AGENTS_DIR / "rag_agent.py"
    tree = ast.parse(source_path.read_text())
    defined_function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden_names = {
        "implied_probability",
        "expected_value",
        "is_positive_ev",
        "best_odds",
        "compare_american_odds",
        "calculate_no_vig_probabilities",
        "calculate_leave_one_out_consensus",
        "calculate_probability_edge",
    }
    assert defined_function_names.isdisjoint(forbidden_names)
