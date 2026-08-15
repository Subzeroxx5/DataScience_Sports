"""Tests for src/evaluation/rag_agent_evaluation.py (Milestone 11) — the
RAG-only agent's end-to-end evaluation harness, filling the gap noted in
milestones/current.md (the RAG-only agent had no dedicated evaluator
before this milestone, unlike the tool-calling and hybrid agents). Mirrors
the coverage in tests/test_tool_agent_evaluation.py / test_tool_agent_e2e.py
and tests/test_hybrid_evaluation.py / test_hybrid_agent_e2e.py. No real
API calls anywhere in this file.
"""

import ast
from pathlib import Path

import pytest

from src.agents.base import AgentRequest
from src.agents.rag_agent import RagAnalysisIncomplete, RagOnlyAgent
from src.evaluation import metrics
from src.evaluation.rag_agent_evaluation import (
    DEFAULT_SCENARIO_IDS,
    DeterministicRagPolicyLLMClient,
    RAG_EVALUATION_TOP_K,
    _classify_failure,
    _detect_hallucination,
    build_default_rag_agent,
    evaluate_scenario,
    evaluate_scenarios,
    summarize_results,
    to_common_result,
)
from src.models import AnalysisStatus, ArchitectureType, BettingAnalysis, GroundTruth, MarketType, QuantGroundTruth
from src.rag.retriever import Retriever

AGENTS_DIR = Path(__file__).resolve().parent.parent / "src" / "agents"
EVALUATION_DIR = Path(__file__).resolve().parent.parent / "src" / "evaluation"


@pytest.fixture(scope="module")
def retriever():
    return Retriever.from_directory()


def _request(**overrides):
    defaults = dict(
        scenario_id="S001", game_id="G-2026-001", market_type=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers",
        query="DraftKings FanDuel BetMGM Caesars moneyline price Los Angeles Lakers Boston Celtics",
    )
    defaults.update(overrides)
    return AgentRequest(**defaults)


def _ground_truth(**overrides) -> GroundTruth:
    defaults = dict(
        scenario_id="S001", expected_best_sportsbook="FanDuel", expected_best_odds=125,
        expected_implied_probability=0.4444, expected_ev=0.0, expected_positive_ev=False,
        expected_sportsbooks=["DraftKings", "FanDuel", "BetMGM", "Caesars"],
        expected_best_sportsbooks=["FanDuel"],
    )
    defaults.update(overrides)
    return GroundTruth(**defaults)


def _quant_ground_truth(**overrides) -> QuantGroundTruth:
    defaults = dict(
        scenario_id="S001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        quant_evaluable=False, ineligibility_reason="not evaluated in this test",
    )
    defaults.update(overrides)
    return QuantGroundTruth(**defaults)


# ---------------------------------------------------------------------------
# Ground-truth isolation
# ---------------------------------------------------------------------------


def test_rag_agent_source_never_imports_evaluation():
    for filename in ("rag_agent.py", "extraction.py"):
        tree = ast.parse((AGENTS_DIR / filename).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any(m.startswith("src.evaluation") for m in imported), filename


def test_evaluator_can_import_ground_truth_generators():
    from src.evaluation.rag_agent_evaluation import generate_all_ground_truth, generate_all_quant_ground_truth

    assert callable(generate_all_ground_truth)
    assert callable(generate_all_quant_ground_truth)


# ---------------------------------------------------------------------------
# Deterministic fake policy — honest, never reads ground truth
# ---------------------------------------------------------------------------


def test_policy_reuses_shared_hybrid_rag_parser_not_a_duplicate():
    from src.evaluation.hybrid_agent_evaluation import extract_honest_rag_evidence
    from src.evaluation.rag_agent_evaluation import DeterministicRagPolicyLLMClient as Policy

    # The class body should delegate, not reimplement, the parsing logic.
    source = (EVALUATION_DIR / "rag_agent_evaluation.py").read_text()
    tree = ast.parse(source)
    policy_class = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "DeterministicRagPolicyLLMClient")
    class_source = ast.get_source_segment(source, policy_class)
    assert "extract_honest_rag_evidence" in class_source
    assert "_parse_rag_evidence_blocks" not in class_source  # no re-implementation


def test_policy_never_references_ground_truth():
    source_path = EVALUATION_DIR / "rag_agent_evaluation.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    policy_class = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "DeterministicRagPolicyLLMClient")
    # Skip the class's own docstring — it explicitly documents what the
    # class must NOT do, which would otherwise trip a naive substring check.
    body = policy_class.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    for node in body:
        code_source = ast.get_source_segment(source, node) or ""
        assert "GroundTruth" not in code_source
        assert "ground_truth" not in code_source.lower()


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def _trace(**overrides):
    from src.agents.rag_agent import RagAgentTrace

    defaults = dict(
        model="fake", query="q", top_k=10, retrieved_document_ids=[], retrieval_scores=[],
        extraction_result=None, rejected_extraction_reasons=[], validation_status="no_valid_prices",
        quant_status="not_attempted", retrieval_latency_seconds=0.0, llm_latency_seconds=0.0,
        quant_latency_seconds=0.0, total_latency_seconds=0.0, errors=[],
    )
    defaults.update(overrides)
    return RagAgentTrace(**defaults)


def test_classify_extraction_failed():
    assert _classify_failure(_trace(validation_status="extraction_failed")) == metrics.FailureCategory.LLM_OUTPUT_INVALID


def test_classify_insufficient_retrieved_evidence():
    from src.agents.extraction import ExtractedMarketEvidence

    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[],
    )
    trace = _trace(validation_status="no_valid_prices", extraction_result=extraction)
    assert _classify_failure(trace) == metrics.FailureCategory.INSUFFICIENT_RETRIEVED_EVIDENCE


def test_classify_provenance_validation_failure():
    from src.agents.extraction import ExtractedMarketEvidence, ExtractedSportsbookPrice

    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="BetRivers", selected_outcome="Los Angeles Lakers", american_odds=130,
                source_document_ids=["fake-doc"],
            )
        ],
    )
    trace = _trace(validation_status="no_valid_prices", extraction_result=extraction)
    assert _classify_failure(trace) == metrics.FailureCategory.PROVENANCE_VALIDATION_FAILURE


def test_classify_none_trace_is_unknown_failure():
    assert _classify_failure(None) == metrics.FailureCategory.UNKNOWN_FAILURE


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------


def _analysis(**overrides) -> BettingAnalysis:
    defaults = dict(
        scenario_id="S001", game_id="G-2026-001", market=MarketType.MONEYLINE,
        selected_outcome="Los Angeles Lakers", best_sportsbook="FanDuel", best_odds=125,
        best_sportsbooks=["FanDuel"], implied_probability=0.4444,
        status=AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE, sportsbooks_considered=["FanDuel"],
        reasoning_summary="test", architecture=ArchitectureType.RAG,
    )
    defaults.update(overrides)
    return BettingAnalysis(**defaults)


def test_hallucination_not_detected_for_real_retrievable_claim(retriever):
    analysis = _analysis(best_sportsbook="FanDuel", best_odds=125)
    assert _detect_hallucination(retriever, _request(), analysis, RAG_EVALUATION_TOP_K) is False


def test_hallucination_detected_for_unsupported_sportsbook(retriever):
    analysis = _analysis(best_sportsbook="BetRivers", best_odds=130, best_sportsbooks=["BetRivers"])
    assert _detect_hallucination(retriever, _request(), analysis, RAG_EVALUATION_TOP_K) is True


def test_hallucination_detected_for_altered_odds(retriever):
    analysis = _analysis(best_sportsbook="DraftKings", best_odds=999, best_sportsbooks=["DraftKings"])
    assert _detect_hallucination(retriever, _request(), analysis, RAG_EVALUATION_TOP_K) is True


# ---------------------------------------------------------------------------
# evaluate_scenario() correctness / incorrectness detection
# ---------------------------------------------------------------------------


class FixedLLMClient:
    def __init__(self, extraction):
        self.extraction = extraction

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        return self.extraction


def test_evaluate_scenario_detects_correct_best_line(retriever):
    from src.agents.extraction import ExtractedMarketEvidence, ExtractedSportsbookPrice

    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="FanDuel", selected_outcome="Los Angeles Lakers", american_odds=125,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-fanduel-v1"],
            )
        ],
    )
    agent = RagOnlyAgent(retriever, llm_client=FixedLLMClient(extraction), top_k=10)
    result = evaluate_scenario(agent, retriever, _request(), _ground_truth(), _quant_ground_truth())
    assert result.best_line_correct is True
    assert result.best_odds_correct is True


def test_evaluate_scenario_detects_incorrect_best_line(retriever):
    from src.agents.extraction import ExtractedMarketEvidence, ExtractedSportsbookPrice

    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="FanDuel", selected_outcome="Los Angeles Lakers", american_odds=125,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-fanduel-v1"],
            )
        ],
    )
    agent = RagOnlyAgent(retriever, llm_client=FixedLLMClient(extraction), top_k=10)
    wrong_gt = _ground_truth(expected_best_sportsbook="DraftKings", expected_best_odds=120, expected_best_sportsbooks=["DraftKings"])
    result = evaluate_scenario(agent, retriever, _request(), wrong_gt, _quant_ground_truth())
    assert result.best_line_correct is False
    assert result.best_odds_correct is False


def test_evaluate_scenario_extraction_failure_classified_correctly(retriever):
    class RaisingLLMClient:
        def generate_structured(self, *, system_prompt, user_prompt, response_model):
            raise ValueError("boom")

    agent = RagOnlyAgent(retriever, llm_client=RaisingLLMClient(), top_k=10)
    result = evaluate_scenario(agent, retriever, _request(), _ground_truth(), _quant_ground_truth())
    assert result.execution_status == metrics.FailureCategory.LLM_OUTPUT_INVALID
    assert result.best_line_correct is None


# ---------------------------------------------------------------------------
# summarize_results()
# ---------------------------------------------------------------------------


def test_summarize_results_excludes_non_evaluable_from_ev_stats(retriever):
    from src.agents.extraction import ExtractedMarketEvidence, ExtractedSportsbookPrice

    extraction = ExtractedMarketEvidence(
        game_id="G-2026-001", market_id="G-2026-001-moneyline", selected_outcome="Los Angeles Lakers",
        sportsbook_prices=[
            ExtractedSportsbookPrice(
                sportsbook="FanDuel", selected_outcome="Los Angeles Lakers", american_odds=125,
                source_document_ids=["g-2026-001-moneyline-los-angeles-lakers-fanduel-v1"],
            )
        ],
    )
    agent = RagOnlyAgent(retriever, llm_client=FixedLLMClient(extraction), top_k=10)
    result = evaluate_scenario(agent, retriever, _request(), _ground_truth(), _quant_ground_truth(quant_evaluable=False))
    summary = summarize_results([result])
    assert summary["ev_classification_accuracy"] is None
    assert summary["best_line_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# End-to-end (real corpus/provider, deterministic policy)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results():
    return evaluate_scenarios()


def test_all_default_scenarios_produce_correct_best_line(results):
    for result in results:
        if result.best_line_correct is not None:
            assert result.best_line_correct is True, result.scenario_id
            assert result.best_odds_correct is True, result.scenario_id


def test_quant_evaluable_scenarios_reach_full_quant_at_evaluation_top_k(results):
    for scenario_id in ("S001", "S007", "S008", "S009"):
        result = next(r for r in results if r.scenario_id == scenario_id)
        assert result.execution_status == metrics.FailureCategory.SUCCESS, scenario_id
        assert result.ev_classification_correct is True
        assert result.ev_absolute_error == pytest.approx(0.0, abs=1e-9)
        assert result.market_reference_absolute_error == pytest.approx(0.0, abs=1e-9)


def test_freshness_scenario_uses_current_not_stale_data(results):
    result = next(r for r in results if r.scenario_id == "S009")
    assert result.freshness_correct is True
    assert result.predicted_best_odds == 140


def test_no_hallucinations_across_default_scenarios(results):
    for result in results:
        assert result.hallucination_detected is False, result.scenario_id


def test_deterministic_evaluation_is_reproducible_ignoring_latency():
    run_1 = evaluate_scenarios(DEFAULT_SCENARIO_IDS)
    run_2 = evaluate_scenarios(DEFAULT_SCENARIO_IDS)
    latency_fields = {
        "retrieval_latency_seconds", "llm_latency_seconds", "quant_latency_seconds", "total_latency_seconds",
    }
    for r1, r2 in zip(run_1, run_2):
        assert r1.model_dump(exclude=latency_fields) == r2.model_dump(exclude=latency_fields), r1.scenario_id


def test_to_common_result_conversion(results):
    for result in results:
        common = to_common_result(result)
        assert common.architecture == ArchitectureType.RAG
        assert common.scenario_id == result.scenario_id
        assert common.execution_status == result.execution_status
