"""RAG-only agent (Milestone 8B) — the first concrete architecture.

    AgentRequest
         |
         v
    RagEvidenceBundle           (src/agents/rag_evidence.py, Milestone 8A,
         |                       reused verbatim — no second retrieval path)
         v
    LLM structured extraction   (src/agents/llm_client.py + extraction.py;
         |                       the LLM only extracts, never calculates)
         v
    Provenance validation       (src/agents/extraction.py — rejects
         |                       hallucinated sportsbooks/odds/source IDs)
         v
    Shared deterministic quant engine
         |                       (src/calculations/market.py + odds_math.py
         |                       — reused exactly as Milestone 7B's ground
         |                       truth generator uses them; never
         |                       reimplemented here)
         v
    BettingAnalysis (architecture=rag)

Architecture isolation (docs/EXPERIMENT_RULES.md, "RAG-Only Boundary"):
this module must never import src.tools or src.providers, and must never
read data/current_odds.json or any ground-truth file. The only path to
sportsbook information is the RAG corpus -> vector index -> retriever ->
evidence pipeline chain above. Enforced by an AST-based test in
tests/test_rag_agent.py.

Freshness: if the only relevant retrieved evidence for a sportsbook is
stale, this agent's result may legitimately be stale — that is the
experimental behavior under study (docs/EXPERIMENT_RULES.md), not a bug
to work around. This module never calls a structured provider to "correct"
a RAG-derived price.
"""

from __future__ import annotations

import time

from pydantic import BaseModel

from src.agents.base import Agent, AgentRequest
from src.agents.extraction import (
    RAG_EXTRACTION_SYSTEM_PROMPT,
    ExtractedMarketEvidence,
    ExtractedSportsbookPrice,
    validate_extraction_provenance,
)
from src.agents.llm_client import DEFAULT_MODEL, AnthropicLLMClient, LLMClient
from src.agents.rag_evidence import (
    DEFAULT_RAG_TOP_K,
    RagEvidenceBundle,
    build_rag_evidence_bundle,
    render_rag_context,
)
from src.calculations.market import (
    MIN_CONSENSUS_BOOKS,
    calculate_leave_one_out_consensus,
    calculate_no_vig_probabilities,
    calculate_probability_edge,
)
from src.calculations.odds_math import (
    best_odds,
    compare_american_odds,
    expected_value,
    implied_probability,
    is_positive_ev,
)
from src.models import AnalysisStatus, ArchitectureType, BettingAnalysis
from src.rag.retriever import Retriever

MIN_QUOTING_BOOKS = MIN_CONSENSUS_BOOKS + 1  # target + MIN_CONSENSUS_BOOKS comparison books


class RagAgentTrace(BaseModel):
    """Deterministic execution record for one RAG-only agent run.

    For hallucination/freshness/latency analysis — never used as an
    inference input. No secrets (API keys, tokens) are recorded here.
    """

    architecture: str = "rag"
    model: str
    query: str
    top_k: int
    retrieved_document_ids: list[str]
    retrieval_scores: list[float]
    extraction_result: ExtractedMarketEvidence | None
    rejected_extraction_reasons: list[str]
    validation_status: str  # "ok" | "no_valid_prices" | "extraction_failed"
    quant_status: str  # "ok" | "insufficient_quant_evidence" | "not_attempted"
    retrieval_latency_seconds: float
    llm_latency_seconds: float
    quant_latency_seconds: float
    total_latency_seconds: float
    errors: list[str]


class RagAnalysisIncomplete(Exception):
    """Raised instead of returning a BettingAnalysis when no sportsbook
    price could be validated at all (extraction failed, or every claimed
    price was rejected as hallucinated). BettingAnalysis.best_sportsbook
    is a required field — there is no honest value to put there when zero
    evidence survives validation, so this is a typed, catchable failure
    rather than either a bare crash or a fabricated result.

    Carries the full RagAgentTrace so a caller (a future experiment
    runner) can log exactly what was retrieved, what the LLM claimed, and
    why every claim was rejected, without the run corrupting a batch.
    """

    def __init__(self, trace: RagAgentTrace) -> None:
        self.trace = trace
        super().__init__(
            f"no sportsbook price could be validated for query={trace.query!r} "
            f"(validation_status={trace.validation_status!r})"
        )


class RagOnlyAgent(Agent):
    """RAG-only architecture: RAG evidence + shared quant engine only.

    Never imports or calls src.tools / OddsProvider / ControlledOddsProvider,
    and never reads data/current_odds.json — see module docstring.
    """

    architecture = ArchitectureType.RAG

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient | None = None,
        top_k: int = DEFAULT_RAG_TOP_K,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client or AnthropicLLMClient()
        self.top_k = top_k
        self.last_trace: RagAgentTrace | None = None

    def analyze(self, request: AgentRequest) -> BettingAnalysis:
        errors: list[str] = []
        run_start = time.monotonic()

        step_start = time.monotonic()
        bundle = build_rag_evidence_bundle(request, self.retriever, k=self.top_k)
        retrieval_latency = time.monotonic() - step_start

        user_prompt = self._render_user_prompt(request, bundle)

        step_start = time.monotonic()
        extraction: ExtractedMarketEvidence | None = None
        try:
            extraction = self.llm_client.generate_structured(
                system_prompt=RAG_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=ExtractedMarketEvidence,
            )
        except Exception as exc:  # malformed/invalid LLM output, network error, etc.
            errors.append(f"LLM extraction failed: {exc!r}")
        llm_latency = time.monotonic() - step_start

        accepted: list[ExtractedSportsbookPrice] = []
        rejected_reasons: list[str] = []
        if extraction is not None:
            accepted, rejected_reasons = validate_extraction_provenance(extraction, bundle)

        if extraction is None:
            validation_status = "extraction_failed"
        elif accepted:
            validation_status = "ok"
        else:
            validation_status = "no_valid_prices"

        step_start = time.monotonic()
        analysis: BettingAnalysis | None = None
        quant_status = "not_attempted"
        if accepted:
            analysis, quant_status = self._run_quant_pipeline(request, accepted)
        quant_latency = time.monotonic() - step_start

        total_latency = time.monotonic() - run_start

        self.last_trace = RagAgentTrace(
            model=getattr(self.llm_client, "model", DEFAULT_MODEL),
            query=request.query,
            top_k=self.top_k,
            retrieved_document_ids=[item.document_id for item in bundle.evidence],
            retrieval_scores=[item.similarity_score for item in bundle.evidence],
            extraction_result=extraction,
            rejected_extraction_reasons=rejected_reasons,
            validation_status=validation_status,
            quant_status=quant_status,
            retrieval_latency_seconds=retrieval_latency,
            llm_latency_seconds=llm_latency,
            quant_latency_seconds=quant_latency,
            total_latency_seconds=total_latency,
            errors=errors,
        )

        if analysis is None:
            raise RagAnalysisIncomplete(self.last_trace)
        return analysis

    @staticmethod
    def _render_user_prompt(request: AgentRequest, bundle: RagEvidenceBundle) -> str:
        return (
            f"Game ID: {request.game_id}\n"
            f"Selected outcome: {request.selected_outcome}\n"
            f"Research query: {request.query}\n\n"
            f"{render_rag_context(bundle)}"
        )

    def _run_quant_pipeline(
        self, request: AgentRequest, accepted: list[ExtractedSportsbookPrice]
    ) -> tuple[BettingAnalysis, str]:
        """Determine the best line from validated prices (always possible
        with >=1 accepted price), then attempt the market-consensus EV
        chain (requires >=MIN_QUOTING_BOOKS sportsbooks with a validated,
        provenance-checked opposing-side price). Every calculation here is
        a direct call into src/calculations/ — nothing is recomputed.
        """
        odds_values = [price.american_odds for price in accepted]
        winning_odds = best_odds(odds_values)
        best_prices = [
            price
            for price in accepted
            if compare_american_odds(price.american_odds, winning_odds) == 0
        ]
        best_sportsbooks = sorted({price.sportsbook for price in best_prices})
        winning_implied_probability = implied_probability(winning_odds)
        sportsbooks_considered = sorted({price.sportsbook for price in accepted})

        complete_pairs = [
            price
            for price in accepted
            if price.opposing_outcome is not None and price.opposing_american_odds is not None
        ]

        market_reference_probability: float | None = None
        probability_edge: float | None = None
        computed_expected_value: float | None = None
        computed_positive_ev: bool | None = None
        status = AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE
        quant_status = "insufficient_quant_evidence"

        if len(complete_pairs) >= MIN_QUOTING_BOOKS:
            no_vig_by_book: dict[str, float] = {}
            for price in complete_pairs:
                fair = calculate_no_vig_probabilities(
                    [price.american_odds, price.opposing_american_odds]
                )
                no_vig_by_book[price.sportsbook] = fair[0]

            # Deterministic (alphabetical) choice among complete-pair books
            # tied for the winning price, matching the tie-break convention
            # already used for GroundTruth/BestLineResult ties elsewhere.
            tied_complete_pairs = sorted(
                (price for price in complete_pairs if price.american_odds == winning_odds),
                key=lambda price: price.sportsbook,
            )
            target = tied_complete_pairs[0] if tied_complete_pairs else None
            if target is not None:
                sportsbook_probabilities = [
                    (price.sportsbook, no_vig_by_book[price.sportsbook])
                    for price in complete_pairs
                ]
                reference_probability = calculate_leave_one_out_consensus(
                    sportsbook_probabilities, target.sportsbook
                )
                book_implied = implied_probability(target.american_odds)
                market_reference_probability = reference_probability
                probability_edge = calculate_probability_edge(reference_probability, book_implied)
                computed_expected_value = expected_value(target.american_odds, reference_probability)
                computed_positive_ev = is_positive_ev(target.american_odds, reference_probability)
                status = AnalysisStatus.OK
                quant_status = "ok"

        reasoning_summary = self._build_reasoning_summary(
            best_sportsbooks=best_sportsbooks,
            winning_odds=winning_odds,
            status=status,
            sportsbooks_considered=sportsbooks_considered,
        )

        analysis = BettingAnalysis(
            scenario_id=request.scenario_id,
            game_id=request.game_id,
            market=request.market_type,
            selected_outcome=request.selected_outcome,
            best_sportsbook=best_sportsbooks[0],
            best_sportsbooks=best_sportsbooks,
            best_odds=winning_odds,
            implied_probability=winning_implied_probability,
            market_reference_probability=market_reference_probability,
            probability_edge=probability_edge,
            expected_value=computed_expected_value,
            positive_ev=computed_positive_ev,
            status=status,
            sportsbooks_considered=sportsbooks_considered,
            reasoning_summary=reasoning_summary,
            architecture=ArchitectureType.RAG,
        )
        return analysis, quant_status

    @staticmethod
    def _build_reasoning_summary(
        *,
        best_sportsbooks: list[str],
        winning_odds: int,
        status: AnalysisStatus,
        sportsbooks_considered: list[str],
    ) -> str:
        books = ", ".join(best_sportsbooks)
        base = (
            f"From retrieved RAG evidence, {books} offered the best validated price "
            f"({winning_odds:+d}) among {len(sportsbooks_considered)} sportsbook(s) considered."
        )
        if status == AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE:
            base += (
                " Insufficient two-sided evidence was retrieved to derive a market "
                "reference probability, so no expected-value verdict is reported."
            )
        return base
