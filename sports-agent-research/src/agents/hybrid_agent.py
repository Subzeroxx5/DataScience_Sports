"""Hybrid RAG + tool-calling agent (Milestone 10A) — the third and final
concrete architecture.

    AgentRequest
         |
         +-------------------------+
         v                         v
    RAG evidence pipeline     Bounded tool-calling loop
    (src/agents/rag_evidence  (src/agents/tool_schemas.py TOOL_SCHEMAS +
     .py, reused verbatim)     execute_tool, reused verbatim; same
         |                     TOOL_AGENT_SYSTEM_PROMPT/MAX_TOOL_ITERATIONS
    LLM structured extraction  as the tool-only agent, src/agents/
    (src/agents/extraction.py, tool_agent.py)
     reused verbatim) -> provenance         |
    validation (reused verbatim)            v
         |                         Structured tool prices
         v                         (folded directly from actual
    RAG prices, per (sportsbook,    SportsbookTools return values —
    outcome), with is_current       never LLM prose)
    preserved                                |
         |                                   |
         +-------------------+---------------+
                              v
                Deterministic source reconciliation
                (src/agents/hybrid_reconciliation.py — current tool data
                 always wins; current-RAG-only is used only when no tool
                 coverage exists; stale-RAG-only is never promoted)
                              |
                              v
                Shared deterministic quant engine
                (src/calculations/market.py + odds_math.py — identical
                 to the RAG-only and tool-calling agents', never
                 reimplemented here)
                              |
                              v
                BettingAnalysis (architecture=hybrid)

Architecture boundary (docs/EXPERIMENT_RULES.md, "Hybrid Boundary"): this
is the ONLY agent module permitted to import both src.rag and
src.tools/src.providers. The RAG-only agent (src/agents/rag_agent.py)
and the tool-calling agent (src/agents/tool_agent.py) are unmodified by
this milestone and retain their original single-channel boundaries —
see tests/test_hybrid_agent.py's architecture-integrity tests.

Anti-hallucination design: exactly like the tool-calling agent, the
final BettingAnalysis is built exclusively from HybridMarketRecord.
authoritative_odds (populated deterministically by
src/agents/hybrid_reconciliation.py from validated RAG extraction +
actual tool-call return values) — never from anything the LLM's final
text says. The RAG side additionally goes through
validate_extraction_provenance() (src/agents/extraction.py, reused
verbatim), so a hallucinated RAG claim cannot even reach reconciliation.
"""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel

from src.agents.base import Agent, AgentRequest
from src.agents.extraction import (
    RAG_EXTRACTION_SYSTEM_PROMPT,
    ExtractedMarketEvidence,
    ExtractedSportsbookPrice,
    validate_extraction_provenance,
)
from src.agents.hybrid_reconciliation import (
    HybridMarketRecord,
    RagPriceObservation,
    reconcile_outcome,
)
from src.agents.llm_client import (
    DEFAULT_MODEL,
    DEFAULT_TOOL_MAX_TOKENS,
    AnthropicLLMClient,
    LLMClient,
    ToolCallingLLMClient,
    ToolUseBlock,
)
from src.agents.rag_evidence import (
    DEFAULT_RAG_TOP_K,
    RagEvidenceBundle,
    build_rag_evidence_bundle,
    render_rag_context,
)
from src.agents.tool_agent import (
    MAX_TOOL_ITERATIONS,
    TOOL_AGENT_SYSTEM_PROMPT,
    ToolCallRecord,
    _summarize_result,
    _turn_to_content_blocks,
)
from src.agents.tool_schemas import TOOL_SCHEMAS, execute_tool
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
from src.models import (
    AnalysisStatus,
    ArchitectureType,
    BettingAnalysis,
    SourceReference,
    SourceType,
)
from src.rag.retriever import Retriever
from src.tools.sportsbook_tools import SportsbookTools

MIN_QUOTING_BOOKS = MIN_CONSENSUS_BOOKS + 1  # target + MIN_CONSENSUS_BOOKS comparison books


class HybridFailureCategory(str, Enum):
    """Explicit failure/outcome classification (milestones/current.md,
    Milestone 10A Step 27) — never collapsed into a generic "error"."""

    SUCCESS = "success"
    RAG_RETRIEVAL_FAILURE = "rag_retrieval_failure"
    TOOL_FAILURE = "tool_failure"
    SOURCE_RECONCILIATION_FAILURE = "source_reconciliation_failure"
    INSUFFICIENT_CURRENT_DATA = "insufficient_current_data"
    LLM_OUTPUT_INVALID = "llm_output_invalid"
    QUANT_INSUFFICIENT_DATA = "quant_insufficient_data"
    FINAL_OUTPUT_INVALID = "final_output_invalid"


class HybridAgentTrace(BaseModel):
    """Deterministic execution record for one hybrid agent run
    (milestones/current.md, Step 26). Never used as an inference input;
    no secrets (API keys, tokens) are recorded here.

    `effort` is recorded instead of `temperature` — see
    src/agents/llm_client.py's module docstring for why DEFAULT_MODEL
    has no temperature parameter at all.
    """

    architecture: str = "hybrid"
    model: str
    effort: str
    query: str

    # RAG side
    retrieved_document_ids: list[str]
    rag_scores: list[float]
    rag_retrieval_latency_seconds: float
    rag_llm_latency_seconds: float
    rag_rejected_reasons: list[str]

    # Tool side
    tool_calls: list[ToolCallRecord]
    tool_call_order: list[str]
    tool_iterations_used: int
    redundant_tool_call_count: int
    tool_llm_latency_seconds: float
    tool_execution_latency_seconds: float

    # Reconciliation
    reconciled_records: list[HybridMarketRecord]
    source_agreements: int
    source_conflicts: int
    rag_only_records: int
    tool_only_records: int
    rag_records_considered: int
    tool_records_considered: int
    reconciliation_latency_seconds: float

    validation_status: HybridFailureCategory
    quant_status: str  # "ok" | "insufficient_quant_evidence" | "not_attempted"

    quant_latency_seconds: float
    total_latency_seconds: float
    errors: list[str]


class HybridAnalysisIncomplete(Exception):
    """Raised instead of returning a BettingAnalysis when reconciliation
    produced no authoritative current price for the requested outcome
    (both channels failed or agreed there is nothing current), or a
    defensive internal failure occurred. Mirrors RagAnalysisIncomplete /
    ToolAnalysisIncomplete.

    Carries the full HybridAgentTrace so a caller can log exactly what
    was retrieved, what tools were called, how sources reconciled, and
    why no analysis could be produced.
    """

    def __init__(self, trace: HybridAgentTrace) -> None:
        self.trace = trace
        super().__init__(
            f"no authoritative current price could be reconciled for query={trace.query!r} "
            f"(validation_status={trace.validation_status.value!r})"
        )


class HybridAgent(Agent):
    """Hybrid architecture: the only agent permitted to access both RAG
    evidence and sportsbook tools. See module docstring for the full
    data flow and the reconciliation policy.
    """

    architecture = ArchitectureType.HYBRID

    def __init__(
        self,
        retriever: Retriever,
        tools: SportsbookTools,
        llm_client: LLMClient | ToolCallingLLMClient | None = None,
        top_k: int = DEFAULT_RAG_TOP_K,
        max_tool_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        self.retriever = retriever
        self.tools = tools
        self.llm_client = llm_client or AnthropicLLMClient(max_tokens=DEFAULT_TOOL_MAX_TOKENS)
        self.top_k = top_k
        self.max_tool_iterations = max_tool_iterations
        self.last_trace: HybridAgentTrace | None = None

    def analyze(self, request: AgentRequest) -> BettingAnalysis:
        run_start = time.monotonic()
        errors: list[str] = []
        category_override: HybridFailureCategory | None = None

        # --- RAG side (Step 15: same retrieval pipeline, same top_k as RAG-only) ---
        bundle: RagEvidenceBundle | None = None
        retrieved_document_ids: list[str] = []
        rag_scores: list[float] = []
        rag_retrieval_latency = 0.0
        rag_llm_latency = 0.0
        rejected_rag_reasons: list[str] = []
        rag_prices_by_outcome: dict[str, dict[str, RagPriceObservation]] = {}

        step_start = time.monotonic()
        try:
            bundle = build_rag_evidence_bundle(request, self.retriever, k=self.top_k)
        except Exception as exc:
            errors.append(f"RAG retrieval failed: {exc!r}")
        rag_retrieval_latency = time.monotonic() - step_start

        if bundle is not None:
            retrieved_document_ids = [item.document_id for item in bundle.evidence]
            rag_scores = [item.similarity_score for item in bundle.evidence]

            rag_user_prompt = self._render_rag_user_prompt(request, bundle)
            step_start = time.monotonic()
            extraction: ExtractedMarketEvidence | None = None
            try:
                extraction = self.llm_client.generate_structured(
                    system_prompt=RAG_EXTRACTION_SYSTEM_PROMPT,
                    user_prompt=rag_user_prompt,
                    response_model=ExtractedMarketEvidence,
                )
            except Exception as exc:
                errors.append(f"RAG LLM extraction failed: {exc!r}")
            rag_llm_latency = time.monotonic() - step_start

            accepted_rag_prices: list[ExtractedSportsbookPrice] = []
            if extraction is not None:
                accepted_rag_prices, rejected_rag_reasons = validate_extraction_provenance(
                    extraction, bundle
                )
            rag_prices_by_outcome = self._fold_rag_prices(accepted_rag_prices)

        # --- Tool side (Step 14: reuse TOOL_SCHEMAS/execute_tool/prompt/loop bound) ---
        (
            tool_prices_by_outcome,
            tool_call_records,
            tool_call_order,
            loop_limit_exceeded,
            tool_iterations_used,
            tool_llm_latency,
            tool_execution_latency,
            tool_errors,
        ) = self._run_tool_loop(request)
        errors.extend(tool_errors)

        # --- Reconciliation (Step 16/19/20: deterministic, no LLM) ---
        reconciliation_start = time.monotonic()
        reconciled_by_outcome: dict[str, list[HybridMarketRecord]] = {}
        try:
            outcomes = sorted(set(rag_prices_by_outcome) | set(tool_prices_by_outcome))
            for outcome in outcomes:
                reconciled_by_outcome[outcome] = reconcile_outcome(
                    outcome,
                    tool_prices_by_outcome.get(outcome, {}),
                    rag_prices_by_outcome.get(outcome, {}),
                )
        except Exception as exc:
            errors.append(f"Source reconciliation failed: {exc!r}")
            reconciled_by_outcome = {}
            category_override = HybridFailureCategory.SOURCE_RECONCILIATION_FAILURE
        reconciliation_latency = time.monotonic() - reconciliation_start

        primary_records = reconciled_by_outcome.get(request.selected_outcome, [])
        primary_authoritative = {
            record.sportsbook: record.authoritative_odds
            for record in primary_records
            if record.authoritative_odds is not None
        }

        # --- Quant (Step 19: single reconciled state -> single quant call) ---
        quant_start = time.monotonic()
        analysis: BettingAnalysis | None = None
        quant_status = "not_attempted"
        if category_override is None and primary_authoritative:
            try:
                analysis, quant_status = self._run_quant_pipeline(request, reconciled_by_outcome)
            except Exception as exc:
                errors.append(f"Quant/final-output construction failed: {exc!r}")
                analysis = None
                category_override = HybridFailureCategory.FINAL_OUTPUT_INVALID
        quant_latency = time.monotonic() - quant_start

        total_latency = time.monotonic() - run_start

        all_records = [record for records in reconciled_by_outcome.values() for record in records]
        source_agreements = sum(
            1 for record in all_records if record.tool_available and record.rag_available and not record.conflict
        )
        source_conflicts = sum(1 for record in all_records if record.conflict)
        rag_only_records = sum(
            1 for record in all_records if record.rag_available and not record.tool_available
        )
        tool_only_records = sum(
            1 for record in all_records if record.tool_available and not record.rag_available
        )
        rag_records_considered = sum(len(prices) for prices in rag_prices_by_outcome.values())
        tool_records_considered = sum(len(prices) for prices in tool_prices_by_outcome.values())
        redundant_tool_call_count = sum(1 for record in tool_call_records if record.is_redundant)

        validation_status = category_override or self._classify_outcome(
            analysis_found=bool(primary_authoritative),
            bundle_missing=bundle is None,
            loop_limit_exceeded=loop_limit_exceeded,
            rag_extraction_attempted=bundle is not None,
            tool_calls_made=bool(tool_call_records),
        )

        self.last_trace = HybridAgentTrace(
            model=getattr(self.llm_client, "model", DEFAULT_MODEL),
            effort=getattr(self.llm_client, "effort", "unknown"),
            query=request.query,
            retrieved_document_ids=retrieved_document_ids,
            rag_scores=rag_scores,
            rag_retrieval_latency_seconds=rag_retrieval_latency,
            rag_llm_latency_seconds=rag_llm_latency,
            rag_rejected_reasons=rejected_rag_reasons,
            tool_calls=tool_call_records,
            tool_call_order=tool_call_order,
            tool_iterations_used=tool_iterations_used,
            redundant_tool_call_count=redundant_tool_call_count,
            tool_llm_latency_seconds=tool_llm_latency,
            tool_execution_latency_seconds=tool_execution_latency,
            reconciled_records=all_records,
            source_agreements=source_agreements,
            source_conflicts=source_conflicts,
            rag_only_records=rag_only_records,
            tool_only_records=tool_only_records,
            rag_records_considered=rag_records_considered,
            tool_records_considered=tool_records_considered,
            reconciliation_latency_seconds=reconciliation_latency,
            validation_status=validation_status,
            quant_status=quant_status,
            quant_latency_seconds=quant_latency,
            total_latency_seconds=total_latency,
            errors=errors,
        )

        if analysis is None:
            raise HybridAnalysisIncomplete(self.last_trace)
        return analysis

    @staticmethod
    def _classify_outcome(
        *,
        analysis_found: bool,
        bundle_missing: bool,
        loop_limit_exceeded: bool,
        rag_extraction_attempted: bool,
        tool_calls_made: bool,
    ) -> HybridFailureCategory:
        if analysis_found:
            return HybridFailureCategory.SUCCESS
        if bundle_missing and loop_limit_exceeded:
            return HybridFailureCategory.RAG_RETRIEVAL_FAILURE
        if bundle_missing:
            return HybridFailureCategory.RAG_RETRIEVAL_FAILURE
        if loop_limit_exceeded:
            return HybridFailureCategory.TOOL_FAILURE
        if not rag_extraction_attempted and not tool_calls_made:
            return HybridFailureCategory.LLM_OUTPUT_INVALID
        return HybridFailureCategory.INSUFFICIENT_CURRENT_DATA

    # -- RAG side helpers --------------------------------------------------

    @staticmethod
    def _render_rag_user_prompt(request: AgentRequest, bundle: RagEvidenceBundle) -> str:
        return (
            f"Game ID: {request.game_id}\n"
            f"Selected outcome: {request.selected_outcome}\n"
            f"Research query: {request.query}\n\n"
            f"{render_rag_context(bundle)}"
        )

    @staticmethod
    def _fold_rag_prices(
        accepted: list[ExtractedSportsbookPrice],
    ) -> dict[str, dict[str, RagPriceObservation]]:
        """Fold provenance-validated RAG prices into outcome -> sportsbook
        -> RagPriceObservation, including the opposing side when the
        (already-validated) extraction carried one — no second retrieval,
        matching Step 15's "do not create another retriever"."""
        by_outcome: dict[str, dict[str, RagPriceObservation]] = {}
        for price in accepted:
            by_outcome.setdefault(price.selected_outcome, {})[price.sportsbook] = RagPriceObservation(
                american_odds=price.american_odds,
                is_current=price.is_current,
                timestamp=price.timestamp,
                document_id=price.source_document_ids[0],
            )
            if price.opposing_outcome is not None and price.opposing_american_odds is not None:
                by_outcome.setdefault(price.opposing_outcome, {})[price.sportsbook] = RagPriceObservation(
                    american_odds=price.opposing_american_odds,
                    is_current=price.is_current,
                    timestamp=price.timestamp,
                    document_id=price.source_document_ids[-1],
                )
        return by_outcome

    # -- Tool side helpers ---------------------------------------------------

    @staticmethod
    def _render_tool_user_prompt(request: AgentRequest) -> str:
        return (
            f"Game ID: {request.game_id}\n"
            f"Market: {request.market_type.value}\n"
            f"Selected outcome: {request.selected_outcome}\n"
            f"Research query: {request.query}\n"
        )

    def _run_tool_loop(
        self, request: AgentRequest
    ) -> tuple[
        dict[str, dict[str, int]], list[ToolCallRecord], list[str], bool, int, float, float, list[str]
    ]:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self._render_tool_user_prompt(request)}
        ]
        tool_prices_by_outcome: dict[str, dict[str, int]] = {}
        tool_call_records: list[ToolCallRecord] = []
        tool_call_order: list[str] = []
        seen_signatures: set[str] = set()
        errors: list[str] = []

        llm_latency_total = 0.0
        tool_latency_total = 0.0
        call_sequence = 0
        iterations_used = 0
        loop_limit_exceeded = False

        for iteration in range(1, self.max_tool_iterations + 1):
            iterations_used = iteration
            step_start = time.monotonic()
            try:
                turn = self.llm_client.create_turn(
                    system_prompt=TOOL_AGENT_SYSTEM_PROMPT,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                )
            except Exception as exc:
                errors.append(f"Tool-call turn failed: {exc!r}")
                llm_latency_total += time.monotonic() - step_start
                break
            llm_latency_total += time.monotonic() - step_start

            messages.append({"role": "assistant", "content": _turn_to_content_blocks(turn)})

            if not turn.tool_uses:
                break

            tool_result_blocks: list[dict[str, Any]] = []
            for tool_use in turn.tool_uses:
                call_sequence += 1
                record, result_block, latency = self._execute_tool_call(
                    request, tool_use, call_sequence, seen_signatures, tool_prices_by_outcome
                )
                tool_call_records.append(record)
                tool_call_order.append(record.tool_name)
                tool_result_blocks.append(result_block)
                tool_latency_total += latency
            messages.append({"role": "user", "content": tool_result_blocks})
        else:
            loop_limit_exceeded = True

        return (
            tool_prices_by_outcome,
            tool_call_records,
            tool_call_order,
            loop_limit_exceeded,
            iterations_used,
            llm_latency_total,
            tool_latency_total,
            errors,
        )

    def _execute_tool_call(
        self,
        request: AgentRequest,
        tool_use: ToolUseBlock,
        call_sequence: int,
        seen_signatures: set[str],
        tool_prices_by_outcome: dict[str, dict[str, int]],
    ) -> tuple[ToolCallRecord, dict[str, Any], float]:
        signature = f"{tool_use.name}:{json.dumps(tool_use.input, sort_keys=True)}"
        is_redundant = signature in seen_signatures
        seen_signatures.add(signature)

        start = time.monotonic()
        try:
            structured_result, raw_result = execute_tool(self.tools, tool_use.name, tool_use.input)
            self._fold_tool_result(request, tool_prices_by_outcome, tool_use.name, tool_use.input, raw_result)
            success = True
            error: str | None = None
            result_summary = _summarize_result(tool_use.name, structured_result)
            result_content: Any = structured_result
        except Exception as exc:
            success = False
            error = str(exc)
            result_summary = f"error: {error}"
            result_content = error
        latency = time.monotonic() - start

        record = ToolCallRecord(
            call_sequence=call_sequence,
            tool_name=tool_use.name,
            arguments=tool_use.input,
            success=success,
            is_redundant=is_redundant,
            result_summary=result_summary,
            error=error,
            latency_seconds=latency,
        )
        result_block = {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": json.dumps(result_content) if success else str(result_content),
            "is_error": not success,
        }
        return record, result_block, latency

    @staticmethod
    def _fold_tool_result(
        request: AgentRequest,
        tool_prices_by_outcome: dict[str, dict[str, int]],
        tool_name: str,
        arguments: dict[str, Any],
        raw_result: Any,
    ) -> None:
        """Record only what a tool call actually, structurally returned —
        never anything derived from LLM prose — and only for this
        request's own game/market, so a tool call about an unrelated
        game can never leak into this analysis."""
        if arguments.get("game_id") != request.game_id:
            return
        if arguments.get("market_type") != request.market_type.value:
            return

        if tool_name == "get_odds":
            prices = tool_prices_by_outcome.setdefault(arguments["selected_outcome"], {})
            for odds in raw_result:
                prices[odds.sportsbook] = odds.american_odds
        elif tool_name == "get_sportsbook_odds":
            prices = tool_prices_by_outcome.setdefault(arguments["selected_outcome"], {})
            prices[raw_result.sportsbook] = raw_result.american_odds
        elif tool_name == "find_best_line":
            prices = tool_prices_by_outcome.setdefault(arguments["selected_outcome"], {})
            for sportsbook in raw_result.sportsbooks:
                prices[sportsbook] = raw_result.best_odds
        # get_games / get_game: no odds to fold.

    # -- Quant / output --------------------------------------------------

    def _run_quant_pipeline(
        self, request: AgentRequest, reconciled_by_outcome: dict[str, list[HybridMarketRecord]]
    ) -> tuple[BettingAnalysis, str]:
        """Determine the best line from the reconciled authoritative
        prices (always possible once >=1 exists for the requested
        outcome), then attempt the market-consensus EV chain (requires
        >=MIN_QUOTING_BOOKS sportsbooks with an authoritative price on
        both this outcome and the single opposing outcome). Every
        calculation here is a direct call into src/calculations/ —
        nothing is recomputed, and every number comes from
        HybridMarketRecord.authoritative_odds — never a raw tool_odds/
        rag_odds field read directly.
        """
        primary_records = reconciled_by_outcome[request.selected_outcome]
        primary_prices = {
            record.sportsbook: record.authoritative_odds
            for record in primary_records
            if record.authoritative_odds is not None
        }

        sportsbooks_considered = sorted(primary_prices)
        odds_values = [primary_prices[book] for book in sportsbooks_considered]
        winning_odds = best_odds(odds_values)
        best_sportsbooks = sorted(
            book
            for book in sportsbooks_considered
            if compare_american_odds(primary_prices[book], winning_odds) == 0
        )
        winning_implied_probability = implied_probability(winning_odds)

        other_outcomes = [
            outcome for outcome in reconciled_by_outcome if outcome != request.selected_outcome
        ]

        market_reference_probability: float | None = None
        probability_edge: float | None = None
        computed_expected_value: float | None = None
        computed_positive_ev: bool | None = None
        status = AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE
        quant_status = "insufficient_quant_evidence"

        if len(other_outcomes) == 1:
            opposing_records = reconciled_by_outcome[other_outcomes[0]]
            opposing_prices = {
                record.sportsbook: record.authoritative_odds
                for record in opposing_records
                if record.authoritative_odds is not None
            }
            complete_books = sorted(book for book in sportsbooks_considered if book in opposing_prices)

            if len(complete_books) >= MIN_QUOTING_BOOKS:
                no_vig_by_book: dict[str, float] = {}
                for book in complete_books:
                    fair = calculate_no_vig_probabilities([primary_prices[book], opposing_prices[book]])
                    no_vig_by_book[book] = fair[0]

                tied_complete_books = sorted(
                    book for book in complete_books if primary_prices[book] == winning_odds
                )
                if tied_complete_books:
                    target = tied_complete_books[0]
                    sportsbook_probabilities = [
                        (book, no_vig_by_book[book]) for book in complete_books
                    ]
                    reference_probability = calculate_leave_one_out_consensus(
                        sportsbook_probabilities, target
                    )
                    book_implied = implied_probability(primary_prices[target])
                    market_reference_probability = reference_probability
                    probability_edge = calculate_probability_edge(reference_probability, book_implied)
                    computed_expected_value = expected_value(primary_prices[target], reference_probability)
                    computed_positive_ev = is_positive_ev(primary_prices[target], reference_probability)
                    status = AnalysisStatus.OK
                    quant_status = "ok"

        conflicts_in_scope = sum(
            1
            for outcome in ([request.selected_outcome] + other_outcomes)
            for record in reconciled_by_outcome.get(outcome, [])
            if record.conflict
        )
        reasoning_summary = self._build_reasoning_summary(
            best_sportsbooks=best_sportsbooks,
            winning_odds=winning_odds,
            status=status,
            sportsbooks_considered=sportsbooks_considered,
            source_conflicts=conflicts_in_scope,
        )
        sources = self._build_source_references(reconciled_by_outcome)

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
            architecture=ArchitectureType.HYBRID,
            sources=sources,
        )
        return analysis, quant_status

    @staticmethod
    def _build_source_references(
        reconciled_by_outcome: dict[str, list[HybridMarketRecord]],
    ) -> list[SourceReference]:
        """Provenance for exactly what fed the final numbers (Step 21) —
        only authoritative contributions, not every observed record (the
        full reconciliation detail, including non-authoritative/
        conflicting observations, lives in HybridAgentTrace.reconciled_records)."""
        sources: list[SourceReference] = []
        for outcome, records in reconciled_by_outcome.items():
            for record in records:
                if record.authoritative_source == SourceType.TOOL:
                    sources.append(
                        SourceReference(
                            source_type=SourceType.TOOL,
                            source_id=f"tool:{record.sportsbook}:{outcome}",
                            sportsbook=record.sportsbook,
                        )
                    )
                elif record.authoritative_source == SourceType.RAG:
                    sources.append(
                        SourceReference(
                            source_type=SourceType.RAG,
                            source_id=record.rag_document_id or f"rag:{record.sportsbook}:{outcome}",
                            sportsbook=record.sportsbook,
                            timestamp=record.rag_timestamp,
                        )
                    )
        return sources

    @staticmethod
    def _build_reasoning_summary(
        *,
        best_sportsbooks: list[str],
        winning_odds: int,
        status: AnalysisStatus,
        sportsbooks_considered: list[str],
        source_conflicts: int,
    ) -> str:
        books = ", ".join(best_sportsbooks)
        base = (
            f"From reconciled RAG and tool evidence, {books} offered the best current "
            f"price ({winning_odds:+d}) among {len(sportsbooks_considered)} sportsbook(s) "
            f"considered ({source_conflicts} source conflict(s) resolved by current-tool-data "
            "precedence)."
        )
        if status == AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE:
            base += (
                " Insufficient two-sided current data was reconciled to derive a market "
                "reference probability, so no expected-value verdict is reported."
            )
        return base
