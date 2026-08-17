"""Mode A — Single Scenario Demo (Milestone 13, Sections 4-10).

Renders one architecture's live BettingAnalysis for one controlled
scenario. All computation happens in the existing agent classes via
dashboard/data_loader.py; this module only lays out widgets and reads
already-computed fields.
"""

from __future__ import annotations

import streamlit as st

from dashboard import data_loader, formatting as fmt
from dashboard.data_loader import DemoRunResult
from src.agents.hybrid_agent import HybridAgentTrace
from src.agents.llm_client import DEFAULT_EFFORT, DEFAULT_MODEL
from src.agents.rag_agent import RagAgentTrace
from src.agents.rag_evidence import DEFAULT_RAG_TOP_K
from src.agents.tool_agent import MAX_TOOL_ITERATIONS, ToolAgentTrace
from src.experiments.config import ExecutionMode
from src.models import AnalysisStatus, ArchitectureType
from src.tools.sportsbook_tools import SportsbookTools

_EFFORT_CHOICES = ["low", "medium", "high"]


def render_demo_mode() -> None:
    manifest = data_loader.full_scenario_manifest()
    scenario_by_id = {s.scenario_id: s for s in manifest}

    with st.sidebar:
        st.subheader("Demo Controls")
        scenario_id = st.selectbox("Scenario", sorted(scenario_by_id), key="demo_scenario_id")
        architecture_value = st.selectbox(
            "Architecture", [a.value for a in ArchitectureType],
            format_func=lambda v: v.upper(), key="demo_architecture",
        )
        execution_mode_value = st.radio(
            "Execution mode", [ExecutionMode.MOCK.value, ExecutionMode.REAL.value],
            index=0, key="demo_execution_mode",
            help="MOCK reuses the deterministic fake-LLM policy (no API cost). "
                 "REAL calls the configured Anthropic model.",
        )
        with st.expander("Advanced settings"):
            model_name = st.text_input("Model", value=DEFAULT_MODEL, key="demo_model_name")
            effort = st.selectbox("Effort", _EFFORT_CHOICES, index=0, key="demo_effort")
            rag_top_k = st.number_input(
                "RAG top_k", min_value=1, value=DEFAULT_RAG_TOP_K, key="demo_rag_top_k"
            )
            max_tool_iterations = st.number_input(
                "Max tool iterations", min_value=1, value=MAX_TOOL_ITERATIONS, key="demo_max_iter"
            )

    scenario = scenario_by_id[scenario_id]
    architecture = ArchitectureType(architecture_value)
    execution_mode = ExecutionMode(execution_mode_value)
    context = data_loader.game_context(scenario_id)
    game = context["game"]

    st.markdown(
        f"**Game:** {game['away_team']} @ {game['home_team']}  \n"
        f"**Market:** {scenario.market_type.value}  \n"
        f"**Selected outcome:** {scenario.selected_outcome}"
    )
    st.markdown(f"**Canonical query:** _{scenario.query}_")

    run_clicked = st.button("Run Analysis", type="primary")

    state_key = f"demo_result::{scenario_id}::{architecture_value}::{execution_mode_value}"
    if run_clicked:
        with st.spinner(f"Running {architecture_value.upper()} analysis..."):
            result = data_loader.run_demo_analysis(
                architecture, scenario, execution_mode,
                model_name, effort, int(rag_top_k), int(max_tool_iterations),
            )
        st.session_state[state_key] = result

    result: DemoRunResult | None = st.session_state.get(state_key)
    if result is None:
        st.info("Click **Run Analysis** to execute the selected architecture on this scenario.")
        return

    if execution_mode == ExecutionMode.MOCK:
        st.caption("MOCK — deterministic fake-LLM policy, no API cost.")

    _render_result_summary(result)
    _render_sportsbook_table(result)
    _render_trace(result)


def _render_result_summary(result: DemoRunResult) -> None:
    st.subheader("Result Summary")

    if result.incomplete or result.analysis is None:
        st.warning(f"No analysis produced (incomplete run): {result.error_message}")
        return

    analysis = result.analysis
    total_latency = getattr(result.trace, "total_latency_seconds", None)

    col1, col2, col3 = st.columns(3)
    col1.metric("Architecture", analysis.architecture.value.upper())
    col1.metric("Status", analysis.status.value)
    col2.metric("Best sportsbook(s)", fmt.format_list(analysis.best_sportsbooks))
    col2.metric("Best odds", fmt.format_odds(analysis.best_odds))
    col3.metric("Positive EV", fmt.format_bool(analysis.positive_ev))
    col3.metric("Total latency", fmt.format_seconds(total_latency))

    st.markdown(
        f"- **Game:** {analysis.game_id}  \n"
        f"- **Market:** {analysis.market.value}  \n"
        f"- **Selected outcome:** {analysis.selected_outcome}  \n"
        f"- **Book implied probability:** {fmt.format_probability(analysis.implied_probability)}  \n"
        f"- **Market reference probability:** {fmt.format_probability(analysis.market_reference_probability)}  \n"
        f"- **Probability edge:** {fmt.format_probability(analysis.probability_edge)}  \n"
        f"- **Expected value:** {fmt.format_error(analysis.expected_value)}  \n"
    )

    if analysis.status == AnalysisStatus.INSUFFICIENT_QUANT_EVIDENCE:
        st.info(fmt.INSUFFICIENT_EVIDENCE + ": a best line was found, but not enough two-sided "
                "data was available to derive a market reference probability / EV verdict.")

    st.markdown("**Reasoning summary (from the agent's own output — not regenerated here):**")
    st.write(analysis.reasoning_summary)


def _render_sportsbook_table(result: DemoRunResult) -> None:
    st.subheader("Sportsbook Comparison")
    trace = result.trace
    if trace is None:
        st.caption(fmt.UNAVAILABLE)
        return

    rows: list[dict] = []
    if isinstance(trace, RagAgentTrace):
        rows = _rag_comparison_rows(trace)
    elif isinstance(trace, ToolAgentTrace):
        rows = _tool_comparison_rows(result)
    elif isinstance(trace, HybridAgentTrace):
        rows = _hybrid_comparison_rows(trace, result.request.selected_outcome)

    if not rows:
        st.caption("No sportsbook data available for this run.")
        return
    st.dataframe(rows, hide_index=True, width="stretch")


def _rag_comparison_rows(trace: RagAgentTrace) -> list[dict]:
    if trace.extraction_result is None:
        return []
    from src.calculations.odds_math import implied_probability

    rows = []
    for price in trace.extraction_result.sportsbook_prices:
        rows.append({
            "Sportsbook": price.sportsbook,
            "American Odds": fmt.format_odds(price.american_odds),
            "Implied Probability": fmt.format_probability(implied_probability(price.american_odds)),
            "Current/Stale": fmt.format_current_stale(price.is_current),
            "Source": "RAG extraction",
        })
    return rows


def _tool_comparison_rows(result: DemoRunResult) -> list[dict]:
    tools = result.aux_handle
    if not isinstance(tools, SportsbookTools):
        return []
    from src.calculations.odds_math import implied_probability

    try:
        odds_list = tools.get_odds(
            result.request.game_id, result.request.market_type, result.request.selected_outcome
        )
    except Exception:
        return []

    rows = []
    for odds in odds_list:
        rows.append({
            "Sportsbook": odds.sportsbook,
            "American Odds": fmt.format_odds(odds.american_odds),
            "Implied Probability": fmt.format_probability(implied_probability(odds.american_odds)),
            "Current/Stale": fmt.format_current_stale(odds.is_current),
            "Source": "Structured tool data",
        })
    return rows


def _hybrid_comparison_rows(trace: HybridAgentTrace, selected_outcome: str) -> list[dict]:
    from src.calculations.odds_math import implied_probability

    rows = []
    for record in trace.reconciled_records:
        if record.selected_outcome != selected_outcome:
            continue
        odds = record.authoritative_odds
        rows.append({
            "Sportsbook": record.sportsbook,
            "American Odds": fmt.format_odds(odds),
            "Implied Probability": (
                fmt.format_probability(implied_probability(odds)) if odds is not None else fmt.NOT_AVAILABLE
            ),
            "Current/Stale": fmt.format_current_stale(record.rag_is_current) if not record.tool_available else "CURRENT",
            "Authoritative Source": fmt.format_enum(record.authoritative_source),
        })
    return rows


def _render_trace(result: DemoRunResult) -> None:
    trace = result.trace
    if trace is None:
        return

    with st.expander("Architecture Trace / Debug", expanded=False):
        if isinstance(trace, RagAgentTrace):
            _render_rag_trace(trace)
        elif isinstance(trace, ToolAgentTrace):
            _render_tool_trace(trace)
        elif isinstance(trace, HybridAgentTrace):
            _render_hybrid_trace(trace)


def _render_rag_trace(trace: RagAgentTrace) -> None:
    st.markdown(f"**Model:** {trace.model} | **top_k:** {trace.top_k}")
    st.markdown(f"**Validation status:** {trace.validation_status} | **Quant status:** {trace.quant_status}")
    rows = [
        {"Document ID": doc_id, "Rank": rank + 1, "Similarity Score": f"{score:.4f}"}
        for rank, (doc_id, score) in enumerate(zip(trace.retrieved_document_ids, trace.retrieval_scores))
    ]
    st.markdown("**Retrieved documents:**")
    st.dataframe(rows, hide_index=True, width="stretch") if rows else st.caption("(none retrieved)")
    if trace.rejected_extraction_reasons:
        st.markdown("**Rejected extraction claims:**")
        for reason in trace.rejected_extraction_reasons:
            st.caption(f"- {reason}")
    if trace.errors:
        st.error("\n".join(trace.errors))


def _render_tool_trace(trace: ToolAgentTrace) -> None:
    st.markdown(f"**Model:** {trace.model} | **Effort:** {trace.effort} | **Iterations used:** {trace.iterations_used}")
    st.markdown(f"**Validation status:** {trace.validation_status} | **Quant status:** {trace.quant_status}")
    rows = [
        {
            "Order": call.call_sequence,
            "Tool": call.tool_name,
            "Arguments": call.arguments,
            "Success": fmt.format_bool(call.success),
            "Redundant": fmt.format_bool(call.is_redundant),
            "Latency": fmt.format_seconds(call.latency_seconds),
        }
        for call in trace.tool_calls
    ]
    st.markdown("**Tool calls:**")
    st.dataframe(rows, hide_index=True, width="stretch") if rows else st.caption("(no tool calls made)")
    if trace.errors:
        st.error("\n".join(trace.errors))


def _render_hybrid_trace(trace: HybridAgentTrace) -> None:
    st.markdown(f"**Model:** {trace.model} | **Effort:** {trace.effort}")
    st.markdown(
        f"**Validation status:** {trace.validation_status.value} | **Quant status:** {trace.quant_status}"
    )

    st.markdown("**RAG evidence:**")
    rag_rows = [
        {"Document ID": doc_id, "Rank": rank + 1, "Similarity Score": f"{score:.4f}"}
        for rank, (doc_id, score) in enumerate(zip(trace.retrieved_document_ids, trace.rag_scores))
    ]
    st.dataframe(rag_rows, hide_index=True, width="stretch") if rag_rows else st.caption("(none retrieved)")

    st.markdown("**Tool calls:**")
    tool_rows = [
        {
            "Order": call.call_sequence, "Tool": call.tool_name, "Arguments": call.arguments,
            "Success": fmt.format_bool(call.success), "Redundant": fmt.format_bool(call.is_redundant),
            "Latency": fmt.format_seconds(call.latency_seconds),
        }
        for call in trace.tool_calls
    ]
    st.dataframe(tool_rows, hide_index=True, width="stretch") if tool_rows else st.caption("(no tool calls made)")

    st.markdown(
        f"**Source agreements:** {trace.source_agreements} | "
        f"**Source conflicts:** {trace.source_conflicts} | "
        f"**RAG-only records:** {trace.rag_only_records} | "
        f"**Tool-only records:** {trace.tool_only_records}"
    )

    _render_freshness_conflicts(trace)

    if trace.errors:
        st.error("\n".join(trace.errors))


def _render_freshness_conflicts(trace: HybridAgentTrace) -> None:
    """Section 9: make freshness/conflict behavior easy to inspect —
    never hides conflicting evidence, even when the RAG side was stale
    and superseded by current tool data."""
    conflicts = [r for r in trace.reconciled_records if r.conflict]
    if not conflicts:
        st.caption("No source conflicts recorded for this run.")
        return

    st.markdown("**Source conflicts (RAG vs. tool):**")
    for record in conflicts:
        st.markdown(
            f"- **{record.sportsbook}** ({record.selected_outcome})  \n"
            f"  RAG snapshot: {fmt.format_odds(record.rag_odds)} "
            f"({fmt.format_current_stale(record.rag_is_current)})  \n"
            f"  Current tool: {fmt.format_odds(record.tool_odds)}  \n"
            f"  Authoritative: {fmt.format_odds(record.authoritative_odds)} "
            f"({fmt.format_enum(record.authoritative_source)})  \n"
            f"  Resolution: {fmt.format_enum(record.conflict_resolution_reason)}"
        )
