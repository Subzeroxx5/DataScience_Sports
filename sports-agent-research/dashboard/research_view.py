"""Mode B — Research Comparison (Milestone 13, Sections 11-22).

Loads a previously generated Milestone 12 experiment directory and
visualizes it. Never recalculates an agent output, and never presents a
MOCK experiment's numbers as final research results.
"""

from __future__ import annotations

import streamlit as st

from dashboard import charts, data_loader, formatting as fmt
from dashboard.data_loader import ExperimentLoadError, LoadedExperiment
from src.experiments.config import ExecutionMode
from src.models import ArchitectureType

_MOCK_BANNER = "🧪 MOCK — INFRASTRUCTURE VALIDATION ONLY. These are not final research results."
_REAL_BANNER = "REAL execution mode."


def render_research_mode() -> None:
    with st.sidebar:
        st.subheader("Research Controls")
        experiment_ids = data_loader.list_experiment_ids()
        if not experiment_ids:
            st.warning(f"No experiments found under `{data_loader.DEFAULT_EXPERIMENTS_DIR}`.")
            st.caption("Run `python -m src.experiments.runner --mode mock` to generate one.")
            return
        experiment_id = st.selectbox("Experiment", experiment_ids, key="research_experiment_id")

    try:
        loaded = data_loader.load_experiment(experiment_id)
    except ExperimentLoadError as exc:
        st.error(str(exc))
        return

    if not loaded.raw_runs:
        st.warning(f"Experiment {experiment_id!r} has no recorded runs.")
        return

    _render_metadata_panel(loaded)

    if data_loader.is_mock_experiment(loaded):
        st.warning(_MOCK_BANNER)
    else:
        st.info(_REAL_BANNER)

    _render_architecture_summaries(loaded)
    _render_charts(loaded)
    _render_scenario_drill_down(loaded)
    _render_failure_analysis(loaded)
    _render_hybrid_conflict_analysis(loaded)
    _render_raw_data(loaded)
    _render_export(loaded)


def _render_metadata_panel(loaded: LoadedExperiment) -> None:
    st.header(f"Experiment: {loaded.metadata.config.experiment_name}")
    config = loaded.metadata.config

    col1, col2, col3 = st.columns(3)
    col1.metric("Experiment ID", config.experiment_id)
    col1.metric("Execution mode", config.execution_mode.value.upper())
    col2.metric("Model", config.model_name)
    col2.metric("RAG top_k", config.rag_top_k)
    col3.metric("Repetitions", config.repetitions)
    col3.metric("Scenario count", len(config.scenario_ids))

    st.markdown(
        f"- **Temperature:** {config.temperature if config.temperature is not None else fmt.NOT_AVAILABLE} "
        f"(effort=`{config.effort}`)  \n"
        f"- **Provider type:** {config.provider_type}  \n"
        f"- **Embedding model:** {config.embedding_model}  \n"
        f"- **Expected run count:** {loaded.summary.expected_runs}  \n"
        f"- **Recorded run count:** {loaded.summary.recorded_runs} "
        f"(successful={loaded.summary.successful_runs}, failed={loaded.summary.failed_runs})"
    )

    with st.expander("Artifact checksums (reproducibility metadata)"):
        st.json(loaded.metadata.artifact_checksums)


def _render_architecture_summaries(loaded: LoadedExperiment) -> None:
    st.subheader("Core Architecture Comparison")
    comparison = loaded.summary.comparison
    rows = []
    for label, summary in (
        ("RAG", comparison.rag_summary),
        ("TOOL", comparison.tool_summary),
        ("HYBRID", comparison.hybrid_summary),
    ):
        if summary is None:
            continue
        rows.append({
            "Architecture": label,
            "Runs": summary.runs,
            "Success rate": fmt.format_percentage(summary.success_rate),
            "Best-line accuracy": fmt.format_percentage(summary.best_line_accuracy),
            "Best-odds accuracy": fmt.format_percentage(summary.best_odds_accuracy),
            "EV classification accuracy": fmt.format_percentage(summary.ev_classification_accuracy),
            "Mean EV abs. error": fmt.format_error(summary.mean_ev_absolute_error),
            "Market-ref. MAE": fmt.format_error(summary.market_reference_mae),
            "Freshness accuracy": fmt.format_percentage(summary.freshness_accuracy),
            "Mean completeness": fmt.format_percentage(summary.mean_completeness),
            "Unsupported-claim rate": fmt.format_percentage(summary.unsupported_claim_rate),
            "Consistency": fmt.format_percentage(summary.consistency),
            "Mean latency (s)": fmt.format_seconds(summary.mean_total_latency_seconds),
        })
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("No architecture is automatically ranked or declared a winner.")


def _render_charts(loaded: LoadedExperiment) -> None:
    st.subheader("Comparison Charts")
    charts.render_comparison_charts(loaded.summary.comparison)


def _render_scenario_drill_down(loaded: LoadedExperiment) -> None:
    st.subheader("Scenario Drill-Down")
    by_scenario = data_loader.group_by_scenario(loaded.raw_runs)
    scenario_id = st.selectbox("Scenario", sorted(by_scenario), key="research_drilldown_scenario")
    runs = by_scenario[scenario_id]

    ground_truth = data_loader.ground_truth_by_scenario().get(scenario_id)
    if ground_truth is not None:
        st.markdown(
            f"**EXPECTED / GROUND TRUTH** — best sportsbook(s): "
            f"{fmt.format_list(ground_truth.expected_best_sportsbooks)}, "
            f"best odds: {fmt.format_odds(ground_truth.expected_best_odds)}"
        )

    by_architecture = data_loader.group_by_architecture(runs)
    columns = st.columns(len(by_architecture)) if by_architecture else []
    for column, (architecture, arch_runs) in zip(columns, sorted(by_architecture.items(), key=lambda kv: kv[0].value)):
        with column:
            st.markdown(f"**{architecture.value.upper()}**")
            for run in sorted(arch_runs, key=lambda r: r.repetition):
                result = run.common_result
                st.markdown(
                    f"Rep {run.repetition}: {fmt.format_list(result.predicted_best_sportsbooks)} "
                    f"{fmt.format_odds(result.predicted_best_odds)}  \n"
                    f"EV: {fmt.format_error(result.predicted_ev)} | "
                    f"Status: {result.execution_status.value} | "
                    f"Freshness: {fmt.format_freshness(result.freshness_correct)}  \n"
                    f"Latency: {fmt.format_seconds(result.latency_metrics.total_latency_seconds)}"
                )

    _render_repetition_inspection(runs, scenario_id)


def _render_repetition_inspection(runs, scenario_id: str) -> None:
    st.subheader("Repetition Inspection")
    by_architecture = data_loader.group_by_architecture(runs)
    for architecture, arch_runs in sorted(by_architecture.items(), key=lambda kv: kv[0].value):
        if len(arch_runs) <= 1:
            continue
        st.markdown(f"**{architecture.value.upper()}** — {scenario_id}")
        rows = [
            {
                "Repetition": run.repetition,
                "Best sportsbook(s)": fmt.format_list(run.common_result.predicted_best_sportsbooks),
                "Best odds": fmt.format_odds(run.common_result.predicted_best_odds),
                "Positive EV": fmt.format_bool(run.common_result.predicted_positive_ev),
                "EV": fmt.format_error(run.common_result.predicted_ev),
                "Status": run.common_result.execution_status.value,
            }
            for run in sorted(arch_runs, key=lambda r: r.repetition)
        ]
        st.dataframe(rows, hide_index=True, width="stretch")


def _render_failure_analysis(loaded: LoadedExperiment) -> None:
    st.subheader("Failure Analysis")
    counts = data_loader.failure_counts_by_architecture(loaded.raw_runs)
    if not counts:
        st.caption("No runs recorded.")
        return
    rows = []
    for architecture, by_status in sorted(counts.items(), key=lambda kv: kv[0].value):
        for status, n in sorted(by_status.items()):
            rows.append({"Architecture": architecture.value.upper(), "Failure category": status, "Count": n})
    st.dataframe(rows, hide_index=True, width="stretch")

    failed = [r for r in loaded.raw_runs if r.common_result.errors]
    if failed:
        with st.expander(f"Error detail ({len(failed)} run(s) with recorded errors)"):
            for run in failed:
                st.markdown(
                    f"- **{run.architecture.value.upper()}** / {run.scenario_id} / rep {run.repetition} "
                    f"({run.common_result.execution_status.value}): {run.common_result.errors[0]}"
                )


def _render_hybrid_conflict_analysis(loaded: LoadedExperiment) -> None:
    summary = data_loader.hybrid_conflict_summary(loaded.raw_runs)
    if summary is None:
        return
    st.subheader("Hybrid Conflict Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Source agreements", summary["source_agreements"])
    col1.metric("Source conflicts", summary["source_conflicts"])
    col2.metric("Correct conflict resolutions", summary["correct_conflict_resolutions"])
    col2.metric("Conflict-resolution accuracy", fmt.format_percentage(summary["conflict_resolution_accuracy"]))
    col3.metric("Stale-RAG conflicts", summary["stale_rag_conflicts"])
    col3.metric("Stale RAG incorrectly promoted", summary["stale_rag_incorrectly_promoted"])
    st.caption(f"Tool-only recoveries: {summary['tool_only_recoveries']}")


def _render_raw_data(loaded: LoadedExperiment) -> None:
    st.subheader("Raw Data Access")
    col1, col2, col3, col4 = st.columns(4)
    architectures = sorted({r.architecture for r in loaded.raw_runs}, key=lambda a: a.value)
    scenario_ids = sorted({r.scenario_id for r in loaded.raw_runs})
    repetitions = sorted({r.repetition for r in loaded.raw_runs})
    statuses = sorted({r.common_result.execution_status.value for r in loaded.raw_runs})

    architecture_filter = col1.selectbox(
        "Architecture", ["(all)"] + [a.value for a in architectures], key="raw_filter_architecture"
    )
    scenario_filter = col2.selectbox("Scenario", ["(all)"] + scenario_ids, key="raw_filter_scenario")
    repetition_filter = col3.selectbox("Repetition", ["(all)"] + repetitions, key="raw_filter_repetition")
    status_filter = col4.selectbox("Status", ["(all)"] + statuses, key="raw_filter_status")

    from src.evaluation.metrics import FailureCategory

    runs = data_loader.filter_raw_runs(
        loaded.raw_runs,
        architecture=ArchitectureType(architecture_filter) if architecture_filter != "(all)" else None,
        scenario_id=scenario_filter if scenario_filter != "(all)" else None,
        repetition=repetition_filter if repetition_filter != "(all)" else None,
        status=FailureCategory(status_filter) if status_filter != "(all)" else None,
    )

    rows = [
        {
            "Architecture": run.architecture.value,
            "Scenario": run.scenario_id,
            "Repetition": run.repetition,
            "Order": run.execution_order_position,
            "Status": run.common_result.execution_status.value,
            "Best sportsbook(s)": fmt.format_list(run.common_result.predicted_best_sportsbooks),
            "Best odds": fmt.format_odds(run.common_result.predicted_best_odds),
            "EV": fmt.format_error(run.common_result.predicted_ev),
            "Timestamp": run.timestamp.isoformat(),
        }
        for run in runs
    ]
    st.caption(f"{len(rows)} of {len(loaded.raw_runs)} raw run(s) shown. This is the research record — read-only.")
    st.dataframe(rows, hide_index=True, width="stretch")


def _render_export(loaded: LoadedExperiment) -> None:
    with st.expander("Export summary (optional)"):
        summary_json = loaded.summary.model_dump_json(indent=2)
        st.download_button(
            "Download summary.json", data=summary_json,
            file_name=f"{loaded.metadata.config.experiment_id}_summary.json", mime="application/json",
        )
        rows = [
            {
                "architecture": run.architecture.value, "scenario_id": run.scenario_id,
                "repetition": run.repetition, "status": run.common_result.execution_status.value,
                "best_sportsbooks": ";".join(run.common_result.predicted_best_sportsbooks),
                "best_odds": run.common_result.predicted_best_odds,
                "predicted_ev": run.common_result.predicted_ev,
            }
            for run in loaded.raw_runs
        ]
        import pandas as pd

        csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download raw results (CSV)", data=csv_bytes,
            file_name=f"{loaded.metadata.config.experiment_id}_raw_results.csv", mime="text/csv",
        )
        st.caption("Exports reflect already-persisted data only — no recalculation, "
                   "and the original experiment files are never modified.")
