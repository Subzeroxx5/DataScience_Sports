# Results Asset Index

Inventory of every final research-ready table and figure produced by
Milestone 14B. All paths are relative to the repository root. No asset is
duplicated here — this file only indexes what already exists under
`results/experiments/final_v1/analysis/`.

## Figures

### Figure 1

- **Path:** `results/experiments/final_v1/analysis/figures/1_best_line_accuracy.png`
- **Shows:** Best-line accuracy by architecture (RAG 88.9%, TOOL 100.0%,
  HYBRID 100.0%), with Wilson 95% confidence-interval error bars.
- **Manuscript section:** Results — Accuracy.
- **Presentation slide:** Slide 8 — Accuracy Results.
- **Source artifact:** `analysis_table.csv`, `pairwise_comparisons.json` (binary, `best_line_correct`).

### Figure 2

- **Path:** `results/experiments/final_v1/analysis/figures/2_ev_classification_accuracy.png`
- **Shows:** EV classification accuracy by architecture — all three bars
  labeled "N/A" (zero valid observations in the dataset; never a fabricated
  zero-height bar).
- **Manuscript section:** Results — Accuracy.
- **Presentation slide:** Slide 8 — Accuracy Results (callout).
- **Source artifact:** `analysis_table.csv`, `pairwise_comparisons.json` (binary, `ev_classification_correct`).

### Figure 3

- **Path:** `results/experiments/final_v1/analysis/figures/3_freshness_accuracy.png`
- **Shows:** Freshness accuracy by architecture (100.0% for all three),
  with Wilson 95% confidence-interval error bars.
- **Manuscript section:** Results — Freshness.
- **Presentation slide:** Slide 9 — Consistency & Freshness.
- **Source artifact:** `descriptive_statistics.json` (`freshness` per architecture).

### Figure 4

- **Path:** `results/experiments/final_v1/analysis/figures/4_consistency.png`
- **Shows:** Consistency by architecture (1.000 for all three, 0-1 scale).
- **Manuscript section:** Results — Consistency.
- **Presentation slide:** Slide 9 — Consistency & Freshness.
- **Source artifact:** `descriptive_statistics.json` (`descriptive.consistency_mean` per architecture), `statistical_tests.json` (Friedman, `consistency`).

### Figure 5

- **Path:** `results/experiments/final_v1/analysis/figures/5_unsupported_claim_rate.png`
- **Shows:** Unsupported-claim (hallucination) rate by architecture (0.0%
  for all three).
- **Manuscript section:** Results — Secondary Metrics.
- **Presentation slide:** Slide 10 — Why the Architectures Behaved Differently.
- **Source artifact:** `analysis_table.csv`, `descriptive_statistics.json` (`hallucination` per architecture).

### Figure 6

- **Path:** `results/experiments/final_v1/analysis/figures/6_median_latency.png`
- **Shows:** Median total latency by architecture, seconds (RAG 16.159,
  TOOL 6.952, HYBRID 22.696).
- **Manuscript section:** Results — Secondary Metrics.
- **Presentation slide:** Slide 9 (mention) / Slide 10 (discussion).
- **Source artifact:** `descriptive_statistics.json` (`descriptive.latency_median` per architecture), `pairwise_comparisons.json` (continuous, `total_latency_seconds`).

## Tables

### Table 1 — Architecture Descriptive Results

- **Path:** `results/experiments/final_v1/analysis/analysis_table.csv`
  (also embedded in `findings.md` and `analysis_summary.json`).
- **Shows:** Per-architecture success rate, best-line/best-odds/EV-
  classification/freshness accuracy, mean completeness, unsupported-claim
  rate, consistency, median latency.
- **Manuscript section:** Results (all subsections reference it).
- **Presentation slide:** Slide 8, Slide 9.
- **Source artifact:** `src/analysis/tables.py::build_table_1_architecture_results`.

### Table 2 — Binary Pairwise Comparisons

- **Path:** `results/experiments/final_v1/analysis/pairwise_comparisons.json` (key: `"binary"`).
- **Shows:** McNemar results for best-line, best-odds, EV-classification,
  and freshness correctness — all three architecture pairs, each with raw
  p, Holm-adjusted p, discordant-pair counts, and observed accuracies.
- **Manuscript section:** Results — Accuracy; Results — Freshness.
- **Presentation slide:** Slide 8, Slide 9.
- **Source artifact:** `src/analysis/tables.py::build_table_2_binary_pairwise`.

### Table 3 — Continuous Metric Comparisons

- **Path:** `results/experiments/final_v1/analysis/pairwise_comparisons.json` (key: `"continuous"`).
- **Shows:** Wilcoxon signed-rank results for EV absolute error,
  market-reference absolute error, completeness, and total latency — all
  three architecture pairs.
- **Manuscript section:** Results — Secondary Metrics.
- **Presentation slide:** Slide 10.
- **Source artifact:** `src/analysis/tables.py::build_table_3_continuous_comparisons`.

### Table 4 — Failure Counts

- **Path:** `results/experiments/final_v1/analysis/failure_analysis.json` (key: `"failure_counts"`).
- **Shows:** Failure category, count, percentage of observations, and
  scenarios affected, by architecture.
- **Manuscript section:** Results — Failure Patterns.
- **Presentation slide:** Slide 10.
- **Source artifact:** `src/analysis/tables.py::build_table_4_failure_counts`.

### Table 5 — Hybrid Reconciliation

- **Path:** `results/experiments/final_v1/analysis/failure_analysis.json` (key: `"hybrid_reconciliation"`).
- **Shows:** Source agreements/conflicts, correct/incorrect conflict
  resolutions, conflict-resolution accuracy, stale-RAG-promotion count,
  tool-only recoveries, source-reconciliation failures.
- **Manuscript section:** Results — Hybrid Conflict Resolution.
- **Presentation slide:** Slide 10.
- **Source artifact:** `src/analysis/tables.py::build_table_5_hybrid_reconciliation`.

### Omnibus Test Table (supplementary)

- **Path:** `results/experiments/final_v1/analysis/statistical_tests.json` (key: `"omnibus"`).
- **Shows:** Friedman test statistic/p-value for completeness, total
  latency, and consistency across all three architectures.
- **Manuscript section:** Results — Consistency; Results — Secondary Metrics.
- **Presentation slide:** Slide 9.
- **Source artifact:** `src/analysis/tables.py::build_omnibus_table`.

## Other Machine-Readable Outputs (not individually numbered as tables)

| File | Contents |
|---|---|
| `descriptive_statistics.json` | Full per-architecture descriptive stats, phase-level latency, freshness detail, hallucination detail. |
| `scenario_analysis.json` | Per-scenario x per-architecture breakdown, plus exploratory/secondary subgroup summary. |
| `analysis_summary.json` | Top-level experiment metadata + Table 1, for quick machine consumption. |
| `findings.md` | The full narrative findings document this index supports; see `docs/FINAL_RESEARCH_SUMMARY.md` for the condensed version. |

## Notes

- All figures are static PNGs (bar charts only, no 3D charts, per Milestone
  14B's own constraints).
- No figure or table listed here was regenerated or altered by Milestone
  15 — this index only points at Milestone 14B's existing outputs.
