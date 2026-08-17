# Final Research Summary

Authoritative, concise summary of this project's completed research. Every
number below is traced to `results/experiments/final_v1/analysis/` (Milestone
14B) — this document performs no new calculation.

## Research Question

How does agent architecture—tool calling, retrieval-augmented generation
(RAG), or a hybrid approach—affect the accuracy, consistency, and freshness
of an AI agent identifying positive expected value opportunities across
multiple sportsbooks?

## Experimental Design

**Independent variable:** agent architecture, at three levels:

- **RAG-only** — retrieves evidence from a vector-indexed controlled
  document corpus; no access to structured sportsbook tools.
- **Tool-calling-only** — calls structured tools backed by `OddsProvider`;
  no access to the RAG corpus.
- **Hybrid** — has access to both channels; current structured tool data
  takes precedence over conflicting RAG-derived data
  (`CURRENT_TOOL_DATA_PRECEDENCE`).

**Held constant across all three architectures:**

- the same locally hosted LLM and configuration (see below)
- the same 11 controlled scenarios
- the same canonical, architecture-neutral query per scenario (no
  per-architecture prompt tuning)
- the same shared deterministic quantitative engine
- the same deterministic ground truth (never exposed to any agent)
- the same repetition count and a deterministic, balanced architecture
  execution order across repetitions
- the same unified evaluation framework and metric definitions

**Execution mode:** REAL — genuine local LLM inference, not the project's
deterministic mock/fake-LLM infrastructure used for earlier infrastructure
validation.

**Exact final configuration** (frozen in `experiments/final_experiment.json`,
Milestone 14A):

| Setting | Value |
|---|---|
| LLM provider | Ollama (local) |
| Exact model | `llama3.1:8b` |
| Temperature | 0.0 |
| RAG top_k | 5 |
| Max tool iterations | 6 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Scenario count | 11 (S001-S009, S012, S013) |
| Repetitions per (architecture, scenario) | 10 |
| Total raw observations | 330 (110 per architecture) |
| Quant methodology | `market_consensus_leave_one_out_v1` |

The benchmark is a **controlled, synthetic** sportsbook dataset with
deterministic ground truth — not live market data.

## Quantitative Method

The shared quantitative engine (`src/calculations/`) — identical across all
three architectures, never reimplemented per-architecture:

1. **American odds -> implied probability**: the standard conversion from a
   sportsbook's American-odds quote to its raw (vig-included) implied
   probability.
2. **No-vig probability**: normalizes a market's raw implied probabilities
   (which sum to more than 1.0 due to bookmaker margin) to fair
   probabilities that sum to 1.0.
3. **Leave-one-sportsbook-out market consensus**: for a given sportsbook's
   price, the reference probability is the mean no-vig probability of the
   *other* sportsbooks quoting that outcome — a sportsbook is never used to
   validate its own price.
4. **Probability edge**: the difference between the market-consensus
   reference probability and a specific sportsbook's own implied
   probability.
5. **Expected value (EV)**: computed from a sportsbook's American odds and
   the market-consensus reference probability.
6. **Market dispersion**: descriptive spread statistics (mean, median,
   population standard deviation, range) of no-vig probabilities across
   sportsbooks for one outcome.

**Critical framing:** the market-implied (no-vig, leave-one-out consensus)
reference probability is **not** the true win probability of the sporting
outcome — it is a market-derived estimate only. "Positive EV" in this
project's results means positive EV *relative to this market-consensus
reference methodology*, not a claim about real-world betting profitability.

## Primary Results

Source: `results/experiments/final_v1/analysis/findings.md`,
`analysis_table.csv`, `pairwise_comparisons.json`. Values not recalculated
here.

### Accuracy

| Architecture | Best-Line Accuracy | Best-Odds Accuracy | EV Classification Accuracy |
|---|---|---|---|
| RAG | 88.9% | 88.9% | N/A |
| TOOL | 100.0% | 100.0% | N/A |
| HYBRID | 100.0% | 100.0% | N/A |

- RAG vs TOOL and RAG vs HYBRID (best-line and best-odds accuracy): the
  paired analysis (exact McNemar) provides evidence of a difference —
  Holm-adjusted p=0.005859 (raw p=0.001953), difference = -11.1 percentage
  points, paired N=90.
- TOOL vs HYBRID (best-line and best-odds accuracy): not statistically
  distinguishable at alpha=.05 (Holm-adjusted p=1.0; zero discordant pairs
  — the two architectures agreed on every paired observation).
- **EV classification accuracy is N/A for all three architectures and all
  pairwise comparisons.** Every successful observation in this dataset
  landed in the `quant_insufficient_data` status: a validated best line was
  produced, but no observation (across all 330) gathered enough two-sided
  market data to reach a full EV verdict. This is reported as-is, never
  imputed.

### Consistency

All three architectures: mean = median = min = max = 1.000 (stdev = 0.000)
across the 11 scenarios with repeated observations. The Friedman omnibus
test on matched per-scenario consistency values is correctly degenerate
(every matched block is a three-way tie; statistic=0.0, p=1.0) — there is no
distinguishable difference because there is no variation to detect.

Consistency measures reproducibility of the architecture's research-relevant
output signature across repetitions, not correctness — a scenario that fails
identically every repetition is also "consistent" by this definition.

### Freshness

All three architectures: 10/10 correct on the one freshness-designated
scenario in this benchmark (100.0%, 95% Wilson CI [72.2%, 100.0%]; 0 used a
known stale value, 0 used another incorrect value). Pairwise comparisons show
zero discordant pairs for every architecture pair (Holm-adjusted p=1.0 in
each case) — not statistically distinguishable by construction, since no
architecture made a freshness error in this dataset.

## Secondary Results

### Completeness

RAG's median completeness (0.75) is significantly lower than TOOL's and
HYBRID's (1.00 each) — Holm-adjusted p=9.223e-11 (raw p=4.611e-11) for both
comparisons, paired N=110. TOOL vs HYBRID: medians identical (1.00 vs 1.00);
every paired difference was exactly zero, so no test was performed.

### Hallucination / Unsupported Claims

Zero unsupported claims for all three architectures (RAG 0/90, TOOL 0/100,
HYBRID 0/100; rate = 0.0% in every case). Zero runs with at least one
unsupported claim.

### Latency

| Architecture | Mean (s) | Median (s) | Stdev (s) | Min (s) | Max (s) |
|---|---|---|---|---|---|
| RAG | 14.502 | 16.159 | 5.792 | 3.660 | 39.681 |
| TOOL | 23.413 | 6.952 | 49.569 | 5.495 | 180.008 |
| HYBRID | 36.632 | 22.696 | 46.821 | 9.211 | 183.865 |

All three pairwise comparisons (Wilcoxon signed-rank) are statistically
significant: RAG vs TOOL Holm-adjusted p=3.844e-08; RAG vs HYBRID
Holm-adjusted p=7.021e-19; TOOL vs HYBRID Holm-adjusted p=5.345e-19 (all
paired N=110). The Friedman omnibus across all three architectures is also
significant (statistic=170.75, p=8.38e-38). TOOL had the lowest median
latency; HYBRID (which performs both the RAG and tool-calling work) had the
highest.

### Failures

40 of 330 observations failed (12.1%); all were preserved, none dropped:

| Architecture | Category | Count | % of Observations | Scenarios Affected |
|---|---|---|---|---|
| RAG | `insufficient_retrieved_evidence` | 20 | 18.2% | S012, S013 |
| TOOL | `llm_output_invalid` | 10 | 9.1% | S012 |
| HYBRID | `insufficient_current_data` | 10 | 9.1% | S012 |

All 10 TOOL and 10 HYBRID failures on scenario S012 are attributable to a
client-side 180-second inference read-timeout on that scenario's
tool-calling turns, not a reasoning failure — an infrastructure
characteristic of this local model/scenario combination.

### Hybrid Conflict Resolution

Source agreements: 310. Source conflicts: **0**. Conflict-resolution
accuracy: N/A (undefined — zero conflicts occurred in this dataset, not a
fabricated zero or one). Stale-RAG conflicts: 0. Stale RAG incorrectly
promoted: 0. Tool-only recoveries: 80. Source-reconciliation failures: 0.

The `CURRENT_TOOL_DATA_PRECEDENCE` reconciliation mechanism itself is
unmodified and was independently verified intact (a direct synthetic-conflict
check plus the dedicated, fully-passing `tests/test_hybrid_reconciliation.py`
suite) — this dataset simply never produced a live disagreement between the
RAG-derived and tool-derived price for the same sportsbook/outcome.

## Direct Answer to Research Question

No single architecture wins across all three primary outcomes.

- **Accuracy:** TOOL and HYBRID statistically outperformed RAG on best-line
  and best-odds accuracy (Holm-adjusted p<.05), but were themselves
  statistically indistinguishable from each other (they agreed on every
  paired observation in this dataset). EV classification accuracy could not
  be evaluated for any architecture — no observation reached a full EV
  verdict.
- **Consistency:** identical across all three architectures (1.000) — no
  distinguishable difference, because there was no variation to detect.
- **Freshness:** identical across all three architectures (100.0%) on the
  one freshness-designated scenario available — no distinguishable
  difference, because no architecture made a freshness error.
- **Tradeoffs:** RAG had markedly lower completeness (~73% vs. ~91%) and
  more failed observations than TOOL/HYBRID, but also lower median latency
  than HYBRID. HYBRID was the slowest architecture overall (it performs both
  the RAG and tool-calling work), with no measured accuracy advantage over
  TOOL in this dataset. Zero hallucinations were observed for any
  architecture.

The data does not support declaring RAG, TOOL, or HYBRID "the best"
architecture overall; it supports architecture-specific tradeoffs between
accuracy/completeness (RAG lower) and latency (HYBRID highest, TOOL lowest),
while consistency and freshness showed no distinguishable difference in this
dataset.

## Limitations

(Reproduced from `findings.md`, Milestone 14B — not restated or reinterpreted.)

- Controlled/synthetic sportsbook benchmark, not live market data.
- Limited scenario count (11) and finite repetition count (10 per
  architecture/scenario) — statistical power is limited, especially for
  subgroups.
- Results are specific to the frozen local Ollama model (`llama3.1:8b`) and
  may not generalize to other local or frontier-hosted models.
- Results are specific to the selected embedding model
  (`sentence-transformers/all-MiniLM-L6-v2`).
- No live sportsbook API validation was performed.
- The market-implied (no-vig, leave-one-out consensus) reference probability
  is not the true win probability of the sporting outcome.
- No independent predictive sports ML model was used or compared against.
- Local LLM behavior may differ substantially from frontier hosted models.
- This experiment does not establish real-world betting profitability.
- **Local-model capability limitations observed in this dataset:** the model
  frequently did not gather two-sided market data (0/330 observations
  reached a full EV verdict), and TOOL/HYBRID's tool-calling turns sometimes
  exceeded the client's 180-second timeout on scenario S012 (10 TOOL + 10
  HYBRID failures, all timeout-attributable, not a reasoning failure).

## Future Work

- **Live sportsbook providers.** A live-data `OddsProvider` implementation
  behind the existing abstraction, requiring zero change to the tools,
  agents, or quant engine above it.
- **Broader scenario coverage.** Extending beyond the current 11-scenario
  representative subset (and the full 14-scenario controlled benchmark) to a
  larger, more varied set of controlled scenarios.
- **Alternative LLMs.** Repeating the final experiment with other local
  models and/or frontier-hosted models, using the same configuration-driven
  provider abstraction (`ExperimentConfig.llm_provider`), to test whether
  this dataset's findings (especially the pervasive `quant_insufficient_data`
  outcome and the S012 timeout failures) are specific to `llama3.1:8b`.
- **Independent predictive ML model.** Future work could introduce an
  independently trained and calibrated predictive model and compare:

  ```text
  ML-derived win probability
           vs.
  market-implied no-vig consensus
  ```

  to test value signals independent of the market-derived reference
  probability, without changing the core architecture-comparison question.
  This was **not** implemented as part of this project.
