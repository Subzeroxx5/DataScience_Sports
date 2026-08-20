*First-draft manuscript. Every numerical claim below is traceable to
`results/experiments/final_v1/analysis/` (Milestone 14B) — none is
recalculated or estimated here. Literature claims are cited inline
(Author, Year); full bibliographic entries are in the References section
at the end. Sources were located and verified via web search against
their original abstract/publication pages — see `docs/CITATION_NEEDS.md`
for the mapping from each claim to its source. This is an
undergraduate/intern-research-report-style first draft, not a
publication-ready manuscript.*

---

# Retrieval, Tools, or Both? A Controlled Comparison of Agent Architectures for Sportsbook Price Identification

## Abstract

Large language model (LLM) agents increasingly rely on external information
— retrieved documents, callable tools, or both — to answer questions that
require current, verifiable data (Wang et al., 2023). We ask: how does agent
architecture (retrieval-augmented generation, tool calling, or a hybrid of
the two) affect the accuracy, consistency, and freshness of an AI agent
identifying positive expected value opportunities across multiple
sportsbooks? We built a controlled research prototype with a synthetic
sportsbook benchmark, deterministic ground truth, a shared deterministic
quantitative engine (implied probability, no-vig normalization,
leave-one-sportsbook-out market consensus, expected value), and three
architectures — RAG-only, tool-calling-only, and hybrid — that share every
experimental control except architecture itself, including one locally
hosted LLM (`llama3.1:8b`, temperature 0.0) used identically across all
three. We ran 330 real observations (11 scenarios x 10 repetitions x 3
architectures) and analyzed them with paired nonparametric statistics
(exact McNemar, Wilcoxon signed-rank, Friedman) under a Holm-corrected
alpha of 0.05. Tool-calling and hybrid architectures significantly
outperformed the RAG-only architecture on best-line and best-odds accuracy
(88.9% vs. 100.0%, Holm-adjusted p=0.0059) and completeness (median 0.75 vs.
1.00, Holm-adjusted p=9.2e-11), but were themselves statistically
indistinguishable from each other on every binary accuracy metric. Hybrid
was significantly slower than tool-calling (median 22.7s vs. 7.0s,
Holm-adjusted p=5.3e-19) with no measured accuracy benefit over tool-calling
alone. Consistency and freshness showed no distinguishable difference across
architectures in this dataset. No architecture reached a full
expected-value verdict in any observation, a data characteristic discussed
below rather than a result. We conclude that no single architecture
dominates across all three primary outcomes studied; architecture selection
in this setting involves an accuracy/completeness-versus-latency tradeoff.

## 1. Introduction

Identifying the best available price for a sports betting market requires
gathering current information from multiple sportsbooks and comparing it
against a reference for value. This is a natural setting to study how AI
agents obtain and use external information, because the correct answer is
determined by verifiable, structured facts (prices) rather than by
subjective judgment, and because "value" itself has a defensible
market-derived reference point (a no-vig consensus across sportsbooks) even
though it is not a claim about the true probability of the sporting outcome.

Modern LLM agents obtain external information in at least two structurally
different ways: retrieval-augmented generation (RAG), where the model reads
snippets pulled from a document corpus (Lewis et al., 2020), and tool
calling (function calling), where the model invokes structured operations
that return exact, typed data (Schick et al., 2023). Hybrid designs
combine both.
Each approach has different failure modes: retrieval can surface stale or
incomplete evidence without the model necessarily recognizing it as stale;
tool calling depends on the model reliably orchestrating multiple calls in
the right sequence. Freshness — whether the information an agent ultimately
relies on is current rather than outdated — is a particularly important
and under-examined property for this domain, since sportsbook prices change
continuously.

This motivates our research question: **how does agent architecture—tool
calling, retrieval-augmented generation (RAG), or a hybrid approach—affect
the accuracy, consistency, and freshness of an AI agent identifying positive
expected value opportunities across multiple sportsbooks?**

Our contribution is a controlled, fully reproducible experimental framework
that isolates architecture as the only meaningfully varying factor —
holding the LLM, its configuration, the scenario set, the query wording,
the quantitative engine, and the evaluation methodology constant across all
three architectures — together with the resulting statistical findings from
a real (non-mock) 330-observation experiment using a single locally hosted
model.

## 2. Background / Related Concepts

Large language model agents that answer questions using external,
non-parametric knowledge sources have become a common design pattern
(Wang et al., 2023). Retrieval-augmented generation supplies the model
with passages retrieved from a corpus via semantic (embedding) similarity
search (Lewis et al., 2020). Tool calling (also called function calling)
instead lets the model invoke predefined operations with structured
arguments and receive structured results, which can make the model's
final answer traceable to an exact data source rather than to free-form
generated text (Schick et al., 2023). Hybrid agents that combine both
channels, and that must reconcile disagreements between them, are a
comparatively less studied design point (Singh et al., 2025).

Sportsbook betting markets quote prices as American odds, which imply a
probability once converted; because sportsbooks build in a margin (the
"vig" or "overround"), the raw implied probabilities across a market's
outcomes sum to more than 1.0, and must be normalized ("no-vig" or "fair"
probabilities) before they can be used as a probability reference
(Shin, 1993; Štrumbelj, 2014). A market-consensus reference probability —
an average of no-vig probabilities across sportsbooks, often computed
leaving out the sportsbook under evaluation to avoid circularity — is one
common way to estimate a "fair" price against which a specific
sportsbook's price can be compared for value (Wolfers & Zitzewitz, 2004).
Expected value (EV) relative to such a reference is a standard framing for
identifying favorably priced bets, though it is important to note that a
market-consensus probability is not a claim about the true probability of
the underlying sporting outcome — bookmakers set prices to manage their
own risk, not solely to forecast outcomes (Levitt, 2004).

Finally, LLM output reliability — including the tendency of models to state
claims not supported by their actual inputs ("hallucination") — is a
relevant concern whenever an agent's output feeds into a decision with real
consequences (Huang et al., 2023). Our architecture designs are built so that
the final structured output can only contain data that was itself returned
by a validated tool call or a provenance-checked retrieval, structurally
limiting (though not by itself proving the absence of) this failure mode.

## 3. Methods

### 3.1 Research Design

The independent variable is agent architecture, at three levels: RAG-only,
tool-calling-only, and hybrid. The primary dependent variables are
accuracy (best-line accuracy, best-odds accuracy, and EV classification
accuracy), consistency, and freshness. Secondary dependent variables are
completeness, unsupported-claim (hallucination) rate, latency, failure
rate/category, and — for the hybrid architecture specifically —
conflict-resolution behavior between its two information channels.

### 3.2 Controlled Benchmark

All experiments use a controlled, synthetic sportsbook benchmark: games,
markets, and per-sportsbook American-odds quotes are fixed, versioned data,
not live market feeds. Ground truth (the correct best sportsbook/price for
each scenario) is computed once, deterministically, directly from this
benchmark data — independent of any agent architecture — and is never
exposed to an agent as part of its input at any point in the pipeline. The
final experiment used 11 scenarios, a representative subset of the full
controlled benchmark spanning positive-odds, negative-odds, and mixed-sign
cases; a best-line tie case; a missing-sportsbook-data case; a
current-vs-stale freshness case; and moneyline, spread, and total market
types.

### 3.3 Architectures

**RAG-only** retrieves the top-k most similar documents from a vector index
built over a controlled document corpus (via a sentence-transformer
embedding model and an exact FAISS inner-product index) and extracts
sportsbook price claims from that retrieved text, validating each claim's
provenance against the retrieved documents before it can reach the final
answer. It has no access to structured sportsbook tools.

**Tool-calling-only** runs a bounded multi-turn loop in which the model
calls structured tools (backed by an `OddsProvider` abstraction) to fetch
current game, market, and price data. The final answer is built exclusively
from the actual structured values returned by tool calls — never from the
model's own generated text — so a hallucinated claim cannot structurally
reach the output. It has no access to the RAG corpus.

**Hybrid** has access to both channels. A deterministic, LLM-free
reconciliation step decides, per sportsbook/outcome, which source is
authoritative: current structured tool data always takes precedence over a
disagreeing RAG-derived price for the same sportsbook and outcome.

### 3.4 Shared Quantitative Engine

All betting-relevant arithmetic — American-odds-to-implied-probability
conversion, no-vig (fair) probability normalization, leave-one-sportsbook-out
market consensus, probability edge, expected value, and market dispersion
statistics — is implemented once and shared identically across all three
architectures; no architecture reimplements any of this math. The
market-consensus reference probability this engine produces is explicitly
not treated, in this study, as a claim about the true probability of the
sporting outcome — only as a market-derived reference point.

### 3.5 Experimental Controls

The following were held identical across all three architectures for the
final experiment: the LLM provider and exact model (`llama3.1:8b`, a
locally hosted Ollama model, chosen so the experiment could run without
per-call API cost), temperature (0.0), RAG retrieval depth (top_k=5), the
tool-calling loop's iteration bound (6), the embedding model
(`sentence-transformers/all-MiniLM-L6-v2`), the 11-scenario set, one
canonical, architecture-neutral natural-language query per scenario (with
no per-architecture wording differences), the repetition count (10 per
architecture/scenario), the shared quantitative engine, and the evaluation
framework and metric definitions. The order in which the three architectures
were executed was rotated deterministically across repetitions (e.g.
RAG->TOOL->HYBRID, then TOOL->HYBRID->RAG, then HYBRID->RAG->TOOL) so that
architecture identity is not confounded with execution order.

### 3.6 Experimental Procedure

The experimental configuration was frozen in a version-controlled file
before execution began. A preflight validation pass — including the full
automated test suite, a check that the deterministic ground-truth generator
still reproduces the persisted ground-truth file, a live retrieval smoke
query against the real vector index, a live smoke check of the structured
sportsbook provider, and a direct check that the hybrid
current-tool-data-precedence policy still behaves correctly — was required
to pass before any real inference call was made. The experiment was then
executed through one unified experiment runner (never a separate script per
architecture), which recorded every individual observation — architecture,
scenario, repetition, and execution-order position, together with the full
agent output, evaluation result, and any failure — to an append-only raw
results file. After execution, the dataset was validated: the recorded run
count matched the expected count (330 of 330) with zero duplicate or
missing (architecture, scenario, repetition) keys, a full scenario-coverage
matrix confirmed every scenario had the expected number of observations
under every architecture, every raw result validated against its schema
with zero errors, ground-truth isolation and architecture-access-boundary
audits both passed, and a nine-way comparison of artifact fingerprints
(covering the benchmark data, both ground-truth files, the RAG corpus, the
RAG index configuration, both system prompts, the canonical-query-template
source, and the frozen configuration file itself) confirmed every controlled
input was unchanged from before the run to after it.

### 3.7 Statistical Analysis

For proportion metrics (best-line accuracy, best-odds accuracy, freshness
accuracy, success rate) we report 95% Wilson score confidence intervals, in
preference to the naive normal approximation, because the latter behaves
poorly near 0% or 100% and this dataset includes several such cases. For
paired binary outcomes (whether a given observation's predicted best line,
best odds, EV classification, or freshness judgment was correct) we used
the exact (binomial) form of McNemar's test throughout, appropriate given
the modest discordant-pair counts involved. For paired continuous outcomes
(EV absolute error, market-reference absolute error, completeness, total
latency) we used the Wilcoxon signed-rank test. For a three-architecture
omnibus comparison on matched observations (completeness and latency at the
individual-observation level; consistency at the scenario level) we used
the Friedman test. Because we made three pairwise architecture comparisons
(RAG-TOOL, RAG-HYBRID, TOOL-HYBRID) for each metric, we applied
Holm-Bonferroni correction within each metric family independently — never
pooling the correction across unrelated metric families. The alpha level
(0.05) and the correction method were fixed before any test was run and
were not changed after observing results.

## 4. Results

*(Reference: `results/experiments/final_v1/analysis/analysis_table.csv`,
`pairwise_comparisons.json`, `statistical_tests.json`, and the figures under
`results/experiments/final_v1/analysis/figures/`. See
`docs/RESULTS_ASSET_INDEX.md` for the full asset list.)*

### Accuracy

RAG's observed best-line accuracy was 88.9% and best-odds accuracy was
88.9% (both out of a valid paired N of 90 against each of TOOL and HYBRID);
TOOL's and HYBRID's were each 100.0% (valid paired N of 100 against each
other) — see Table 1 and Figure 1. The paired comparison between RAG and
TOOL, and between RAG and HYBRID, on both best-line and best-odds accuracy
provides evidence of a difference (exact McNemar, Holm-adjusted p=0.005859,
raw p=0.001953; observed difference -11.1 percentage points in both cases).
The paired comparison between TOOL and HYBRID on both metrics was not
statistically distinguishable at alpha=.05 (Holm-adjusted p=1.0); the two
architectures agreed on every one of the 100 paired observations
(zero discordant pairs).

EV classification accuracy was N/A for all three architectures and every
pairwise comparison (Figure 2 shows this directly as labeled "N/A" bars,
not a fabricated zero). Every successful observation across all 330 landed
in a "quant insufficient data" status: a valid best line was identified,
but no observation gathered enough two-sided market data (both outcomes of
a market, from enough sportsbooks) to compute a full EV verdict. This
pattern held for RAG, TOOL, and HYBRID alike; it is reported as an observed
dataset characteristic (discussed in Section 5), not corrected, imputed, or
hidden.

### Consistency

All three architectures showed mean, median, minimum, and maximum
consistency of 1.000 (standard deviation 0.000) across the 11 scenarios
with repeated observations (Figure 4). The Friedman omnibus test on the
per-scenario matched consistency values was correctly degenerate — every
matched block was a three-way tie across architectures, so there was no
rank variation to test (statistic=0.0, p=1.0). Because consistency measures
the reproducibility of an architecture's research-relevant output signature
across repetitions rather than its correctness, a scenario on which an
architecture fails identically on every repetition also counts as
"consistent" under this definition.

### Freshness

All three architectures answered correctly on 10 of 10 observations of the
single freshness-designated scenario in this benchmark (100.0%, 95% Wilson
CI [72.2%, 100.0%] in each case; Figure 3). Because no architecture made a
freshness error, every pairwise McNemar comparison had zero discordant
pairs (Holm-adjusted p=1.0 in each case) — not statistically distinguishable
by construction.

### Secondary Metrics

RAG's median completeness was 0.75, compared to 1.00 for both TOOL and
HYBRID; the RAG-TOOL and RAG-HYBRID comparisons (Wilcoxon signed-rank) both
show evidence of a difference (Holm-adjusted p=9.223e-11, raw p=4.611e-11,
paired N=110); TOOL and HYBRID had identical medians and zero paired
differences (no test performed). Unsupported-claim (hallucination) rate was
0.0% for all three architectures (Figure 5) — zero unsupported claims out
of 90 (RAG), 100 (TOOL), and 100 (HYBRID) verifiable claims respectively,
and zero observations with at least one unsupported claim for any
architecture.

Median total latency was 16.159s for RAG, 6.952s for TOOL, and 22.696s for
HYBRID (Figure 6; Table 1). All three pairwise Wilcoxon comparisons were
statistically significant (RAG-TOOL Holm-adjusted p=3.844e-08; RAG-HYBRID
Holm-adjusted p=7.021e-19; TOOL-HYBRID Holm-adjusted p=5.345e-19; all
paired N=110), and the Friedman omnibus across all three was likewise
significant (statistic=170.75, p=8.377e-38).

### Failure Patterns

Of 330 total observations, 40 (12.1%) were recorded as failures; none were
dropped from the dataset (Table 4). RAG had 20 failures
(`insufficient_retrieved_evidence`, on scenarios S012 and S013 — the spread
and total market scenarios). TOOL had 10 failures (`llm_output_invalid`, on
scenario S012) and HYBRID had 10 failures (`insufficient_current_data`,
also on scenario S012). All 20 of the TOOL and HYBRID failures on S012 are
attributable to a client-side 180-second inference read-timeout on that
scenario's tool-calling turns, an infrastructure characteristic of this
local model/scenario pairing rather than a reasoning failure.

### Hybrid Conflict Resolution

Across all 110 hybrid observations, the two information channels agreed
310 times and never disagreed (zero source conflicts; Table 5). Because no
conflicts occurred, conflict-resolution accuracy is correctly reported as
N/A rather than a fabricated value. Tool-only recoveries (sportsbook price
data available only from the tool channel) occurred 80 times; zero stale
RAG-only prices were incorrectly promoted to authoritative status, and zero
source-reconciliation failures occurred.

## 5. Discussion

The clearest architecture effect in this dataset is that RAG's best-line
and best-odds accuracy (88.9%) and completeness (median 0.75) were both
significantly lower than TOOL's and HYBRID's (100.0% and 1.00
respectively), while TOOL and HYBRID were statistically indistinguishable
from each other on both metrics. A plausible mechanism is that the
canonical query used across all three architectures was deliberately kept
architecture-neutral (it names no sportsbooks and includes no
per-architecture tuning), and semantic retrieval at a fixed top_k does not
guarantee that both sides of a two-sided market, across enough
sportsbooks, will be retrieved — whereas the tool-calling loop can request
exactly the structured records it needs by name. This is consistent with,
but not proven by, RAG's 20 `insufficient_retrieved_evidence` failures on
the spread and total scenarios (S012, S013), which are market types with
different underlying document coverage in the controlled corpus than the
moneyline-heavy majority of scenarios.

A second, cross-architecture finding is that zero of the 330 observations
reached a full expected-value verdict. Because this pattern is present for
RAG, TOOL, and HYBRID alike, it more plausibly reflects a
model-capability characteristic of `llama3.1:8b` on this specific
multi-step task (gathering both sides of a market before a value
comparison can be computed) than an architecture-design effect — the same
tool schema, system prompts, and quantitative engine are used regardless of
which local model is configured. We did not modify the model, prompts, or
tool schemas after observing this pattern; it is reported as an honest
dataset characteristic and a candidate hypothesis for future work with a
different model.

TOOL-calling and HYBRID's structural guarantee that a final answer can only
contain values actually returned by a validated tool call (never
free-form model text) is consistent with the zero unsupported claims
observed for all three architectures, including RAG (whose provenance
validation serves an analogous role for retrieved evidence).

The hybrid architecture's deterministic conflict-resolution mechanism
(current tool data always overriding a disagreeing RAG-derived price) is
correctly implemented and independently verified, but this dataset never
exercised it in a live disagreement: the RAG-side extraction did not retain
a validated price that disagreed with the tool-derived current price in any
of the 110 hybrid observations. Observing the mechanism in action would
require a scenario design in which the local model reliably extracts a
disagreeing, provenance-valid stale price from the RAG corpus — something
this study's canonical, architecture-neutral query and this specific local
model did not reliably produce.

Finally, HYBRID's latency (median 22.696s) reflects the combined cost of
running both the RAG and tool-calling pipelines; it was the slowest
architecture and offered no measured accuracy advantage over TOOL alone in
this dataset. Whether that additional latency is worthwhile in a real
deployment depends on whether the deployment specifically needs RAG's
contextual/historical coverage in addition to TOOL's current-data
reliability — a question this controlled, synthetic-benchmark study is not
positioned to answer on its own.

We emphasize that these are observations from one controlled study, using
one local LLM, one embedding model, and 11 scenarios with 10 repetitions
each; we do not claim these findings generalize to other models, other
benchmarks, or production betting systems (see Limitations).

## 6. Limitations

- Controlled/synthetic sportsbook benchmark, not live market data.
- Limited scenario count (11) and finite repetition count (10 per
  architecture/scenario) — statistical power is limited, especially for
  subgroup analyses.
- Results are specific to the frozen local Ollama model (`llama3.1:8b`) and
  may not generalize to other local or frontier-hosted models.
- Results are specific to the selected embedding model
  (`sentence-transformers/all-MiniLM-L6-v2`).
- No live sportsbook API validation was performed.
- The market-implied (no-vig, leave-one-out consensus) reference
  probability is not the true win probability of the sporting outcome.
- No independent predictive sports ML model was used or compared against.
- Local LLM behavior may differ substantially from frontier hosted models.
- This experiment does not establish real-world betting profitability.
- Local-model capability limitations observed directly in this dataset:
  the model frequently did not gather two-sided market data (0 of 330
  observations reached a full EV verdict), and TOOL's/HYBRID's
  tool-calling turns sometimes exceeded the inference client's 180-second
  timeout on scenario S012 (10 TOOL + 10 HYBRID failures, all
  timeout-attributable, not a reasoning failure).

## 7. Future Work

- **Live sportsbook providers.** Implementing a live-data `OddsProvider`
  behind the project's existing provider abstraction, requiring no change
  to the tools, agents, or quantitative engine layered above it.
- **Broader scenario coverage.** Extending beyond the 11-scenario
  representative subset used here to a larger and more varied set of
  controlled scenarios.
- **Alternative LLMs.** Repeating this experiment with other local models
  and with frontier-hosted models, using the project's existing
  configuration-driven provider selection, to test whether this study's
  findings — in particular the pervasive lack of a full EV verdict and the
  S012 timeout failures — are specific to `llama3.1:8b`.
- **Independent predictive ML model.** Future work could introduce an
  independently trained and calibrated predictive model and compare its
  win-probability estimates against the existing market-implied no-vig
  consensus, to test value signals that are independent of the
  market-derived reference probability — without changing the core
  architecture-comparison question studied here. This extension was **not**
  implemented as part of this project.

## 8. Conclusion

This controlled study finds no single agent architecture that dominates
across accuracy, consistency, and freshness. Tool-calling and hybrid
architectures achieved significantly higher best-line/best-odds accuracy
and completeness than RAG-only, but were statistically indistinguishable
from each other on every binary accuracy metric measured; the hybrid
architecture was significantly slower than tool-calling alone with no
measured accuracy benefit in this dataset. Consistency and freshness showed
no statistically distinguishable difference between any pair of
architectures, because no architecture in this study produced inconsistent
or stale-affected output on the scenarios evaluated. Directly answering the
research question: architecture affects accuracy and completeness (favoring
tool-calling and hybrid over RAG-only) and latency (favoring tool-calling
over hybrid), but did not produce a measurable difference in consistency or
freshness in this controlled experiment — so the answer is architecture-
and metric-dependent, not a single ranking.

## References

All entries below were located and verified against their original
abstract/publication page (arXiv, NeurIPS proceedings, or journal page) via
web search; none is cited from memory alone. See `docs/CITATION_NEEDS.md`
for the mapping from each manuscript claim to its source.

- Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q.,
  Peng, W., Feng, X., Qin, B., & Liu, T. (2023). *A Survey on Hallucination
  in Large Language Models: Principles, Taxonomy, Challenges, and Open
  Questions*. arXiv:2311.05232.
- Levitt, S. D. (2004). Why Are Gambling Markets Organised So Differently
  from Financial Markets? *The Economic Journal*, 114(495), 223–246.
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
  Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D.
  (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks. *Advances in Neural Information Processing Systems (NeurIPS) 33*.
  arXiv:2005.11401.
- Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M.,
  Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). *Toolformer:
  Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761.
- Shin, H. S. (1993). Measuring the Incidence of Insider Trading in a
  Market for State-Contingent Claims. *The Economic Journal*, 103(420),
  1141–1153.
- Singh, A., Ehtesham, A., Kumar, S., Khoei, T. T., & Vasilakos, A. V.
  (2025). *Agentic Retrieval-Augmented Generation: A Survey on Agentic
  RAG*. arXiv:2501.09136.
- Štrumbelj, E. (2014). On Determining Probability Forecasts from Betting
  Odds. *International Journal of Forecasting*, 30(4), 934–943.
- Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z.,
  Tang, J., Chen, X., Lin, Y., Zhao, W. X., Wei, Z., & Wen, J. (2023).
  *A Survey on Large Language Model based Autonomous Agents*.
  arXiv:2308.11432.
- Wolfers, J., & Zitzewitz, E. (2004). Prediction Markets. *Journal of
  Economic Perspectives*, 18(2), 107–126.
