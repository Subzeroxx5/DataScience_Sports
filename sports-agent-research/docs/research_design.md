# Research Design

## Research Question

**How does agent architecture—tool calling, retrieval-augmented generation (RAG), or a hybrid approach—affect the accuracy, consistency, and freshness of an AI agent identifying positive expected value opportunities across multiple sportsbooks?**

## Purpose

This is a controlled research prototype comparing three AI agent architectures on their ability to identify positive expected value (+EV) betting opportunities across multiple sportsbooks. It is **not** a production gambling application. All sportsbook data used in experiments will be synthetic and controlled.

---

## Independent Variable

**Agent architecture**, with three levels:

1. **RAG-only** — retrieves information from a vector database of documents.
2. **Tool-calling-only** — calls structured tools/functions to fetch current data and perform calculations.
3. **Hybrid (RAG + tool calling)** — combines both, with tools as the authoritative source for current structured data and RAG for historical/contextual information.

All other experimental factors (model, prompts, scenarios, schema, etc.) are held constant across architectures — see [Experimental Controls](#experimental-controls).

---

## Dependent Variables

### Best-Line Accuracy

Whether the system identifies the sportsbook offering the mathematically best available odds for the specified outcome.

Potential metric:

```text
correct best-line selections / total evaluated scenarios
```

### EV Classification Accuracy

Whether the system correctly classifies an opportunity as positive EV or non-positive EV.

Potential metric:

```text
correct EV classifications / total scenarios
```

### Completeness

Whether the system considers all sportsbooks available in the controlled scenario.

Potential metric:

```text
sportsbooks correctly considered / sportsbooks expected
```

### Hallucination Rate

Whether the system produces sportsbook names, odds, markets, games, or factual claims unsupported by its available data (retrieved documents and/or tool outputs).

### Consistency

Whether repeated executions of the same scenario produce the same key conclusion. Compared across at least:

- selected sportsbook
- selected odds
- positive/negative EV classification

### Freshness

Whether the architecture uses the current version of sportsbook data instead of intentionally stale information (see the RAG staleness condition under [Architecture Boundaries](#architecture-boundaries)).

### Latency

Time required to produce the final structured result.

> Note: None of these metrics are implemented in this milestone. They are defined conceptually only, to be implemented in Milestone 10.

---

## Ground Truth Methodology

Each controlled test scenario will eventually contain:

- scenario ID
- game ID
- teams
- market
- selected outcome
- sportsbook odds
- current/stale status
- estimated true probability
- expected best sportsbook
- expected best odds
- expected implied probability
- expected expected-value calculation
- expected positive-EV classification

**Ground truth must be generated or validated using deterministic Python calculations rather than an LLM.**

### Why deterministic ground truth is essential

The purpose of this study is to measure how *agent architecture* affects accuracy, consistency, and freshness. If ground truth were itself produced by an LLM, any errors, biases, or inconsistencies in that LLM would contaminate the baseline every architecture is measured against — making it impossible to distinguish "the architecture got it wrong" from "the ground truth was wrong to begin with." Deterministic, auditable Python calculations (fixed formulas over fixed inputs) guarantee that ground truth is stable, reproducible, and independent of any model's behavior, so that observed differences between architectures can be attributed to the architectures themselves.

---

## Architecture Boundaries

### RAG-Only Architecture

**Allowed:**
- vector database
- retrieved documents
- metadata associated with retrieved documents
- calculations performed on successfully retrieved information

**Not allowed:**
- structured current sportsbook lookup tools
- direct access to current structured odds dataset
- tool calls that retrieve current sportsbook information

**Important experimental condition:** Some RAG documents will intentionally contain stale odds so that freshness failures can be measured.

### Tool-Calling-Only Architecture

**Allowed:**
- structured sportsbook tools
- current controlled odds dataset through those tools
- deterministic calculation functions

**Not allowed:**
- vector database
- embeddings
- RAG retriever
- RAG documents

### Hybrid Architecture

**Allowed:**
- RAG retrieval
- structured sportsbook tools
- deterministic calculations

**Responsibility separation:**

- **Tools** are used for: current odds, structured sportsbook data, deterministic calculations.
- **RAG** is used for: historical context, unstructured information, contextual supporting information.

**Precedence rule:** If RAG contains stale odds and tools contain newer odds, current tool data takes precedence.

---

## Mathematics

> Documented for reference only. Not implemented until Milestone 3.

### Implied Probability — Positive American Odds

For odds greater than zero:

```text
implied_probability = 100 / (odds + 100)
```

Example: `+150` → `100 / 250 = 0.40`

### Implied Probability — Negative American Odds

For odds below zero:

```text
implied_probability = abs(odds) / (abs(odds) + 100)
```

Example: `-200` → `200 / 300 = 0.6667`

### Profit on a $1 Stake — Positive Odds

```text
profit_if_win = odds / 100
```

Example: `+150` → `$1.50` profit

### Profit on a $1 Stake — Negative Odds

```text
profit_if_win = 100 / abs(odds)
```

Example: `-200` → `$0.50` profit

### Expected Value (assume a $1 stake)

```text
EV = (true_probability × profit_if_win) - ((1 - true_probability) × 1)
```

Example:

```text
Odds: +150
True probability: 0.45

EV = (0.45 × 1.50) - (0.55 × 1)
EV = 0.125
```

`EV > 0` represents positive expected value.

---

## Experimental Controls

To make the comparison across architectures defensible, later experiments should hold the following constant wherever possible:

- same LLM model
- same model version
- same temperature
- same user query
- same test scenarios
- same expected output schema
- same deterministic calculation library
- same ground truth
- same number of repeated runs
- same hardware/environment where possible

The main intentionally changing variable is:

```text
Agent Architecture
```

---

## Threats to Validity

### Synthetic Data
Controlled sportsbook data may not capture all characteristics of real sportsbook APIs.
**Mitigation:** Clearly state that this study measures architecture behavior under controlled conditions, not real-world deployment performance.

### Estimated True Probability
Positive EV requires an estimate of the true probability of an outcome. That estimate itself may be imperfect.
**Mitigation:** Treat the supplied probability as experimental ground truth rather than claiming it represents actual real-world probability.

### LLM Randomness
Responses may differ between runs.
**Mitigation:** Use low temperature and repeated trials.

### Retrieval Quality
Poor retrieval could cause RAG failure independently of reasoning ability.
**Mitigation:** Evaluate retrieval separately before evaluating the complete RAG architecture.

### Prompt Sensitivity
Slight prompt differences could affect architectures differently.
**Mitigation:** Standardize prompts and output schemas as much as possible across architectures.

### Information Asymmetry
RAG-only and tool-only architectures intentionally receive information differently.
**Mitigation:** Clearly define architecture boundaries and interpret results as architecture-level performance rather than claiming every architecture has identical information access.

### Model Dependency
Results using one LLM might not generalize to another.
**Mitigation:** Document the exact model used and list multi-model testing as future work.

### Sample Size
Small test sets may produce unstable conclusions.
**Mitigation:** Use enough controlled scenarios and repeated runs to support descriptive comparison.

---

## Future Implementation Milestones

| Milestone | Description |
|---|---|
| 1 | Project initialization and experimental design |
| 2 | Core Pydantic data models |
| 3 | Deterministic odds and EV calculations |
| 4 | Controlled sportsbook dataset and ground truth |
| 5 | Sportsbook lookup tools |
| 6 | RAG ingestion and retrieval |
| 7 | RAG-only agent |
| 8 | Tool-calling-only agent |
| 9 | Hybrid agent |
| 10 | Evaluation metrics |
| 11 | Experiment runner |
| 12 | Full experiment |
| 13 | Statistical analysis and visualizations |

No milestone beyond Milestone 1 is implemented in this document or repository state.
