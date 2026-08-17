# Architecture

See `docs/QUANT_STRATEGY.md` for the Shared Quant Engine's calculations,
`docs/EXPERIMENT_RULES.md` for the access boundaries between architectures,
and `docs/REPRODUCIBILITY.md` for how to run each layer described below.
This document reflects the completed system as of Milestone 15 — every
layer described here is implemented, not planned.

## System Diagram

```text
Controlled Benchmark
        │
        ├──────────────────────────────┐
        │                              │
        ↓                              ↓
OddsProvider / Sportsbook Tools    RAG Corpus
        │                              │
        │                          Retriever
        │                              │
        └───────────────┬──────────────┘
                         │
                Agent Architectures
             RAG-only / Tool-only / Hybrid
                         │
                Shared Quant Engine
                         │
                  BettingAnalysis
                         │
              Unified Evaluation Framework
                         │
              Experiment Runner / Analysis
                         │
                     Dashboard
```

## Layer Responsibilities

### Models (`src/models.py`)

Data contracts and validation. Every other layer passes data around as
these Pydantic models — `Game`, `Market`, `SportsbookOdds`, `TestScenario`,
`GroundTruth`, `QuantGroundTruth`, `BestLineResult`, `BettingAnalysis`,
`SourceReference`, and related enums (`MarketType`, `ArchitectureType`,
`SourceType`, `AnalysisStatus`).

### Calculations (`src/calculations/`)

Deterministic betting mathematics — the single source of truth, never
reimplemented by any other layer:

- `odds_math.py`: implied probability, decimal odds, profit, expected
  value, positive-EV classification, American-odds comparison, best-odds
  selection.
- `market.py`: overround, no-vig (fair) probability normalization, market
  consensus, leave-one-sportsbook-out consensus, probability edge, market
  dispersion.

No LLM involvement anywhere in this layer.

### Providers (`src/providers/`)

Structured sportsbook data access behind the abstract `OddsProvider`
interface (`base.py`). `ControlledOddsProvider` (`controlled.py`) is the
current implementation, reading the controlled JSON benchmark. A future
live-API provider would implement the same interface with zero change to
anything above it (see `docs/FINAL_RESEARCH_SUMMARY.md`, Future Work).

### Tools (`src/tools/sportsbook_tools.py`)

Agent-facing structured operations (`SportsbookTools`): `get_games`,
`get_game`, `get_odds`, `get_sportsbook_odds`, `find_best_line`. Depends
only on `OddsProvider`, never on a concrete provider or on raw JSON. This
is the only path a tool-calling or hybrid agent may use to reach current
structured sportsbook data.

### RAG (`src/rag/`)

- `documents.py` / `build_corpus.py`: controlled document corpus
  (`data/rag_documents/corpus.jsonl`), deterministically generated from the
  benchmark — no ground-truth leakage (enforced by
  `tests/test_rag_corpus.py`).
- `embeddings.py`: local sentence-transformer embeddings
  (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim, normalized).
- `vector_index.py` / `build_index.py`: an exact FAISS inner-product index
  (`data/rag_index/`).
- `retriever.py`: the one semantic retrieval path, preserving retrieval
  rank/score/freshness exactly as returned — stale documents are never
  filtered or "corrected" at this layer.

### Shared Quant Engine

See `docs/QUANT_STRATEGY.md`. Implemented in `src/calculations/`, consumed
identically by all three architectures via
`src/agents/*_agent.py`'s quant pipeline — never reimplemented per
architecture. This is what keeps agent architecture the only meaningfully
varying factor in the controlled experiment.

### Agents (`src/agents/`)

Three concrete architectures, each fixing `Agent.architecture` to one
`ArchitectureType` and enforcing its own access boundary (AST-verified by
tests, see `docs/EXPERIMENT_RULES.md`):

- `rag_agent.py` (`RagOnlyAgent`): RAG evidence -> LLM structured
  extraction -> provenance validation -> shared quant engine ->
  `BettingAnalysis`. No access to `src.tools` / `src.providers`.
- `tool_agent.py` (`ToolCallingAgent`): a bounded multi-turn tool-calling
  loop -> `SportsbookTools` -> shared quant engine -> `BettingAnalysis`.
  No access to `src.rag`. The final output is built exclusively from
  actual tool-call return values, never the LLM's own text.
- `hybrid_agent.py` (`HybridAgent`) + `hybrid_reconciliation.py`: the only
  architecture with access to both channels. Deterministic, LLM-free
  source reconciliation (`reconcile_outcome`) decides per
  (sportsbook, outcome) which source is authoritative — current structured
  tool data always takes precedence over a conflicting RAG-derived price.

`llm_client.py` provides the provider-agnostic `LLMClient` /
`ToolCallingLLMClient` protocols, implemented by both `AnthropicLLMClient`
(Anthropic API) and `OllamaLLMClient` (locally hosted Ollama model) —
provider selection is configuration-driven
(`ExperimentConfig.llm_provider`), never architecture-specific.

### Evaluation (`src/evaluation/`)

Architecture-independent measurement against deterministic ground truth:
`ground_truth.py` / `quant_ground_truth.py` (generation, from the
benchmark only), `dataset.py` (benchmark loading), `metrics.py` (the
unified metric/failure-taxonomy/consistency framework shared by all
per-architecture evaluators), and one evaluator per architecture
(`rag_agent_evaluation.py`, `tool_agent_evaluation.py`,
`hybrid_agent_evaluation.py`). Ground truth is generated once, independent
of any architecture, and is never given to an agent as input — enforced by
AST-based tests.

### Experiments (`src/experiments/`)

The controlled experiment runner: `config.py` (`ExperimentConfig`, the
deterministic scenario manifest, the balanced architecture-rotation
policy, artifact-checksum reproducibility metadata), `agent_factory.py`
(`create_agent`, the one factory every architecture is built through),
`runner.py` (`run_experiment`, the unified execution engine — never a
per-architecture script), `fingerprint.py` / `preflight.py` /
`validation.py` (the final-experiment freeze/preflight/postflight
integrity checks added for the real final experiment).

### Analysis (`src/analysis/`)

Read-only statistical analysis of a frozen experiment directory: paired
architecture alignment, Wilson confidence intervals, exact McNemar,
Wilcoxon signed-rank, Friedman omnibus tests with Holm correction, and
generation of tables/figures/`findings.md` — see
`docs/FINAL_RESEARCH_SUMMARY.md` for the results this layer produced for
the final dataset. Never modifies the raw experiment it analyzes.

### Dashboard (`dashboard/`)

A thin Streamlit UI layer above the experimental system — Demo mode (live
single-scenario execution through `create_agent`) and Research Comparison
mode (read-only visualization of a persisted experiment directory,
including the final dataset). Contains no sportsbook or quant business
logic of its own (verified by an AST-based structural test); MOCK results
are always visibly labeled as infrastructure validation, never presented
as a research finding.
