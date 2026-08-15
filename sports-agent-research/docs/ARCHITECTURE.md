# Architecture

See `docs/QUANT_STRATEGY.md` for the planned Shared Quant Engine's
calculations, and `docs/EXPERIMENT_RULES.md` for the access boundaries
between architectures.

## System Diagram

```text
Controlled Benchmark
        │
        ├─────────────────────────────┐
        │                             │
        ↓                             ↓
ControlledOddsProvider          Controlled RAG Corpus
        ↓                             ↓
OddsProvider                 Future Vector Retriever
        ↓                             ↓
Sportsbook Tools              RAG Architecture
        ↓
Tool-Calling Architecture

                 Shared Quant Engine
                        ↓
       ┌────────────────┼────────────────┐
       ↓                ↓                ↓
      RAG             Tools           Hybrid

                        ↓
              Common Structured Output
                        ↓
                Evaluation Framework
```

## Layer Responsibilities

### Models (`src/models.py`)

Data contracts and validation. Every other layer passes data around as
these Pydantic models — `Game`, `Market`, `SportsbookOdds`, `TestScenario`,
`GroundTruth`, `BestLineResult`, `BettingAnalysis`, `SourceReference`, and
related enums (`MarketType`, `ArchitectureType`, `SourceType`).

### Calculations (`src/calculations/odds_math.py`)

Deterministic betting mathematics: implied probability, decimal odds,
profit, expected value, positive-EV classification, American-odds
comparison, best-odds selection. No LLM involvement, ever. This is the
single source of truth for odds math — no other layer reimplements it.

### Providers (`src/providers/`)

Structured sportsbook data access, behind the abstract `OddsProvider`
interface (`base.py`). `ControlledOddsProvider` (`controlled.py`) is the
current implementation, reading the controlled JSON benchmark. A future
live-API provider would implement the same interface with zero change to
anything above it.

### Tools (`src/tools/sportsbook_tools.py`)

Agent-facing structured operations (`SportsbookTools`): `get_games`,
`get_game`, `get_odds`, `get_sportsbook_odds`, `find_best_line`. Depends
only on `OddsProvider`, never on a concrete provider or on raw JSON.
This is the only path a tool-calling or hybrid agent may use to reach
current structured sportsbook data.

### RAG (`src/rag/`)

Controlled documents (`documents.py`: `RagDocument`, `RagSourceType`) and
their deterministic generator (`build_corpus.py`), producing
`data/rag_documents/corpus.jsonl`. Embeddings, vector indexing, and
semantic retrieval do not exist yet — see `milestones/current.md` for
Milestone 6C, which adds them.

### Shared Quant Engine (future)

Not yet implemented (see `docs/QUANT_STRATEGY.md`). Will provide implied
probability, no-vig probability, market consensus, leave-one-out
consensus, expected value, and market dispersion — identical across all
three architectures, so agent architecture remains the only meaningfully
varying factor in the experiment.

### Agents (future)

Architecture-specific orchestration only (RAG-only, tool-calling-only,
hybrid). Agents call into the Shared Quant Engine and the Tools/RAG
layers; they must not reimplement calculations or bypass the access
boundaries in `docs/EXPERIMENT_RULES.md`.

### Evaluation (`src/evaluation/`)

Architecture-independent measurement against deterministic ground truth
(`ground_truth.py`, plus `dataset.py` for loading the benchmark). Ground
truth is generated once, from the benchmark, independent of any
architecture, and is never given to an agent as input.

### Dashboard / UI (future)

Will sit above the experimental system for visualization only. It must
not define or influence experimental logic, quant calculations, or
ground truth.
