# Experiment Rules

See `docs/ARCHITECTURE.md` for the system layers referenced below and
`docs/research_design.md` for the full experimental design (dependent
variables, threats to validity, mathematics reference).

## Independent Variable

```text
Agent architecture
```

Levels: RAG-only, tool-calling-only, hybrid.

## Controlled Variables

Held constant across architectures wherever practical:

- LLM model
- LLM version
- temperature
- user query
- scenario set
- deterministic ground truth
- structured output schema
- shared quant implementation
- repetition count
- runtime environment

## RAG-Only Boundary

May use:

- RAG retrieval results
- RAG metadata
- the shared deterministic quant engine, after successful extraction from
  retrieved content

May **not** use:

- sportsbook lookup tools
- the structured current-odds provider (`OddsProvider` /
  `ControlledOddsProvider`)

## Tool-Calling-Only Boundary

May use:

- sportsbook tools (`SportsbookTools`)
- structured current data through `OddsProvider`
- the shared quant engine

May **not** use:

- the RAG retriever
- a vector database
- RAG documents

## Hybrid Boundary

May use both RAG and tools.

**Freshness rule:** current structured tool data takes precedence over
stale RAG sportsbook data for current-line analysis.

## Ground Truth

- deterministic (see `src/evaluation/ground_truth.py`)
- architecture-independent
- never exposed as an information source to any architecture

## RAG Leakage Restrictions

RAG documents must never expose:

- the correct/best sportsbook
- expected best odds
- expected EV
- expected EV classification (positive/negative)
- a market-consensus answer
- any deterministic ground-truth label

This is enforced today by `tests/test_rag_corpus.py`'s automated leakage
scan over the controlled corpus, and must continue to be enforced for any
future corpus content.
