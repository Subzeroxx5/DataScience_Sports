# Sports Agent Research

## Research Question

**How does agent architecture—tool calling, retrieval-augmented generation (RAG), or a hybrid approach—affect the accuracy, consistency, and freshness of an AI agent identifying positive expected value opportunities across multiple sportsbooks?**

## Purpose

This repository is a controlled research prototype for comparing AI agent architectures on their ability to identify positive expected value (+EV) betting opportunities across multiple sportsbooks. It exists to answer a research question about agent design, not to serve as a production or consumer-facing gambling tool. All sportsbook data used in experiments is synthetic and controlled.

## Architectures Compared

1. **RAG-only** — retrieves information from a vector database of documents.
2. **Tool-calling-only** — calls structured tools/functions to fetch current data and perform calculations.
3. **Hybrid (RAG + tool calling)** — tools serve as the authoritative source for current structured data; RAG supplies historical/contextual information.

See [docs/research_design.md](docs/research_design.md) for the full experimental design, including dependent variables, ground truth methodology, architecture boundaries, mathematics, experimental controls, and threats to validity.

## Current Project Status

See [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) for the authoritative,
up-to-date project state (completed milestones, current test baseline,
what exists vs. what doesn't yet) and [docs/ROADMAP.md](docs/ROADMAP.md)
for the full milestone list. This README is not kept in sync milestone by
milestone — those two files are the source of truth.

## Python Version

Recommended: **Python 3.11+** (developed/tested against Python 3.13).

## Project Structure

```text
sports-agent-research/
├── docs/               research design and documentation
├── data/               controlled datasets and RAG source documents
├── src/
│   ├── calculations/   deterministic odds/EV math
│   ├── providers/      OddsProvider abstraction + controlled JSON provider
│   ├── tools/          sportsbook lookup tools
│   ├── rag/            RAG document schema + controlled corpus (retrieval: future)
│   ├── agents/         RAG-only, tool-only, and hybrid agents (future)
│   └── evaluation/     dataset loaders + deterministic ground truth
├── experiments/        experiment runners (future)
├── results/            experiment output (gitignored)
└── tests/              test suite
```
