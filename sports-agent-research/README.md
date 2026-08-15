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

**Design / foundation stage.** The repository structure and experimental design have been established. No calculations, tools, RAG, agents, or LLM integrations have been implemented yet.

## Milestone Roadmap

| Milestone | Description | Status |
|---|---|---|
| 1 | Project initialization and experimental design | In progress |
| 2 | Core Pydantic data models | Not started |
| 3 | Deterministic odds and EV calculations | Not started |
| 4 | Controlled sportsbook dataset and ground truth | Not started |
| 5 | Sportsbook lookup tools | Not started |
| 6 | RAG ingestion and retrieval | Not started |
| 7 | RAG-only agent | Not started |
| 8 | Tool-calling-only agent | Not started |
| 9 | Hybrid agent | Not started |
| 10 | Evaluation metrics | Not started |
| 11 | Experiment runner | Not started |
| 12 | Full experiment | Not started |
| 13 | Statistical analysis and visualizations | Not started |

## Python Version

Recommended: **Python 3.11+** (developed/tested against Python 3.13).

## Project Structure

```text
sports-agent-research/
├── docs/               research design and documentation
├── data/               controlled datasets and RAG source documents
├── src/
│   ├── calculations/   deterministic odds/EV math (future)
│   ├── tools/          sportsbook lookup tools (future)
│   ├── rag/            RAG ingestion and retrieval (future)
│   ├── agents/         RAG-only, tool-only, and hybrid agents (future)
│   └── evaluation/     evaluation metrics (future)
├── experiments/        experiment runners (future)
├── results/            experiment output (gitignored)
└── tests/              test suite
```
