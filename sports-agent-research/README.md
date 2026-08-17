# Sports Agent Research

## Research Questions

**How does agent architecture—tool calling, retrieval-augmented generation (RAG), or a hybrid approach—affect the accuracy, consistency, and freshness of an AI agent identifying positive expected value opportunities across multiple sportsbooks?**
Can an LLM-assisted betting tracker reliably identify best available sportsbook lines?
Which architecture performs best: RAG, TOOL, or HYBRID?
Can a local Ollama model, specifically llama3.1:8b, handle the orchestration reliably enough?

## Purpose

This repository is a controlled research prototype for comparing AI agent architectures on their ability to identify positive expected value (+EV) betting opportunities across multiple sportsbooks. It exists to answer a research question about agent design, not to serve as a production or consumer-facing gambling tool. All sportsbook data used in experiments is synthetic and controlled.

## Architectures Compared

1. **RAG-only** — retrieves information from a vector database of documents.
2. **Tool-calling-only** — calls structured tools/functions to fetch current data and perform calculations.
3. **Hybrid (RAG + tool calling)** — tools serve as the authoritative source for current structured data; RAG supplies historical/contextual information.

See [docs/research_design.md](docs/research_design.md) for the full experimental design, including dependent variables, ground truth methodology, architecture boundaries, mathematics, experimental controls, and threats to validity.

## LLM Providers

All three architectures share one provider-agnostic LLM abstraction (`src/agents/llm_client.py`) with two concrete, interchangeable implementations selected by configuration (`ExperimentConfig.llm_provider`), never by architecture:

- **Anthropic** (`AnthropicLLMClient`) — the Anthropic API. Requires `ANTHROPIC_API_KEY`.
- **Ollama** (`OllamaLLMClient`) — a locally hosted model via [Ollama](https://ollama.com), used for the final controlled experiment so it can run without per-call API cost. Requires `ollama serve` (or `brew services start ollama`) running locally and the model already pulled (`ollama pull llama3.1:8b`); no API key needed. The same locally hosted model is held constant across RAG, TOOL, and HYBRID.

Current default local model: `llama3.1:8b`, temperature `0.0` (the lowest deterministic setting Ollama supports — see `DEFAULT_OLLAMA_MODEL`/`DEFAULT_OLLAMA_TEMPERATURE` in `src/agents/llm_client.py`). Verified locally to support both structured/schema-constrained JSON output and native multi-turn tool calling — see `experiments/run_ollama_three_architecture_smoke_test.py` for the manual verification script.

Local inference through `OllamaLLMClient` counts as **REAL** execution, not the project's deterministic MOCK fakes — MOCK mode remains available separately for infrastructure validation regardless of which provider is configured.

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
├── experiments/        manual, credential-gated real-API smoke tests
├── dashboard/          Streamlit research dashboard (Milestone 13)
├── results/            experiment output (gitignored)
└── tests/              test suite
```

## Dashboard

A lightweight Streamlit dashboard (`dashboard/`) sits above the
experimental system for demonstration and inspection — it never defines
its own betting/quant math, ground truth, or agent behavior; it only
calls into the existing agent/experiment/evaluation modules and displays
what they already computed.

**Install:** `pip install -r requirements.txt` (adds `streamlit` and
`pandas` on top of the core dependencies).

**Launch:**

```bash
streamlit run dashboard/app.py
```

**Demo vs. Research Comparison:**

- **Demo** — pick one scenario and one architecture (RAG / TOOL /
  HYBRID) from the controlled scenario manifest, then click **Run
  Analysis** to execute that architecture live (via
  `src.experiments.agent_factory.create_agent`, the same factory the
  experiment runner uses) and inspect its `BettingAnalysis`, sportsbook
  comparison table, and full architecture trace (RAG retrieval, tool
  calls, hybrid source reconciliation/freshness conflicts). MOCK
  execution mode (default) reuses the deterministic fake-LLM policy at
  no API cost; REAL calls the configured LLM provider (Anthropic API or
  a locally hosted Ollama model — see "LLM Providers" below) and fails
  clearly, without crashing, if it is not reachable.
- **Research Comparison** — loads a previously generated experiment
  directory from `results/experiments/<experiment_id>/` (produced by
  `python -m src.experiments.runner`, Milestone 12: `config.json`,
  `manifest.json`, `raw_results.jsonl`, `summary.json`) and visualizes
  architecture summaries, comparison charts, per-scenario drill-down,
  repetition/consistency inspection, failure analysis, hybrid conflict
  analysis, and raw-result access — read-only, never recalculated.

**MOCK results are infrastructure-validation data, not research
conclusions.** Any experiment run in MOCK execution mode is labeled
"MOCK — INFRASTRUCTURE VALIDATION ONLY" wherever its numbers appear in
the dashboard; it must never be read as a finding about which
architecture performs better (see `docs/EXPERIMENT_RULES.md`).
