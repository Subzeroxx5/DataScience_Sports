# Current Milestone

## Milestone 12 — Controlled Experiment Runner

## Objective

Build the runner that actually executes the controlled experiment: run
each of the three architectures (RAG-only, tool-calling, hybrid) across
the full scenario set, repeated a fixed number of times per scenario,
using the unified evaluation framework (`src/evaluation/metrics.py`,
Milestone 11) to score every run and compute per-architecture
`ArchitectureSummary`/cross-architecture `ArchitectureComparison`
objects — including, for the first time, real `consistency` values
(Milestone 11 only defined and unit-tested that calculation; this
milestone is what actually produces repeated-run data for it to
measure).

See `docs/ARCHITECTURE.md` ("Evaluation") and `docs/EXPERIMENT_RULES.md"`
("Controlled Variables" — repetition count, runtime environment) for how
this fits the overall experimental design.

This file is a placeholder recording the next milestone's identity only;
detailed scope (repetition count, how real-LLM vs. deterministic-mock
runs are selected, output persistence format, resumability/failure
handling across a long-running batch, test requirements) should be
defined when Milestone 12 begins.

## Explicitly Out of Scope (until stated otherwise)

- a dashboard
- statistical significance testing / research conclusions
- changes to the RAG-only agent (`src/agents/rag_agent.py`,
  `src/agents/extraction.py`, Milestone 8B — complete)
- changes to the tool-calling agent (`src/agents/tool_agent.py`,
  `src/agents/tool_schemas.py`, Milestones 9A/9B — complete)
- changes to the hybrid agent (`src/agents/hybrid_agent.py`,
  `src/agents/hybrid_reconciliation.py`, Milestones 10A/10B — complete)
- changes to the unified evaluation framework's metric definitions
  (`src/evaluation/metrics.py`, Milestone 11 — complete) unless a
  verified defect exists
- changes to the shared LLM abstraction (`src/agents/llm_client.py`), the
  provider/tool subsystem's existing behavior, the RAG pipeline, or the
  quant engine's formulas
- exposing ground truth (`data/ground_truth.json`,
  `data/quant_ground_truth.json`) to any agent in any form
