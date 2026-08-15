# Sports Agent Research — Project Instructions

**Research question:** How does agent architecture—tool calling,
retrieval-augmented generation (RAG), or a hybrid approach—affect the
accuracy, consistency, and freshness of an AI agent identifying positive
expected value opportunities across multiple sportsbooks?

This is a controlled research prototype, not a production betting tool.

## Working Rules

- Read `docs/PROJECT_STATE.md` before starting milestone work.
- Read `milestones/current.md` before modifying code.
- Preserve completed milestone behavior unless a verified defect exists.
- Implement only the requested/current milestone — do not start the next one.
- Run the full pytest suite before declaring a milestone complete.
- Record a baseline test result before making significant changes.
- Never remove or weaken legitimate tests merely to obtain a passing suite.
- Never fabricate sportsbook data.
- Keep deterministic betting/quantitative calculations in Python, never in LLM reasoning.
- Keep the shared quantitative engine identical across RAG, tool-calling, and hybrid architectures.
- Keep the controlled benchmark as the primary experiment.
- Preserve `OddsProvider` as the structured data-source abstraction.
- Prevent ground-truth answers from entering the RAG corpus or agent inputs.
- Do not automatically begin the next milestone.
- Stop for user review after completing the requested milestone.

## Where to Look

- `docs/PROJECT_STATE.md` — current status, test baseline, what exists / doesn't yet.
- `docs/ARCHITECTURE.md` — system layers and data flow.
- `docs/EXPERIMENT_RULES.md` — independent/controlled variables, architecture access boundaries.
- `docs/QUANT_STRATEGY.md` — implemented vs. planned quantitative calculations.
- `milestones/current.md` — the active milestone's objective and scope.
