# Architecture Boundaries

```text
models
 ↑
calculations

models
 ↑
providers
 ↑
tools
 ↑
agents
```

An upward arrow means "depends on" — e.g. `providers` depends on
`models`, `tools` depends on `providers`. See `docs/ARCHITECTURE.md` for
the full layer diagram and responsibilities.

## Rules

- Providers must not depend on agents.
- Providers must not depend on RAG.
- Tools must not directly read benchmark JSON.
- Tools should use `OddsProvider`, never a concrete provider or raw file
  path.
- The RAG-only architecture must not access structured current
  sportsbook tools.
- The tool-only architecture must not access RAG.
- The hybrid architecture may access both, with current structured data
  taking precedence over stale RAG data (see `docs/EXPERIMENT_RULES.md`).
- Agents orchestrate deterministic components rather than duplicate
  calculations — all quant math lives in the Shared Quant Engine
  (`docs/QUANT_STRATEGY.md`), not inside an agent.
- Future APIs belong behind `OddsProvider`.
- The dashboard/UI remains outside core experimental logic.
