# Testing Rules

- Run a baseline test suite before modifying completed functionality.
- Run the full test suite after implementation.
- A milestone cannot pass with failing tests.
- Do not delete, weaken, or skip legitimate tests merely to obtain green
  status.
- Add regression tests for discovered defects.
- Use `pytest.approx` where appropriate for floating-point calculations.
- Record actual test counts — never invent or reuse an old number.
- Manually verify critical deterministic examples where useful (e.g. hand
  calculations for odds/EV, printed before/after comparisons for
  freshness or reproducibility checks).
