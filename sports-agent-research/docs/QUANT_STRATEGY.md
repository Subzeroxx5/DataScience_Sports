# Quant Strategy

This file documents the quantitative calculation engine. All layers are
implemented as of Milestone 7B: `src/calculations/odds_math.py`
(Milestone 3), `src/calculations/market.py` (Milestone 7A), and the
ground-truth integration in `src/evaluation/quant_ground_truth.py`
(Milestone 7B), which writes `data/quant_ground_truth.json` — kept
entirely separate from the Milestone 4 controlled-reference
`data/ground_truth.json` (unmodified; see "Value Analysis" below).

## Existing Calculations — Odds Math (Milestone 3)

Implemented in `src/calculations/odds_math.py`:

- American odds → implied probability
- American odds → decimal odds
- profit calculation
- expected value
- positive/non-positive EV classification
- American-odds comparison (never by absolute value — see
  `compare_american_odds`)

These are reused, not duplicated, by every layer above them (ground truth
generation, `SportsbookTools.find_best_line`, and `market.py` below).

## Existing Calculations — Market Math (Milestone 7A)

Implemented in `src/calculations/market.py`, built on top of
`odds_math.py`'s `implied_probability` (never reimplemented):

- `calculate_overround(odds)` — sum of raw implied probabilities across a
  mutually exclusive market
- `remove_vig_from_probabilities(raw_probabilities)` /
  `calculate_no_vig_probabilities(odds)` — fair/no-vig probability
  normalization (see below)
- `calculate_market_consensus(probabilities)` — unweighted arithmetic
  mean across sportsbooks
- `calculate_leave_one_out_consensus(sportsbook_probabilities,
  excluded_sportsbook)` — consensus excluding the target sportsbook,
  requiring `MIN_CONSENSUS_BOOKS = 2` remaining comparison books
- `calculate_probability_edge(market_reference_probability,
  book_implied_probability)` — raw decimal edge, never pre-formatted as
  a percentage
- `calculate_market_dispersion(probabilities)` → `MarketDispersion`
  (mean, median, population standard deviation, range, book count)
- `calculate_signed_distance_from_consensus` /
  `calculate_absolute_distance_from_consensus`

EV integration reuses `odds_math.expected_value` directly — `market.py`
has no EV function of its own (see `test_market_module_does_not_reimplement_ev`
in `tests/test_market_quant.py`).

## No-Vig Calculation (implemented)

For a two-outcome market:

```text
raw_A = implied probability of outcome A
raw_B = implied probability of outcome B

overround = raw_A + raw_B

fair_A = raw_A / overround
fair_B = raw_B / overround
```

`fair_A + fair_B ≈ 1`.

## Market Consensus (implemented)

Calculate no-vig probabilities separately for multiple sportsbooks, then
combine them with an unweighted arithmetic mean
(`calculate_market_consensus`). No sportsbook/liquidity/"sharp book"
weighting or predictive modeling — deferred as out of scope for the
current research question.

## Leave-One-Sportsbook-Out Consensus (implemented)

When evaluating sportsbook X, exclude sportsbook X from the market
consensus used to evaluate its own value.

Example — evaluating FanDuel:

```text
Consensus =
mean(
    DraftKings fair probability,
    BetMGM fair probability,
    Caesars fair probability
)
```

**Reason:** reduces circularity — a sportsbook should not be used to
validate itself. Enforced by `MIN_CONSENSUS_BOOKS = 2`: at least 2 other
sportsbooks must remain after excluding the target, or
`calculate_leave_one_out_consensus` raises explicitly.

## Value Analysis (implemented — Milestone 7B)

For a given sportsbook and outcome, recorded in
`SportsbookValueGroundTruth` (`src/models.py`), one entry per
quant-evaluable target sportsbook in `data/quant_ground_truth.json`:

```text
book_implied_probability      -- odds_math.implied_probability
market_reference_probability  -- market.calculate_leave_one_out_consensus
probability_edge              -- market.calculate_probability_edge
expected_value                -- odds_math.expected_value (reused, not reimplemented)
positive_ev                   -- odds_math.is_positive_ev (reused, not reimplemented)
```

A market is quant-evaluable only if it's a two-outcome moneyline market
with at least 3 sportsbooks (`MIN_CONSENSUS_BOOKS + 1`) quoting current
prices on both sides. As of Milestone 7B, 4 of the 14 controlled
scenarios qualify (the two-sided markets from Milestone 6A); the other
10 are recorded with `quant_evaluable=False` and an explicit
`ineligibility_reason` — never fabricated, never silently omitted.

This is entirely separate from — and does not replace — the Milestone 4
controlled-reference `data/ground_truth.json`
(`estimated_true_probability`-based). `ReferenceProbabilityMode`
(`src/models.py`) distinguishes the two: `CONTROLLED_REFERENCE` vs.
`MARKET_CONSENSUS`.

## Market Dispersion (implemented)

Metrics across sportsbooks for the same outcome, via
`calculate_market_dispersion` → `MarketDispersion`:

```text
mean
median
population standard deviation
range
distance from consensus (signed and absolute variants)
```

## Critical Framing

> Market consensus is a market-implied reference probability and must
> not be described as the true probability of the sporting outcome.

This mirrors the existing framing of `estimated_true_probability` in the
controlled benchmark (see `data/README.md`): it is an experimental input
used to compute deterministic ground truth, not a claim about real-world
probability. Market consensus, once implemented, will be an additional,
separate derived reference — also not a claim of objective truth.
