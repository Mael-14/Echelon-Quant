# ADR-004: No directional model at intraday horizons

**Status:** Accepted
**Date:** 2026-09-20
**Deciders:** Project owner and implementation agent
**Supersedes:** the momentum-model gate of ADR-001 and the ICT Q-table

## Context

Five learning systems were built and all five landed on the base rate:

| model | features | result |
|---|---|---|
| GRU 6→16 | 6 indicators | 0.5149 vs 0.517 baseline |
| GRU 6→32, 18x data | 6 indicators | 0.504 vs 0.508 |
| Logistic, triple-barrier label | 6 indicators | Brier 0.23928 vs 0.23909 |
| Q-table, 34 configurations | ICT | best +0.032R at t = 1.01 |
| Decision tree | ICT, continuous | +0.202R at t = 3.56 |

Each failure was investigated on its own terms — too little data, too little
capacity, the wrong label, the wrong state cardinality, the wrong session,
the wrong execution frame. Each fix was reasonable and none of them worked.

The question nobody had asked was whether there was anything to find.

## The measurement

For a driftless random walk with absorbing barriers at `-1` and `+b`, the
probability of touching `+b` first is exactly `1/(1+b)`. Measured on M15
majors across five symbols:

| reward `b` | `1/(1+b)` | measured | deviation |
|---|---|---|---|
| 0.5 | 0.6667 | 0.6566 | −0.010 |
| 1.0 | 0.5000 | 0.4965 | −0.004 |
| 1.5 | 0.4000 | 0.3947 | −0.005 |
| 2.0 | 0.3333 | 0.3287 | −0.005 |
| 3.0 | 0.2500 | 0.2528 | +0.003 |

**The series is a martingale at this horizon.** The base rate every model
converged to is the no-information answer, and the models were not failing
to find signal — they were correctly reporting its absence.

The consequence is algebraic rather than empirical. With `p = 1/(1+b)`:

```
EV = p·b − (1−p) − cost = 0 − cost = −cost
```

**Expectancy is minus the cost at every reward ratio.** No barrier geometry
helps. That also settles a proposal that was on the table — anchoring stops
at structural invalidation instead of an ATR multiple cannot create
expectancy, because rearranging barriers in a martingale changes nothing.
Worse, structural stops measured 0.55x as wide as the 1.5x ATR stop, which
nearly doubles cost in R.

## Decision

**No directional model runs in the gate chain.** The momentum GRU, the
logistic win model and the ICT Q-table are deleted rather than demoted, with
their trainers, artifacts and tests.

Demotion was the previous answer and it is not good enough. A demoted model
still loads, still logs, still invites someone to retrain it, and still
carried a defect that would have blocked every trade had it ever been
promoted (the pipeline computed an 8-field state key against a table trained
on 5-field keys, so every lookup missed and returned `hold`).

`strategy/ict.py` survives. The pattern definitions are correct and tested,
and the failure was not in the features. `strategy/expectancy.py` survives
because it holds the one result that generalises.

## What the cost model implies instead

Cost in R is `spread / stop_distance`. The spread is fixed in price; ATR
grows with the square root of time. **Cost in R therefore falls as
1/√time**, and that is arithmetic rather than a hypothesis:

| frame | drag | breakeven win rate | history available |
|---|---|---|---|
| M5 | 0.095R | 58.6% | 89 days |
| M15 | 0.045R | 41.8% | 364 days |
| H4 | 0.017R | 40.7% | 20 years |
| D1 | 0.007R | 40.3% | 55 years |

At M5 a strategy must clear an 18-point handicap over the martingale rate.
At H4 it must clear 0.7 points.

This is the whole finding. The project spent its effort trying to out-predict
a random walk by 18 points, when moving the execution frame removes 17 of
them. The technique book already measured `trendline_continuation` at
+0.209R — an edge that was never collectable at M5 and is at H4.

The sample-size problem disappears at the same time: 20 years of H4 against
89 days of M5.

## Consequences

- **The gate chain loses a step.** It had no effect — the artifact was
  demoted — and the step was broken. Nothing regresses.
- **`REQUIRE_MOMENTUM_MODEL` and `REQUIRE_ICT_MODEL` are gone**, with their
  paths, from config, `.env.example` and `render.yaml`.
- **No artifact is committed.** `artifacts/` is empty and `.gitignore` no
  longer needs its exemption.
- **`data/` is untracked.** 26MB of regenerable broker history was committed
  by a `git add -A` in 7488c74. It is out of the index; **the blobs remain in
  history** and removing them needs a rewrite, which is a separate decision.
- **The martingale result is horizon-specific, not universal.** It says
  nothing about R_75, whose ATR is an order of magnitude larger in
  percentage terms and which was the only instrument that ever won on Deriv.
  It says nothing about H4 or D1 either, where the measured deviation turns
  slightly positive (+0.0086 at D1) — though not significantly so.

## Before building another model

Run the null test first. `p` against `1/(1+b)` at three reward ratios is
twenty lines, and it tells you the maximum edge available before anyone
spends a week trying to capture it. Had it been run on day one, four of the
five systems above would never have been written.
