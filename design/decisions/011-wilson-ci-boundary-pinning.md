# 011 — Wilson CI boundary pinning

**Status:** accepted

## Context
The C2 run report includes a Wilson 95% confidence interval for the corpus fault rate. The standard Wilson formula produces values slightly outside [0, 1] at the boundaries due to floating-point drift: at `k = n`, `ci_high` can be `0.9999999999999999`; at `k = 0`, `ci_low` can be a few ulps above zero. These are visually confusing and semantically wrong — a CI for a proportion is always in [0, 1].

## Decision
**Wilson CI endpoints are pinned to [0, 1] at the boundaries.** `wilson_interval` returns exactly `(0, ...)` when `k = 0` and `(..., 1)` when `k = n`. The pinning is applied after the standard Wilson calculation, before rounding.

The formula itself is direct implementation of the Wilson score interval without continuity correction, chosen over bootstrapping for the point estimate (bootstrap is used for AUC CIs elsewhere — see 012).

## Consequences
- Report output is always in the valid range; no `ci_high = 1 - 1e-16` in the printed report.
- The pinning is a cosmetic boundary fix, not a statistical adjustment; intervals at `k = 0` or `k = n` are degenerate by definition.
- The Wilson interval is also used for C3's baseline base-rate reporting, where the same pinning applies.