# Design decisions

One indexed line per decision. Newest at the top.

## Source resolution

- 015 — Source path resolution: sibling checkout default + env override — accepted

## Training methodology

- 014 — Gold exclusion from weak-label training — accepted

## Baseline methodology

- 013 — JEV re-score state derivation: three variants — accepted
- 012 — Pure-Python analytics (no numpy/scipy) — accepted
- 011 — Wilson CI boundary pinning — accepted

## Adjudication resilience

- 010 — Resume keyed on union of manifest and existing results — accepted
- 009 — Frame arithmetic — accepted
- 008 — Disk cache pattern for LLM calls — accepted
- 007 — Null LLM completion as transient failure → skip+backfill — accepted
- 006 — S2 reuse: import, don't shell out — accepted

## Data pipeline & fault attribution

- 005 — Origin derivation from S2 evidence text, not parentId — accepted
- 004 — Determinism: byte-identical re-runs — accepted
- 003 — Artifact projection via explicit ROW_FIELDS — accepted
- 002 — Prompt resolution model — accepted
- 001 — Label derivation authority — accepted