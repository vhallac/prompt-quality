# 001 — Label derivation authority

**Status:** accepted

## Context
The fault pipeline produces `fault-pipeline.jsonl` with S2 verdict bins (`no-fault-within-round`, `prompt-misread`, `ambiguous`, `other`, `stale-context`, `external`, `retrieval-noise`). The dataset build (C1, E1) maps these bins to gold labels: fault bins are positive; `no-fault-within-round` and `external` are negative; `ambiguous` is excluded. Every downstream stage (C2–C6) consumes the labeled artifact `dataset/rounds-labeled.jsonl`.

Without a declared authority, a downstream stage could re-derive labels from raw bin names — silently diverging from the C1 split, breaking comparability between stages, and making the gold set's operating-point semantics untrustworthy.

## Decision
**C1's `label_for_bin` mapping is the single source of truth for gold labels.** No downstream stage re-derives labels from bin names.

- `POSITIVE_BINS`, `NEGATIVE_BINS`, and `EXCLUDED_BINS` are defined once in `build-dataset.py` and reused by import in C2/C3.
- C2's `classify_bin` imports from C1, not from a local copy.
- C3's issue explicitly forbids label re-derivation ("issue decision 1").
- A stage that needs the label rule reads the committed artifact or imports the canonical function; it never inlines a bin→label mapping.

## Consequences
- Gold label drift between stages is structurally impossible — one function, one outcome.
- A bin-vocabulary change flows through one site (C1) and downstream stages pick it up via the artifact or import.
- The 8 `ambiguous` rows are in the artifact but excluded from metrics; consumers must filter on `label`, not on bin name.
- The spec-prose double-count ("205+3 negative") is a documentation artifact, not a code one; the code's negative count is 205 = 202 `no-fault-within-round` + 3 `external`.