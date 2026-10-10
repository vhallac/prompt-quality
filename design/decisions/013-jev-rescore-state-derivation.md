# 013 — JEV re-score state derivation

**Status:** accepted

## Context
The jev-round-scan.py scorer normally builds state from the round's own response (`build_state`) or, in `--refine` mode, from the parent round's response (`build_refine_state`). The baseline stage (C3) needs three variants to measure context leakage:

1. **prompt-only:** no response context at all — the scorer sees only the prompt.
2. **own-response:** the scorer's normal context (the round's own response) — measures how much the scorer leans on the response it is judging.
3. **parent-response:** the scorer sees the parent round's response — measures the value of prior-round context.

The question wording and clipping logic must mirror the original scanner exactly, or the comparison to weak labels is invalid.

## Decision
**Three state builders, each mirroring the jev-round-scan.py shape.** The builders:

- **prompt-only:** substitutes a marker (`[not shown]`) in the response slot, keeping the state structure identical.
- **own-response:** `build_state` shape — the round's `responseSequence` is passed as a string (tool calls included verbatim, matching the original).
- **parent-response:** `build_refine_state` shape — tool-call segments in the parent response are redacted with a count note (`[2 tool calls redacted]`).

All three clip the final state to `JEV_STATE_CHARS = 12000` characters, matching the original scanner's limit. The questions are copied verbatim from `jev-round-scan.py`. The model is `typesafe/jev-1.13`. The score is `max(correction_p, frustration / 4)` — monotone with the scan's OR-flag rule (`fr >= 3 or corr >= 0.7 ⇒ score >= 0.7`).

"Matching the original" is checked, not asserted: the reference scorer is pinned
in this repository at `scripts/reference/jev-round-scan.py` (a byte-for-byte copy
of semblr's file at `ba10970`), `baseline-metrics.json` records its digest under
`inputs.jev_reference`, and `scripts/baselines.test.py` compares the questions,
model, endpoint, state limit and both builders against that copy.

## Consequences
- The three variants are comparable: only the response context changes; the clipping, questions, model, and scoring rule are identical.
- "Leak delta" (own-response AUC − prompt-only AUC) and "session delta" (parent-response AUC − prompt-only AUC) are reported on exactly the paired rows where both variants have valid scores.
- Parentless rounds and rounds with missing parent files are excluded from the session delta denominator only (per issue #3 extensions 4a/4b).
- The marker `[not shown]` is a deliberate placeholder — not an empty string — so the state structure is preserved and the LLM sees a missing-response signal rather than a truncated prompt.