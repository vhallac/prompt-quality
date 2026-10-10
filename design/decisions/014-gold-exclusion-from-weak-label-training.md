# 014 — Gold exclusion from weak-label training

**Status:** accepted

## Context
The trained scorer stage (C4, E3) trains on the weak-label corpus (`dataset/prompt-corpus.jsonl`, 5430 rows) and validates on the gold set (`dataset/rounds-labeled.jsonl`, 242 rows). As emitted by C1, the corpus contains **all 242 gold ids** (`corpus ∩ gold = 242`). If C4 trains on the full corpus without filtering, the validation rows are inside the training set — textbook train/test leakage.

The C1 corpus contract is "the pinned 5430 scan ids" — correct as emitted. The exclusion belongs to C4, not C1. C2's issue explicitly states the exclusion; C4's issue (as originally written) did not.

## Decision
**C4 must exclude the 242 gold ids from its weak-label training frame.** The training frame is the same 5159 eligible frame that C2 uses — re-derived from `dataset/base-rate-sample.json`'s recorded `missing_ids` ledger, not from a round-store scan (decision 009); a live scan today no longer returns 29 absent ids. C4's issue was amended to state this explicitly:

- Training on `prompt-corpus` weak labels **minus the adjudicated gold ids**.
- Training and validation sets MUST be disjoint.
- The run report MUST name the training-frame size and excluded-gold count.

C1 does not filter the corpus — that would break C2's frame arithmetic and the corpus contract. The exclusion is applied at consumption time by C4.

## Consequences
- Train/test leakage is prevented at the consumer, not the producer — C1's artifact contract stays simple.
- C4's issue now carries the exclusion as an explicit acceptance criterion, so a C4 implementation that forgets to filter will fail its own acceptance tests.
- C2 (which also excludes gold) already defines the eligible-frame arithmetic; C4 reuses the same formula.