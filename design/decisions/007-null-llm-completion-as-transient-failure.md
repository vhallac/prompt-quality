# 007 — Null LLM completion as transient failure

**Status:** accepted

## Context
During the first live C2 base-rate run, after 80 successful adjudications, the OpenRouter API returned a response with `choices[0].message.content == null`. The imported `parse_json_loose` crashed on `NoneType.strip()`, killing the entire paid pass. The 80 cached API calls were lost because the cache was persisted only on clean exit.

The live pass costs money and minutes; a single null completion should not abort an entire run.

## Decision
**Null LLM completions are transient failures, not fatal errors.** `adjudicate_one` wraps `TypeError`, `AttributeError`, and `ValueError` from the imported machinery as a typed `AdjudicationUnavailable` exception. `run_adjudication` treats `AdjudicationUnavailable` like a missing round file: drop the id, record it as unresolved, let backfill replace it.

The disk cache is persisted **after every adjudicated row**, not only on clean exit. The append closure calls `save_llm_cache` inside the loop, so a crash mid-run preserves all prior paid calls.

## Consequences
- A transient API hiccup costs at most one row's worth of work; the run continues.
- Unresolved ids from API failures appear in the run report alongside missing-round-file ids, distinguished by reason.
- Per-row cache persistence means a resume after a crash skips all already-adjudicated rows (keyed on union of manifest and existing results — see 010).
- The cache file grows with each adjudication; it is a write-only append pattern (JSON is rewritten whole, but keyed so old entries are stable).