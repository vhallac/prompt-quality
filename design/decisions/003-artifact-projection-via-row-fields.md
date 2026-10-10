# 003 — Artifact projection via explicit ROW_FIELDS

**Status:** accepted

## Context
The source `fault-pipeline.jsonl` carries fields beyond what downstream stages need — including `prompt_preview` (a degraded 120-character substitute) and internal pipeline fields. Passing raw source records into the emitted artifacts would silently carry `prompt_preview` alongside full prompts and let internal fields leak into the permanent dataset contract.

## Decision
**Emitted artifacts are an explicit projection, not a raw passthrough.** A pinned `ROW_FIELDS` tuple defines the exact set of keys written to `dataset/rounds-labeled.jsonl` and `dataset/prompt-corpus.jsonl`. `row_for` builds rows from this tuple and asserts the key set, so adding a field requires a deliberate change to `ROW_FIELDS` rather than silent passthrough.

The projection carries: `id`, `prompt`, `prompt_status`, `bin` (gold only), `label` (gold only, null in corpus), `origin_id`, `origin_source`, `timestamp`, `s0_f`, `s0_filter_pass`, `s1_fault_type`, `s1_prob`, `s2_verdict`, `s2_fault_type`, `s2_root_cause`, `s2_evidence`.

## Consequences
- `prompt_preview` is structurally excluded from both artifacts — verified by acceptance tests against the written bytes, not just the in-memory rows.
- The projection is the artifact contract; downstream stages depend on these fields and nothing else.
- Adding a field is a deliberate, reviewable change to the pinned tuple.
- `raw` passthrough is never the default; a consumer that needs an unpinned field must be explicitly added to the projection.