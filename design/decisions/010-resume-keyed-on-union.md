# 010 — Resume keyed on union of manifest and existing results

**Status:** accepted

## Context
The base-rate adjudication loop (C2) may be interrupted and resumed. A naive resume that only checks the manifest ids would re-adjudicate rows that were already written before the crash, producing duplicates. A naive resume that only checks existing results would miss new rows added to the manifest.

The source `fault-pipeline.jsonl` already carries 4 duplicate ids (from its own resume reruns), so the existing output is not a reliable dedupe signal on its own.

## Decision
**Resume keys on the union of manifest ids and existing result ids.** `resume_plan` takes the manifest's sampled ids and the already-adjudicated ids from the results file, and returns the set difference — only ids in the manifest that are not already in the results. `append_result` additionally guards with an in-process duplicate-id check, so even if the union logic is wrong, a duplicate is never written.

The `backfill_ids` replacement pool also excludes already-adjudicated ids, so replaced rows are drawn from the eligible frame minus (sampled ∪ adjudicated).

## Consequences
- An interrupted run resumes without duplication and without re-adjudication.
- The union logic is testable offline: synthetic manifests and result sets can prove the set-difference without live API calls.
- The `fault-pipeline.jsonl` duplicate precedent is not repeated — the C2 output has exactly one row per id.