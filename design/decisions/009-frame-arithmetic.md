# 009 — Frame arithmetic

**Status:** accepted

## Context
The base-rate estimation (C2) must draw a uniform random sample from the pool of rounds that have *not* been adjudicated by the fault pipeline. The pool is: all 5430 scan ids, minus the 244 deduped fault-pipeline ids (gold), minus the 29 round-store ids that had no round file when C2 was drawn.

The arithmetic looks simple but has a subtlety: the "244 gold" and "29 absent" sets overlap. The two Q2-dropped pipeline ids (`83ebde…`, `91853e…`) are both gold *and* absent-round-file, so subtracting 244 and then 29 removes them twice and lands on 5157. The frame is the union subtraction: `5430 − |gold ∪ absent| = 5430 − 271 = 5159`.

## Decision
**Eligible frame = 5430 scan ids − (244 pipeline ids ∪ 29 absent-round ids) = 5159.**
The frame is a *pinned snapshot*: its missing-round-file term is the
`missing_ids` ledger recorded in `dataset/base-rate-sample.json`, not a fresh
scan of the live round store. Corpus and gold come from the committed artifacts
(`dataset/prompt-corpus.jsonl`, `dataset/rounds-labeled.jsonl`), the absent ids
from the ledger, and `seed` + that snapshot reproduce the identical sample on any
machine — including one with no round store present.

The live store is consulted only by `check_store_drift`, which names the fields
that would differ (`missing_size`, `missing_ids`, `frame_size`) and reports the
pinned and live frame sizes side by side. It never redraws the sample. A store
that lost a round file belonging to the recorded sample raises, because the
snapshot can then no longer honour what it claims to reproduce. `assert_anchors`
keeps the four numbers as hand-maintained anchors of the snapshot, so an edited
ledger fails naming the field it broke rather than silently widening the frame.

The subtraction is a **set union**, so overlap is absorbed rather than excluded:
`gold ∩ missing` is exactly the two Q2-dropped pipeline ids (`83ebde…`,
`91853e…`), which is why the arithmetic lands on 5159 and not 5157. The scan in
`build_manifest` runs over the whole corpus, so those two ids are visible in both
sets and `eligible_frame` subtracts them once.

## Reconciliation (2026-10-10)
The upstream semblr sweep recovered rounds after C2 was drawn, so the live store
no longer matches the ledger: of the 29 recorded absent ids **15 now have a round
file, 14 are still absent, and none went missing that were not already recorded**
(the live store holds 9 301 round files at the time of this measurement, and
keeps growing — re-measure it with `ls ~/.pi/agent/semblr/rounds/*.json | wc -l`).
That moves the store-derived frame
from 5159 to 5174 — the reason the old live-scan anchor could not hold. All 100
sampled ids still have their round files, `gold ∩ missing` is still exactly the
two Q2 ids, and the ledger still reproduces 5159, so the C2 sample stands and was
**not** re-adjudicated (`dataset/base-rate-results.jsonl` untouched). A larger
frame is a future stage with its own recorded seed and its own anchors — never a
mutation of this snapshot. C1's corpus side of the same movement (15 of its 30
recorded unresolved ids have round files again) is pinned by
`dataset/c1-build-snapshot.json` under `design/decisions/002`; the two ledgers
agree on the same 29 no-round-file ids.

## Consequences
- Frame drift (round files appearing or going missing) is disclosed at build
  time by name, and the disclosure cannot be mistaken for a redraw.
- The C2 spec's prose "5430 − 244 − 29 = 5159" is arithmetically 5157; the value 5159 reconciles because the 2 extra pipeline ids are covered by the missing-file term. The code produces the correct value.
- The frame is stored in `base-rate-sample.json` alongside the seed and sample, so downstream consumers (C4) can reproduce it without re-scanning — and *must* use the stored ledger, because the scan no longer returns it.
- `dataset/base-rate-sample.json` is canonical JSON (sorted keys, sorted and
  deduplicated ledger). `main` refuses to write unless the reproduced manifest
  renders to the exact bytes already on disk.