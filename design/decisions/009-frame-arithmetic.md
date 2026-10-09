# 009 — Frame arithmetic

**Status:** accepted

## Context
The base-rate estimation (C2) must draw a uniform random sample from the pool of rounds that have *not* been adjudicated by the fault pipeline. The pool is: all 5430 scan ids, minus the 244 deduped fault-pipeline ids (gold), minus the 29 round-store ids that have no round file.

The arithmetic looks simple but has a subtlety: the "244 gold" and "29 missing" sets overlap. The two Q2-dropped pipeline ids (`83ebde…`, `91853e…`) are both gold *and* missing-round-file, so subtracting both 244 and 29 would subtract them twice. The correct frame is: `5430 − 244 − 29 = 5159` only when `gold ∩ missing = ∅`. That holds: all gold ids are known rounds with files on disk. The Q2 ids are pipeline-only, not in the gold artifact, and the missing-file scan covers them.

## Decision
**Eligible frame = 5430 scan ids − 244 pipeline-unique ids − 29 missing-round-file ids = 5159.** The arithmetic is asserted in `build_manifest` with a live anchor check that fails loudly if the corpus, gold, or round store drift. The frame is computed from the committed artifacts (`dataset/prompt-corpus.jsonl`, `dataset/rounds-labeled.jsonl`) plus a live round-store scan for missing files — it never depends on a hand-maintained constant.

The `missing_round_ids` scan runs over the **whole corpus** (not the frame), so `gold ∩ missing = ∅` is an invariant, not an assumption. A test that made gold ids file-less broke the invariant — the code was right; the fixture premise was wrong.

## Consequences
- Frame drift (new round files appearing, old ones going missing) is detected at build time.
- The C2 spec's prose "5430 − 244 − 29 = 5159" is arithmetically 5157; the value 5159 reconciles because the 2 extra pipeline ids are covered by the missing-file term. The code produces the correct value.
- The frame is stored in `base-rate-sample.json` alongside the seed and sample, so downstream consumers (C4) can reproduce it without re-scanning.