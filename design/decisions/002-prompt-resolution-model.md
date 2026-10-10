# 002 — Prompt resolution model

**Status:** accepted

## Context
Prompt text must be resolved from the semblr round store (`~/.pi/agent/semblr/rounds/<id>.json`)
at dataset-build time. As at the C1 build, 29 of the 5430 scan ids had no round file, 1 id had an
empty `userPrompt` field, and some files may be malformed. The round record also carries a
`prompt_preview` field (first 120 characters), which is cheap to fall back to but is a degraded
input — a preview is not the full prompt, and downstream consumers must not silently train or
score on truncated text.

Those store-time counts are an observation of a store that keeps moving: the same sweep that
decision `009` records recovered 15 of the 29 missing rounds after C1 was built. Re-deriving the
ledger from the live store therefore re-draws the corpus.

## Decision
**Prompt resolution is binary: resolved or unresolved.** A row is `prompt_status: unresolved` when
the round file is absent, unreadable, malformed, lacks `userPrompt`, or that field is not a
non-empty string. `prompt_preview` is **never substituted** for a missing prompt. Resolution walks
past per-id misses and never drops a row — unresolved rows stay in the artifact so downstream
stages can decide what to do with them.

**The ledger is recorded, not observed.** The unresolved ids of the build that produced the
committed artifacts are the `corpus_unresolved` ledger of
`dataset/c1-build-snapshot.json`, together with its store-time classification
(`no_round_file_ids`, `empty_prompt_ids`) — the classification no artifact can show, since both
kinds of row carry a null prompt. A pinned build forces that ledger (`resolve_prompts(…,
force_unresolved=…)`): ledger ids are emitted unresolved without consulting the store, so a round
file that reappeared upstream cannot rewrite them. The ledger never fabricates text — an id
outside it still goes to the store, and a store miss there is still recorded unresolved. Leaving
the ledger out (`--redraw`, or no snapshot pinned) is the fresh-draw path, which records what it
saw.

**The live store is a report, graded by what it invalidates.** `check_store_drift` names the
snapshot fields a live scan would have moved and lists recovered ids; the rows that came back
widen only a *hypothetical* corpus, so they are disclosed, not acted on. A row recorded
`resolved` whose round file is gone, or whose stored text no longer redacts to the committed
prompt, raises `StoreDriftError`, because the artifact can then no longer be re-derived from its
own inputs. `main` then refuses to write unless the build renders to the exact committed bytes,
naming the divergent row.

## Consequences
- Consumers can trust that `prompt_status: resolved` means the full prompt text is present byte-for-byte.
- Consumers that need full prompts (C3 gold, C5 calibration) filter on `prompt_status`; consumers that don't (C2 frame arithmetic) ignore it.
- The 30 unresolved corpus rows (29 missing round file + 1 empty prompt, `a363b8d1…`) are carried in the artifact, named in the run report, and recorded in the snapshot; they are a fact about the C1 build, not about today's store, which resolves 15 of them again.
- Gold is fully resolved (0 unresolved of 242) as recorded. If that stops being true the C1 build fails twice over: the drift report raises on the lost prompt text, and the byte gate refuses to overwrite the artifact the record describes.
- `dataset/c1-build-snapshot.json` is a committed artifact like the two it describes: a re-run of C1 on a moved store reproduces `rounds-labeled.jsonl` and `prompt-corpus.jsonl` byte-for-byte, or stops naming the field.
- The C1 no-round-file ledger and C2's frame ledger (`missing_ids` in `dataset/base-rate-sample.json`) are the same 29 ids, seen a few hours apart from two stages. The cross-check lives in the test layer (`test_c1_no_round_file_ledger_is_c2_frame_ledger`) so C1's build stays independent of C2's artifact; an edit to either fails naming which one moved.
