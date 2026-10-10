# 016 — Pinned external reference copies

**Status:** accepted

## Context
Three stages made claims that depend on bytes living outside this repository: C3's
parity claim about `jev-round-scan.py`'s state builders and rubric (review F3), C2's
frame arithmetic over the round store, and C1's unresolved-id ledger over the same
store (U1/U2). Two failure modes followed from citing instead of copying. The
reference scorer lived only in the sibling `../semblr/` checkout, so a clean clone
could not check the parity claim at all, and the anchors cited a path that does not
exist here with a line range that was never correct. The store facts were read live,
so when the upstream semblr sweep recovered 15 previously-missing rounds and the store
grew past 9,300 files, the committed artifacts no longer described anything
reproducible — the sample could not be redrawn from its own seed, and the tests failed
against a moving input rather than against a wrong program.

A citation is not a pin: it rots silently, and it cannot be verified where it is read.

## Decision
**When this repository makes a claim about behaviour or about a corpus state that it
does not own, commit the bytes or the facts, and keep the provenance outside the
copy.** Three shapes, all keyed the same way:

1. **Reference implementation** — commit a byte-for-byte copy under
   `scripts/reference/` (`jev-round-scan.py`, from semblr at `ba10970`, sha256
   `4e294839a341c953...`). The copy
   itself carries no metadata; its `sha256`, upstream naming, and the field pointing at
   the rubric digest are published in the artifact it feeds (`baseline-metrics.json` →
   `inputs.jev_reference`). Parity tests import the pinned copy and compare the
   builders, questions, model, endpoint and state limit against it — they never assert
   local string literals, and they never fall back to the sibling checkout.
2. **Corpus snapshot** — commit the data the stage actually scored
   (`dataset/prompt-corpus.jsonl`), so a re-run does not re-read a mutable upstream file.
3. **Store-state record** — commit a sidecar that records the store-time facts the
   artifact was keyed on (`dataset/c1-build-snapshot.json`; C2's manifest inside
   `dataset/base-rate-sample.json`): the id ledger, its classification, the counts, and
   a digest over the id universe. Reproduction reads the record; the live store becomes
   a **drift check** that names the divergent field and fails loudly, and a stage
   refuses any flag that would redraw a pinned snapshot.

The arbiter in every case is an executable check, not prose: a digest test for the
copy, a byte-identical re-derivation for the snapshot, a named-field failure for drift.

## Consequences
- The claim is checkable from a clean clone with no sibling checkout, no round store,
  and no API key — that configuration is now a gate, and the provenance tests *run*
  there instead of skipping (the `inputs` key-table, gold-digest and display-path tests
  all execute on a data-less clone).
- Provenance in the artifact rather than in the copy means the copy can be diffed
  against upstream byte-for-byte forever; a header comment added to it would break that.
- Editing a pinned copy is a tampering event that the digest test names, rather than a
  silent change of behaviour (`test_pinned_reference_tampering_is_named_not_published`).
- A moved external input degrades to a reported condition, never to a redrawn artifact:
  drift is printed by field name and the committed numbers stay reproducible. Recorded
  facts and live facts are therefore never conflated in one artifact.
- Cost: pinned copies go stale relative to upstream. That is the intent — a baseline
  must be measured against one named version — and it is disclosed by recording the
  upstream commit, so "which scorer was this AUC taken against" is answerable from the
  artifact alone.
- Anything not in a pinned copy or a committed record (a `temp/` file, a live store,
  `../semblr/`) cannot be cited from a durable artifact; the minimal fact gets copied in
  instead.
