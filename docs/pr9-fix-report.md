# PR !9 remediation report — F1–F10, U1–U2

**Source:** PR #9 review comment (`vhallac/prompt-quality#9`, 2026-10-09) and its
fix-scope addendum "live-round-store drift (U1–U2)". **Branch:** `feat/3-baselines`,
commits `d27c6cd..HEAD`. **Tracker:** work item #3 (C3).
**Reading convention:** one line per finding — reference, defect, shipped approach, the
test that discriminates it, the commit. Evidence for the artifact regeneration is below.

## The fixes

**F1 (Blocking)** published `lexical_weights` were off by one — the intercept was published as `length` and the last feature's weight was dropped, so the artifact declared a combiner whose rebuilt AUC is 0.4775, not the reported 0.6364 → the block is now `{intercept, features{weight, mean, std}}`, i.e. the combiner is re-derivable from the metrics file alone (standardisation included) → `test_published_combiner_is_rederivable` rebuilds every row's logit from the published fields and equals `lexical_combined` → `d27c6cd`

**F2 (Important)** the jev cache key omitted the questions payload, so editing the rubric silently served stale answers and *still* passed the byte-identical determinism gate → the key is payload-complete `sha256(state, model, questions_sha256)`, `jev.questions_sha256` is published, and the 564 paid entries are re-keyed through a legacy-adoption path gated on the digest recorded for that file → `test_rubric_change_does_not_reproduce_byte_identical_scores`, `test_legacy_cache_is_not_adopted_for_a_different_rubric`, `test_pinned_legacy_digest_is_what_authorizes_the_shipped_cache` → `4ea28fb`

**F3 (Important)** baseline (c)'s reference scorer was out-of-repo and unpinned, and the anchors cited a path that does not exist in this repository with a line range that was never correct → the scorer is committed as `scripts/reference/jev-round-scan.py` (byte-for-byte copy of semblr's file at `ba10970`), its digest and upstream naming are published under `inputs.jev_reference`, the parity tests import that copy instead of asserting local string literals, and `attack-plan.md`'s anchors are corrected (`build_state` 78–92, `build_refine_state` 154–173, `QUESTIONS` 46–76, `MODEL` 44, `--state-chars` 272) → `test_pinned_reference_copy_matches_its_recorded_digest`, `test_pinned_reference_build_state_parity`, `test_pinned_reference_absent_copy_fails_the_build_loudly` → `5ab0ad5`

**F4 (Important)** operating point (ii) was reported for `base_rate` at an achieved alarm rate of 0.0 with no disclosure, and `dff63bf`'s threshold-candidate change was never reconciled with the tracker → every FNR cell publishes `target_alarm_rate`, `achieved_alarm_rate`, `alarm_rate_tolerance` and a `note` naming the degeneracy when the two disagree beyond tolerance; two (ii) cells are now marked non-rate-matched — `base_rate` (the constant-score baseline has 1 distinct value, so the target rate is unreachable and the selected threshold alarms nothing: achieved 0.0) and `jev_parent_response` (77 distinct values, achieved 0.198473 vs target 0.193878, tolerance 0.003817) — and `base_rate` (ii) FNR reads 1.000, not the 0.000 the tracker shows. The disclosure itself moved nothing — `test_alarm_rate_disclosure_does_not_move_threshold_or_fnr` pins that; the value `dff63bf` had already changed was the (ii) threshold candidate set, and the tracker never reported the consequence → `test_unreachable_target_alarm_rate_is_marked_non_rate_matched`, `test_alarm_rate_disclosure_does_not_move_threshold_or_fnr` → `5b47acf`

**F5 (Minor)** the session delta point was not the difference of the two AUCs the same artifact reports (0.321 − 0.481 = −0.160 ≠ the published −0.0239) → each delta block names its population (`n_pairs`, `n_reference_rows`, a `population` note) and carries `auc_reference_paired`, both arms' AUCs computed on the paired rows, whose difference *is* the published point → `test_paired_delta_block_names_its_own_population`, `test_paired_delta_equals_marginal_difference_when_arms_share_every_row` → `d97dfb2`

**F6 (Minor)** `output_contract_absence`'s published definition contradicted its implementation — an absence indicator scores 1 when it matches nothing, while extension 3a declared "matches nothing scores 0" → the implementation-facing key is `no_marker_score: 1.0`, and an `ext_3a_scope` member confines extension 3a's zero rule to the count/ratio features (`length`, `imperative_density`, `deixis`) → `test_feature_definitions_match_the_implementations` → `e0cbdf9`

**F7 (Minor)** extension 5a (single-class stratum) and the zero-shared-rows delta path were implemented but untested → both null paths are asserted: null AUC, null FNRs, note present, stratum absent from `run_report.thresholds_used`, and an empty-pairs delta that stays null but keeps its shape → `test_single_class_stratum_yields_null_auc_and_null_fnrs_with_note`, `test_single_class_stratum_is_absent_from_thresholds_used`, `test_paired_delta_without_shared_valid_rows_is_null_but_keeps_its_shape` → `d97dfb2`

**F8 (Minor)** a truncated jev cache crashed with a raw `JSONDecodeError`, and the cache was written *before* the determinism gate, so a rejected run mutated a tracked artifact → an empty/absent cache is cold and a corrupt one raises `JevCacheError` naming the path; saves are atomic (temp file + `os.replace`) and happen only after a successful `emit_artifacts` → `test_corrupt_cache_raises_naming_the_path`, `test_failed_save_leaves_the_warm_cache_and_no_temp_file`, `test_main_writes_no_cache_when_the_artifact_gate_rejects` → `4ea28fb`

**F9 (Observation)** issue #3's prose still described the re-score as "a live OpenRouter call over 242 rounds" → the tracker record is corrected to the measured denominators — 242 gold rows, 234 scored (8 `ambiguous` excluded), 599 states (234 prompt-only + 234 own-response + 131 parent-response) — together with the F1 weights, the F4 (ii) cell and the F5 delta prose; publish-ready old→new replacement text is in `docs/issue3-reconciliation.md` (nothing posted without a go) → docs-only correction; every number verified against `dataset/baseline-metrics.json` rather than against the superseded comment

**F10 (Observation)** `fit_logistic` accepted a `seed` it never used, presenting the combiner as seed-sensitive → the dead parameter is removed from the signature and its call sites; the fit is deterministic by construction → `test_fit_logistic_is_not_seed_sensitive` → `e0cbdf9`

**U1 (Drift — C2)** the committed 100-id sample was not reproducible from the seed alone because its frame came from the live round store, which has since moved (15 of the 29 missing rounds recovered by the semblr sweep, store grown past 9,300 files) → `dataset/base-rate-sample.json`'s manifest *is* the pinned snapshot: seed + the recorded missing-id ledger + the recorded store facts re-derive the identical sample, the live scan became a drift check that names the drifted field (`StoreDriftError`, never a silent redraw), and both retargeted tests pass offline → `test_real_frame_anchor`, `test_determinism_committed_manifest_reproduces_from_seed`, `test_cli_refuses_to_redraw_a_pinned_snapshot` → `b6430b6`

**U2 (Drift — C1)** the same rule did not hold for C1's unresolved-id anchors, which asserted 30 against a store that now yields 15 → the build's ledger is pinned in the new committed `dataset/c1-build-snapshot.json` (30 unresolved ids = 29 no-round-file + 1 empty prompt, 5,400 resolved of 5,430, the id-universe digest); anchors assert against the record, a disagreeing store fails by field name, and a byte-level pre-write gate refuses to emit changed corpus text → `test_anchor_corpus_unresolved_split`, `test_committed_snapshot_is_a_complete_pinned_record`, `test_drift_fails_loudly_when_recorded_prompt_text_changed` → `6b83f73`

## Addendum items 1–6, disposition

| Addendum item | Where it landed |
|---|---|
| 1 Pin the input snapshot (store facts in the manifest/sidecar) | U1 `b6430b6`; C1 analogue U2 `6b83f73`; C3's `inputs` block `f3fe033` |
| 2 Retarget the two failing tests to the pinned manifest | U1 — both tests now read the record, not the store |
| 3 Loud drift check naming the drifted field | U1 `check_store_drift`/`assert_anchors`; U2 `SnapshotError`/`StoreDriftError` |
| 4 Record the reconciliation; same requirement for C3's inputs | U1/U2 evidence below; C3 `inputs.gold_sha256` + portable `round_store` in `f3fe033` |
| 5 Sequence the questions-in-key fix before any future re-score | F2 `4ea28fb` landed first; C3 regenerated from the warm cache with zero API calls in `f3fe033` |
| 6 A bigger frame later is a new stage | enforced: `--redraw` on a pinned snapshot is refused by both stages |

**C3's inputs block, and why it names the gold by digest.** The addendum's item 4 asked
for "the store snapshot facts" in `baseline-metrics.json`. C3 reads the round store, but
it does not read C1's snapshot record, so naming that file would have been a false
provenance claim. Instead `inputs` now carries `gold_sha256` — a digest over the scored
`(id, label, prompt)` rows — as the true tie to C1's output, and `round_store` is
rendered home-relative (`~/…`) so the artifact re-emits byte-identically off this
machine. Tests: `test_metrics_inputs_records_a_digest_of_the_rows_scored`,
`test_metrics_inputs_records_no_machine_absolute_path`,
`test_committed_metrics_inputs_reemit_off_this_machine`.

## Regeneration evidence (`f3fe033`)

- `dataset/baseline-metrics.json` re-emitted offline (`env -u OPENROUTER_API_KEY`), twice;
  all 8 `dataset/*` md5s identical across runs, `baseline-scores.jsonl` md5 unchanged.
- Cache: 564 entries → 564 entries, 0 key overlap (legacy → payload keys), the answer
  payload set identical (553 distinct), `__questions_sha256__` recorded and equal to
  `JEV_QUESTIONS_SHA256` ⇒ **zero API calls**, the paid work preserved.
- 51 divergent paths against the previous artifact, all inside the blocks the findings
  declare changed — `lexical_weights` (F1), the `fnr` cells (F4), `deltas` (F5/F7),
  `feature_definitions` (F6/F10), `jev.questions_sha256` (F2), `inputs.jev_reference`
  (F3) — plus `inputs.gold_sha256` (added) and `inputs.round_store`, the single VALUE
  change in the document. `auc`, `fnr` points, `ci95`, thresholds, `feature_aucs`, `gold`
  and `run_report` are otherwise byte-equal.
- The corrected published combiner: intercept `-2.006206`; `length +0.053816`,
  `imperative_density +0.249498`, `deixis -0.227257`, `output_contract_absence +0.071459`
  (the pre-F1 mapping had shifted every one of these and dropped the last).

## Gates

- Targeted per unit; file gate `pytest -q scripts/baselines.test.py` → 128 passed.
- Commit gate `pytest -q` → **380 passed / 0 failed** (was 373 before unit-009; the suite
  went fully green at unit-011 after the drift remediation).
- Clean clone (`git archive HEAD`), no round store, no sibling semblr checkout, empty
  `HOME`, no API key → 361 passed / 19 skipped, with the input-provenance tests
  *running* rather than skipping.
- Mutation probes — 63 in total, each reverted by copy with the source file's md5
  verified afterwards: 19 (U2 ledger/drift/byte-gate), 13 (U1 pinning), 7 (F3 pinned
  copy), 6 (F5/F7 provenance and null paths), 6 (C3 `inputs`), 5 (F2/F8 cache), 3 (F4
  disclosure), 3 (F6/F10 declarations), 1 (F1 weight shift). Every behavioural mutation
  is caught by a named test; the one probe recorded as *not* caught (re-basing
  `StoreDriftError` on `RuntimeError`) is an exception-hierarchy claim — documentation,
  not behaviour — and is reported as such rather than smoothed over.

## Open

- Publishing: the PR comment, the push of `feat/3-baselines`, and the issue #3 edits in
  `docs/issue3-reconciliation.md` are drafted but **not posted** — they need an explicit go.
- Out of this PR by the review's own note: U3 (issue #1 arithmetic collision) — tracker
  grooming only.
