# Issue #3 reconciliation — publish-ready corrections (F1, F3, F4, F5, F9)

Work item #3 (`vhallac/prompt-quality#3`, C3 — Baselines / E2) carries three stale
records: the description's re-score prose and scorer anchor (F9, F3), the grooming
comment's line anchors (F3), and the step-3 results comment (F1 weights, F4 operating
point (ii), F5 delta prose).
The replacements below are **exact-substring edits** — quote-for-quote `OLD` → `NEW` —
taken from the live issue text and checked against `dataset/baseline-metrics.json` as
committed. Nothing here has been posted; posting needs an explicit go.

Objects: description (body, edits A and A2), comment `6071190398` (grooming, edit B),
comment `6074600109` (step-3 record, edits C1–C4), plus one new comment announcing the
remediation (D).

## A. Description — the re-score denominators (F9)

```
OLD: This is a live OpenRouter call over 242 rounds; results are cached to disk for determinism.
NEW: The re-score runs over the 234 scored gold rows (of 242 loaded; 8 `ambiguous` excluded) in three variants — prompt-only, own-response, parent-response — so 599 states are requested (234 + 234 + 131; the 103 parentless rows yield `missing_parent` without an API call). The scorer under test is pinned in this repository as `scripts/reference/jev-round-scan.py` and its digest is recorded in `baseline-metrics.json` → `inputs.jev_reference`. Answers are cached to disk (`dataset/jev-cache.json`) keyed on the full request payload — state, model, and the questions digest — so the run is offline and byte-identical on re-run.
```

## A2. Description — the reference-scorer anchor (F3, owed since `5ab0ad5`)

The review asked for the path and line range to be corrected "in `attack-plan.md` **and
in the issue #3 anchors**"; the plan corrected the former in unit-006, and this is the
latter half of the same finding.

```
OLD: - The jev state is prompt + **own** response: `scripts/jev-round-scan.py` `build_state()` concatenates `USER PROMPT` and `responseSequence` (lines 66–77). This is the leak C3 measures.
NEW: - The jev state is prompt + **own** response: the scorer pinned as `scripts/reference/jev-round-scan.py` (byte-for-byte copy of semblr's `scripts/jev-round-scan.py` at `ba10970`, sha256 recorded in `baseline-metrics.json` → `inputs.jev_reference`) — its `build_state()` concatenates `USER PROMPT` and `responseSequence` (lines 78–92). Only `--refine`'s `build_refine_state` (`:154-173`) uses the parent's response. This is the leak C3 measures.
```

Two further description passages are stale by implication but are **not** edited here,
because they belong to tracker grooming rather than to this PR: extension 3a's "a prompt
that matches nothing scores 0" (its scope is now confined explicitly in the artifact —
`feature_definitions.ext_3a_scope`) and the combiner bullet's "fixed seed" (the dead
`seed` parameter is gone, review F10; the fit is deterministic without one).

## B. Grooming comment `6071190398` — the reference-scorer anchors (F3)

```
OLD: the scorer being baseline-tested is `scripts/jev-round-scan.py`, whose `build_state()` concatenates `USER PROMPT` with the round's **own** `responseSequence` (lines 66–77, verified). Only `--refine` (`build_refine_state`) uses the parent's response.
NEW: the scorer being baseline-tested is semblr's `scripts/jev-round-scan.py`, pinned in this repository as `scripts/reference/jev-round-scan.py` (byte-for-byte copy at upstream `ba10970`, sha256 `4e294839a341c953…`), whose `build_state()` (`:78-92`) concatenates `USER PROMPT` with the round's **own** `responseSequence`. Only `--refine` (`build_refine_state`, `:154-173`) uses the parent's response. `QUESTIONS` is `:46-76`, `MODEL` `:44`, the `--state-chars` default `:272`.
```

## C. Step-3 comment `6074600109` — four edits

### C1. Combiner weights (F1)

```
OLD: Lexical combiner weights: `length −2.006`, `deixis +0.249`, `output_contract_absence −0.227`, `imperative_density +0.054`.
NEW: Lexical combiner, as published in `lexical_weights` after the off-by-one fix (review F1 — the pre-fix mapping named the intercept `length` and dropped the last weight): intercept `-2.006206`; `length +0.053816` (standardisation mean 33.7735, std 38.3762), `imperative_density +0.249498` (0.065810, 0.168274), `deixis -0.227257` (4.629418, 4.273716), `output_contract_absence +0.071459` (0.666667, 0.471405). The published score is the logit `intercept + sum(weight * (feature - mean) / std)`; `test_published_combiner_is_rederivable` rebuilds every row from these fields.
```

Note for the record: the direction claim that repeated the old numbers ("`deixis` is
positive") also flips — `deixis` carries the **negative** weight, and
`output_contract_absence` is the small positive that was dropped entirely.

### C2. Operating point (ii) for `base_rate` (F4)

```
OLD: | base_rate | 234 | 0.500 [0.500, 0.500] | 0.000 (0.1239) | 0.000 (0.1239) |
NEW: | base_rate | 234 | 0.500 [0.500, 0.500] | 0.000 (0.1239; alarms everything) | 1.000 (0.1239; **non-rate-matched** — achieved alarm rate 0.0 vs target 0.193878) |
```

```
OLD: | jev parent-response | 131 | 0.321 [0.100, 0.581] | 0.750 (0.8387) | 0.750 (0.8387) |
NEW: | jev parent-response | 131 | 0.321 [0.100, 0.581] | 0.750 (0.8387) | 0.750 (0.8387; **non-rate-matched** — achieved 0.198473 vs target 0.193878) |
```

And insert after the table:

```
NEW (insert): Every FNR cell now discloses the alarm rate it achieved as well as the rate it was asked for. Point (ii) matches the C2 corpus base rate `0.193878` only where the score vector can express it: `base_rate` holds a single distinct score, so the nearest candidate threshold alarms nothing and FNR is 1.0 by construction; `jev_parent_response` has 77 distinct values and lands 0.00460 above target, outside its `1/(2n)` tolerance of 0.003817. Both cells carry a `note` naming the degeneracy. No threshold and no FNR point moved with this change — only the disclosure, and the `base_rate` (ii) cell as corrected by `dff63bf` (alarm-nothing threshold included in the candidate set), which this comment had not yet reflected.
```

### C3. Delta prose (F5)

```
OLD: **Deltas** (paired, same rows): leak `AUC(prompt+own-response) − AUC(prompt-only)` = **−0.0962** [−0.1808, −0.0131], n_pairs 234; session `AUC(prompt+parent-response) − AUC(prompt-only)` = **−0.0239** [−0.1940, 0.1048], n_pairs 131.
NEW: **Deltas** (paired on the rows both arms score non-null, reported with the population they differ on): leak = **-0.0962** [-0.1808, -0.0131], n_pairs 234 of 234 reference rows - here the paired arms' AUCs are the published marginals (0.3848 - 0.4810), so point and marginal difference agree. session = **-0.0239** [-0.1940, 0.1048], n_pairs 131 of 234: on those 131 rows the prompt-only arm scores AUC 0.3450 and the parent-response arm 0.3211, so the paired point is *not* the difference of the two marginals in the table above (0.321 - 0.481 = -0.160). Each delta block publishes `auc_reference_paired` (both arms on the paired rows, whose difference is the point) beside `auc_marginal`, plus `n_pairs`, `n_reference_rows` and a `population` note.
```

### C4. Determinism paragraph (F2, F8, regeneration)

```
OLD: **Determinism.** `python3 scripts/baselines.py` twice; `cmp` on `dataset/baseline-metrics.json`, `dataset/baseline-scores.jsonl`, and `dataset/jev-cache.json` all identical. Second run is offline (cache 564 entries).
NEW: **Determinism.** `python3 scripts/baselines.py` twice, offline (`OPENROUTER_API_KEY` unset); all `dataset/*` artifacts byte-identical across runs and `baseline-scores.jsonl` unchanged by the remediation. The cache keys are now payload-complete — `sha256(state, model, questions_sha256)` — and the 564 paid entries were re-keyed in place (0 key overlap, the answer set unchanged: 553 distinct payloads), so the regenerated run still costs zero API calls. Saves are atomic and happen only after the artifact gate passes, so a rejected run leaves the tracked cache untouched. A rubric edit is now a cache miss with a loud failure rather than a silent stale answer.
```

## D. New comment on #3 — announce the remediation (draft)

```markdown
**PR #9 review remediation landed on `feat/3-baselines` — F1–F10 + the U1/U2 drift pinning.**

Ten findings, one unit each: `d27c6cd` F1 (combiner published as `{intercept, features{weight, mean, std}}`, re-derivable from the artifact), `e0cbdf9` F6+F10 (feature declarations match implementations; dead `seed` dropped), `5b47acf` F4 (achieved vs target alarm rate disclosed at both operating points), `d97dfb2` F5+F7 (paired deltas name their population; null paths tested), `4ea28fb` F2+F8 (cache keyed on the whole payload, written atomically after the emit), `5ab0ad5` F3 (reference scorer pinned at `scripts/reference/jev-round-scan.py`, parity tests import it), `b6430b6` U1 (C2 manifest is the pinned snapshot; live store is a drift check), `6b83f73` U2 (C1 ledger pinned in `dataset/c1-build-snapshot.json`), `f3fe033` C3 artifacts regenerated offline.

Corrections to this issue's own record are drafted in `docs/pr9-fix-report.md` and applied by the edits to this comment, the description, and the grooming comment: the combiner weights in the step-3 record were shifted by one slot, the `base_rate` operating-point-(ii) cell reads 1.000 (non-rate-matched) after `dff63bf`, the session delta is a paired difference on 131 rows rather than the difference of the two tabled AUCs, the re-score is over 234 scored rows / 599 states and not "242 rounds", and the scorer anchor points at the pinned copy (`build_state` 78–92).

No reported statistic moved except the fields those findings declare changed. Suite: 380 passed / 0 failed; clean clone with no round store, no sibling checkout and no API key: 361 passed / 19 skipped; zero API calls spent.

🤖 Content created by LLM
```

## Applying these edits

Each block is a literal substring replacement on the fetched body/comment, verified to
match exactly once before writing. `$WORK` is an ephemeral scratch directory chosen by
the operator — nothing under it is a source of truth or a reference target:

```bash
gh api repos/vhallac/prompt-quality/issues/3 -q .body > "$WORK/i3.md"        # description
gh api repos/vhallac/prompt-quality/issues/comments/6071190398 -q .body > "$WORK/groom.md"
gh api repos/vhallac/prompt-quality/issues/comments/6074600109 -q .body > "$WORK/step3.md"
# apply the OLD→NEW pairs above (A + A2 on the description; C1–C4 on the step-3 record;
# assert exactly one match per OLD string before writing), then:
gh api repos/vhallac/prompt-quality/issues/3 -X PATCH -F body=@"$WORK/i3.new.md"
gh api repos/vhallac/prompt-quality/issues/comments/6071190398 -X PATCH -F body=@"$WORK/groom.new.md"
gh api repos/vhallac/prompt-quality/issues/comments/6074600109 -X PATCH -F body=@"$WORK/step3.new.md"
gh api repos/vhallac/prompt-quality/issues/3/comments -F body=@"$WORK/announce.md"   # section D
```

Edits to a *record of what was reported* keep the audit trail: C1–C4 correct the step-3
comment in place and the new comment in section D names the commits that superseded those
numbers, so the original reading stays attributable rather than silently rewritten.
