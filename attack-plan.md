# Attack Plan: Prompt-Only Failure Prediction

## What the fault pipeline actually does (verified against scripts, Sep 27/28 outputs)

Pipeline = `jev-round-scan.py` → `fault-pipeline.py`, with `jev-chain-fault.py` as a side tool.

The scan-stage scorer is pinned in this repository at
`scripts/reference/jev-round-scan.py` — a byte-for-byte copy of semblr's
`scripts/jev-round-scan.py` at commit `ba10970`, sha256
`4e294839a341c953d38dba17dadfd87be53d36f536700d3274da16f387bf8704`. Line
citations below are to that pinned copy (`build_state` 78–92,
`build_refine_state` 154–173, `QUESTIONS` 46–76, `MODEL` 44, `--state-chars`
default 272), so they resolve from a clean clone. `baseline-metrics.json`'s
`inputs.jev_reference` records the same digest, and `scripts/baselines.test.py`
asserts C3's copies against this file.

1. **Scan (weak supervision).** `jev-round-scan.py` scores all 5430 scanned rounds on
   (a) user frustration 0–4 and (b) correction-of-round-discovery-failure probability.
   Thresholds trip → examination bin. Refine pass rebuilds state from the prompt +
   **parent round's response** and filters out parentless/benign rounds
   (`false-positive.txt`: 216; survivors: 17).
2. **Staged adjudication.** `fault-pipeline.py` takes ranked candidates
   (S0: jev fault score 0–1, rank-only, never a gate; top-k 50 / top-frac 0.25),
   then S1 (jev fault-type from a fixed taxonomy), then S2 (LLM root-cause with the
   first-class escapes "no fault within round" / external / ambiguous), with one
   bounded S2→S3→S4 re-route cycle.
3. **Chain walk (origin).** `jev-chain-fault.py` walks the parentId chain and produces
   per-round fault bands plus an origin posterior (earliest-faulty-round semantics).
   Only 4 records exist so far — underused.

## Ground truth status (temp/fault-pipeline.jsonl, deduped)

- 248 rows → **244 unique rounds** (resume reruns duplicated ids — dedupe by id, last write wins).
- Bins: 203 no-fault, 17 prompt-misread, 9 ambiguous, 8 other, 3 stale-context,
  3 external, 1 retrieval-noise → **29 confirmed faults, 3 external, 9 ambiguous**.
- **Critical caveat:** these are all top-ranked jev candidates, so the *corpus base
  rate* of faultiness is unmeasured. Also: the pipeline judged the *correction target*
  round — S2 evidence repeatedly says "the fault occurred in the parent round". Many
  true origin rounds are therefore NOT in the 244.

## Refinements to the earlier experiment design

1. **Failure definition changes.** Not regex-supersession. Ground truth = S2 verdict
   (fault bins are positives; no-fault and external are negatives; ambiguous excluded).
   Weak labels = jev-scan frustration/correction scores.
2. **Known leak.** `jev-round-scan.py` passes the round's **own** response to the
   scorer (`build_state`, `scripts/reference/jev-round-scan.py:78-92`); only
   `--refine` (`build_refine_state`, `:154-173`) uses the **parent**
   response. A prompt-only re-score strips that context. Measure both deltas:
   **leak delta** = `AUC(prompt+own-response) − AUC(prompt-only)` — how much the scorer
   leans on the response it is judging, the headline "context value"; and **session
   delta** = `AUC(prompt+parent-response) − AUC(prompt-only)` — the value of
   prior-round context. Both are reported; neither is dropped (issue #3).
3. **Label the origin, not the trigger.** Use S2 root-cause evidence ("fault belongs
   to the parent round") to back-propagate positive labels to origin rounds. This
   aligns the dataset with the real question: *which prompt initiated the doomed
   work?* Chain-walk (`jev-chain-fault.py`) on the 29 confirmed faults would deepen
   this (origin posterior per fault) — cheap to extend since the tool exists.
4. **Estimate the base rate.** Sample ~100 random unscanned rounds, push them through
   S2 only, to measure the true corpus fault rate. Without this, "beat the base rate"
   is meaningless.
5. **Continuation stratification stays as designed.** Classify each prompt
   fresh-topic vs continuation (parental adjacency + deictic cues: "it", "that",
   "yes", pronouns-without-antecedent). Prediction: prompt-only signal lives in
   fresh-topic prompts; continuations need a context-sufficiency flag instead of a
   score.

## Experiment stages

- **E1 — dataset build.** Dedupe fault-pipeline.jsonl → 244 gold-ish rounds; extract
  prompt text, S2 bin, fault type, root-cause evidence; back-propagate origins;
  sample ~100 random rounds for base-rate S2 adjudication. Output:
  `dataset/rounds-labeled.jsonl` (gold, ~350), `dataset/prompt-corpus.jsonl` (5430 weak).
- **E2 — baselines.** (a) base rate; (b) lexical heuristics (length, imperative
  density, deixis, output-contract absence); (c) jev re-score in three variants —
  prompt-only (context stripped), prompt+own-response, and prompt+parent-response —
  yielding the leak and session deltas of refinement 2.
- **E3 — trained scorer.** Small classifier on weak labels (with the leak measured
  per refinement 2), validated on the adjudicated gold only.
- **E4 — calibration & ablation.** ECE binned; stratify fresh vs continuation;
  report per-stratum AUC, FNR, calibration.
- **E5 — decision gate.** AUC ≥ 0.7 & ECE ≤ 0.15 → shadow-mode scorer tool.
  0.55–0.7 → fresh-topic-only advisory. < 0.55 → prompt-only dead; pivot to
  session-aware scoring (which the chain-walk machinery already half-implements).

## Concrete next actions

1. Write `build-dataset.py` (dedupe, label extraction, origin back-propagation,
   corpus extraction) → `~/work/personal/prompt-quality/`.
2. Write `sample-base-rate.py` (random 100 rounds → S2-only adjudication, resumable,
   same API pattern as fault-pipeline.py).
3. Run E2 baselines against the gold set.
4. Decide E3 investment based on E2 results (if lexical features alone hit the gate,
   skip the model).
