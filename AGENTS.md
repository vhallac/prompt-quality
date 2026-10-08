# Prompt Quality

Measuring whether a user prompt alone predicts its round's failure — and at
what operating point the prediction is trustworthy enough to act on. The
research design is in [attack-plan.md](attack-plan.md); the work breakdown is
in the tracker (issues `C1`–`C6`, one per experiment stage `E1`–`E5`).

## What This Project Is

An experiment suite, not a service. Each issue produces a reproducible
artifact — a dataset, a metrics file, a model, a report — that the next stage
consumes. There is no long-running process and no user-facing surface.

Two consequences shape everything below:

- **Reproducibility is a first-class requirement.** Artifacts are keyed by a
  recorded seed and a pinned input snapshot; re-running the same stage on the
  same inputs must reproduce byte-identical output, or fail loudly naming the
  non-deterministic field. Never silently substitute a degraded input
  (`prompt_preview` for a full prompt) or drop a row without recording it in
  the run report.
- **Labels are defined once and reused.** The fault/negative/excluded label
  rule lives in the dataset build and every downstream stage consumes it; no
  stage re-derives labels.

## Project Structure

- `attack-plan.md` — research design: pipeline description, ground-truth
  caveats, experiment stages `E1`–`E5`, concrete next actions
- `AGENTS.md` — this file (project context for AI agents)
- `dataset/` — experiment artifacts (labeled data, metrics, score dumps)
- `scripts/` — experiment code (one script per stage), tests colocated as
  `*.test.*` beside the module they protect
- `docs/` — findings and write-ups promoted to permanent artifacts

Adopt semblr's shape — `scripts/` for stage code, tests beside sources,
`docs/` for write-ups — but **do not import its TypeScript tooling**. This is a
Python experiment suite; `uv` is the available dependency manager and the
standard library is preferred. No `npm`/`justfile`/`knip`/`stryker`.

## Development Methodology

> This section replaces the default Development Methodology in the global
> agent configuration; follow it as written.

For implementation tasks, use code-and-test-together development.

1. **Understand**
   - Restate the required behavior and the stage it belongs to.
   - Identify behavioral anchors: the issue's use case and success scenario,
     the "Verified anchors" section, the attack plan, and the pinned input
     snapshots. An anchor is a fact checked against source, not a claim.
   - Surface assumptions before changing code.

2. **Plan code, tests, and artifacts together**
   - For tasks spanning more than four meaningful steps, track scope in the
     issue tracker; keep it updated as steps complete.
   - Identify production changes (the stage script).
   - Identify tests that protect the stage's contract: dedupe, label split,
     resume behavior, and byte-identical re-run.
   - Identify which issue's artifacts or definitions are affected.
   - Refactor only as needed to expose test seams; preserve behavior.

3. **Implement first pass**
   - Write the stage script and its matching tests together.
   - Tests MUST check externally anchored behavior (the success scenario and
     its extensions), not mirror the implementation.
   - Prefer focused regression tests over broad coverage.

4. **Validate externally**
   - Run the stage's declared gates from the repository root and report
     exactly what was run: the test command, then the stage end-to-end on the
     pinned inputs.
   - Re-run the stage a second time and compare artifacts byte-for-byte;
     a non-deterministic output is a failure, not a note.
   - Passing tests alone are insufficient if the success scenario was not
     re-checked end-to-end.

5. **Diagnose failures before fixing**
   - Classify each failure by source:
     - **code:** wrong behavior, edge case, integration
     - **data:** input drift, unresolved id, missing round file
     - **test:** wrong expectation, setup, fixture, harness
     - **mechanical:** syntax, type, lint, formatting
     - **requirement:** ambiguous, contradicted by existing behavior
   - Fix the diagnosed source.
   - Mechanical fixes may be direct.
   - NEVER weaken tests merely to pass; a changed expectation needs an anchor.

6. **Harden**
   - Add edge-case and regression tests after the main behavior works: empty
     strata, single-class denominators, interrupted-and-resumed runs, ids with
     no round file.
   - For risky label or metric logic, exercise every branch with targeted
     parameterized tests.

7. **Reconcile artifacts and documentation**
   - Correct `attack-plan.md` and any issue definition the change invalidated —
     a decision, formula, or bin tally that moved during implementation gets
     reflected back.
   - Report only permanent artifacts: issues, `attack-plan.md`, committed
     code. Never cite `temp/` paths, run logs, or scratch files as a source
     of truth; copy the minimal fact instead.
   - Findings that outlive the experiment go under `docs/`.

## Attributions

- When committing code that is entirely written by you, add

🤖 LLM authored

- When committing code that is partially written by you (50% or less), and the rest is written by a human, add

🤖 LLM assisted

- When writing code forge pull requests, comments, issues, or contributing to discussions, add

🤖 Content created by LLM

as the last line of the text.

When only creating commit messages to code fully written by a human; do not add an LLM attibution.
