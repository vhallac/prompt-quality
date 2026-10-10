# 015 — Source path resolution

**Status:** accepted

## Context
The prompt-quality project depends on inputs from the sibling semblr checkout: `semblr/temp/fault-pipeline.jsonl` (fault adjudication results), `semblr/temp/jev-scan-results.jsonl` (scan snapshot), `semblr/scripts/fault-pipeline.py` (S2 machinery), `semblr/scripts/jev-round-scan.py` (JEV scorer), and the round store (`~/.pi/agent/semblr/rounds/`). The plan and issues refer to paths under `semblr/`, but this repo has no `semblr/` directory — the real source is the sibling checkout at `../semblr/`.

Two of those inputs are claims this repository makes about *behaviour*, not just
data, so they no longer resolve to the sibling checkout: the scan snapshot is
committed as `dataset/prompt-corpus.jsonl` (C2) and the JEV scorer is pinned as
`scripts/reference/jev-round-scan.py` (C3, review F3). What still lives only in
`../semblr/` is `fault-pipeline.py` (imported for its S2 machinery) and the round
store.

Hard-coding `../semblr/` assumes a specific directory layout. Environment variables allow the paths to be overridden without editing code — useful for CI, vendoring, or a different checkout name.

## Decision
**Every external source path has a default derived from the sibling checkout and an environment-variable override.** The defaults point at `../semblr/` (resolved from the script's location via `Path(__file__).parents[2] / "semblr"`). The overrides are:

- `PROMPT_QUALITY_FAULT_PIPELINE` — fault-pipeline.jsonl (C1)
- `PROMPT_QUALITY_SCAN_RESULTS` — jev-scan-results.jsonl (C1)
- `PROMPT_QUALITY_ROUND_STORE` — round store directory (C1, C2, C3)
- `PROMPT_QUALITY_CORPUS` — raw scan snapshot (C2, for direct read)
- `PROMPT_QUALITY_FAULT_PIPELINE` (reused) — fault-pipeline.py script path (C2)
- `PROMPT_QUALITY_LLM_CACHE` / `PROMPT_QUALITY_JEV_CACHE` — LLM cache files (C2, C3)

The round store path is also configurable: `~/.pi/agent/semblr/rounds` by default.

**A location written into a committed artifact is written home-relative, never
machine-absolute.** C3's `metrics.inputs.round_store` used to carry
`/home/<user>/.pi/agent/semblr/rounds`, so the artifact could not re-emit
byte-identically on another box — the determinism gate would name the field and
report a username as non-determinism. `baselines.input_display_path` renders any
absolute path under home as `~/…`; a `--round-store` or
`PROMPT_QUALITY_ROUND_STORE` override outside home is recorded verbatim, because
then the artifact has to say what it actually read. C1 and C2 keep store paths in
their *ephemeral* drift reports, which are not committed, so they need no such
rendering.

## Consequences
- Scripts are runnable from any directory layout without editing source.
- A pinned copy inside this repository (corpus, jev reference scorer) is readable
  with no sibling checkout at all, which is what makes C3's parity claim
  checkable from a clean clone.
- A missing sibling checkout is detected at import/resolution time with a clear error, not a confusing `FileNotFoundError` deep in the code.
- The env-var overrides are documented in each script's `--help` and in the attack plan as-needed.
- No script hard-codes a bare `../semblr/` string; all go through a configurable default function.