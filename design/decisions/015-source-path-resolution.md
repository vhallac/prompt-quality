# 015 — Source path resolution

**Status:** accepted

## Context
The prompt-quality project depends on inputs from the sibling semblr checkout: `semblr/temp/fault-pipeline.jsonl` (fault adjudication results), `semblr/temp/jev-scan-results.jsonl` (scan snapshot), `semblr/scripts/fault-pipeline.py` (S2 machinery), `semblr/scripts/jev-round-scan.py` (JEV scorer), and the round store (`~/.pi/agent/semblr/rounds/`). The plan and issues refer to paths under `semblr/`, but this repo has no `semblr/` directory — the real source is the sibling checkout at `../semblr/`.

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

## Consequences
- Scripts are runnable from any directory layout without editing source.
- A missing sibling checkout is detected at import/resolution time with a clear error, not a confusing `FileNotFoundError` deep in the code.
- The env-var overrides are documented in each script's `--help` and in the attack plan as-needed.
- No script hard-codes a bare `../semblr/` string; all go through a configurable default function.