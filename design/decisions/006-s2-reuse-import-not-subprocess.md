# 006 — S2 reuse: import, don't shell out

**Status:** accepted

## Context
The base-rate estimation stage (C2, E1 refinement 4) needs to adjudicate ~100 randomly sampled rounds through the S2 (LLM root-cause) stage of the fault pipeline. The pipeline machinery lives in `../semblr/scripts/fault-pipeline.py`. Two approaches: shell out to the script's CLI, or import it as a Python module.

Shelling out would require serializing round state to disk, invoking a subprocess, and parsing stdout — fragile, slow, and impossible to cache or mock for offline tests. Importing gives direct access to `process_round`, `parse_json_loose`, and `llm_request`, enabling a test seam for the LLM call.

## Decision
**S2 adjudication reuses the fault pipeline by importing it, not by shelling out.** `load_s2_module` resolves the pipeline path (default `../semblr/scripts/fault-pipeline.py`, overridable via `PROMPT_QUALITY_FAULT_PIPELINE`) and imports it with `importlib`. The imported `llm_request` function is swapped out via `cached_llm` — a closure that intercepts the call and routes to a disk cache — so offline tests and warm re-runs make zero API calls.

S2-only adjudication is invoked as `process_round(state, stages={'S2'})` — no S0/S1 ranking, per the base-rate sampling design.

## Consequences
- The LLM call is behind a cache seam: tests pass without network, and warm re-runs are deterministic.
- The import path is configurable, so the pipeline can move or be vendored without changing the C2 script.
- A transient import failure (e.g., missing dependency in the fault pipeline) is caught at `load_s2_module` time and reported, not discovered mid-run.
- The subprocess alternative was considered and rejected: it would lose the test seam and make the cache impossible.