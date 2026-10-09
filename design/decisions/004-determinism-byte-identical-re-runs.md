# 004 — Determinism: byte-identical re-runs

**Status:** accepted

## Context
Reproducibility is a first-class requirement (from AGENTS.md): re-running the same stage on the same inputs must produce byte-identical output, or fail loudly naming the non-deterministic field. Without an enforcement mechanism, non-determinism can creep in through JSON key ordering, float representation, or API call variation, and downstream stages would silently consume drifted artifacts.

## Decision
**Every stage that emits an artifact enforces byte-identical determinism.** The mechanism has three layers:

1. **Serialization determinism.** Rows sorted by `id`; JSON keys via `sort_keys=True`; floats via `repr` (stable). Chosen over a canonicalization library to avoid an external dependency.
2. **In-process drift check.** `check_determinism` emits, re-serializes the same records in memory, byte-compares, and on mismatch raises naming the first divergent row and byte offset — aborting the build. A single process can observe re-run drift without a second pass.
3. **External re-run check.** The stage's verify command includes a `build → cp → build → diff -r` cycle; a second run must produce byte-identical artifacts. This catches non-determinism that the in-process check can't reach (e.g., live API calls with a populated cache).

For stages with live API calls (C2, C3), the disk cache (see 008) makes the re-run deterministic: warm cache → no API calls → byte-identical output.

## Consequences
- Drift is detected at build time, not discovered by a confused downstream consumer.
- The in-process check catches serialization bugs; the external diff catches cache/persistence bugs.
- Live-run stages are still non-deterministic on first run (API calls vary), but the warm-cache re-run is deterministic — the committed artifact is reproducible from the recorded seed and cache.
- Extending the mechanism to a new stage requires only the same three layers, not a new determinism framework.