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

Determinism also needs the artifact to *name* the inputs it re-derives from, in
a form a re-run can reproduce. C3's `inputs` therefore records `gold_sha256` — a
digest of the gold rows scored (id, label, prompt), not a file path — and every
location home-relative. A path names a moving file; the digest states which bytes
the published numbers came from, and the fields that must not move (`auc`, `fnr`,
`ci95`, thresholds, `run_report`) are then reconcilable against it.

## Consequences
- Drift is detected at build time, not discovered by a confused downstream consumer.
- The in-process check catches serialization bugs; the external diff catches cache/persistence bugs.
- Live-run stages are still non-deterministic on first run (API calls vary), but the warm-cache re-run is deterministic — the committed artifact is reproducible from the recorded seed and cache.
- Extending the mechanism to a new stage requires only the same three layers, not a new determinism framework.
- A stage whose inputs are *recorded* snapshots (C1: `design/decisions/002`, C2: `design/decisions/009`) adds one gate on top of the three: before writing, the build must render to the exact bytes already committed, or stop naming the first divergent row. A hand-edited artifact therefore halts a re-run instead of being silently normalised by it.