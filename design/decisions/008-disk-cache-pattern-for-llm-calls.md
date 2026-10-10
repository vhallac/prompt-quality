# 008 — Disk cache pattern for LLM calls

**Status:** accepted

## Context
Both C2 (base-rate S2 adjudication) and C3 (jev re-score) make paid LLM API calls. Without a cache, every test run re-spends money, determinism is impossible, and interrupted runs lose paid work. A cache format that differs between stages would create fragmentation and force each stage to re-implement the same resume/eviction/validation logic.

## Decision
**All LLM-calling stages share one cache pattern.** The pattern, first implemented in C2 (`sample-base-rate.py`) and reused in C3 (`baselines.py`), is:

1. **Key: payload-complete.** Every field of the request that can change the answer is in the key, delimited by null bytes — for C2 `sha256(system, user, model)`, for C3 `sha256(state, model, questions_sha256)`. The questions enter as their digest so the key stays fixed-width hex. A key that omits any request field is a defect: the review F2 counterexample is a rubric edit that C3's old `(state, model)` key answered with the *previous* rubric's verdicts — and passed the byte-identical determinism gate, because determinism was measured against stale answers rather than against the rubric in the source.
2. **Storage:** sorted JSON at a configurable path (`PROMPT_QUALITY_LLM_CACHE` / `PROMPT_QUALITY_JEV_CACHE`), loaded on startup and saved after every row.
3. **Seam:** `cached_llm(llm_request, cache, cache_path)` returns a closure that checks the cache before calling the real `llm_request`, and appends to the cache after every call. The closure is swapped in for the imported `llm_request` function, so the calling code is unchanged.
4. **Determinism:** a warm cache means zero API calls — the re-run is byte-identical.
5. **Provenance beside the entries, not in the key alone.** The cache document carries `__questions_sha256__` (a reserved non-hex member, so it cannot collide with an entry key), stating which payload its answers belong to. A cache that predates this metadata is admitted only through a **gated legacy-adoption path**: the digest of the payload that produced it is pinned in source as a fact of git history, and adoption is allowed only while the live payload still hashes to it. Edit the rubric and the paid entries become unreachable — a miss, and without an API key a loud failure — instead of answering questions they were never asked.
6. **Load policy:** absent or empty file ⇒ cold cache, run normally. Present but unparseable ⇒ raise naming the path (`JevCacheError`); silently treating a half-written cache as cold would discard paid answers and re-bill for them without telling anyone.
7. **Write atomically, and last.** Saves go through a temp file in the same directory plus `os.replace`, so a kill mid-save cannot leave a tracked cache truncated, and the cache is persisted only **after** the artifact emit has passed its determinism gate — a rejected run must not mutate a committed input.
8. **A migration is invisible in the published artifact.** The count of adopted legacy entries is deliberately *not* recorded in metrics: it is 564 on the first run after re-keying and 0 on every run after, so publishing it would make the artifact non-reproducible. Digests and counts that describe *how many entries moved* belong in the run report or the commit message, never in the keyed artifact.

## Consequences
- C2 and C3 share the same cache keying and persistence logic; a new stage adopting it only needs to provide the state serialization and the real `llm_request`.
- Cache files can grow large (C3's jev cache reached 564 entries); the sorted-JSON format is simple and inspectable, not optimized for size.
- Cache invalidation is implicit and complete: any change to model, state, or rubric produces a different hash → cache miss → fresh API call. No stale answer can be served under a key it does not belong to.
- Per-row persistence (see 007) means an interrupted run keeps all prior entries; atomic write means an interrupted *save* keeps them too.
- Re-keying an existing paid cache is a one-time, self-authorized operation gated on a pinned digest, so fixing the key does not force re-billing (the 564-entry jev cache survived F2's fix with zero API calls — see `docs/pr9-fix-report.md`).
- C2's `(system, user, model)` key already satisfied the rule; C3's `(state, model)` key is the case that motivated restating it, because it omitted a field the stage reads from source rather than from the CLI.