# 008 — Disk cache pattern for LLM calls

**Status:** accepted

## Context
Both C2 (base-rate S2 adjudication) and C3 (jev re-score) make paid LLM API calls. Without a cache, every test run re-spends money, determinism is impossible, and interrupted runs lose paid work. A cache format that differs between stages would create fragmentation and force each stage to re-implement the same resume/eviction/validation logic.

## Decision
**All LLM-calling stages share one cache pattern.** The pattern, first implemented in C2 (`sample-base-rate.py`) and reused in C3 (`baselines.py`), is:

1. **Key:** `sha256(state_bytes \0 model_name)` — the full serialized state + a null byte delimiter + the model identifier — so different prompts or models produce different cache entries.
2. **Storage:** sorted JSON at a configurable path (`PROMPT_QUALITY_LLM_CACHE` / `PROMPT_QUALITY_JEV_CACHE`), loaded on startup and saved after every row.
3. **Seam:** `cached_llm(llm_request, cache, cache_path)` returns a closure that checks the cache before calling the real `llm_request`, and appends to the cache after every call. The closure is swapped in for the imported `llm_request` function, so the calling code is unchanged.
4. **Determinism:** a warm cache means zero API calls — the re-run is byte-identical.

## Consequences
- C2 and C3 share the same cache keying and persistence logic; a new stage adopting it only needs to provide the state serialization and the real `llm_request`.
- Cache files can grow large (C3's jev cache reached 564 entries); the sorted-JSON format is simple and inspectable, not optimized for size.
- Cache invalidation is implicit: a different model name or state change produces a different hash → cache miss → fresh API call.
- Per-row persistence (see 007) means an interrupted run keeps all prior entries.