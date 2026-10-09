# 002 — Prompt resolution model

**Status:** accepted

## Context
Prompt text must be resolved from the semblr round store (`~/.pi/agent/semblr/rounds/<id>.json`) at dataset-build time. Some round files are missing (29 of 5430 scan ids), some have an empty `userPrompt` field (1 id), and some may be malformed. The round record also carries a `prompt_preview` field (first 120 characters), which is cheap to fall back to but is a degraded input — a preview is not the full prompt, and downstream consumers must not silently train or score on truncated text.

## Decision
**Prompt resolution is binary: resolved or unresolved.** A row is `prompt_status: unresolved` when the round file is absent, unreadable, malformed, lacks `userPrompt`, or that field is not a non-empty string. `prompt_preview` is **never substituted** for a missing prompt. Resolution walks past per-id misses and never drops a row — unresolved rows stay in the artifact so downstream stages can decide what to do with them.

## Consequences
- Consumers can trust that `prompt_status: resolved` means the full prompt text is present byte-for-byte.
- Consumers that need full prompts (C3 gold, C5 calibration) filter on `prompt_status`; consumers that don't (C2 frame arithmetic) ignore it.
- The 30 unresolved corpus rows (29 missing round file + 1 empty prompt) are carried in the artifact and named in the run report.
- Gold is fully resolved (0 unresolved of 242) by construction; if that ever changes, the C1 build fails loudly.