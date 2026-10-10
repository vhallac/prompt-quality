# 005 — Origin derivation from S2 evidence text

**Status:** accepted

## Context
The attack plan's refinement 3 says: "Label the origin, not the trigger. Use S2 root-cause evidence to back-propagate positive labels to origin rounds." The round store carries a structural `parentId` field that could also be used for origin walks. Issue #1 reserves the structural `parentId` chain-walk for C6 (`jev-chain-fault.py`), and using it in C1 would duplicate the C6 posterior and risk a split signal.

The S2 evidence text (`s2.root_cause` / `s2.evidence`) names round ids as 6–32-char hex tokens — sometimes as 7–8-char prefixes (`6547ceb`), sometimes as full ids — in narrative prose about where the fault originated.

## Decision
**Origin is derived from S2 evidence text, not from `parentId`.** The derivation extracts the **first** hex token (6–32 chars) in first-appearance order that resolves to a known round id (exact match preferred, else unique prefix). A token that matches nothing or matches multiple ids is not treated as evidence.

Origin classification:
- **self:** the row's own id, or no resolvable parent token found.
- **in-set-parent:** names a different id that is a gold row.
- **external-parent:** names a different id not in the gold set — not back-labeled.

External parents are never back-labeled into the gold set; that is the C6 chain-walk's job.

## Consequences
- The signal boundary between C1 (text heuristic) and C6 (chain-walk posterior) is explicit: C1 uses prose evidence, C6 uses structural `parentId`.
- The text heuristic is noisy: a named round may be a *related* round, not the true origin. C6's posterior is designed to close that gap.
- ~49 of 242 gold rows name a resolvable round token; the live split is 193 self / 6 in-set-parent / 43 external-parent.
- The spec's "76 pipeline records carry parent language" anchor is qualitative (prose mentions of a parent) and is distinct from the ~49 records that name a concrete round id.