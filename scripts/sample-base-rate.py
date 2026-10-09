#!/usr/bin/env python3
"""Estimate the corpus base rate by S2-adjudicating a random sample.

The measured goal (issue #2, attack-plan.md E2): the 244 existing gold rows are
top-ranked jev candidates, so their ~12% fault prevalence is a *candidate*
prevalence, not a corpus prevalence. This script draws a uniform random sample
from the eligible sampling frame and adjudicates each round with S2 only.

Pipeline (built incrementally across the plan's work units):

  unit-001  frame arithmetic + seeded uniform sample manifest   (this file now)
  unit-002  S2 adjudication reuse + append + resume-on-union
  unit-003  backfill + run report (k/n, bins, Wilson 95% CI)
  unit-004  determinism + frame/gold cross-check

Definitions (issue #2):
  frame     = 5430 scan ids − 244 gold ids − missing-round-file ids = 5159
  positive  : fault-type bins (prompt-misread, stale-context, other, retrieval-noise)
  negative  : no-fault-within-round + external
  excluded  : ambiguous

The gold split lives in the committed ``dataset/rounds-labeled.jsonl`` (242 rows)
plus the two ids C1 dropped by its Q2 decision (``DROP_IDS``) — together exactly
the 244 deduped fault-pipeline ids. Keeping gold as committed artifacts only
means frame arithmetic never has to reach into the sibling ``semblr`` checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

# --- Constants ---------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# The frozen 5430-id weak corpus (scan snapshot, committed sorted copy).
#
# The scan snapshot proper lives in the sibling checkout at
# ``../semblr/temp/jev-scan-results.jsonl``; its id *set* is identical to the
# committed corpus (verified: 5430 ids, set-equal, corpus is the sorted copy).
# The frame depends only on the id set, so the committed artifact is the
# self-contained default; point ``PROMPT_QUALITY_SCAN_RESULTS`` at the snapshot
# to run against it directly.
DEFAULT_CORPUS = Path(
    os.environ.get(
        "PROMPT_QUALITY_CORPUS",
        PROJECT_ROOT / "dataset" / "prompt-corpus.jsonl",
    )
)

# The committed gold labels (242 rows = 29 positive + 205 negative + 8 excluded).
DEFAULT_GOLD = Path(
    os.environ.get(
        "PROMPT_QUALITY_GOLD",
        PROJECT_ROOT / "dataset" / "rounds-labeled.jsonl",
    )
)

# One JSON record per round id: where the full prompt text lives.
DEFAULT_ROUND_STORE = Path(
    os.environ.get(
        "PROMPT_QUALITY_ROUND_STORE",
        Path.home() / ".pi" / "agent" / "semblr" / "rounds",
    )
)

# The sample manifest emitted by the frame+sample step.
DEFAULT_MANIFEST = Path(
    os.environ.get(
        "PROMPT_QUALITY_SAMPLE_MANIFEST",
        PROJECT_ROOT / "dataset" / "base-rate-sample.json",
    )
)

# Expected size of the pinned scan snapshot (issue #2 verified anchor).
EXPECTED_CORPUS = 5430

# The 244 deduped gold ids = rounds-labeled.jsonl (242) ∪ these two.
#
# C1 dropped these by its Q2 decision (build-dataset.py ``DROP_IDS``); they are
# still inside the 5430 scan frame, so they must still be excluded here.
DROP_IDS = frozenset(
    {
        "83ebde302dd2fa4d00bed2a1a65b39f5",  # no-fault-within-round
        "91853eb0e0115771ceb582755d72a017",  # ambiguous
    }
)

# Eligible frame = 5430 − 244 gold − 29 missing-round-file = 5159 (issue #2).
EXPECTED_GOLD = 244
EXPECTED_FRAME = 5159

# Default sample size (issue #2: "sample ~100 random unscanned rounds").
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 20261009

# --- I/O helpers -------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def corpus_ids(path: str | Path = DEFAULT_CORPUS) -> list[str]:
    """The pinned weak-corpus id list (5430 ids)."""
    return [row["id"] for row in read_jsonl(path)]


def gold_ids(
    gold_path: str | Path = DEFAULT_GOLD,
    drop_ids: Iterable[str] = DROP_IDS,
) -> set[str]:
    """The 244 adjudicated gold ids: labelled rows ∪ C1's dropped ids.

    ``rounds-labeled.jsonl`` holds 242 rows (the two Q2 ids were dropped from
    it); re-adding those two gives the full 244 that must leave the frame.
    """
    return {row["id"] for row in read_jsonl(gold_path)} | set(drop_ids)


def missing_round_ids(
    ids: Iterable[str], round_store: str | Path = DEFAULT_ROUND_STORE
) -> list[str]:
    """Ids with no ``<id>.json`` in the round store, in input order."""
    store = Path(round_store)
    return [rid for rid in ids if not (store / f"{rid}.json").exists()]


# --- Frame arithmetic --------------------------------------------------------


def eligible_frame(
    corpus: Iterable[str],
    gold: set[str],
    missing: Iterable[str],
) -> list[str]:
    """The eligible sampling frame: scan ids − gold − missing-round-file ids.

    Ids are kept in corpus order so the frame is deterministic without sorting.
    The subtraction is exact: an id cannot be both gold and missing (a gold id
    is a known round; if its file vanished that is upstream drift we want to
    see, not silently absorb), so ``len(frame) == len(set(corpus)) − |gold| −
    |missing|`` over the corpus.
    """
    gold = set(gold)
    missing = set(missing)
    return [rid for rid in corpus if rid not in gold and rid not in missing]


# --- Seeded uniform sample ---------------------------------------------------


def draw_sample(
    frame: Iterable[str],
    size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Draw ``size`` distinct ids uniformly at random from ``frame``.

    ``random.Random(seed).sample`` is a uniform sample without replacement; the
    same seed over the same frame yields the same ids, which is the
    determinism obligation. Order is the draw order, not sorted, so a
    re-draw with the same seed is byte-identical.
    """
    frame = list(frame)
    if size > len(frame):
        raise ValueError(f"sample size {size} exceeds frame size {len(frame)}")
    return random.Random(seed).sample(frame, size)


def build_manifest(
    seed: int = DEFAULT_SEED,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    corpus_path: str | Path = DEFAULT_CORPUS,
    gold_path: str | Path = DEFAULT_GOLD,
    round_store: str | Path = DEFAULT_ROUND_STORE,
    drop_ids: Iterable[str] = DROP_IDS,
) -> dict:
    """Assemble the sample manifest: seed, frame arithmetic, sampled ids.

    The manifest is the durable record of *what* was sampled and *how the frame
    was derived*, so a later run can re-derive the identical sample from the
    seed alone and audit the arithmetic.

    ``drop_ids`` are C1's two Q2-dropped gold ids; they are threaded so a
    synthetic fixture can supply its own set and stay fully isolated from the
    live gold split.
    """
    corpus = corpus_ids(corpus_path)
    gold = gold_ids(gold_path, drop_ids)
    missing = missing_round_ids(corpus, round_store)
    frame = eligible_frame(corpus, gold, missing)
    sampled = draw_sample(frame, sample_size, seed)
    return {
        "seed": seed,
        "sample_size": sample_size,
        "corpus_size": len(corpus),
        "gold_size": len(gold),
        "missing_size": len(missing),
        "frame_size": len(frame),
        "missing_ids": sorted(missing),
        "sampled_ids": sampled,
    }


def write_manifest(manifest: dict, path: str | Path = DEFAULT_MANIFEST) -> Path:
    """Write the manifest deterministically (sorted keys, trailing newline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def assert_anchors(manifest: dict) -> None:
    """Fail loudly if the frame arithmetic drifts from the issue anchors.

    A drift means the corpus, the gold split, or the round store changed under
    us; the base-rate estimate would then no longer be comparable to C1.
    """
    if manifest["corpus_size"] != EXPECTED_CORPUS:
        raise SystemExit(
            f"corpus drift: expected {EXPECTED_CORPUS}, got {manifest['corpus_size']}"
        )
    if manifest["gold_size"] != EXPECTED_GOLD:
        raise SystemExit(
            f"gold drift: expected {EXPECTED_GOLD}, got {manifest['gold_size']}"
        )
    if manifest["frame_size"] != EXPECTED_FRAME:
        raise SystemExit(
            f"frame drift: expected {EXPECTED_FRAME}, got {manifest['frame_size']}"
        )

# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--rounds-dir", default=str(DEFAULT_ROUND_STORE))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = ap.parse_args(argv)

    manifest = build_manifest(
        seed=args.seed,
        sample_size=args.sample_size,
        corpus_path=args.corpus,
        gold_path=args.gold,
        round_store=args.rounds_dir,
    )
    assert_anchors(manifest)
    out = write_manifest(manifest, args.manifest)
    print(
        f"frame={manifest['frame_size']} "
        f"(corpus={manifest['corpus_size']} − gold={manifest['gold_size']} "
        f"− missing={manifest['missing_size']}) "
        f"seed={manifest['seed']} sampled={len(manifest['sampled_ids'])}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
