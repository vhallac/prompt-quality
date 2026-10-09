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
import contextlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable, Iterable

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

# The S2 root-cause machinery lives in the sibling semblr checkout. It is
# imported (not shelled out to) so the LLM call has an in-process seam: a test
# swaps ``llm_request`` for a cache-backed function and runs the real S2 prompt
# assembly, chain walk, and response parsing offline.
DEFAULT_FAULT_PIPELINE = Path(
    os.environ.get(
        "PROMPT_QUALITY_FAULT_PIPELINE",
        PROJECT_ROOT.parent / "semblr" / "scripts" / "fault-pipeline.py",
    )
)

# Default adjudication transcript (one record per adjudicated round).
DEFAULT_RESULTS = Path(
    os.environ.get(
        "PROMPT_QUALITY_RESULTS",
        PROJECT_ROOT / "dataset" / "base-rate-results.jsonl",
    )
)

DEFAULT_STATE_CHARS = 12000
DEFAULT_CHAIN_CHARS = 24000
DEFAULT_LLM_MODEL = "z-ai/glm-5.3-flash"

# S2 is the only stage this script runs (issue #2: no S0/S1 ranking).
S2_STAGES = frozenset({"S2"})


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


# --- S2 adjudication reuse ---------------------------------------------------


def load_s2_module(path: str | Path = DEFAULT_FAULT_PIPELINE):
    """Import ``fault-pipeline.py`` and return the module.

    Loading by path (rather than package import) keeps the sibling script's
    hyphenated filename usable and pins exactly which copy is driven. The
    module object is also the monkeypatch target for the LLM cache seam.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"S2 machinery not found: {path}")
    spec = importlib.util.spec_from_file_location("fault_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fault_pipeline"] = module
    spec.loader.exec_module(module)
    return module


class S2Args:
    """The argument surface ``process_round``/``s2_root_cause`` read.

    Mirrors the subset of the fault-pipeline argparse namespace the S2 path
    touches, so the imported machinery runs unmodified.
    """

    def __init__(
        self,
        state_chars: int = DEFAULT_STATE_CHARS,
        chain_chars: int = DEFAULT_CHAIN_CHARS,
        llm_model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self.state_chars = state_chars
        self.chain_chars = chain_chars
        self.llm_model = llm_model


def cache_key(system: str, user: str, model: str) -> str:
    """Stable cache key for one LLM request."""
    import hashlib

    h = hashlib.sha256()
    for part in (system, user, model):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_llm_cache(path: str | Path) -> dict[str, str]:
    """Load a ``{cache_key: response_text}`` JSON cache (empty if absent)."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_llm_cache(cache: dict[str, str], path: str | Path) -> Path:
    """Persist the cache deterministically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


@contextlib.contextmanager
def cached_llm(s2_module, cache: dict[str, str]):
    """Swap ``s2_module.llm_request`` for a cache-backed stand-in.

    On a cache hit the stored response is returned without a network call; on a
    miss the real ``llm_request`` runs and its text is stored under the key.
    Restores the original callable on exit, so the seam never leaks.
    """
    original = s2_module.llm_request

    def wrapped(system: str, user: str, api_key: str, model: str, max_retries: int = 3) -> str:
        key = cache_key(system, user, model)
        if key in cache:
            return cache[key]
        text = original(system, user, api_key, model, max_retries=max_retries)
        cache[key] = text
        return text

    s2_module.llm_request = wrapped
    try:
        yield cache
    finally:
        s2_module.llm_request = original


def adjudicate_one(
    rid: str,
    rounds_dir: str | Path = DEFAULT_ROUND_STORE,
    s2_module=None,
    args: S2Args | None = None,
    api_key: str | None = None,
) -> dict:
    """Adjudicate one round with S2 only and return its result record.

    Delegates to the imported ``process_round`` with ``stages={'S2'}``: no
    S0/S1 ranking, no S3/S4 re-route. The record carries the round id, the
    S2 verdict/fault_type/root_cause, and the derived ``bin``.
    """
    if s2_module is None:
        s2_module = load_s2_module()
    if args is None:
        args = S2Args()
    return s2_module.process_round(rid, Path(rounds_dir), api_key, args, set(S2_STAGES))


# --- Results transcript + resume-on-union ------------------------------------


def load_results(path: str | Path = DEFAULT_RESULTS) -> list[dict]:
    """Read the results transcript, skipping blank/unparseable lines."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def resume_plan(manifest: dict, results: Iterable[dict]) -> list[str]:
    """Ids still to adjudicate: manifest ids minus ids already in results.

    Keyed on the **union** of manifest ids and existing result ids (issue #2 /
    C1 finding): the transcript is the source of truth for *done*, and the
    manifest is the source of truth for *wanted*. An id present in either set
    is never double-written and never re-adjudicated; order follows the
    manifest draw so a resumed run is deterministic.
    """
    done = {row["id"] for row in results if row.get("id")}
    seen: set[str] = set(done)
    todo = []
    for rid in manifest.get("sampled_ids", []):
        if rid in seen:
            continue
        seen.add(rid)
        todo.append(rid)
    return todo


def append_result(record: dict, path: str | Path = DEFAULT_RESULTS) -> bool:
    """Append one result row, refusing to duplicate an existing id.

    Returns True when the row was written, False when the id is already in the
    transcript. The guard is what makes resume-on-union safe: even if a caller
    mis-orders the todo list, an id can never be written twice.
    """
    path = Path(path)
    existing = {row["id"] for row in load_results(path) if row.get("id")}
    if record.get("id") in existing:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


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
