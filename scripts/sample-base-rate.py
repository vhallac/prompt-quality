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

The frame's missing-round-file term is a *recorded* fact: the manifest's
``missing_ids`` ledger. The live round store keeps growing, so deriving the
frame from it would silently redraw the sample on a different population; the
store is now consulted only to report drift (``check_store_drift``), never to
re-derive ``sampled_ids``. Seed + pinned snapshot reproduce the identical
sample on any machine, with or without a round store present.
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
#
# These anchor the *pinned snapshot*, not the live round store. The store today
# reports 14 absent ids and a 5174-entry frame, because 15 of the 29 recorded
# rounds were recovered upstream after C2 was drawn; the recorded frame stays the
# sample's frame. ``EXPECTED_MISSING`` exists so a hand-edited ``missing_ids``
# ledger fails naming the field it broke instead of changing the frame quietly.
EXPECTED_GOLD = 244
EXPECTED_MISSING = 29
EXPECTED_FRAME = 5159

# The manifest field each hand-maintained anchor governs, in check order.
FRAME_ANCHORS = (
    ("corpus_size", EXPECTED_CORPUS),
    ("gold_size", EXPECTED_GOLD),
    ("missing_size", EXPECTED_MISSING),
    ("frame_size", EXPECTED_FRAME),
)

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

# Bin classification (issue #2: reuses C1's fault definition).
POSITIVE_BINS = frozenset(
    {"prompt-misread", "stale-context", "other", "retrieval-noise"}
)
NEGATIVE_BINS = frozenset({"no-fault-within-round", "external"})
EXCLUDED_BINS = frozenset({"ambiguous"})

# Minimum adjudicated rows before a CI may be reported (issue #2 extension 6a).
MIN_ROWS = 100


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
    missing_ids: Iterable[str] | None = None,
) -> dict:
    """Assemble the sample manifest: seed, frame arithmetic, sampled ids.

    The manifest is the durable record of *what* was sampled and *how the frame
    was derived*, so a later run can re-derive the identical sample from the
    seed alone and audit the arithmetic.

    ``drop_ids`` are C1's two Q2-dropped gold ids; they are threaded so a
    synthetic fixture can supply its own set and stay fully isolated from the
    live gold split.

    ``missing_ids`` is the recorded ledger: when supplied it *replaces* the
    round-store scan, so the frame is re-derived from what was absent at build
    time rather than from what the live store happens to hold now. Leaving it
    ``None`` is the fresh-draw path, used only when there is no snapshot to pin.
    """
    corpus = corpus_ids(corpus_path)
    gold = gold_ids(gold_path, drop_ids)
    missing = (
        missing_round_ids(corpus, round_store)
        if missing_ids is None
        else sorted(set(missing_ids))
    )
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


def manifest_bytes(manifest: dict) -> bytes:
    """The pinned serialization of a manifest: sorted keys, trailing newline."""
    return (
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def write_manifest(manifest: dict, path: str | Path = DEFAULT_MANIFEST) -> Path:
    """Write the manifest deterministically (sorted keys, trailing newline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(manifest_bytes(manifest))
    return path


# --- Pinned frame snapshot ----------------------------------------------------

# The fields a snapshot must re-derive from its own recorded store facts. Named
# in one place so a reproduction failure can say which of them moved.
SNAPSHOT_FIELDS = (
    "seed",
    "sample_size",
    "corpus_size",
    "gold_size",
    "missing_size",
    "frame_size",
    "missing_ids",
    "sampled_ids",
)


class SnapshotError(RuntimeError):
    """A pinned snapshot no longer reproduces from its own recorded facts."""


class StoreDriftError(SnapshotError):
    """The live round store contradicts the snapshot where the sample cares."""


def _brief(value: object) -> str:
    """A short, stable rendering of a snapshot field for an error message."""
    if isinstance(value, list):
        if not value:
            return "len=0"
        return f"len={len(value)} first={value[0]!r}"
    return repr(value)


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict | None:
    """Read the snapshot at ``path``, or ``None`` when nothing is pinned there."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def reproduce_snapshot(
    snapshot: dict,
    corpus_path: str | Path = DEFAULT_CORPUS,
    gold_path: str | Path = DEFAULT_GOLD,
    drop_ids: Iterable[str] = DROP_IDS,
) -> dict:
    """Re-derive a manifest from the snapshot's own recorded ledger.

    The frame's three terms are all pinned inputs: the corpus id list and the
    gold split from the committed artifacts, the absent-round-file ids from
    ``snapshot["missing_ids"]``. No round store is read, so the same seed over
    the same snapshot yields the identical sample on any machine — which is the
    determinism obligation, now independent of a live store that keeps moving.

    Every ``SNAPSHOT_FIELDS`` entry is compared, and the error names each field
    that does not reproduce instead of failing on a bare equality.
    """
    absent = [field for field in SNAPSHOT_FIELDS if field not in snapshot]
    if absent:
        raise SnapshotError(
            "pinned snapshot is not a complete frame record — missing fields: "
            + ", ".join(absent)
        )
    rebuilt = build_manifest(
        seed=snapshot["seed"],
        sample_size=snapshot["sample_size"],
        corpus_path=corpus_path,
        gold_path=gold_path,
        drop_ids=drop_ids,
        missing_ids=snapshot["missing_ids"],
    )
    diverged = [
        f"{field}: recorded {_brief(snapshot.get(field))} vs "
        f"re-derived {_brief(rebuilt.get(field))}"
        for field in SNAPSHOT_FIELDS
        if rebuilt.get(field) != snapshot.get(field)
    ]
    if diverged:
        raise SnapshotError(
            "pinned snapshot does not reproduce — " + "; ".join(diverged)
        )
    return rebuilt


def check_store_drift(
    snapshot: dict,
    round_store: str | Path = DEFAULT_ROUND_STORE,
    corpus_path: str | Path = DEFAULT_CORPUS,
    gold_path: str | Path = DEFAULT_GOLD,
    drop_ids: Iterable[str] = DROP_IDS,
) -> dict:
    """Report how the live round store differs from the pinned ledger.

    Reading only: the frame the sample was drawn from stays the recorded one, so
    a moved store is disclosed by name rather than silently redrawn. ``drifted_fields``
    lists the snapshot fields the live store would have produced differently.

    Raises ``StoreDriftError`` when the difference reaches the recorded sample —
    a sampled id whose round file is gone can no longer be re-adjudicated, so the
    snapshot is no longer honest about what it can reproduce.
    """
    corpus = corpus_ids(corpus_path)
    gold = gold_ids(gold_path, drop_ids)
    recorded = set(snapshot["missing_ids"])
    live_missing = set(missing_round_ids(corpus, round_store))
    recovered = sorted(recorded - live_missing)
    newly_missing = sorted(live_missing - recorded)
    sampled_missing = missing_round_ids(snapshot["sampled_ids"], round_store)
    live_frame = len(eligible_frame(corpus, gold, sorted(live_missing)))

    drifted: list[str] = []
    if snapshot["missing_size"] != len(live_missing):
        drifted.append("missing_size")
    if recovered or newly_missing:
        drifted.append("missing_ids")
    if snapshot["frame_size"] != live_frame:
        drifted.append("frame_size")

    report = {
        "round_store": str(round_store),
        "drifted_fields": drifted,
        "recorded_missing_size": snapshot["missing_size"],
        "live_missing_size": len(live_missing),
        "recovered_ids": recovered,
        "newly_missing_ids": newly_missing,
        "sampled_missing_ids": sampled_missing,
        "pinned_frame_size": snapshot["frame_size"],
        "live_frame_size": live_frame,
    }
    if sampled_missing:
        raise StoreDriftError(
            "live round store invalidates the pinned sample: sampled_missing_ids "
            f"({len(sampled_missing)}) first={sampled_missing[0]!r} "
            f"store={round_store}"
        )
    return report


def drift_summary(report: dict) -> str:
    """One line naming what moved, and stating that nothing was redrawn."""
    if not report["drifted_fields"]:
        return "store drift: none — the live round store matches the pinned ledger"
    return (
        "store drift (the pinned snapshot stays authoritative; the frame was not "
        "redrawn): fields="
        + ",".join(report["drifted_fields"])
        + f" recorded_missing={report['recorded_missing_size']}"
        + f" live_missing={report['live_missing_size']}"
        + f" recovered={len(report['recovered_ids'])}"
        + f" newly_missing={len(report['newly_missing_ids'])}"
        + f" frame={report['pinned_frame_size']}->{report['live_frame_size']}"
    )


def assert_anchors(manifest: dict) -> None:
    """Fail loudly if the frame arithmetic drifts from the pinned anchors.

    The anchors describe the pinned snapshot. On the live-scan path a mismatch
    means the corpus, the gold split or the round store moved under the frame —
    the base-rate estimate would no longer be comparable to C1, so a new draw
    needs a new stage with its own recorded numbers, not a silent redraw. On the
    snapshot path a mismatch means the recorded ledger was edited.
    """
    for field, expected in FRAME_ANCHORS:
        got = manifest[field]
        if got != expected:
            raise SystemExit(f"{field} drift: expected {expected}, got {got}")


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


class AdjudicationUnavailable(RuntimeError):
    """A round could not be adjudicated because the LLM gave no usable reply.

    Raised when the upstream machinery crashes on a null/empty completion (the
    provider can return ``choices[0].message.content == null``). A transient
    provider hiccup must not abort a long paid pass: ``run_adjudication``
    treats this like a missing round file — drop the id, record it as
    unresolved, and let backfill find a replacement.
    """


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

    A null/unparseable completion raises :class:`AdjudicationUnavailable`
    rather than propagating an ``AttributeError`` from deep inside the
    imported parser, so one bad reply cannot kill a paid pass.
    """
    if s2_module is None:
        s2_module = load_s2_module()
    if args is None:
        args = S2Args()
    try:
        return s2_module.process_round(
            rid, Path(rounds_dir), api_key, args, set(S2_STAGES)
        )
    except (TypeError, AttributeError, ValueError) as exc:
        # The imported parser calls ``text.strip()`` and ``json.loads`` on the
        # completion; a null completion surfaces as one of these. Re-raise as
        # an explicit "this round is unresolved" signal.
        raise AdjudicationUnavailable(f"{rid}: unusable LLM reply ({exc})") from exc


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


# --- Run report + backfill ---------------------------------------------------


def classify_bin(bin_name: str) -> str:
    """Map a bin name to ``positive`` | ``negative`` | ``excluded``.

    Reuses C1's fault definition (issue #2): fault-type bins are positive;
    ``no-fault-within-round`` and ``external`` are negative; ``ambiguous`` is
    excluded from the denominator. An unknown bin is treated as ``ambiguous``
    rather than silently counted, so vocabulary drift surfaces instead of
    skewing the rate.
    """
    if bin_name in POSITIVE_BINS:
        return "positive"
    if bin_name in NEGATIVE_BINS:
        return "negative"
    return "excluded"


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than Wald (issue #2): better small-``n`` coverage, and it
    stays inside [0, 1] for a zero count. ``Z`` defaults to the 95% normal
    quantile. ``n == 0`` has no estimable rate, so it is rejected.
    """
    if n <= 0:
        raise ValueError("wilson_interval needs n > 0")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} out of range for n={n}")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    low, high = centre - margin, centre + margin
    # At a boundary proportion the exact Wilson endpoint is 0 or 1; the formula
    # lands a few ulps short, so pin it to keep n=0/k=n reports clean.
    if k == 0:
        low = 0.0
    if k == n:
        high = 1.0
    return (max(0.0, low), min(1.0, high))


def summarize(rows: Iterable[dict], unresolved: Iterable[str] = ()) -> dict:
    """Tally adjudicated rows into the run report's counts and rate.

    ``bin`` is read from the record; a record without one is tallied under
    ``ambiguous`` (the pipeline's own fallback), never dropped. The fault rate
    is ``k / (positive + negative)`` — ``ambiguous`` is excluded from the
    denominator, per C1's rule.
    """
    bin_counts: dict[str, int] = {}
    class_counts = {"positive": 0, "negative": 0, "excluded": 0}
    for row in rows:
        bin_name = row.get("bin") or "ambiguous"
        bin_counts[bin_name] = bin_counts.get(bin_name, 0) + 1
        class_counts[classify_bin(bin_name)] += 1

    n = class_counts["positive"] + class_counts["negative"]
    k = class_counts["positive"]
    low, high = wilson_interval(k, n) if n > 0 else (0.0, 0.0)
    return {
        "n": sum(bin_counts.values()),
        "denominator": n,
        "k": k,
        "fault_rate": (k / n) if n > 0 else 0.0,
        "ci_low": low,
        "ci_high": high,
        "bins": dict(sorted(bin_counts.items())),
        "positive": class_counts["positive"],
        "negative": class_counts["negative"],
        "ambiguous": bin_counts.get("ambiguous", 0),
        "excluded": class_counts["excluded"],
        "external": bin_counts.get("external", 0),
        "unresolved": sorted(set(unresolved)),
    }


def build_report(
    rows: Iterable[dict],
    manifest: dict | None = None,
    unresolved: Iterable[str] = (),
    min_rows: int = MIN_ROWS,
) -> dict:
    """Assemble the run report, failing loudly on a shortfall.

    Issue #2 extension 6a: fewer than ``min_rows`` adjudicated rows must abort
    with the shortfall and the unresolved list, never a CI over too few rows.
    The report also carries the frame arithmetic so the number is auditable.
    """
    rows = list(rows)
    report = summarize(rows, unresolved)
    if report["n"] < min_rows:
        raise SystemExit(
            f"shortfall: adjudicated {report['n']} rows, need {min_rows}; "
            f"unresolved={report['unresolved']}"
        )
    if manifest is not None:
        report["frame"] = {
            "corpus_size": manifest.get("corpus_size"),
            "gold_size": manifest.get("gold_size"),
            "missing_size": manifest.get("missing_size"),
            "frame_size": manifest.get("frame_size"),
            "sample_size": manifest.get("sample_size"),
            "seed": manifest.get("seed"),
        }
    return report


def backfill_ids(
    manifest: dict,
    results: Iterable[dict],
    unresolved: Iterable[str] = (),
    corpus_path: str | Path = DEFAULT_CORPUS,
    gold_path: str | Path = DEFAULT_GOLD,
    round_store: str | Path = DEFAULT_ROUND_STORE,
    drop_ids: Iterable[str] = DROP_IDS,
) -> list[str]:
    """Draw replacement ids for sampled ids with no usable result.

    Issue #2 extension 1a: an unresolvable sampled id is dropped and replaced
    by the next eligible id, so the adjudicated count still reaches 100. Two
    kinds of id are unresolvable: those with no round file (``missing_ids`` on
    the manifest) and those the live pass saw fail (``unresolved`` — e.g. a
    round whose LLM reply was unusable). Both need a replacement, so the
    caller supplies the live unresolved set and it is unioned with the
    manifest's missing set.

    The replacements come from the full eligible frame (a superset of the
    sample) and exclude every id already sampled, adjudicated, or drawn as a
    replacement, so no id is used twice. Order is deterministic given the
    manifest's seed.
    """
    missing = set(manifest.get("missing_ids", []))
    missing |= set(unresolved)
    sampled = manifest.get("sampled_ids", [])
    unresolved = [rid for rid in sampled if rid in missing]
    if not unresolved:
        return []

    corpus = corpus_ids(corpus_path)
    gold = gold_ids(gold_path, drop_ids)
    still_missing = missing_round_ids(corpus, round_store)
    frame = eligible_frame(corpus, gold, still_missing)
    used = set(sampled) | {row["id"] for row in results if row.get("id")}
    pool = [rid for rid in frame if rid not in used]
    if len(pool) < len(unresolved):
        raise SystemExit(
            f"cannot backfill {len(unresolved)} unresolved ids: pool has {len(pool)}"
        )
    return draw_sample(pool, len(unresolved), manifest.get("seed", DEFAULT_SEED))


def run_adjudication(
    manifest: dict,
    results: Iterable[dict] = (),
    adjudicate: Callable[[str], dict] | None = None,
    append: Callable[[dict], bool] | None = None,
    backfill: Callable[[dict, Iterable[dict], Iterable[str]], list[str]] | None = None,
    min_rows: int = MIN_ROWS,
    max_rounds: int = 5,
) -> dict:
    """Drive adjudication over the manifest, backfilling to reach ``min_rows``.

    Walks the resume plan (manifest minus already-done ids), adjudicates each
    id, and appends the result. Ids whose round file is missing — or whose
    adjudication raised :class:`AdjudicationUnavailable` — are dropped and
    replaced by drawing from the remaining eligible pool, repeating until the
    adjudicated count reaches ``min_rows`` or the backfill pool dries up
    (issue #2 extension 1a). The live unresolved set is passed to ``backfill``
    so a failed replacement is itself replaced. Returns the final run report.

    ``adjudicate``, ``append``, and ``backfill`` are injectable so the loop is
    exercised fully offline; the defaults wire the real S2 adapter, transcript,
    and frame arithmetic.
    """
    adjudicate = adjudicate or (lambda rid: adjudicate_one(rid))
    append = append or append_result
    backfill = backfill or backfill_ids

    all_results = list(results)
    unresolved: list[str] = []
    todo = resume_plan(manifest, all_results)
    for _ in range(max_rounds):
        for rid in todo:
            try:
                record = adjudicate(rid)
            except (FileNotFoundError, AdjudicationUnavailable):
                unresolved.append(rid)
                continue
            if append(record):
                all_results.append(record)

        done = {row["id"] for row in all_results}
        if len(done) >= min_rows:
            break
        replacements = backfill(manifest, all_results, unresolved)
        todo = [rid for rid in replacements if rid not in done]
        if not todo:
            break

    return build_report(all_results, manifest=manifest, unresolved=unresolved, min_rows=min_rows)


# --- CLI ---------------------------------------------------------------------


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Resolve the OpenRouter key: explicit flag, then ``OPENROUTER_API_KEY``.

    The key is never logged or persisted; only its presence is reported.
    """
    return explicit or os.environ.get("OPENROUTER_API_KEY") or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--rounds-dir", default=str(DEFAULT_ROUND_STORE))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument(
        "--out",
        default=None,
        help="adjudicated results transcript (default: --results / DEFAULT_RESULTS)",
    )
    ap.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
        help="where adjudicated rows are appended (resume reads this too)",
    )
    ap.add_argument(
        "--min-rows",
        type=int,
        default=MIN_ROWS,
        help="minimum adjudicated rows before a report is emitted",
    )
    ap.add_argument(
        "--cache",
        default=None,
        help="optional {cache_key: response} JSON file to reuse/save LLM calls",
    )
    ap.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter key (defaults to $OPENROUTER_API_KEY)",
    )
    ap.add_argument(
        "--manifest-only",
        action="store_true",
        help="only rebuild/write the sample manifest (no adjudication)",
    )
    ap.add_argument(
        "--redraw",
        action="store_true",
        help="derive the frame from a live round-store scan; refuses to overwrite a "
        "pinned snapshot, so a new draw needs a new --manifest path",
    )
    args = ap.parse_args(argv)

    snapshot = load_manifest(args.manifest)
    if snapshot is None:
        # Fresh-draw path: nothing is pinned at this manifest, so the frame is
        # scanned from the live store and assert_anchors is what makes a moved
        # store fail loudly instead of silently drawing a different sample.
        manifest = build_manifest(
            seed=args.seed,
            sample_size=args.sample_size,
            corpus_path=args.corpus,
            gold_path=args.gold,
            round_store=args.rounds_dir,
        )
        frame_source = "live store scan (nothing pinned at the manifest path)"
    else:
        if args.redraw:
            raise SystemExit(
                f"--redraw refuses to overwrite the pinned snapshot at {args.manifest}: "
                "the recorded frame is the sample that was adjudicated. A bigger frame "
                "is a new stage — new seed, new --manifest path."
            )
        if args.seed != snapshot["seed"] or args.sample_size != snapshot["sample_size"]:
            raise SystemExit(
                "seed/sample_size conflict with the pinned snapshot: it was drawn with "
                f"seed={snapshot['seed']} sample_size={snapshot['sample_size']}, "
                f"this run asked for seed={args.seed} sample_size={args.sample_size}. "
                "A new draw is a new stage: pass --redraw with a new --manifest path."
            )
        manifest = reproduce_snapshot(
            snapshot,
            corpus_path=args.corpus,
            gold_path=args.gold,
        )
        frame_source = "pinned snapshot (the live round store did not draw the sample)"

    assert_anchors(manifest)

    if snapshot is None:
        manifest_out = write_manifest(manifest, args.manifest)
    else:
        drift = check_store_drift(
            manifest,
            round_store=args.rounds_dir,
            corpus_path=args.corpus,
            gold_path=args.gold,
        )
        print(drift_summary(drift))
        existing = Path(args.manifest).read_bytes()
        if existing != manifest_bytes(manifest):
            raise SystemExit(
                f"reproducing the pinned snapshot would change {args.manifest}; "
                "the recorded sample no longer matches its own bytes, so nothing was "
                "written"
            )
        manifest_out = Path(args.manifest)

    print(
        f"frame={manifest['frame_size']} "
        f"(corpus={manifest['corpus_size']} − gold={manifest['gold_size']} "
        f"− missing={manifest['missing_size']}) "
        f"seed={manifest['seed']} sampled={len(manifest['sampled_ids'])} "
        f"source={frame_source}"
    )
    print(f"manifest: {manifest_out}")
    if args.manifest_only:
        return 0

    # ``--out`` is the plan's verify-command flag for the results transcript.
    results_path = Path(args.out or args.results)
    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise SystemExit(
            "no OpenRouter key: pass --api-key or set OPENROUTER_API_KEY"
        )

    existing = load_results(results_path)
    s2_module = load_s2_module()
    cache: dict[str, str] = {}
    cache_path = Path(args.cache) if args.cache else None
    if cache_path is not None:
        cache = load_llm_cache(cache_path)

    def adjudicate(rid: str) -> dict:
        return adjudicate_one(rid, s2_module=s2_module, api_key=api_key)

    def append(rec: dict) -> bool:
        written = append_result(rec, results_path)
        # Persist the cache after every adjudicated row: the pass is paid and
        # long, so a crash or kill must never discard completed LLM calls.
        if cache_path is not None:
            save_llm_cache(cache, cache_path)
        return written

    with cached_llm(s2_module, cache):
        report = run_adjudication(
            manifest,
            results=existing,
            adjudicate=adjudicate,
            append=append,
            min_rows=args.min_rows,
        )
        if cache_path is not None:
            save_llm_cache(cache, cache_path)
            print(f"cache: {len(cache)} entries → {cache_path}")

    print(
        f"adjudicated n={report['n']} rows "
        f"(positive={report['positive']} negative={report['negative']} "
        f"ambiguous={report['ambiguous']})"
    )
    print(
        f"fault_rate={report['fault_rate']:.4f} "
        f"(95% CI {report['ci_low']:.4f}–{report['ci_high']:.4f})"
    )
    if report["unresolved"]:
        print(f"unresolved={len(report['unresolved'])} ids")
    print(f"wrote {results_path}")
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
