#!/usr/bin/env python3
"""Build the labelled prompt dataset for the prompt-quality experiment.

The pipeline: load fault-pipeline records, dedupe by id (last write wins),
drop the two Q2 ids, and derive the gold label split; resolve the full
``userPrompt`` from the round store (unresolved rows flagged, never
substituted); back-propagate the origin from S2 text; and emit both artifacts
deterministically (byte-identical re-run). Free-text fields are redacted at
emit time so no artifact can ship a credential.

Definitions (issue #1 / attack-plan.md E1):
  positive  : fault bins  -> prompt-misread, stale-context, other, retrieval-noise
  negative  : no fault within round + external
  excluded  : ambiguous

The round store keeps growing and rounds reappear in it upstream, so the
store-time facts of the build that produced the committed artifacts are a
*recorded* input, not an observation: the snapshot's ``corpus_unresolved``
ledger is what defines the unresolved rows, and the live store is consulted
only to report drift by field name (``check_store_drift``). Seed-equivalent
statement for C1: the pinned snapshot plus the committed artifacts reproduce
the identical corpus and gold bytes whether or not 15 recorded-missing rounds
have since come back.

Source input is the sibling ``semblr`` checkout by default; override with
``PROMPT_QUALITY_FAULT_PIPELINE`` or the ``path`` argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# --- Constants ---------------------------------------------------------------

DEFAULT_FAULT_PIPELINE = Path(
    os.environ.get(
        "PROMPT_QUALITY_FAULT_PIPELINE",
        Path(__file__).resolve().parents[2] / "semblr" / "temp" / "fault-pipeline.jsonl",
    )
)

# The frozen weak-label corpus: the 5430 ids recorded by the scan snapshot.
DEFAULT_SCAN_RESULTS = Path(
    os.environ.get(
        "PROMPT_QUALITY_SCAN_RESULTS",
        Path(__file__).resolve().parents[2] / "semblr" / "temp" / "jev-scan-results.jsonl",
    )
)

# Where full prompt text lives: one JSON record per round id.
DEFAULT_ROUND_STORE = Path(
    os.environ.get(
        "PROMPT_QUALITY_ROUND_STORE",
        Path.home() / ".pi" / "agent" / "semblr" / "rounds",
    )
)

# Expected size of the pinned scan snapshot (issue #1 verified anchor).
EXPECTED_CORPUS = 5430

# ``prompt_status`` values.
PROMPT_RESOLVED = "resolved"
PROMPT_UNRESOLVED = "unresolved"

# ``origin_source`` values (issue #1 extensions 5a/5b and the in-set middle).
ORIGIN_SELF = "self"
ORIGIN_IN_SET = "in-set-parent"
ORIGIN_EXTERNAL = "external-parent"

# Hex-length window for the round-id tokens the S2 text names. The store uses
# 32-char opaque ids; the S2 narrative cites them by 7- or 8-char prefix
# (e.g. ``6547ceb``, ``9566700f``) and occasionally in full (``9566700f...``).
ORIGIN_TOKEN_MIN = 6
ORIGIN_TOKEN_MAX = 32

# The record field the spec forbids substituting for a missing full prompt.
PREVIEW_FIELD = "prompt_preview"

# Output artifact names (issue #1 step 6).
GOLD_ARTIFACT = "rounds-labeled.jsonl"
CORPUS_ARTIFACT = "prompt-corpus.jsonl"
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"

# The committed artifacts are also the *record* the pinned build reads its input
# facts from: the corpus rows carry the id universe and the per-row resolution
# status, the gold rows carry the gold id set.
DEFAULT_GOLD_ARTIFACT = DEFAULT_DATASET_DIR / GOLD_ARTIFACT
DEFAULT_CORPUS_ARTIFACT = DEFAULT_DATASET_DIR / CORPUS_ARTIFACT

# The C1 build snapshot: what the round store looked like for the build that
# emitted those artifacts. Override with ``PROMPT_QUALITY_SNAPSHOT``.
SNAPSHOT_ARTIFACT = "c1-build-snapshot.json"
DEFAULT_SNAPSHOT = Path(
    os.environ.get(
        "PROMPT_QUALITY_SNAPSHOT", str(DEFAULT_DATASET_DIR / SNAPSHOT_ARTIFACT)
    )
)

# Ids dropped by the C1 Q2 decision (see .todo/findings.md and issue #1).
DROP_IDS = frozenset(
    {
        "83ebde302dd2fa4d00bed2a1a65b39f5",  # no-fault-within-round
        "91853eb0e0115771ceb582755d72a017",  # ambiguous
    }
)

FAULT_BINS = ("prompt-misread", "stale-context", "other", "retrieval-noise")
NEGATIVE_BINS = ("no-fault-within-round", "external")
EXCLUDED_BINS = ("ambiguous",)

LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
LABEL_EXCLUDED = "excluded"

# Expected post-dedupe tallies, asserted by the split tests.
#
# Post-drop gold = 242 rows = 29 positive + 205 negative + 8 excluded.
# The issue prose writes "205+3 negative": 205 already includes the 3 external
# rows (202 no-fault-within-round + 3 external), so the negative count is 205.
EXPECTED_DEDUPED = 244
EXPECTED_GOLD = 242
EXPECTED_POSITIVE = 29
EXPECTED_NEGATIVE = 205  # 202 no-fault-within-round + 3 external
EXPECTED_EXCLUDED = 8

# Resolution tallies of the *pinned snapshot*, not of the live round store.
#
# The store today resolves 15 of the 30 recorded-unresolved corpus ids: those
# rounds reappeared upstream after C1 was built (the same movement C2 records in
# ``design/decisions/009``). The recorded numbers below stay the ones this build
# emitted, so a hand-edited ledger fails naming the field it broke instead of
# quietly re-drawing the corpus.
EXPECTED_CORPUS_RESOLVED = 5400
EXPECTED_CORPUS_UNRESOLVED = 30
EXPECTED_NO_ROUND_FILE = 29
EXPECTED_EMPTY_PROMPT = 1
EXPECTED_GOLD_UNRESOLVED = 0

# The snapshot field each hand-maintained anchor governs, in check order.
SNAPSHOT_ANCHORS = (
    ("corpus_size", EXPECTED_CORPUS),
    ("gold_size", EXPECTED_GOLD),
    ("corpus_resolved_size", EXPECTED_CORPUS_RESOLVED),
    ("corpus_unresolved_size", EXPECTED_CORPUS_UNRESOLVED),
    ("no_round_file_size", EXPECTED_NO_ROUND_FILE),
    ("empty_prompt_size", EXPECTED_EMPTY_PROMPT),
    ("gold_unresolved_size", EXPECTED_GOLD_UNRESOLVED),
)


# --- Records -----------------------------------------------------------------


@dataclass
class Record:
    """A deduped fault-pipeline record plus its derived gold label."""

    id: str
    bin: str
    label: str | None = None
    raw: dict = field(default_factory=dict)
    prompt: str | None = None
    prompt_status: str | None = None
    origin_id: str | None = None
    origin_source: str | None = None

    def __post_init__(self) -> None:
        if self.label is None:
            self.label = label_for_bin(self.bin)


def label_for_bin(bin_name: str) -> str | None:
    """Map a fault-pipeline bin to a gold label.

    Returns ``None`` for bins outside the taxonomies, so callers can surface
    unknown bins loudly rather than silently mislabelling.
    """
    if bin_name in FAULT_BINS:
        return LABEL_POSITIVE
    if bin_name in NEGATIVE_BINS:
        return LABEL_NEGATIVE
    if bin_name in EXCLUDED_BINS:
        return LABEL_EXCLUDED
    return None


# --- Loading and dedupe ------------------------------------------------------


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(path: str | Path = DEFAULT_FAULT_PIPELINE) -> list[Record]:
    """Load fault-pipeline.jsonl and dedupe by id, last write wins.

    Raises ``KeyError`` if a row lacks ``id`` or ``bin``.
    """
    deduped: dict[str, Record] = {}
    for row in read_jsonl(path):
        record = Record(id=row["id"], bin=row["bin"], raw=row)
        deduped[record.id] = record  # last write wins
    return [deduped[key] for key in deduped]


def drop_ids(records: Iterable[Record], ids: Iterable[str] = DROP_IDS) -> list[Record]:
    """Remove records whose id is in ``ids``."""
    drop = set(ids)
    return [record for record in records if record.id not in drop]


def gold_records(records: Iterable[Record]) -> list[Record]:
    """The 242-row gold set: drop the two Q2 ids.

    The ``ambiguous`` rows stay in the gold file and are excluded only by
    consumers that train on labelled rows; the plan's goal memory counts gold
    as 29 positive + 205 negative + 8 excluded = 242.
    """
    return drop_ids(records)


# --- Prompt resolution -------------------------------------------------------


def load_prompt(id: str, round_store: str | Path = DEFAULT_ROUND_STORE) -> str | None:
    """Return the full ``userPrompt`` for ``id``, or ``None`` if unrecoverable.

    Unrecoverable means the round file is absent, unreadable/malformed, has no
    ``userPrompt`` field, or that field is not a non-empty string. Callers must
    mark such rows unresolved -- never substitute the truncated
    ``prompt_preview`` and never drop the row.
    """
    path = Path(round_store) / f"{id}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    prompt = record.get("userPrompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return None


def resolve_prompts(
    records: Iterable[Record],
    round_store: str | Path = DEFAULT_ROUND_STORE,
    force_unresolved: Iterable[str] | None = None,
) -> list[Record]:
    """Attach ``prompt`` / ``prompt_status`` to every record in place.

    Per-id misses are recorded as ``unresolved`` and the walk continues; one
    missing round file never aborts the build (issue #1 extension 2a).

    ``force_unresolved`` is the pinned ledger: those ids are recorded unresolved
    and the store is never consulted for them, so a round file that reappeared
    upstream cannot silently rewrite a row the snapshot says has no prompt. It
    never *fabricates* a resolution either -- an id outside the ledger still goes
    to the store, and a store miss there is still recorded as unresolved.
    Leaving it ``None`` is the fresh-draw path, used only when no snapshot is
    pinned.
    """
    ledger = set(force_unresolved) if force_unresolved is not None else None
    resolved: list[Record] = []
    for record in records:
        if ledger is not None and record.id in ledger:
            record.prompt = None
            record.prompt_status = PROMPT_UNRESOLVED
        else:
            prompt = load_prompt(record.id, round_store)
            if prompt is None:
                record.prompt = None
                record.prompt_status = PROMPT_UNRESOLVED
            else:
                record.prompt = prompt
                record.prompt_status = PROMPT_RESOLVED
        resolved.append(record)
    return resolved


def read_scan_ids(path: str | Path = DEFAULT_SCAN_RESULTS) -> list[str]:
    """The pinned weak-corpus id list, in scan order (5430 ids)."""
    return [row["id"] for row in read_jsonl(path)]


def corpus_records(
    scan_ids: Iterable[str],
    round_store: str | Path = DEFAULT_ROUND_STORE,
    force_unresolved: Iterable[str] | None = None,
) -> list[Record]:
    """Build weak-corpus records from the pinned id list.

    The corpus is defined by the ids, not by whatever the live round store
    happens to contain, so ids whose round file has since drifted are emitted
    ``unresolved`` rather than dropped (issue #1 Q1). ``force_unresolved``
    threads through to ``resolve_prompts``: on the pinned path it is the
    snapshot's recorded ledger, which is what makes a moved store unable to
    re-draw the corpus.
    """
    records = [Record(id=id, bin="", raw={}) for id in scan_ids]
    return resolve_prompts(records, round_store, force_unresolved)


def unresolved_ids(records: Iterable[Record]) -> list[str]:
    """Ids with no recoverable prompt, sorted for a deterministic report."""
    return sorted(r.id for r in records if r.prompt_status == PROMPT_UNRESOLVED)


def build_report(
    gold: Iterable[Record],
    corpus: Iterable[Record],
    round_store: str | Path = DEFAULT_ROUND_STORE,
) -> dict:
    """The run report: per-bin counts and unresolved ids for both sets."""
    gold = list(gold)
    corpus = list(corpus)
    return {
        "round_store": str(round_store),
        "gold": {
            "total": len(gold),
            "by_bin": tally(gold),
            "by_label": label_tally(gold),
            "by_origin_source": origin_tally(gold),
            "unresolved": unresolved_ids(gold),
        },
        "corpus": {
            "total": len(corpus),
            "unresolved": unresolved_ids(corpus),
        },
    }


# --- Pinned build snapshot ---------------------------------------------------
#
# C1 reads prompt text from a round store that keeps moving: of the 30 corpus ids
# the build recorded as unresolved, 15 have round files today. A build that
# re-derives that fact from the live store would rewrite those 15 rows and emit a
# different corpus -- the same disease U1 removed from C2's frame, on the corpus
# side. The snapshot is the cure: the store-time facts of the build are committed
# next to the artifacts they describe, the ledger decides resolution
# (``resolve_prompts(force_unresolved=...)``), and the live store is read only to
# *report* drift by field name.

# The fields a snapshot must carry. ``corpus_size`` … ``gold_unresolved`` are
# re-derived from the committed artifacts; the ``no_round_file_*`` and
# ``empty_prompt_*`` ledgers are the store-time classification, which no artifact
# can show (both kinds of row have a null prompt) and which the live store can no
# longer reproduce.
SNAPSHOT_FIELDS = (
    "corpus_size",
    "corpus_ids_sha256",
    "corpus_resolved_size",
    "corpus_unresolved_size",
    "corpus_unresolved",
    "no_round_file_size",
    "no_round_file_ids",
    "empty_prompt_size",
    "empty_prompt_ids",
    "gold_size",
    "gold_unresolved_size",
    "gold_unresolved",
)


class SnapshotError(RuntimeError):
    """A pinned snapshot no longer reproduces from its own recorded facts."""


class StoreDriftError(SnapshotError):
    """The live round store contradicts the snapshot where the snapshot claims."""


def _brief(value: object) -> str:
    """A short, stable rendering of a snapshot field for an error message."""
    if isinstance(value, list):
        if not value:
            return "len=0"
        return f"len={len(value)} first={value[0]!r}"
    return repr(value)


def _canonical(ids: Iterable[str]) -> list[str]:
    """The canonical ledger form: sorted and free of duplicates."""
    return sorted(set(ids))


def artifact_ids(path: str | Path = DEFAULT_CORPUS_ARTIFACT) -> list[str]:
    """The corpus id universe as recorded by a committed artifact, canonical."""
    return sorted(row["id"] for row in read_jsonl(path))


def corpus_ids_sha256(ids: Iterable[str]) -> str:
    """Digest of the canonical id universe: sorted ids, newline-joined.

    The scan snapshot this corpus was built from lives in an ephemeral sibling
    checkout; the digest is the portable record that the id universe behind the
    committed artifact is still the one the build used.
    """
    joined = "\n".join(_canonical(ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def classify_unresolved(
    records: Iterable[Record],
    round_store: str | Path = DEFAULT_ROUND_STORE,
) -> tuple[list[str], list[str]]:
    """Split unresolved ids into ``(no round file, round file unusable)``.

    Only meaningful against the store the build actually ran on: it is how a
    fresh draw fills the snapshot's classification ledgers. The ``empty_prompt``
    side covers every unusable file (absent, blank or non-string
    ``userPrompt``, malformed).
    """
    store = Path(round_store)
    no_file, unusable = [], []
    for record in sorted(
        (r for r in records if r.prompt_status == PROMPT_UNRESOLVED),
        key=lambda r: r.id,
    ):
        if (store / f"{record.id}.json").exists():
            unusable.append(record.id)
        else:
            no_file.append(record.id)
    return no_file, unusable


def build_snapshot(
    corpus: Iterable[Record],
    gold: Iterable[Record],
    no_round_file_ids: Iterable[str],
    empty_prompt_ids: Iterable[str],
) -> dict:
    """Assemble the build snapshot from the records just resolved.

    The classification ledgers are inputs rather than observations here, which is
    what lets the committed snapshot be re-recorded from another permanent record
    (the ``no_round_file`` ledger of a pinned build cross-checks against
    ``dataset/base-rate-sample.json``'s frame ledger in the test layer).
    """
    corpus = list(corpus)
    gold = list(gold)
    unresolved = unresolved_ids(corpus)
    gold_unresolved = unresolved_ids(gold)
    no_file = _canonical(no_round_file_ids)
    empty = _canonical(empty_prompt_ids)
    return {
        "corpus_size": len(corpus),
        "corpus_ids_sha256": corpus_ids_sha256(r.id for r in corpus),
        "corpus_resolved_size": len(corpus) - len(unresolved),
        "corpus_unresolved_size": len(unresolved),
        "corpus_unresolved": unresolved,
        "no_round_file_size": len(no_file),
        "no_round_file_ids": no_file,
        "empty_prompt_size": len(empty),
        "empty_prompt_ids": empty,
        "gold_size": len(gold),
        "gold_unresolved_size": len(gold_unresolved),
        "gold_unresolved": gold_unresolved,
    }


def snapshot_bytes(snapshot: dict) -> bytes:
    """The pinned serialization of a snapshot: sorted keys, trailing newline."""
    return (
        json.dumps(snapshot, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def write_snapshot(snapshot: dict, path: str | Path = DEFAULT_SNAPSHOT) -> Path:
    """Write the snapshot deterministically (sorted keys, trailing newline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(snapshot_bytes(snapshot))
    return path


def load_snapshot(path: str | Path = DEFAULT_SNAPSHOT) -> dict | None:
    """Read the snapshot at ``path``, or ``None`` when nothing is pinned there."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def derive_from_artifacts(
    corpus_path: str | Path = DEFAULT_CORPUS_ARTIFACT,
    gold_path: str | Path = DEFAULT_GOLD_ARTIFACT,
) -> dict:
    """Re-derive every artifact-derivable snapshot field from committed output.

    The artifacts carry the id universe, the per-row resolution status and the
    gold split, so those fields are checkable with no round store present at all.
    The store-time classification is not, which is why it is recorded.
    """
    corpus_rows = list(read_jsonl(corpus_path))
    gold_rows = list(read_jsonl(gold_path))
    corpus_unresolved = sorted(
        row["id"]
        for row in corpus_rows
        if row.get("prompt_status") != PROMPT_RESOLVED
    )
    gold_unresolved = sorted(
        row["id"]
        for row in gold_rows
        if row.get("prompt_status") != PROMPT_RESOLVED
    )
    return {
        "corpus_size": len(corpus_rows),
        "corpus_ids_sha256": corpus_ids_sha256(row["id"] for row in corpus_rows),
        "corpus_resolved_size": len(corpus_rows) - len(corpus_unresolved),
        "corpus_unresolved_size": len(corpus_unresolved),
        "corpus_unresolved": corpus_unresolved,
        "gold_size": len(gold_rows),
        "gold_unresolved_size": len(gold_unresolved),
        "gold_unresolved": gold_unresolved,
    }


def _ledger_problems(snapshot: dict) -> list[str]:
    """Why a recorded snapshot is not a self-consistent store-time record.

    Each problem is prefixed with the snapshot field it blames.
    """
    problems: list[str] = []
    unresolved = set(snapshot["corpus_unresolved"])
    no_file = set(snapshot["no_round_file_ids"])
    empty = set(snapshot["empty_prompt_ids"])
    for field, value in (
        ("corpus_unresolved_size", snapshot["corpus_unresolved"]),
        ("no_round_file_size", snapshot["no_round_file_ids"]),
        ("empty_prompt_size", snapshot["empty_prompt_ids"]),
        ("gold_unresolved_size", snapshot["gold_unresolved"]),
    ):
        if snapshot[field] != len(value):
            problems.append(f"{field}: recorded {snapshot[field]} vs len {len(value)}")
    if snapshot["corpus_resolved_size"] + snapshot["corpus_unresolved_size"] != (
        snapshot["corpus_size"]
    ):
        problems.append(
            "corpus_resolved_size: "
            f"recorded {snapshot['corpus_resolved_size']} + unresolved "
            f"{snapshot['corpus_unresolved_size']} != corpus_size "
            f"{snapshot['corpus_size']}"
        )
    if no_file | empty != unresolved:
        outside = sorted((no_file | empty) ^ unresolved)
        problems.append(
            "corpus_unresolved: the two classification ledgers must cover it "
            f"exactly ({len(outside)} id(s) differ first={outside[0]!r})"
        )
    if no_file & empty:
        problems.append(
            f"empty_prompt_ids: {len(no_file & empty)} id(s) classified both "
            "no-round-file and unusable-file"
        )
    for field, value in (
        ("no_round_file_ids", no_file),
        ("empty_prompt_ids", empty),
    ):
        outside = sorted(value - unresolved)
        if outside:
            problems.append(
                f"{field}: {len(outside)} id(s) not in corpus_unresolved "
                f"first={outside[0]!r}"
            )
    return problems


def reproduce_snapshot(
    snapshot: dict,
    corpus_path: str | Path = DEFAULT_CORPUS_ARTIFACT,
    gold_path: str | Path = DEFAULT_GOLD_ARTIFACT,
) -> dict:
    """Re-derive a snapshot from the committed artifacts and its own ledgers.

    No round store is read, so the pinned record is checkable on any machine. The
    artifact-derivable fields must match what the artifacts say today; the
    classification ledgers are kept as recorded after being checked for internal
    consistency (they are canonically sorted-and-distinct, so a hand-reordered
    ledger fails naming its own field). Every field that does not reproduce is
    named -- never a bare dict equality.
    """
    absent = [field for field in SNAPSHOT_FIELDS if field not in snapshot]
    if absent:
        raise SnapshotError(
            "pinned snapshot is not a complete build record — missing fields: "
            + ", ".join(absent)
        )
    rebuilt = dict(derive_from_artifacts(corpus_path, gold_path))
    rebuilt["no_round_file_ids"] = _canonical(snapshot["no_round_file_ids"])
    rebuilt["no_round_file_size"] = len(rebuilt["no_round_file_ids"])
    rebuilt["empty_prompt_ids"] = _canonical(snapshot["empty_prompt_ids"])
    rebuilt["empty_prompt_size"] = len(rebuilt["empty_prompt_ids"])

    problems = _ledger_problems(rebuilt)
    diverged = [
        f"{field}: recorded {_brief(snapshot.get(field))} vs "
        f"re-derived {_brief(rebuilt.get(field))}"
        for field in SNAPSHOT_FIELDS
        if rebuilt.get(field) != snapshot.get(field)
    ]
    # Both phases are reported together: a flipped row in the corpus artifact
    # contradicts the record on three fields *and* breaks the ledger that was
    # built from it, and a reviewer needs to see all of it to tell the two apart.
    complaints = diverged + problems
    if complaints:
        raise SnapshotError(
            "pinned snapshot does not reproduce — " + "; ".join(complaints)
        )
    return rebuilt


def assert_anchors(snapshot: dict) -> None:
    """Fail loudly if the pinned record drifts from the hand-maintained anchors.

    The anchors describe the recorded build, so a mismatch means either the
    snapshot's ledgers were edited (``*_size`` disagrees with its own list) or
    the committed artifacts no longer say what the snapshot says they do.
    """
    for field, expected in SNAPSHOT_ANCHORS:
        got = snapshot[field]
        if got != expected:
            raise SystemExit(f"{field} drift: expected {expected}, got {got}")


def check_store_drift(
    snapshot: dict,
    round_store: str | Path = DEFAULT_ROUND_STORE,
    corpus_path: str | Path = DEFAULT_CORPUS_ARTIFACT,
) -> dict:
    """Report how the live round store differs from the pinned build record.

    Reading only: the ledger the artifacts were emitted with stays authoritative,
    so a moved store is disclosed by name rather than silently re-drawn. Each
    entry blames the snapshot fields a live scan would have moved.

    Raises ``StoreDriftError`` when the difference reaches what the snapshot
    claims it can supply -- a corpus row recorded ``resolved`` whose round file is
    gone, or whose stored text no longer redacts to the committed prompt, means
    ``prompt-corpus.jsonl`` can no longer be re-derived from its own inputs. Rows
    recorded unresolved that the store can now resolve are ``recovered_ids``: they
    widen a *hypothetical* corpus and are only disclosed.
    """
    rows = {row["id"]: row for row in read_jsonl(corpus_path)}
    ledger = set(snapshot["corpus_unresolved"])
    no_file = set(snapshot["no_round_file_ids"])
    recovered: list[str] = []
    recovered_no_file: list[str] = []
    recovered_unusable: list[str] = []
    lost: list[str] = []
    changed: list[str] = []
    for rid in sorted(rows):
        prompt = load_prompt(rid, round_store)
        if rid in ledger:
            if prompt is not None:
                recovered.append(rid)
                if rid in no_file:
                    recovered_no_file.append(rid)
                else:
                    recovered_unusable.append(rid)
        elif prompt is None:
            lost.append(rid)
        elif redact(prompt) != rows[rid].get("prompt"):
            changed.append(rid)

    drifted: list[str] = []
    if recovered or lost:
        drifted += [
            "corpus_unresolved",
            "corpus_unresolved_size",
            "corpus_resolved_size",
        ]
    if recovered_no_file:
        drifted += ["no_round_file_ids", "no_round_file_size"]
    if recovered_unusable:
        drifted += ["empty_prompt_ids", "empty_prompt_size"]

    report = {
        "round_store": str(round_store),
        "drifted_fields": drifted,
        "recorded_unresolved_size": snapshot["corpus_unresolved_size"],
        "live_unresolved_size": snapshot["corpus_unresolved_size"]
        - len(recovered)
        + len(lost),
        "recovered_ids": recovered,
        "recovered_no_round_file_ids": recovered_no_file,
        "recovered_unusable_file_ids": recovered_unusable,
        "lost_round_file_ids": lost,
        "prompt_text_mismatch_ids": changed,
        "store_round_files": len(round_store_ids(round_store)),
        "recorded_corpus_size": snapshot["corpus_size"],
    }
    if lost or changed:
        first_lost = lost[0] if lost else None
        first_changed = changed[0] if changed else None
        raise StoreDriftError(
            "live round store invalidates the pinned snapshot: the committed "
            "corpus claims these rows have prompt text it can no longer re-derive; "
            f"lost_round_file_ids={len(lost)} first={first_lost!r} "
            f"prompt_text_mismatch_ids={len(changed)} first={first_changed!r} "
            f"store={round_store}"
        )
    return report


def drift_summary(report: dict) -> str:
    """One line naming what moved, and stating that nothing was re-drawn."""
    if not report["drifted_fields"]:
        return "store drift: none — the live round store matches the pinned ledger"
    return (
        "store drift (the pinned snapshot stays authoritative; the corpus was not "
        "re-drawn): fields="
        + ",".join(report["drifted_fields"])
        + f" recorded_unresolved={report['recorded_unresolved_size']}"
        + f" live_unresolved={report['live_unresolved_size']}"
        + f" recovered={len(report['recovered_ids'])}"
        + f" lost={len(report['lost_round_file_ids'])}"
        + f" text_changed={len(report['prompt_text_mismatch_ids'])}"
        + f" store_files={report['store_round_files']}"
        + f" pinned_ids={report['recorded_corpus_size']}"
    )


def report_divergences(report: dict, snapshot: dict) -> list[str]:
    """Name where a freshly built run report contradicts the pinned record.

    The pinned build reads its ids from the committed artifacts and forces the
    recorded ledger, so the report it produces must agree with the snapshot field
    for field. Anything else means the build stopped being a reproduction.
    """
    checks = (
        ("corpus_size", report["corpus"]["total"], snapshot["corpus_size"]),
        (
            "corpus_unresolved",
            report["corpus"]["unresolved"],
            snapshot["corpus_unresolved"],
        ),
        ("gold_size", report["gold"]["total"], snapshot["gold_size"]),
        (
            "gold_unresolved",
            report["gold"]["unresolved"],
            snapshot["gold_unresolved"],
        ),
    )
    return [
        f"{field}: snapshot {_brief(expected)} vs built {_brief(got)}"
        for field, got, expected in checks
        if got != expected
    ]


# --- Origin back-propagation -------------------------------------------------

# Token is any run of hex chars long enough to be a round id, bounded below so
# short words (``add``, ``dec``) do not appear, and above by the id length.
_ORIGIN_TOKEN_RE = re.compile(
    rf"\b[0-9a-f]{{{ORIGIN_TOKEN_MIN},{ORIGIN_TOKEN_MAX}}}\b"
)


def round_store_ids(round_store: str | Path = DEFAULT_ROUND_STORE) -> set[str]:
    """Every round id present in the store (the universe for external-parent)."""
    store = Path(round_store)
    if not store.is_dir():
        return set()
    return {path.stem for path in store.glob("*.json")}


def origin_tokens(record: Record) -> list[str]:
    """Round-id-looking tokens named in the record's S2 root cause + evidence.

    Order of first appearance is preserved so the choice below is deterministic.
    """
    s2 = record.raw.get("s2") or {}
    text = f"{s2.get('root_cause') or ''}\n{s2.get('evidence') or ''}"
    seen: list[str] = []
    for token in _ORIGIN_TOKEN_RE.findall(text):
        if token not in seen:
            seen.append(token)
    return seen


def _resolve_token(token: str, ids: set[str]) -> str | None:
    """Resolve a hex token to exactly one id, or ``None`` if absent/ambiguous.

    A token matches every id it prefixes. Prefer an exact id; otherwise, if the
    token prefixes exactly one id, take it. A token prefixing several ids is
    ambiguous and yields ``None`` (no evidence), rather than guessing.
    """
    if token in ids:
        return token
    matches = sorted(i for i in ids if i.startswith(token))
    if len(matches) == 1:
        return matches[0]
    return None


def derive_origin(
    record: Record, gold_ids: set[str], all_ids: set[str]
) -> tuple[str, str]:
    """Return ``(origin_id, origin_source)`` for one record.

    Reads the S2 evidence text (issue #1 step 5): the first token that resolves
    to a known round id is the origin. A token resolving into the gold set is an
    ``in-set-parent``; one resolving to any other stored round is an
    ``external-parent`` (never back-labelled into gold). With no resolvable
    origin evidence the round is its own origin (``self`` / extension 5a).
    """
    for token in origin_tokens(record):
        # A gold id is a known round even if the live store no longer holds it.
        resolved = _resolve_token(token, all_ids | gold_ids)
        if resolved is None:
            continue
        if resolved == record.id:
            return record.id, ORIGIN_SELF
        if resolved in gold_ids:
            return resolved, ORIGIN_IN_SET
        return resolved, ORIGIN_EXTERNAL
    return record.id, ORIGIN_SELF


def attach_origins(
    records: Iterable[Record],
    gold_ids: set[str] | None = None,
    all_ids: set[str] | None = None,
    round_store: str | Path = DEFAULT_ROUND_STORE,
) -> list[Record]:
    """Attach ``origin_id`` / ``origin_source`` to every record in place.

    ``gold_ids`` defaults to the ids of ``records`` themselves (origin inference
    is over the gold set); ``all_ids`` defaults to the live round store.
    """
    records = list(records)
    if gold_ids is None:
        gold_ids = {r.id for r in records}
    if all_ids is None:
        all_ids = round_store_ids(round_store)
    for record in records:
        record.origin_id, record.origin_source = derive_origin(
            record, gold_ids, all_ids
        )
    return records


def origin_tally(records: Iterable[Record]) -> dict[str, int]:
    """Count records by ``origin_source``."""
    counts: dict[str, int] = {}
    for record in records:
        key = record.origin_source or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# --- Redaction ---------------------------------------------------------------

# Credential-shaped tokens are replaced with a stable placeholder before any
# text reaches an artifact. Harvested prompts are user-pasted content and have
# carried live API keys; the dataset is published, so redaction runs on every
# build rather than as a one-off patch. Patterns are anchored on known key
# prefixes to avoid mangling ordinary prose. Placeholder keeps the prompt
# recognizable and the row count stable.
REDACTION_PLACEHOLDER = "[REDACTED]"

_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),   # OpenRouter
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),     # OpenAI-style
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),      # GitHub personal access
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),  # GitHub fine-grained
    re.compile(r"gho_[A-Za-z0-9]{20,}"),      # GitHub OAuth
    re.compile(r"AKIA[0-9A-Z]{16}"),          # AWS access key id
    re.compile(r"xox[baprs]-[A-Za-z0-9-]+"),  # Slack
)


def redact(text: str | None) -> str | None:
    """Replace credential-shaped tokens in ``text`` with a placeholder.

    ``None`` passes through unchanged so unresolved rows stay unresolved.
    """
    if text is None:
        return None
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTION_PLACEHOLDER, text)
    return text


# --- Emit --------------------------------------------------------------------

# The projection written per row. Explicit (never ``raw``) so ``prompt_preview``
# and other load-time-only fields cannot leak into an artifact (issue #1
# extension 3a), and so the on-disk schema is pinned by one list.
ROW_FIELDS = (
    "id",
    "bin",
    "label",
    "prompt",
    "prompt_status",
    "origin_id",
    "origin_source",
    "fault_type",
    "root_cause",
    "evidence",
    "timestamp",
    "s0_f",
    "s0_filter_pass",
    "s1_fault_type",
    "s1_prob",
)


def row_for(record: Record) -> dict:
    """Project a record onto ``ROW_FIELDS`` (the artifact row schema).

    Attaches the S2 verdict metadata the spec requires (``fault_type``,
    ``root_cause``, ``evidence``) plus the provenance fields, and never the
    truncated ``prompt_preview``.
    """
    s2 = record.raw.get("s2") or {}
    row = {
        "id": record.id,
        "bin": record.bin,
        "label": record.label,
        "prompt": redact(record.prompt),
        "prompt_status": record.prompt_status,
        "origin_id": record.origin_id,
        "origin_source": record.origin_source,
        "fault_type": s2.get("fault_type"),
        "root_cause": redact(s2.get("root_cause")),
        "evidence": redact(s2.get("evidence")),
        "timestamp": record.raw.get("timestamp"),
        "s0_f": record.raw.get("s0_f"),
        "s0_filter_pass": record.raw.get("s0_filter_pass"),
        "s1_fault_type": record.raw.get("s1_fault_type"),
        "s1_prob": record.raw.get("s1_prob"),
    }
    assert set(row) == set(ROW_FIELDS), "row schema drifted from ROW_FIELDS"
    return row


def serialize_rows(rows: Iterable[dict]) -> str:
    """Serialize rows to JSONL deterministically: sorted rows, sorted keys.

    Rows are ordered by id; dict keys are sorted; ``json.dumps`` renders floats
    via ``repr`` (stable). Two builds over equal input therefore differ only if
    an input value itself drifted.
    """
    ordered = sorted(rows, key=lambda r: r["id"])
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in ordered
    )


def write_artifact(path: str | Path, records: Iterable[Record]) -> Path:
    """Write one artifact: sorted rows, sorted keys, trailing newline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_rows(row_for(r) for r in records), encoding="utf-8")
    return path


def emit_datasets(
    gold: Iterable[Record],
    corpus: Iterable[Record],
    out_dir: str | Path = DEFAULT_DATASET_DIR,
) -> dict[str, Path]:
    """Emit both artifacts and return their paths."""
    out_dir = Path(out_dir)
    return {
        "gold": write_artifact(out_dir / GOLD_ARTIFACT, gold),
        "corpus": write_artifact(out_dir / CORPUS_ARTIFACT, corpus),
    }


def _first_divergent_field(a: bytes, b: bytes) -> str:
    """Name the field where two serialized artifacts first differ.

    Used for the extension 7a report so a drifting build says *what* drifted.
    Falls back to a byte-offset description when the difference is structural
    (row count, ordering) rather than a single field value.
    """
    if a == b:
        return ""
    # A difference in length only (one side is a prefix of the other, as when a
    # row's text grew) never breaks the walk, so the fallback offset is set first.
    i = min(len(a), len(b))
    for pos, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            i = pos
            break
    line_no = a[:i].count(b"\n")
    try:
        line = a.splitlines()[line_no].decode("utf-8")
        row = json.loads(line)
        rid = row.get("id", "?")
    except (IndexError, ValueError):
        return f"byte {i} (row {line_no})"
    return f"row {rid} at byte {i} (line {line_no})"


def check_determinism(
    gold: Iterable[Record],
    corpus: Iterable[Record],
    out_dir: str | Path = DEFAULT_DATASET_DIR,
) -> dict[str, Path]:
    """Emit both artifacts, re-serialize in memory, and fail on any drift.

    The build emits to ``out_dir`` then serializes the same records a second
    time and byte-compares. A mismatch means a field value or ordering is not
    reproducible; the error names the first divergent row (extension 7a) and
    the build aborts rather than shipping a drifting artifact.
    """
    paths = emit_datasets(gold, corpus, out_dir)
    checks = {"gold": gold, "corpus": corpus}
    for name, records in checks.items():
        first = paths[name].read_bytes()
        second = serialize_rows(row_for(r) for r in records).encode("utf-8")
        if first != second:
            field = _first_divergent_field(first, second)
            raise RuntimeError(
                f"non-deterministic output in {paths[name].name}: {field}"
            )
    return paths


# --- Reporting ---------------------------------------------------------------


def tally(records: Iterable[Record]) -> dict[str, int]:
    """Count records by bin, sorted by descending count then bin name."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record.bin] = counts.get(record.bin, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def label_tally(records: Iterable[Record]) -> dict[str, int]:
    """Count records by derived gold label."""
    counts: dict[str, int] = {}
    for record in records:
        key = record.label or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def check_committed_bytes(
    out_dir: str | Path = DEFAULT_DATASET_DIR,
    gold: Iterable[Record] = (),
    corpus: Iterable[Record] = (),
) -> None:
    """Stop before writing unless the build renders to the bytes already there.

    The C1 analogue of the pin's byte gate: a pinned build claims to reproduce
    the committed artifacts, so a hand-reformatted or semantically different
    rendering halts the run -- naming the first divergent row -- rather than
    being silently normalised over the record it is supposed to protect.
    """
    out_dir = Path(out_dir)
    for name, records, artifact in (
        ("gold", gold, GOLD_ARTIFACT),
        ("corpus", corpus, CORPUS_ARTIFACT),
    ):
        target = out_dir / artifact
        if not target.exists():
            continue
        committed = target.read_bytes()
        rendered = serialize_rows(row_for(r) for r in records).encode("utf-8")
        if rendered != committed:
            raise SystemExit(
                f"pinned build would change {target}: "
                f"{_first_divergent_field(committed, rendered)}; the recorded "
                "snapshot no longer matches its own bytes, so nothing was written"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(DEFAULT_FAULT_PIPELINE))
    ap.add_argument(
        "--scan-results",
        default=None,
        help="scan snapshot to take the id universe from on a fresh draw; refused "
        "when a snapshot is pinned, because the committed corpus is then the record",
    )
    ap.add_argument("--rounds-dir", default=str(DEFAULT_ROUND_STORE))
    ap.add_argument("--out-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    ap.add_argument(
        "--redraw",
        action="store_true",
        help="derive the resolution ledger from a live round-store scan; refuses to "
        "overwrite a pinned snapshot, so a new build needs a new --snapshot path",
    )
    args = ap.parse_args(argv)

    pinned = load_snapshot(args.snapshot)
    if pinned is not None and args.scan_results is not None:
        raise SystemExit(
            "--scan-results names a scan snapshot, but a snapshot is pinned at "
            f"{args.snapshot}: the id universe of a pinned build is the corpus "
            f"artifact it reproduces ({args.out_dir}/{CORPUS_ARTIFACT}), checked by "
            "digest. A new scan is a new build -- point --snapshot at a new path "
            "and use --redraw."
        )
    scan_results = args.scan_results or str(DEFAULT_SCAN_RESULTS)
    corpus_artifact = Path(args.out_dir) / CORPUS_ARTIFACT
    gold_artifact = Path(args.out_dir) / GOLD_ARTIFACT
    records = load_records(args.source)
    print(f"loaded+deduped: {len(records)} (expected {EXPECTED_DEDUPED})")
    for name, count in tally(records).items():
        print(f"  {name:24} {count}")

    if pinned is None:
        # Fresh-draw path: nothing is pinned, so the ledger is whatever the live
        # store says today, and the snapshot records that observation.
        if not Path(scan_results).exists():
            raise SystemExit(
                f"no pinned snapshot at {args.snapshot} and no scan snapshot at "
                f"{scan_results}: there is no id universe to build the corpus from"
            )
        gold = resolve_prompts(gold_records(records), args.rounds_dir)
        attach_origins(gold, round_store=args.rounds_dir)
        corpus = corpus_records(read_scan_ids(scan_results), args.rounds_dir)
        no_file, unusable = classify_unresolved(corpus, args.rounds_dir)
        snapshot = build_snapshot(corpus, gold, no_file, unusable)
        build_source = "live round store scan (nothing pinned at --snapshot)"
    else:
        if args.redraw:
            raise SystemExit(
                f"--redraw refuses to overwrite the pinned snapshot at "
                f"{args.snapshot}: the recorded ledger is the one the committed "
                "artifacts were emitted with. A corpus resolved against today's "
                "store is a new build -- new --snapshot path, new artifacts, and a "
                "re-derivation of every label that consumes them."
            )
        if not corpus_artifact.exists() or not gold_artifact.exists():
            raise SystemExit(
                f"a snapshot is pinned at {args.snapshot} but the artifacts it records "
                f"are not in {args.out_dir}: a pinned build reproduces them, it does "
                "not create them -- pass the --out-dir they live in, or --redraw with "
                "a new --snapshot path for a genuinely fresh build"
            )
        snapshot = reproduce_snapshot(pinned, corpus_artifact, gold_artifact)
        assert_anchors(snapshot)
        if Path(args.rounds_dir).exists():
            print(
                drift_summary(
                    check_store_drift(
                        snapshot,
                        round_store=args.rounds_dir,
                        corpus_path=corpus_artifact,
                    )
                )
            )
        else:
            print(
                f"store drift: not checked — no round store at {args.rounds_dir}; "
                "the pinned snapshot stays authoritative and no prompt text can be "
                "re-derived from here"
            )
        gold = resolve_prompts(
            gold_records(records), args.rounds_dir, snapshot["gold_unresolved"]
        )
        attach_origins(gold, round_store=args.rounds_dir)
        corpus = corpus_records(
            artifact_ids(corpus_artifact),
            args.rounds_dir,
            snapshot["corpus_unresolved"],
        )
        build_source = (
            "pinned snapshot (the live round store did not define the ledger)"
        )

    report = build_report(gold, corpus, args.rounds_dir)
    print(f"gold: {len(gold)} (expected {EXPECTED_GOLD})")
    for name, count in label_tally(gold).items():
        print(f"  {name:24} {count}")
    for name, count in origin_tally(gold).items():
        print(f"  origin:{name:17} {count}")
    print(f"corpus: {len(corpus)} (expected {EXPECTED_CORPUS})")
    print(f"gold unresolved:   {len(report['gold']['unresolved'])}")
    print(f"corpus unresolved: {len(report['corpus']['unresolved'])}")
    for id_ in report["corpus"]["unresolved"]:
        print(f"  {id_}")
    print(
        f"corpus={snapshot['corpus_size']} "
        f"resolved={snapshot['corpus_resolved_size']} "
        f"unresolved={snapshot['corpus_unresolved_size']} "
        f"(no_round_file={snapshot['no_round_file_size']} "
        f"empty_prompt={snapshot['empty_prompt_size']}) "
        f"gold={snapshot['gold_size']} source={build_source}"
    )

    if pinned is not None:
        diverged = report_divergences(report, snapshot)
        if diverged:
            raise SystemExit(
                "built report contradicts the pinned snapshot — " + "; ".join(diverged)
            )
        check_committed_bytes(args.out_dir, gold, corpus)
        snapshot_out = Path(args.snapshot)
    else:
        snapshot_out = None

    paths = check_determinism(gold, corpus, args.out_dir)
    for name, path in paths.items():
        print(f"emitted {name:6} {path} ({path.stat().st_size} bytes)")
    if snapshot_out is None:
        snapshot_out = write_snapshot(snapshot, args.snapshot)
    print(f"snapshot: {snapshot_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
