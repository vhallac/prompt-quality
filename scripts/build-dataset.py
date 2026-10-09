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

Source input is the sibling ``semblr`` checkout by default; override with
``PROMPT_QUALITY_FAULT_PIPELINE`` or the ``path`` argument.
"""

from __future__ import annotations

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
    records: Iterable[Record], round_store: str | Path = DEFAULT_ROUND_STORE
) -> list[Record]:
    """Attach ``prompt`` / ``prompt_status`` to every record in place.

    Per-id misses are recorded as ``unresolved`` and the walk continues; one
    missing round file never aborts the build (issue #1 extension 2a).
    """
    resolved: list[Record] = []
    for record in records:
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
    scan_ids: Iterable[str], round_store: str | Path = DEFAULT_ROUND_STORE
) -> list[Record]:
    """Build weak-corpus records from the pinned id list.

    The corpus is defined by the ids, not by whatever the live round store
    happens to contain, so ids whose round file has since drifted are emitted
    ``unresolved`` rather than dropped (issue #1 Q1).
    """
    records = [Record(id=id, bin="", raw={}) for id in scan_ids]
    return resolve_prompts(records, round_store)


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
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
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


def main() -> int:
    records = load_records()
    print(f"loaded+deduped: {len(records)} (expected {EXPECTED_DEDUPED})")
    for name, count in tally(records).items():
        print(f"  {name:24} {count}")

    gold = resolve_prompts(gold_records(records))
    attach_origins(gold)
    print(f"gold: {len(gold)} (expected {EXPECTED_GOLD})")
    for name, count in label_tally(gold).items():
        print(f"  {name:24} {count}")
    for name, count in origin_tally(gold).items():
        print(f"  origin:{name:17} {count}")

    scan_ids = read_scan_ids()
    corpus = corpus_records(scan_ids)
    print(f"corpus: {len(corpus)} (expected {EXPECTED_CORPUS})")

    report = build_report(gold, corpus)
    print(f"gold unresolved:   {len(report['gold']['unresolved'])}")
    print(f"corpus unresolved: {len(report['corpus']['unresolved'])}")
    for id_ in report["corpus"]["unresolved"]:
        print(f"  {id_}")

    paths = check_determinism(gold, corpus)
    for name, path in paths.items():
        print(f"emitted {name:6} {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
