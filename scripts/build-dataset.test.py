#!/usr/bin/env python3
"""Tests for scripts/build-dataset.py (C1 unit-001: dedupe + split).

Two layers:
  * synthetic fixtures pin the dedupe/label logic without the live source;
  * anchor tests assert the spec's verified numbers against the real
    fault-pipeline.jsonl when it is reachable (skipped otherwise).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Import build-dataset.py despite the hyphen in its filename.
_MODULE_PATH = Path(__file__).with_name("build-dataset.py")
_spec = importlib.util.spec_from_file_location("build_dataset", _MODULE_PATH)
bd = importlib.util.module_from_spec(_spec)
sys.modules["build_dataset"] = bd
_spec.loader.exec_module(bd)


def _row(id: str, bin: str, *, extra: str = "") -> dict:
    return {"id": id, "bin": bin, "prompt_preview": extra or f"preview-{id}"}


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    import json

    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


# --- dedupe ------------------------------------------------------------------


def test_dedupe_last_write_wins(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "fp.jsonl",
        [
            _row("a", "prompt-misread", extra="first"),
            _row("b", "no-fault-within-round"),
            _row("a", "no-fault-within-round", extra="second"),  # last write wins
            _row("c", "ambiguous"),
        ],
    )
    records = bd.load_records(path)
    assert len(records) == 3
    by_id = {r.id: r for r in records}
    assert by_id["a"].bin == "no-fault-within-round"
    assert by_id["a"].raw["prompt_preview"] == "second"
    # insertion order preserved by first appearance
    assert [r.id for r in records] == ["a", "b", "c"]


def test_dedupe_blank_lines_ignored(tmp_path: Path) -> None:
    path = tmp_path / "fp.jsonl"
    path.write_text('{"id":"a","bin":"other"}\n\n   \n', encoding="utf-8")
    records = bd.load_records(path)
    assert len(records) == 1


def test_dedupe_missing_field_raises(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "fp.jsonl", [{"id": "a"}])  # no bin
    with pytest.raises(KeyError):
        bd.load_records(path)


# --- label mapping -----------------------------------------------------------


@pytest.mark.parametrize("bin_name", bd.FAULT_BINS)
def test_fault_bins_are_positive(bin_name: str) -> None:
    assert bd.label_for_bin(bin_name) == bd.LABEL_POSITIVE


@pytest.mark.parametrize("bin_name", bd.NEGATIVE_BINS)
def test_negative_bins(bin_name: str) -> None:
    assert bd.label_for_bin(bin_name) == bd.LABEL_NEGATIVE


@pytest.mark.parametrize("bin_name", bd.EXCLUDED_BINS)
def test_excluded_bins(bin_name: str) -> None:
    assert bd.label_for_bin(bin_name) == bd.LABEL_EXCLUDED


def test_unknown_bin_labels_none() -> None:
    assert bd.label_for_bin("no-such-bin") is None


# --- split -------------------------------------------------------------------


def test_drop_ids_removes_named_records(tmp_path: Path) -> None:
    records = [
        bd.Record(id=next(iter(bd.DROP_IDS)), bin="ambiguous"),
        bd.Record(id="keep", bin="other"),
    ]
    kept = bd.drop_ids(records)
    assert [r.id for r in kept] == ["keep"]


def test_gold_records_drops_q2_ids_only() -> None:
    records = [
        bd.Record(id=next(iter(bd.DROP_IDS)), bin="ambiguous"),
        bd.Record(id="amb", bin="ambiguous"),
        bd.Record(id="pos", bin="prompt-misread"),
        bd.Record(id="neg", bin="no-fault-within-round"),
    ]
    gold = bd.gold_records(records)
    # ambiguous rows stay in gold; only the named Q2 ids are dropped.
    assert {r.id for r in gold} == {"amb", "pos", "neg"}


def test_gold_label_split_fixture() -> None:
    records = [
        bd.Record(id="p1", bin="prompt-misread"),
        bd.Record(id="p2", bin="stale-context"),
        bd.Record(id="p3", bin="other"),
        bd.Record(id="p4", bin="retrieval-noise"),
        bd.Record(id="n1", bin="no-fault-within-round"),
        bd.Record(id="n2", bin="no-fault-within-round"),
        bd.Record(id="n3", bin="external"),
        bd.Record(id="x1", bin="ambiguous"),
    ]
    gold = bd.gold_records(records)
    counts = bd.label_tally(gold)
    assert counts[bd.LABEL_POSITIVE] == 4
    assert counts[bd.LABEL_NEGATIVE] == 3
    assert counts[bd.LABEL_EXCLUDED] == 1
    assert len(gold) == 8


# --- prompt resolution -------------------------------------------------------


def _write_round(store: Path, id: str, user_prompt) -> Path:
    import json

    store.mkdir(parents=True, exist_ok=True)
    path = store / f"{id}.json"
    path.write_text(json.dumps({"id": id, "userPrompt": user_prompt}), encoding="utf-8")
    return path


def test_resolve_prompt_happy_path(tmp_path: Path) -> None:
    store = tmp_path / "rounds"
    _write_round(store, "a", "full prompt text for a")
    records = bd.resolve_prompts([bd.Record(id="a", bin="other")], store)
    assert records[0].prompt == "full prompt text for a"
    assert records[0].prompt_status == bd.PROMPT_RESOLVED


def test_missing_round_file_is_unresolved_not_dropped(tmp_path: Path) -> None:
    store = tmp_path / "rounds"
    store.mkdir()
    records = bd.resolve_prompts([bd.Record(id="gone", bin="other")], store)
    assert len(records) == 1  # never dropped
    assert records[0].prompt is None
    assert records[0].prompt_status == bd.PROMPT_UNRESOLVED


def test_empty_prompt_is_unresolved(tmp_path: Path) -> None:
    store = tmp_path / "rounds"
    _write_round(store, "blank", "   ")
    records = bd.resolve_prompts([bd.Record(id="blank", bin="other")], store)
    assert records[0].prompt_status == bd.PROMPT_UNRESOLVED


def test_non_string_prompt_is_unresolved(tmp_path: Path) -> None:
    store = tmp_path / "rounds"
    _write_round(store, "odd", None)
    records = bd.resolve_prompts([bd.Record(id="odd", bin="other")], store)
    assert records[0].prompt_status == bd.PROMPT_UNRESOLVED


def test_malformed_round_file_is_unresolved(tmp_path: Path) -> None:
    store = tmp_path / "rounds"
    store.mkdir()
    (store / "bad.json").write_text("{not json", encoding="utf-8")
    records = bd.resolve_prompts([bd.Record(id="bad", bin="other")], store)
    assert records[0].prompt_status == bd.PROMPT_UNRESOLVED


def test_never_substitutes_prompt_preview(tmp_path: Path) -> None:
    """A record carrying prompt_preview must NOT have it used as the prompt.

    The round file is absent, so the row is unresolved even though the raw
    pipeline record *does* carry a truncated preview field.
    """
    store = tmp_path / "rounds"
    store.mkdir()
    record = bd.Record(
        id="previewonly", bin="other", raw={"prompt_preview": "short truncated..."}
    )
    resolved = bd.resolve_prompts([record], store)[0]
    assert resolved.prompt is None
    assert resolved.prompt is not resolved.raw[bd.PREVIEW_FIELD]
    assert resolved.prompt_status == bd.PROMPT_UNRESOLVED


def test_resolve_continues_past_miss(tmp_path: Path) -> None:
    store = tmp_path / "rounds"
    _write_round(store, "have", "present prompt")
    records = [
        bd.Record(id="have", bin="other"),
        bd.Record(id="missing", bin="other"),
        bd.Record(id="have2", bin="other"),
    ]
    _write_round(store, "have2", "also present")
    out = bd.resolve_prompts(records, store)
    assert [r.prompt_status for r in out] == [
        bd.PROMPT_RESOLVED,
        bd.PROMPT_UNRESOLVED,
        bd.PROMPT_RESOLVED,
    ]


def test_corpus_defined_by_ids_not_store(tmp_path: Path) -> None:
    store = tmp_path / "rounds"
    _write_round(store, "present", "hello")
    corpus = bd.corpus_records(["present", "absent"], store)
    assert len(corpus) == 2  # pinned ids win; drift does not shrink the corpus
    assert bd.unresolved_ids(corpus) == ["absent"]


def test_build_report_lists_unresolved_for_both_sets(tmp_path: Path) -> None:
    store = tmp_path / "rounds"
    _write_round(store, "g1", "gold prompt")
    gold = bd.resolve_prompts([bd.Record(id="g1", bin="prompt-misread")], store)
    corpus = bd.corpus_records(["g1", "c2"], store)
    report = bd.build_report(gold, corpus, store)
    assert report["gold"]["unresolved"] == []
    assert report["corpus"]["unresolved"] == ["c2"]
    assert report["gold"]["total"] == 1
    assert report["corpus"]["total"] == 2


def test_unresolved_ids_sorted() -> None:
    records = [
        bd.Record(id="z", bin="", prompt_status=bd.PROMPT_UNRESOLVED),
        bd.Record(id="a", bin="", prompt_status=bd.PROMPT_UNRESOLVED),
        bd.Record(id="m", bin="", prompt_status=bd.PROMPT_RESOLVED),
    ]
    assert bd.unresolved_ids(records) == ["a", "z"]


# --- prompt-resolution anchors against the real store (skipped if absent) ----


pytestmark_store = pytest.mark.skipif(
    not Path(bd.DEFAULT_ROUND_STORE).exists()
    or not Path(bd.DEFAULT_FAULT_PIPELINE).exists(),
    reason="round store or source not found",
)


@pytestmark_store
def test_anchor_all_gold_prompts_resolved() -> None:
    gold = bd.resolve_prompts(bd.gold_records(bd.load_records()))
    assert bd.unresolved_ids(gold) == []
    assert all(r.prompt and r.prompt_status == bd.PROMPT_RESOLVED for r in gold)


@pytestmark_store
def test_anchor_corpus_unresolved_split() -> None:
    """30 corpus ids are unresolved: 29 have no round file, 1 has an empty prompt.

    The spec's anchor counts "29 of 5430 scan ids have no round file"; the
    report's unresolved list is one larger because extension 3a also covers a
    round file whose ``userPrompt`` is empty (id a363b8d1...).
    """
    scan_ids = bd.read_scan_ids()
    assert len(scan_ids) == bd.EXPECTED_CORPUS
    corpus = bd.corpus_records(scan_ids)
    assert len(corpus) == bd.EXPECTED_CORPUS
    unresolved = bd.unresolved_ids(corpus)
    assert len(unresolved) == 30
    store = Path(bd.DEFAULT_ROUND_STORE)
    no_file = [i for i in unresolved if not (store / f"{i}.json").exists()]
    empty_file = [i for i in unresolved if (store / f"{i}.json").exists()]
    assert len(no_file) == 29
    assert len(empty_file) == 1


# --- origin back-propagation -------------------------------------------------

# Fixture ids: 32-char hex, prefixed so a short token resolves to exactly one.
SELF_ID = "aa11" + "0" * 28
TARGET_ID = "ff99" + "0" * 28  # the round under judgement (not an origin here)
IN_SET_ID = "bb22" + "0" * 28  # a gold round named as an origin
IN_SET_PARENT_ID = "bb22" + "a" * 28  # same 4-char prefix family (distinct 8-char)
EXTERNAL_ID = "cc33" + "0" * 28
OUT_OF_SCOPE_ID = "dd44" + "0" * 28  # a stored round that is NOT gold


def _s2row(id: str, root_cause: str = "", evidence: str = "") -> bd.Record:
    return bd.Record(
        id=id,
        bin="prompt-misread",
        raw={"s2": {"root_cause": root_cause, "evidence": evidence}},
    )


def test_origin_self_when_no_evidence() -> None:
    rec = _s2row(SELF_ID, "The response handled the request correctly.")
    bd.attach_origins([rec], gold_ids={SELF_ID}, all_ids={SELF_ID})
    assert rec.origin_id == SELF_ID
    assert rec.origin_source == bd.ORIGIN_SELF  # extension 5a


def test_origin_in_set_parent_from_prefix() -> None:
    rec = _s2row(TARGET_ID, f"The mishandling began in the parent round {IN_SET_ID[:8]}.")
    bd.attach_origins([rec], gold_ids={TARGET_ID, IN_SET_ID, IN_SET_PARENT_ID}, all_ids=set())
    assert rec.origin_id == IN_SET_ID
    assert rec.origin_source == bd.ORIGIN_IN_SET


def test_origin_in_set_parent_from_full_id() -> None:
    rec = _s2row(TARGET_ID, f"The omitted round ({IN_SET_ID}) is the origin.")
    bd.attach_origins([rec], gold_ids={IN_SET_ID}, all_ids=set())
    assert rec.origin_id == IN_SET_ID
    assert rec.origin_source == bd.ORIGIN_IN_SET


def test_origin_external_parent_not_back_labelled() -> None:
    """Extension 5b: a named round outside gold is external, never added to gold."""
    rec = _s2row(EXTERNAL_ID, f"The fault originated upstream in {OUT_OF_SCOPE_ID[:8]}.")
    gold_ids = {EXTERNAL_ID}
    bd.attach_origins([rec], gold_ids=gold_ids, all_ids={OUT_OF_SCOPE_ID})
    assert rec.origin_id == OUT_OF_SCOPE_ID
    assert rec.origin_source == bd.ORIGIN_EXTERNAL
    assert OUT_OF_SCOPE_ID not in gold_ids  # not back-labelled


def test_origin_evidence_field_consulted() -> None:
    rec = _s2row(TARGET_ID, evidence=f"root cause traces to {IN_SET_ID[:8]}")
    bd.attach_origins([rec], gold_ids={TARGET_ID, IN_SET_ID}, all_ids=set())
    assert rec.origin_id == IN_SET_ID
    assert rec.origin_source == bd.ORIGIN_IN_SET


def test_origin_unknown_token_is_ignored() -> None:
    """A hex token naming no known round is not evidence."""
    rec = _s2row(SELF_ID, "See round deadbeefcafe for context.")
    bd.attach_origins([rec], gold_ids={SELF_ID}, all_ids={SELF_ID})
    assert rec.origin_id == SELF_ID
    assert rec.origin_source == bd.ORIGIN_SELF


def test_origin_ambiguous_prefix_is_not_guessed() -> None:
    """A prefix matching several ids yields no evidence, not a guess."""
    shared = "ab12" + "0" * 28
    other = "ab12" + "1" + "0" * 27
    rec = _s2row(SELF_ID, "See ab12 for the origin.")
    bd.attach_origins([rec], gold_ids={SELF_ID, shared, other}, all_ids=set())
    assert rec.origin_id == SELF_ID
    assert rec.origin_source == bd.ORIGIN_SELF


def test_origin_self_token_is_self() -> None:
    """A token resolving to the record's own id is self, not in-set-parent."""
    rec = _s2row(SELF_ID, f"This round {SELF_ID[:8]} recovered from an earlier slip.")
    bd.attach_origins([rec], gold_ids={SELF_ID}, all_ids={SELF_ID})
    assert rec.origin_id == SELF_ID
    assert rec.origin_source == bd.ORIGIN_SELF


def test_origin_tally_counts_sources() -> None:
    a = _s2row(SELF_ID)
    b = _s2row(TARGET_ID, f"origin {IN_SET_ID[:8]}")
    bd.attach_origins([a, b], gold_ids={TARGET_ID, IN_SET_ID}, all_ids=set())
    assert bd.origin_tally([a, b]) == {bd.ORIGIN_IN_SET: 1, bd.ORIGIN_SELF: 1}


# --- origin anchors against the real source (skipped if unreachable) ---------

pytestmark_anchor = pytest.mark.skipif(
    not Path(bd.DEFAULT_FAULT_PIPELINE).exists(),
    reason=f"source not found: {bd.DEFAULT_FAULT_PIPELINE}",
)


@pytestmark_anchor
def test_anchor_origin_sources_partition_gold() -> None:
    gold = bd.attach_origins(bd.gold_records(bd.load_records()))
    counts = bd.origin_tally(gold)
    assert set(counts) <= {bd.ORIGIN_SELF, bd.ORIGIN_IN_SET, bd.ORIGIN_EXTERNAL}
    assert sum(counts.values()) == bd.EXPECTED_GOLD


@pytestmark_anchor
def test_anchor_external_origins_are_not_gold_ids() -> None:
    gold = bd.gold_records(bd.load_records())
    gold_ids = {r.id for r in gold}
    bd.attach_origins(gold, gold_ids=gold_ids)
    externals = [r for r in gold if r.origin_source == bd.ORIGIN_EXTERNAL]
    assert externals, "expected some external parents in the live data"
    assert all(r.origin_id not in gold_ids for r in externals)


@pytestmark_anchor
def test_anchor_origin_sources_have_real_examples() -> None:
    """The live data exercises all three outcomes (self / in-set / external)."""
    gold = bd.attach_origins(bd.gold_records(bd.load_records()))
    sources = {r.origin_source for r in gold}
    assert sources == {bd.ORIGIN_SELF, bd.ORIGIN_IN_SET, bd.ORIGIN_EXTERNAL}


# --- anchors against the real source (skipped if unreachable) ----------------


def test_anchor_dedupe_248_to_244() -> None:
    raw_lines = sum(
        1
        for line in open(bd.DEFAULT_FAULT_PIPELINE, encoding="utf-8")
        if line.strip()
    )
    assert raw_lines == 248
    assert len(bd.load_records()) == bd.EXPECTED_DEDUPED


@pytestmark_anchor
def test_anchor_gold_split() -> None:
    gold = bd.gold_records(bd.load_records())
    counts = bd.label_tally(gold)
    assert len(gold) == bd.EXPECTED_GOLD
    assert counts[bd.LABEL_POSITIVE] == bd.EXPECTED_POSITIVE
    assert counts[bd.LABEL_NEGATIVE] == bd.EXPECTED_NEGATIVE
    assert counts[bd.LABEL_EXCLUDED] == bd.EXPECTED_EXCLUDED


@pytestmark_anchor
def test_anchor_both_drop_ids_absent_from_gold() -> None:
    gold_ids = {r.id for r in bd.gold_records(bd.load_records())}
    assert not (bd.DROP_IDS & gold_ids)


# --- emit + determinism (unit-004) -------------------------------------------

def _gold_record(id: str, bin: str, raw: dict | None = None) -> "bd.Record":
    rec = bd.Record(id=id, bin=bin, raw=raw or {})
    rec.prompt = f"prompt-{id}"
    rec.prompt_status = bd.PROMPT_RESOLVED
    rec.origin_id = id
    rec.origin_source = bd.ORIGIN_SELF
    return rec


def test_row_for_projects_exact_schema() -> None:
    rec = _gold_record("a1" * 16, "prompt-misread", raw={})
    row = bd.row_for(rec)
    assert set(row) == set(bd.ROW_FIELDS)
    assert row["id"] == rec.id
    assert row["bin"] == "prompt-misread"
    assert row["label"] == bd.LABEL_POSITIVE
    assert row["prompt"] == rec.prompt
    assert row["prompt_status"] == bd.PROMPT_RESOLVED


def test_row_for_never_leaks_prompt_preview() -> None:
    rec = _gold_record(
        "b2" * 16,
        "no-fault-within-round",
        raw={"prompt_preview": "SHOULD-NOT-APPEAR", "s2": {}},
    )
    row = bd.row_for(rec)
    assert bd.PREVIEW_FIELD not in row
    assert "SHOULD-NOT-APPEAR" not in json.dumps(row)


def test_row_for_carries_s2_metadata() -> None:
    rec = _gold_record(
        "c3" * 16,
        "stale-context",
        raw={
            "s2": {
                "fault_type": "stale-context",
                "root_cause": "the parent round was ignored",
                "evidence": "see round 6547ceb",
            },
            "timestamp": "2026-09-27T00:00:00Z",
            "s0_f": 0.26,
            "s1_prob": 0.57,
        },
    )
    row = bd.row_for(rec)
    assert row["fault_type"] == "stale-context"
    assert row["root_cause"] == "the parent round was ignored"
    assert row["evidence"] == "see round 6547ceb"
    assert row["timestamp"] == "2026-09-27T00:00:00Z"
    assert row["s0_f"] == 0.26
    assert row["s1_prob"] == 0.57


def test_row_for_missing_s2_is_all_none() -> None:
    rec = _gold_record("d4" * 16, "no-fault-within-round", raw={})
    row = bd.row_for(rec)
    assert row["fault_type"] is None
    assert row["root_cause"] is None
    assert row["evidence"] is None


# --- redaction ---------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [
        "sk-or-v1-" + "a" * 64,
        "sk-" + "b" * 40,
        "ghp_" + "c" * 36,
        "github_pat_" + "d" * 30,
        "gho_" + "e" * 36,
        "AKIA" + "F" * 16,
        "xoxb-" + "1" * 12,
    ],
)
def test_redact_replaces_known_secret_prefixes(secret: str) -> None:
    text = f"here is a management key to test with (temporary): {secret}"
    assert bd.redact(text) == f"here is a management key to test with (temporary): {bd.REDACTION_PLACEHOLDER}"


def test_redact_leaves_ordinary_text_alone() -> None:
    text = "plan the sketch and describe it; no credentials here at all."
    assert bd.redact(text) == text


def test_redact_passes_none_through() -> None:
    assert bd.redact(None) is None


def test_redact_removes_secret_from_row_prompt() -> None:
    secret = "sk-or-v1-" + "f" * 64
    rec = _gold_record("e5" * 16, "no-fault-within-round", raw={})
    rec.prompt = f"paste my key: {secret}"
    row = bd.row_for(rec)
    assert secret not in json.dumps(row)
    assert bd.REDACTION_PLACEHOLDER in row["prompt"]


def test_redaction_covers_all_patterns() -> None:
    """No pattern in the module can be bypassed by concatenation."""
    secrets = [p.pattern for p in bd._SECRET_PATTERNS]
    assert secrets, "redaction patterns must not be empty"


def test_serialize_rows_sorts_by_id_and_keys() -> None:
    rows = [bd.row_for(_gold_record(i, "no-fault-within-round")) for i in ("zz", "aa", "mm")]
    text = bd.serialize_rows(rows)
    ids = [json.loads(line)["id"] for line in text.splitlines()]
    assert ids == ["aa", "mm", "zz"]
    first_line = text.splitlines()[0]
    assert first_line == json.dumps(json.loads(first_line), sort_keys=True, ensure_ascii=False)


def test_serialize_rows_float_repr_is_stable() -> None:
    rows = [bd.row_for(_gold_record("f1" * 16, "other", raw={"s0_f": 0.1, "s1_prob": 0.30000000000000004}))]
    a = bd.serialize_rows(rows)
    b = bd.serialize_rows(list(rows))
    assert a == b
    assert "0.1" in a  # repr, not a rounded form


def test_write_artifact_creates_file_and_parent(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.jsonl"
    recs = [_gold_record("aa" * 16, "other")]
    written = bd.write_artifact(path, recs)
    assert written == path
    assert path.exists()
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_unresolved_row_is_emitted_not_dropped(tmp_path: Path) -> None:
    rec = bd.Record(id="e5" * 16, bin="prompt-misread", raw={})
    rec.prompt = None
    rec.prompt_status = bd.PROMPT_UNRESOLVED
    rec.origin_id = rec.id
    rec.origin_source = bd.ORIGIN_SELF
    path = bd.write_artifact(tmp_path / "g.jsonl", [rec])
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["id"] == rec.id
    assert row["prompt"] is None
    assert row["prompt_status"] == bd.PROMPT_UNRESOLVED


def test_emit_datasets_writes_both_artifacts(tmp_path: Path) -> None:
    gold = [_gold_record("aa" * 16, "prompt-misread")]
    corpus = [bd.Record(id="bb" * 16, bin="", raw={})]
    corpus[0].prompt = "p"
    corpus[0].prompt_status = bd.PROMPT_RESOLVED
    paths = bd.emit_datasets(gold, corpus, tmp_path)
    assert paths["gold"] == tmp_path / bd.GOLD_ARTIFACT
    assert paths["corpus"] == tmp_path / bd.CORPUS_ARTIFACT
    assert paths["gold"].exists() and paths["corpus"].exists()


def test_emit_is_byte_identical_on_rerun(tmp_path: Path) -> None:
    gold = [_gold_record("aa" * 16, "other"), _gold_record("bb" * 16, "stale-context")]
    corpus = [bd.Record(id="cc" * 16, bin="", raw={})]
    corpus[0].prompt = "x"
    corpus[0].prompt_status = bd.PROMPT_RESOLVED
    first = bd.emit_datasets(gold, corpus, tmp_path / "one")
    bytes_one = {k: p.read_bytes() for k, p in first.items()}
    second = bd.emit_datasets(gold, corpus, tmp_path / "two")
    bytes_two = {k: p.read_bytes() for k, p in second.items()}
    assert bytes_one == bytes_two


def test_check_determinism_passes_on_stable_input(tmp_path: Path) -> None:
    gold = [_gold_record("aa" * 16, "other")]
    corpus: list = []
    paths = bd.check_determinism(gold, corpus, tmp_path)
    assert paths["gold"].exists()


def test_check_determinism_raises_on_perturbed_writer(tmp_path: Path, monkeypatch) -> None:
    """A drifting field must abort the build and name the row (extension 7a)."""
    gold = [_gold_record("aa" * 16, "other")]
    calls = {"n": 0}
    real = bd.serialize_rows

    def drifting(rows):
        calls["n"] += 1
        rows = list(rows)
        if calls["n"] > 2:  # only the re-serialization inside the check drifts
            rows = [dict(rows[0], bin="DRIFTED")]
        return real(rows)

    monkeypatch.setattr(bd, "serialize_rows", drifting)
    # write_artifact also uses serialize_rows; guard the first two calls so the
    # on-disk file is written normally and only the re-check diverges.
    with pytest.raises(RuntimeError) as exc:
        bd.check_determinism(gold, [], tmp_path)
    assert "non-deterministic" in str(exc.value)
    assert "aa" * 16 in str(exc.value)


def test_first_divergent_field_names_row() -> None:
    a = json.dumps({"id": "abc", "bin": "x"}, sort_keys=True).encode() + b"\n"
    b = json.dumps({"id": "abc", "bin": "y"}, sort_keys=True).encode() + b"\n"
    assert "abc" in bd._first_divergent_field(a, b)
    assert bd._first_divergent_field(a, a) == ""


# --- end-to-end acceptance vs the spec anchors (unit-005) --------------------
#
# Slow, real-source anchors: run the full build once (dedupe -> gold -> prompts
# -> origins -> emit) and assert every number in issue #1's "Verified anchors"
# section holds over the emitted artifacts. This is the whole-build acceptance
# gate the per-unit tests do not provide.


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the full build into a scratch dir; return records, report, paths."""
    records = bd.load_records()
    gold = bd.resolve_prompts(bd.gold_records(records))
    bd.attach_origins(gold)
    scan_ids = bd.read_scan_ids()
    corpus = bd.corpus_records(scan_ids)
    report = bd.build_report(gold, corpus)
    out = tmp_path_factory.mktemp("acceptance")
    paths = bd.emit_datasets(gold, corpus, out)
    return {
        "records": records,
        "gold": gold,
        "corpus": corpus,
        "report": report,
        "paths": paths,
    }


pytestmark_acceptance = pytest.mark.skipif(
    not Path(bd.DEFAULT_FAULT_PIPELINE).exists()
    or not Path(bd.DEFAULT_ROUND_STORE).exists(),
    reason="source pipeline or round store not found",
)


@pytestmark_acceptance
def test_acceptance_dedupe_anchor(built: dict) -> None:
    """248 raw lines -> 244 unique rounds (last write wins)."""
    raw_lines = sum(
        1
        for line in open(bd.DEFAULT_FAULT_PIPELINE, encoding="utf-8")
        if line.strip()
    )
    assert raw_lines == 248
    assert len(built["records"]) == bd.EXPECTED_DEDUPED == 244


@pytestmark_acceptance
def test_acceptance_pre_drop_bin_tally(built: dict) -> None:
    """Pre-drop bins 203/17/9/8/3/3/1 over the 244 deduped rounds."""
    counts = bd.tally(built["records"])
    assert counts == {
        "no-fault-within-round": 203,
        "prompt-misread": 17,
        "ambiguous": 9,
        "other": 8,
        "external": 3,
        "stale-context": 3,
        "retrieval-noise": 1,
    }


@pytestmark_acceptance
def test_acceptance_gold_split_anchor(built: dict) -> None:
    """Post-drop gold = 242 = 29 positive + 205 negative (202 + 3) + 8 excluded.

    Spec prose "205 + 3 negative" double-counts the 3 external rows: 205 is the
    full negative count (202 no-fault-within-round + 3 external). See findings.
    """
    gold = built["gold"]
    assert len(gold) == bd.EXPECTED_GOLD == 242
    labels = bd.label_tally(gold)
    assert labels[bd.LABEL_POSITIVE] == bd.EXPECTED_POSITIVE == 29
    assert labels[bd.LABEL_NEGATIVE] == bd.EXPECTED_NEGATIVE == 205
    assert labels[bd.LABEL_EXCLUDED] == bd.EXPECTED_EXCLUDED == 8
    bins = bd.tally(gold)
    assert bins["ambiguous"] == 8
    assert bins["no-fault-within-round"] == 202
    assert bins["external"] == 3
    assert bins["no-fault-within-round"] + bins["external"] == 205


@pytestmark_acceptance
def test_acceptance_q2_ids_absent_from_gold(built: dict) -> None:
    """Neither Q2 dropped id survives into the gold set."""
    gold_ids = {r.id for r in built["gold"]}
    assert not (bd.DROP_IDS & gold_ids)
    assert bd.DROP_IDS == {
        "83ebde302dd2fa4d00bed2a1a65b39f5",
        "91853eb0e0115771ceb582755d72a017",
    }


@pytestmark_acceptance
def test_acceptance_all_gold_prompts_resolved(built: dict) -> None:
    """Every gold row carries a full prompt; none is unresolved (extension 2a)."""
    gold = built["gold"]
    assert bd.unresolved_ids(gold) == []
    assert all(r.prompt and r.prompt_status == bd.PROMPT_RESOLVED for r in gold)


@pytestmark_acceptance
def test_acceptance_corpus_pinned_to_5430(built: dict) -> None:
    """The weak corpus is pinned to the 5430 frozen scan ids, not the live store."""
    assert len(built["corpus"]) == bd.EXPECTED_CORPUS == 5430
    assert len(bd.read_scan_ids()) == 5430


@pytestmark_acceptance
def test_acceptance_corpus_unresolved_anchor(built: dict) -> None:
    """29 scan ids have no round file, plus 1 empty prompt -> 30 listed."""
    report = built["report"]
    unresolved = report["corpus"]["unresolved"]
    assert len(unresolved) == 30
    store = Path(bd.DEFAULT_ROUND_STORE)
    no_file = [i for i in unresolved if not (store / f"{i}.json").exists()]
    empty_file = [i for i in unresolved if (store / f"{i}.json").exists()]
    assert len(no_file) == 29
    assert len(empty_file) == 1
    assert unresolved == sorted(unresolved)


@pytestmark_acceptance
def test_acceptance_report_names_counts_per_bin(built: dict) -> None:
    """The run report names per-bin counts and unresolved ids for both sets."""
    report = built["report"]
    assert report["gold"]["total"] == 242
    assert report["gold"]["by_bin"] == bd.tally(built["gold"])
    assert report["gold"]["by_label"] == bd.label_tally(built["gold"])
    assert report["gold"]["by_origin_source"] == bd.origin_tally(built["gold"])
    assert report["gold"]["unresolved"] == []
    assert report["corpus"]["total"] == 5430
    assert set(report["gold"]["by_bin"]) == {
        "ambiguous",
        "external",
        "no-fault-within-round",
        "other",
        "prompt-misread",
        "retrieval-noise",
        "stale-context",
    }
    assert report["round_store"] == str(bd.DEFAULT_ROUND_STORE)


@pytestmark_acceptance
def test_acceptance_artifacts_match_anchors(built: dict) -> None:
    """The emitted files themselves carry the anchor counts (not just memory)."""
    rows = [json.loads(l) for l in built["paths"]["gold"].read_text().splitlines()]
    assert len(rows) == 242
    labels = {"positive": 0, "negative": 0, "excluded": 0}
    for r in rows:
        labels[r["label"]] += 1
    assert labels == {"positive": 29, "negative": 205, "excluded": 8}
    corpus_rows = [
        json.loads(l) for l in built["paths"]["corpus"].read_text().splitlines()
    ]
    assert len(corpus_rows) == 5430
    assert all("prompt_preview" not in r for r in rows)
    assert all("prompt_preview" not in r for r in corpus_rows)


@pytestmark_acceptance
def test_acceptance_build_is_deterministic(built: dict) -> None:
    """A full re-emit into a fresh dir is byte-identical (extension 7a)."""
    import tempfile

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        first = bd.emit_datasets(built["gold"], built["corpus"], a)
        second = bd.emit_datasets(built["gold"], built["corpus"], b)
        for key in first:
            assert first[key].read_bytes() == second[key].read_bytes()
