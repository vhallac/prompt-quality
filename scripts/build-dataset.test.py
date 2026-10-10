#!/usr/bin/env python3
"""Tests for scripts/build-dataset.py (C1: dedupe, split, resolution, origins).

Three layers:
  * synthetic fixtures pin the dedupe/label/resolution logic without the live
    source;
  * pinned-snapshot anchors assert the spec's verified numbers against the
    committed artifacts and ``dataset/c1-build-snapshot.json`` -- no round store
    is read, so they are green on a store that has moved since C1 was built;
  * the live store appears only in the drift layer, which reports what moved by
    field name and fails loudly where the snapshot's own claims break.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
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


def test_anchor_all_gold_prompts_resolved() -> None:
    """The pinned record says gold is fully resolved; the artifacts agree.

    U2: this used to re-resolve the 242 gold rows against the live store, so a
    round file going missing upstream made the suite red for a fact that is
    already settled -- the build that produced the committed gold artifact had
    every prompt. The live store's ability to honour that claim is what
    ``check_store_drift`` reports, not what this anchor asserts.
    """
    snapshot = _pinned_snapshot_or_skip()
    assert snapshot["gold_unresolved"] == []
    assert snapshot["gold_unresolved_size"] == 0
    rows = [json.loads(l) for l in _read_artifact(bd.DEFAULT_GOLD_ARTIFACT)]
    assert len(rows) == snapshot["gold_size"] == bd.EXPECTED_GOLD
    assert [r["id"] for r in rows if r["prompt_status"] != bd.PROMPT_RESOLVED] == []
    assert all(r["prompt"] for r in rows)


def test_anchor_corpus_unresolved_split() -> None:
    """30 corpus ids are unresolved: 29 have no round file, 1 has an empty prompt.

    The spec's anchor counts "29 of 5430 scan ids have no round file"; the
    report's unresolved list is one larger because extension 3a also covers a
    round file whose ``userPrompt`` is empty (id a363b8d1...).

    U2 retargets this from the live store to the pinned snapshot: the store today
    resolves 15 of the 30 again (those rounds reappeared upstream), so scanning
    it yields 15 and the spec's 30 becomes an unreachable claim. The numbers here
    are literals from issue #1, and every one of them is a fact about the
    committed record -- no round store present required.
    """
    snapshot = _pinned_snapshot_or_skip()
    unresolved = snapshot["corpus_unresolved"]
    assert len(unresolved) == 30
    assert len(snapshot["no_round_file_ids"]) == 29
    assert snapshot["empty_prompt_ids"] == ["a363b8d13575101a0226e8d0d054f2e7"]
    # The classification is a partition of the unresolved ledger.
    assert set(unresolved) == set(snapshot["no_round_file_ids"]) | set(
        snapshot["empty_prompt_ids"]
    )
    assert set(snapshot["no_round_file_ids"]).isdisjoint(snapshot["empty_prompt_ids"])
    # The artifacts say the same thing as the record of them.
    rebuilt = bd.reproduce_snapshot(snapshot)
    assert rebuilt["corpus_unresolved"] == unresolved
    rows = [json.loads(l) for l in _read_artifact(bd.DEFAULT_CORPUS_ARTIFACT)]
    assert len(rows) == snapshot["corpus_size"] == 5430
    assert sorted(r["id"] for r in rows if r["prompt"] is None) == unresolved
    assert bd.load_snapshot(bd.DEFAULT_SNAPSHOT) == snapshot


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


@pytestmark_anchor
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


def test_redact_covers_all_free_text_fields() -> None:
    """Every free-text field, not just ``prompt``, is redacted at emit.

    S2 is an LLM summary of user content, so it can quote a credential; the
    whole row must be scrubbed regardless of which field carries it.
    """
    or_secret = "sk-or-v1-" + "a" * 64
    gh_secret = "ghp_" + "b" * 36
    rec = _gold_record(
        "e5" * 16,
        "prompt-misread",
        raw={
            "s2": {
                "root_cause": f"user pasted {or_secret} into the prompt",
                "evidence": f"log line contained {gh_secret}",
            }
        },
    )
    rec.prompt = "ordinary prompt"
    row = bd.row_for(rec)
    blob = json.dumps(row)
    assert or_secret not in blob and gh_secret not in blob
    assert row["root_cause"] == (
        f"user pasted {bd.REDACTION_PLACEHOLDER} into the prompt"
    )
    assert row["evidence"] == f"log line contained {bd.REDACTION_PLACEHOLDER}"


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
    """Run the pinned full build into a scratch dir; return records, report, paths.

    U2: this is the build the committed artifacts came from, not a fresh scan of
    today's store -- ids come from the pinned corpus record and the unresolved
    ledger comes from the snapshot, so the acceptance numbers below are the
    recorded ones. The live store still supplies prompt *text*, which is what
    ``check_store_drift`` audits.
    """
    snapshot = bd.load_snapshot(bd.DEFAULT_SNAPSHOT)
    records = bd.load_records()
    gold = bd.resolve_prompts(
        bd.gold_records(records),
        bd.DEFAULT_ROUND_STORE,
        snapshot["gold_unresolved"],
    )
    bd.attach_origins(gold)
    corpus = bd.corpus_records(
        bd.artifact_ids(bd.DEFAULT_CORPUS_ARTIFACT),
        bd.DEFAULT_ROUND_STORE,
        snapshot["corpus_unresolved"],
    )
    report = bd.build_report(gold, corpus)
    out = tmp_path_factory.mktemp("acceptance")
    paths = bd.emit_datasets(gold, corpus, out)
    return {
        "records": records,
        "gold": gold,
        "corpus": corpus,
        "report": report,
        "paths": paths,
        "snapshot": snapshot,
    }


def _pinned_snapshot_or_skip() -> dict:
    """The committed C1 snapshot, or a skip when the clone has no artifacts."""
    for path in (
        bd.DEFAULT_SNAPSHOT,
        bd.DEFAULT_CORPUS_ARTIFACT,
        bd.DEFAULT_GOLD_ARTIFACT,
    ):
        if not Path(path).exists():
            pytest.skip(f"committed C1 artifact not present: {path}")
    return bd.load_snapshot(bd.DEFAULT_SNAPSHOT)


def _read_artifact(path: Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


pytestmark_acceptance = pytest.mark.skipif(
    not Path(bd.DEFAULT_FAULT_PIPELINE).exists()
    or not Path(bd.DEFAULT_ROUND_STORE).exists()
    or not Path(bd.DEFAULT_SNAPSHOT).exists(),
    reason="source pipeline, pinned snapshot or round store not found",
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
    # The scan snapshot lives in a sibling checkout, so it is pinned by digest:
    # the id universe behind the committed record must still be the one it holds.
    assert bd.corpus_ids_sha256(bd.read_scan_ids()) == built["snapshot"][
        "corpus_ids_sha256"
    ]


@pytestmark_acceptance
def test_acceptance_corpus_unresolved_anchor(built: dict) -> None:
    """29 scan ids have no round file, plus 1 empty prompt -> 30 listed.

    U2: the numbers are asserted against the pinned snapshot, and the built
    report must agree with it field for field. A live store that resolves 15 of
    those 30 again is disclosed by ``check_store_drift`` and changes nothing here.
    """
    report = built["report"]
    snapshot = built["snapshot"]
    unresolved = report["corpus"]["unresolved"]
    assert unresolved == snapshot["corpus_unresolved"]
    assert bd.report_divergences(report, snapshot) == []
    assert len(unresolved) == 30
    assert len(snapshot["no_round_file_ids"]) == 29
    assert len(snapshot["empty_prompt_ids"]) == 1
    assert unresolved == sorted(unresolved)
    # Every unresolved row reaches the artifact: the ledger is carried, not dropped.
    corpus_rows = [
        json.loads(l) for l in built["paths"]["corpus"].read_text().splitlines()
    ]
    assert sorted(r["id"] for r in corpus_rows if r["prompt"] is None) == unresolved
    assert all(r["prompt"] is None for r in corpus_rows if r["id"] in set(unresolved))


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


# --- unit-011 (U2): the pinned C1 build snapshot ------------------------------
#
# The round store moves: of the 30 corpus ids C1 recorded as unresolved, 15 have
# round files again upstream. Tests in this layer assert the *recorded* snapshot
# and the committed artifacts, never a live scan, and the live store appears only
# where a test is explicitly about drift. Fixture shape: 12 corpus ids -- 8
# resolved from the store, 3 with no round file, 1 whose round file has a blank
# prompt -- which is the real 5430 / 29 / 1 at a size a reader can hold in mind.

PIN_IDS = [f"id{i:03d}" for i in range(12)]
PIN_LEDGER = PIN_IDS[8:]  # the recorded unresolved ids (4)
PIN_NO_FILE = PIN_IDS[8:11]  # 3 of them have no round file at all
PIN_UNUSABLE = PIN_IDS[11:]  # 1 has a file whose prompt is unusable
PIN_GOLD = PIN_IDS[:2]

# The issue #1 resolution numbers, as literals. Not read from bd.EXPECTED_* or
# bd.SNAPSHOT_ANCHORS: a test that generates its cases from the code's own anchor
# table cannot notice an anchor being deleted from it.
ISSUE_ONE_RESOLUTION = {
    "corpus_size": 5430,
    "corpus_resolved_size": 5400,
    "corpus_unresolved_size": 30,
    "no_round_file_size": 29,
    "empty_prompt_size": 1,
    "gold_size": 242,
    "gold_unresolved_size": 0,
}

# Every field the pinned record must carry, as literals for the same reason.
PIN_RECORD_FIELDS = (
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


def _pin_fixture(tmp_path: Path) -> dict:
    """A committed corpus/gold pair, a round store, and the snapshot pinning them."""
    store = tmp_path / "rounds"
    store.mkdir()
    for rid in PIN_IDS:
        if rid in PIN_NO_FILE:
            continue
        _write_round(store, rid, "" if rid in PIN_UNUSABLE else f"prompt for {rid}")
    corpus = bd.corpus_records(PIN_IDS, store, PIN_LEDGER)
    gold = bd.resolve_prompts(
        [bd.Record(id=rid, bin="other") for rid in PIN_GOLD], store, []
    )
    bd.attach_origins(gold, round_store=store)
    snapshot = bd.build_snapshot(corpus, gold, PIN_NO_FILE, PIN_UNUSABLE)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    # The artifacts are written through the build's own serializer, so a pinned
    # run has the exact committed bytes to reproduce (``_write_jsonl`` would not
    # sort keys and the byte gate would fire on formatting).
    corpus_path = dataset / bd.CORPUS_ARTIFACT
    corpus_path.write_text(
        bd.serialize_rows(bd.row_for(r) for r in corpus), encoding="utf-8"
    )
    gold_path = dataset / bd.GOLD_ARTIFACT
    gold_path.write_text(
        bd.serialize_rows(bd.row_for(r) for r in gold), encoding="utf-8"
    )
    snapshot_path = bd.write_snapshot(snapshot, dataset / bd.SNAPSHOT_ARTIFACT)
    source = _write_jsonl(
        tmp_path / "fault-pipeline.jsonl",
        [{"id": rid, "bin": "other"} for rid in PIN_GOLD],
    )
    return {
        "ids": PIN_IDS,
        "ledger": PIN_LEDGER,
        "no_file": PIN_NO_FILE,
        "unusable": PIN_UNUSABLE,
        "store": store,
        "dataset": dataset,
        "corpus_path": corpus_path,
        "gold_path": gold_path,
        "snapshot_path": snapshot_path,
        "source": source,
        "snapshot": snapshot,
        "corpus": corpus,
        "gold": gold,
    }


def _diverged_fields(message: str) -> list[str]:
    """The field names a ``SnapshotError`` blames, parsed out of its message."""
    tail = message.split("— ", 1)[1]
    return [part.split(":", 1)[0].strip() for part in tail.split("; ")]


def _bump(value):
    return value + 1


def _scramble(value):
    return value[::-1]


def _drop_last(value):
    return value[:-1]


def _drop_first(value):
    return value[1:]


def _rewrite_artifact(path: Path, mutate) -> None:
    """Edit a committed artifact in place, keeping its canonical rendering."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    _write_jsonl(path, rows)


def _rewrite_record_artifact(path: Path, mutate) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    path.write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


# --- the pinned ledger is what decides resolution -----------------------------


def test_pinned_ledger_keeps_a_reappeared_round_unresolved(tmp_path: Path) -> None:
    """A round file that came back upstream cannot re-draw a recorded row.

    The control half makes it discriminate: without the ledger the same store
    resolves the id, so a regression to a live scan would fail here rather than
    silently emit a different corpus.
    """
    fx = _pin_fixture(tmp_path)
    rid = PIN_LEDGER[0]
    _write_round(fx["store"], rid, "the round reappeared upstream")

    pinned = bd.corpus_records(PIN_IDS, fx["store"], PIN_LEDGER)
    assert bd.unresolved_ids(pinned) == PIN_LEDGER
    rebuilt = [r for r in pinned if r.id == rid][0]
    assert rebuilt.prompt is None
    assert rebuilt.prompt_status == bd.PROMPT_UNRESOLVED

    unpinned = bd.corpus_records(PIN_IDS, fx["store"])
    assert [r for r in unpinned if r.id == rid][0].prompt_status == bd.PROMPT_RESOLVED
    assert bd.unresolved_ids(unpinned) == PIN_LEDGER[1:]


def test_pinned_ledger_never_fabricates_a_prompt(tmp_path: Path) -> None:
    """The ledger marks rows unresolved; it does not invent text for them."""
    fx = _pin_fixture(tmp_path)
    corpus = bd.corpus_records(PIN_IDS, fx["store"], PIN_LEDGER)
    for record in corpus:
        if record.id in set(PIN_LEDGER):
            assert record.prompt is None
        else:
            assert record.prompt == f"prompt for {record.id}"


def test_pinned_ledger_ignores_ids_that_are_not_in_the_build(tmp_path: Path) -> None:
    """A ledger entry for an absent id neither crashes nor adds a row."""
    fx = _pin_fixture(tmp_path)
    corpus = bd.corpus_records(PIN_IDS, fx["store"], [*PIN_LEDGER, "not-a-corpus-id"])
    assert len(corpus) == len(PIN_IDS)
    assert bd.unresolved_ids(corpus) == PIN_LEDGER


def test_force_unresolved_none_is_the_fresh_draw_path(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    assert bd.unresolved_ids(bd.corpus_records(PIN_IDS, fx["store"])) == [
        *PIN_NO_FILE,
        *PIN_UNUSABLE,
    ]


def test_classify_unresolved_splits_by_round_file(tmp_path: Path) -> None:
    """The store-time classification the artifacts themselves cannot show."""
    fx = _pin_fixture(tmp_path)
    assert bd.classify_unresolved(fx["corpus"], fx["store"]) == (
        PIN_NO_FILE,
        PIN_UNUSABLE,
    )


# --- the snapshot record ------------------------------------------------------


def test_build_snapshot_is_the_record_of_a_build(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    snapshot = fx["snapshot"]
    assert set(snapshot) == set(PIN_RECORD_FIELDS)
    assert snapshot["corpus_size"] == 12
    assert snapshot["corpus_resolved_size"] == 8
    assert snapshot["corpus_unresolved"] == PIN_LEDGER
    assert snapshot["no_round_file_ids"] == PIN_NO_FILE
    assert snapshot["empty_prompt_ids"] == PIN_UNUSABLE
    assert snapshot["gold_unresolved"] == []
    assert snapshot["corpus_ids_sha256"] == bd.corpus_ids_sha256(PIN_IDS)


def test_snapshot_round_trips_byte_identical(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    again = bd.load_snapshot(fx["snapshot_path"])
    assert again == fx["snapshot"]
    assert bd.snapshot_bytes(again) == fx["snapshot_path"].read_bytes()
    other = bd.write_snapshot(again, tmp_path / "again.json")
    assert other.read_bytes() == fx["snapshot_path"].read_bytes()


def test_reproduce_snapshot_round_trips_the_fixture(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    assert bd.reproduce_snapshot(
        fx["snapshot"], fx["corpus_path"], fx["gold_path"]
    ) == fx["snapshot"]


def test_reproduce_snapshot_needs_no_round_store(tmp_path: Path) -> None:
    """The record is checkable on a machine with no rounds at all."""
    fx = _pin_fixture(tmp_path)
    shutil.rmtree(fx["store"])
    assert bd.reproduce_snapshot(
        fx["snapshot"], fx["corpus_path"], fx["gold_path"]
    ) == fx["snapshot"]


@pytest.mark.parametrize(
    "field, tamper, expected",
    [
        ("corpus_size", _bump, {"corpus_size"}),
        ("corpus_ids_sha256", _scramble, {"corpus_ids_sha256"}),
        ("corpus_resolved_size", _bump, {"corpus_resolved_size"}),
        ("corpus_unresolved_size", _bump, {"corpus_unresolved_size"}),
        ("corpus_unresolved", _drop_last, {"corpus_unresolved"}),
        ("gold_size", _bump, {"gold_size"}),
        ("gold_unresolved_size", _bump, {"gold_unresolved_size"}),
        ("gold_unresolved", lambda value: [PIN_GOLD[0]], {"gold_unresolved"}),
        # A classification ledger is a record about the store, so tampering one
        # blames its own count and the unresolved set it exists to cover.
        ("no_round_file_size", _bump, {"no_round_file_size"}),
        ("no_round_file_ids", _drop_first, {"no_round_file_size", "corpus_unresolved"}),
        ("empty_prompt_size", _bump, {"empty_prompt_size"}),
        (
            "empty_prompt_ids",
            lambda value: [*value, "zzz-not-a-round"],
            {"empty_prompt_size", "empty_prompt_ids", "corpus_unresolved"},
        ),
    ],
)
def test_reproduce_snapshot_names_the_field_that_diverges(
    tmp_path: Path, field: str, tamper, expected: set[str]
) -> None:
    """U2's loud failure: a mismatch blames the field it broke, by name."""
    fx = _pin_fixture(tmp_path)
    tampered = dict(fx["snapshot"])
    tampered[field] = tamper(tampered[field])
    with pytest.raises(bd.SnapshotError) as excinfo:
        bd.reproduce_snapshot(tampered, fx["corpus_path"], fx["gold_path"])
    message = str(excinfo.value)
    assert set(_diverged_fields(message)) == expected, message
    assert "recorded" in message and "re-derived" in message


def test_reproduce_snapshot_requires_a_complete_record(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    partial = {"corpus_size": 12, "corpus_unresolved": PIN_LEDGER}
    with pytest.raises(bd.SnapshotError) as excinfo:
        bd.reproduce_snapshot(partial, fx["corpus_path"], fx["gold_path"])
    message = str(excinfo.value)
    assert "missing fields" in message
    for field in ("corpus_ids_sha256", "no_round_file_ids", "gold_unresolved"):
        assert field in message, message


def test_reproduce_snapshot_keeps_the_ledger_canonical(tmp_path: Path) -> None:
    """A hand-reordered or duplicated ledger is not the pinned ledger."""
    fx = _pin_fixture(tmp_path)
    reordered = dict(fx["snapshot"])
    reordered["no_round_file_ids"] = list(reversed(PIN_NO_FILE))
    with pytest.raises(bd.SnapshotError) as excinfo:
        bd.reproduce_snapshot(reordered, fx["corpus_path"], fx["gold_path"])
    assert set(_diverged_fields(str(excinfo.value))) == {"no_round_file_ids"}

    duplicated = dict(fx["snapshot"])
    duplicated["no_round_file_ids"] = sorted([*PIN_NO_FILE, PIN_NO_FILE[0]])
    with pytest.raises(bd.SnapshotError) as excinfo:
        bd.reproduce_snapshot(duplicated, fx["corpus_path"], fx["gold_path"])
    assert set(_diverged_fields(str(excinfo.value))) == {"no_round_file_ids"}


def test_reproduce_snapshot_rejects_an_overlapping_classification(
    tmp_path: Path,
) -> None:
    """One id cannot be both a missing file and an unusable file."""
    fx = _pin_fixture(tmp_path)
    both = dict(fx["snapshot"])
    both["no_round_file_ids"] = sorted([*PIN_NO_FILE, *PIN_UNUSABLE])
    both["no_round_file_size"] = len(both["no_round_file_ids"])
    with pytest.raises(bd.SnapshotError) as excinfo:
        bd.reproduce_snapshot(both, fx["corpus_path"], fx["gold_path"])
    assert "unusable-file" in str(excinfo.value)


def test_digest_pins_the_id_universe(tmp_path: Path) -> None:
    """Swap one id for an unseen one and only the digest can tell."""
    fx = _pin_fixture(tmp_path)
    swapped = sorted([*PIN_IDS[:11], "zzz-unseen-id"])
    corpus_rows = [
        json.loads(line) for line in fx["corpus_path"].read_text().splitlines()
    ]
    corpus_rows[0]["id"] = "zzz-unseen-id"
    # keep the ledger meaningful: the swapped row is a resolved one
    _write_jsonl(fx["corpus_path"], corpus_rows)
    snapshot = dict(fx["snapshot"])
    snapshot["corpus_ids_sha256"] = bd.corpus_ids_sha256(swapped)
    with pytest.raises(bd.SnapshotError) as excinfo:
        bd.reproduce_snapshot(snapshot, fx["corpus_path"], fx["gold_path"])
    assert "corpus_ids_sha256" in _diverged_fields(str(excinfo.value))


def test_digest_is_independent_of_row_order(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    rows = [
        json.loads(line) for line in fx["corpus_path"].read_text().splitlines()
    ]
    _write_jsonl(fx["corpus_path"], list(reversed(rows)))
    assert bd.reproduce_snapshot(
        fx["snapshot"], fx["corpus_path"], fx["gold_path"]
    ) == fx["snapshot"]


def test_reproduce_detects_a_flipped_status_row(tmp_path: Path) -> None:
    """An artifact edited to claim a recorded-missing round has text fails loudly."""

    def flip(rows):
        for row in rows:
            if row["prompt_status"] != bd.PROMPT_RESOLVED:
                row["prompt_status"] = bd.PROMPT_RESOLVED
                return

    fx = _pin_fixture(tmp_path)
    _rewrite_artifact(fx["corpus_path"], flip)
    with pytest.raises(bd.SnapshotError) as excinfo:
        bd.reproduce_snapshot(fx["snapshot"], fx["corpus_path"], fx["gold_path"])
    named = set(_diverged_fields(str(excinfo.value)))
    assert {
        "corpus_unresolved",
        "corpus_unresolved_size",
        "corpus_resolved_size",
    } <= named


# --- the hand-maintained anchors ----------------------------------------------


@pytest.mark.parametrize("field", sorted(ISSUE_ONE_RESOLUTION))
def test_assert_anchors_names_the_field_that_drifted(field: str) -> None:
    snapshot = dict(ISSUE_ONE_RESOLUTION)
    snapshot[field] += 1
    with pytest.raises(SystemExit) as excinfo:
        bd.assert_anchors(snapshot)
    message = str(excinfo.value)
    assert message.startswith(f"{field} drift: "), message
    got = ISSUE_ONE_RESOLUTION[field] + 1
    assert f"expected {ISSUE_ONE_RESOLUTION[field]}, got {got}" in message


def test_assert_anchors_accepts_the_issue_one_numbers() -> None:
    bd.assert_anchors(dict(ISSUE_ONE_RESOLUTION))


def test_anchor_table_and_field_list_are_the_declared_record() -> None:
    """The anchor table governs the declared fields -- checked as literals.

    Both tables are written out here rather than read from the module, so an
    anchor or a snapshot field deleted from the code cannot delete its own test.
    """
    assert bd.SNAPSHOT_FIELDS == PIN_RECORD_FIELDS
    assert set(bd.SNAPSHOT_ANCHORS) == {
        (field, value) for field, value in ISSUE_ONE_RESOLUTION.items()
    }


# --- the live store is a report, graded by what it invalidates ----------------


def test_drift_is_quiet_on_a_matching_store(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    report = bd.check_store_drift(
        fx["snapshot"], round_store=fx["store"], corpus_path=fx["corpus_path"]
    )
    assert report["drifted_fields"] == []
    assert report["recovered_ids"] == []
    assert report["lost_round_file_ids"] == []
    assert report["prompt_text_mismatch_ids"] == []
    assert report["live_unresolved_size"] == report["recorded_unresolved_size"] == 4
    assert "drift: none" in bd.drift_summary(report)


def test_drift_reports_a_reappeared_round_without_redrawing_it(tmp_path: Path) -> None:
    """Recovered rounds widen a hypothetical corpus; the record stays put."""
    fx = _pin_fixture(tmp_path)
    rid = PIN_NO_FILE[0]
    _write_round(fx["store"], rid, "this round came back")
    report = bd.check_store_drift(
        fx["snapshot"], round_store=fx["store"], corpus_path=fx["corpus_path"]
    )
    assert report["recovered_ids"] == [rid]
    assert report["recovered_no_round_file_ids"] == [rid]
    assert set(report["drifted_fields"]) == {
        "corpus_unresolved",
        "corpus_unresolved_size",
        "corpus_resolved_size",
        "no_round_file_ids",
        "no_round_file_size",
    }
    assert report["live_unresolved_size"] == 3
    assert "the corpus was not re-drawn" in bd.drift_summary(report)
    # The pin is undisturbed by what the store now says.
    assert bd.reproduce_snapshot(
        fx["snapshot"], fx["corpus_path"], fx["gold_path"]
    ) == fx["snapshot"]
    assert bd.unresolved_ids(
        bd.corpus_records(PIN_IDS, fx["store"], fx["snapshot"]["corpus_unresolved"])
    ) == PIN_LEDGER


def test_drift_classifies_a_repaired_file_as_an_empty_prompt_recovery(
    tmp_path: Path,
) -> None:
    fx = _pin_fixture(tmp_path)
    rid = PIN_UNUSABLE[0]
    _write_round(fx["store"], rid, "the prompt was filled in")
    report = bd.check_store_drift(
        fx["snapshot"], round_store=fx["store"], corpus_path=fx["corpus_path"]
    )
    assert report["recovered_unusable_file_ids"] == [rid]
    assert set(report["drifted_fields"]) == {
        "corpus_unresolved",
        "corpus_unresolved_size",
        "corpus_resolved_size",
        "empty_prompt_ids",
        "empty_prompt_size",
    }


def test_drift_fails_loudly_when_a_recorded_prompt_is_gone(tmp_path: Path) -> None:
    """A row the artifacts claim is resolved must still be re-derivable."""
    fx = _pin_fixture(tmp_path)
    victim = next(i for i in PIN_IDS if i not in set(PIN_LEDGER))
    (fx["store"] / f"{victim}.json").unlink()
    with pytest.raises(bd.StoreDriftError) as excinfo:
        bd.check_store_drift(
        fx["snapshot"], round_store=fx["store"], corpus_path=fx["corpus_path"]
    )
    message = str(excinfo.value)
    assert "lost_round_file_ids=1" in message
    assert victim in message


def test_drift_fails_loudly_when_recorded_prompt_text_changed(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    victim = next(i for i in PIN_IDS if i not in set(PIN_LEDGER))
    _write_round(fx["store"], victim, "rewritten after the build")
    with pytest.raises(bd.StoreDriftError) as excinfo:
        bd.check_store_drift(
        fx["snapshot"], round_store=fx["store"], corpus_path=fx["corpus_path"]
    )
    message = str(excinfo.value)
    assert "prompt_text_mismatch_ids=1" in message
    assert victim in message


def test_drift_compare_is_redaction_aware(tmp_path: Path) -> None:
    """A committed prompt that is the redacted store text is not a mismatch.

    The artifact holds ``[REDACTED]`` where the round file still holds the key the
    user pasted; comparing raw text would report every such row as drift.
    """
    fx = _pin_fixture(tmp_path)
    victim = next(i for i in PIN_IDS if i not in set(PIN_LEDGER))
    secret = "sk-or-v1-" + "a" * 64
    _write_round(fx["store"], victim, f"paste my key: {secret}")
    _rewrite_record_artifact(
        fx["corpus_path"],
        lambda rows: [
            row.update(prompt=f"paste my key: {bd.REDACTION_PLACEHOLDER}")
            for row in rows
            if row["id"] == victim
        ],
    )
    report = bd.check_store_drift(
        fx["snapshot"], round_store=fx["store"], corpus_path=fx["corpus_path"]
    )
    assert report["prompt_text_mismatch_ids"] == []


def test_drift_ignores_rounds_that_are_not_in_the_pinned_universe(
    tmp_path: Path,
) -> None:
    """The store growing past the corpus is not drift in anything the pin claims."""
    fx = _pin_fixture(tmp_path)
    for n in range(50):
        _write_round(fx["store"], f"unrelated{n:03d}", "a round from later")
    report = bd.check_store_drift(
        fx["snapshot"], round_store=fx["store"], corpus_path=fx["corpus_path"]
    )
    assert report["drifted_fields"] == []
    assert report["store_round_files"] == len(PIN_IDS) + 50 - len(PIN_NO_FILE)


def test_drift_without_a_store_is_a_loud_absence_not_a_silence(tmp_path: Path) -> None:
    """No store means no prompt text: the snapshot cannot honour its rows."""
    fx = _pin_fixture(tmp_path)
    shutil.rmtree(fx["store"])
    with pytest.raises(bd.StoreDriftError):
        bd.check_store_drift(
        fx["snapshot"], round_store=fx["store"], corpus_path=fx["corpus_path"]
    )


def test_store_drift_error_is_a_snapshot_error() -> None:
    assert issubclass(bd.StoreDriftError, bd.SnapshotError)


# --- the built report must agree with the record ------------------------------


def test_report_divergences_accepts_the_pinned_build(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    report = bd.build_report(fx["gold"], fx["corpus"], fx["store"])
    assert bd.report_divergences(report, fx["snapshot"]) == []


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda r: r["corpus"]["unresolved"].append("zzz"), "corpus_unresolved"),
        (lambda r: r["corpus"].update(total=99), "corpus_size"),
        (lambda r: r["gold"]["unresolved"].append(PIN_GOLD[0]), "gold_unresolved"),
        (lambda r: r["gold"].update(total=99), "gold_size"),
    ],
)
def test_report_divergences_names_the_contradiction(
    tmp_path: Path, mutate, field: str
) -> None:
    fx = _pin_fixture(tmp_path)
    report = bd.build_report(fx["gold"], fx["corpus"], fx["store"])
    mutate(report)
    diverged = bd.report_divergences(report, fx["snapshot"])
    assert [line.split(":", 1)[0] for line in diverged] == [field]
    assert "snapshot" in diverged[0] and "built" in diverged[0]


# --- committed-artifact anchors (no round store required) ---------------------


def test_committed_snapshot_is_a_complete_pinned_record() -> None:
    """Every field the pin needs is in the committed artifact, and consistent."""
    snapshot = _pinned_snapshot_or_skip()
    assert set(snapshot) == set(PIN_RECORD_FIELDS)
    for field, expected in ISSUE_ONE_RESOLUTION.items():
        assert snapshot[field] == expected, field
    for list_field, size_field in (
        ("corpus_unresolved", "corpus_unresolved_size"),
        ("no_round_file_ids", "no_round_file_size"),
        ("empty_prompt_ids", "empty_prompt_size"),
        ("gold_unresolved", "gold_unresolved_size"),
    ):
        assert snapshot[list_field] == sorted(set(snapshot[list_field]))
        assert snapshot[size_field] == len(snapshot[list_field])
    assert set(snapshot["no_round_file_ids"]) | set(snapshot["empty_prompt_ids"]) == set(
        snapshot["corpus_unresolved"]
    )
    assert set(snapshot["no_round_file_ids"]).isdisjoint(snapshot["empty_prompt_ids"])
    assert snapshot["corpus_ids_sha256"] == bd.corpus_ids_sha256(
        bd.artifact_ids()
    )
    bd.assert_anchors(snapshot)
    bd.reproduce_snapshot(snapshot)


def test_committed_snapshot_is_its_own_canonical_bytes() -> None:
    snapshot = _pinned_snapshot_or_skip()
    committed = Path(bd.DEFAULT_SNAPSHOT).read_bytes()
    assert bd.snapshot_bytes(snapshot) == committed
    elsewhere = Path(bd.DEFAULT_SNAPSHOT).with_name("s2.json")
    assert bd.write_snapshot(snapshot, elsewhere).read_bytes() == committed
    elsewhere.unlink()


def test_c1_no_round_file_ledger_is_c2_frame_ledger() -> None:
    """Two stages, one fact: the ids with no round file agree across artifacts.

    C2's frame ledger and C1's resolution ledger were drawn from the same store
    state, so the cross-check is what keeps either one from being edited quietly
    -- the classification no artifact can otherwise show.
    """
    snapshot = _pinned_snapshot_or_skip()
    manifest_path = bd.DEFAULT_DATASET_DIR / "base-rate-sample.json"
    if not manifest_path.exists():
        pytest.skip("C2 sample manifest not present")
    c2 = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert sorted(snapshot["no_round_file_ids"]) == sorted(c2["missing_ids"])
    # The extra id in C1's ledger is the empty-prompt row extension 3a covers.
    assert set(snapshot["corpus_unresolved"]) - set(c2["missing_ids"]) == set(
        snapshot["empty_prompt_ids"]
    )


def test_pinned_corpus_build_reproduces_the_committed_bytes() -> None:
    """The whole point of U2, on the real data: the moved store re-draws nothing.

    Rebuilding the corpus from the pinned ids and the pinned ledger, taking prompt
    text from today's store, renders the exact committed bytes.
    """
    if not Path(bd.DEFAULT_ROUND_STORE).exists():
        pytest.skip("round store not found")
    snapshot = _pinned_snapshot_or_skip()
    corpus = bd.corpus_records(
        bd.artifact_ids(bd.DEFAULT_CORPUS_ARTIFACT),
        bd.DEFAULT_ROUND_STORE,
        snapshot["corpus_unresolved"],
    )
    rendered = bd.serialize_rows(bd.row_for(r) for r in corpus).encode("utf-8")
    assert rendered == Path(bd.DEFAULT_CORPUS_ARTIFACT).read_bytes()


def test_live_store_drift_is_named_and_leaves_the_corpus_pinned() -> None:
    """The real moved store, asserted without asserting any live count.

    What must hold on any store: nothing the snapshot claims is broken, whatever
    differs is named by field, and the record still reproduces. The counts are
    deliberately not asserted -- that was U2's disease.
    """
    if not Path(bd.DEFAULT_ROUND_STORE).exists():
        pytest.skip("round store not found")
    snapshot = _pinned_snapshot_or_skip()
    report = bd.check_store_drift(
        snapshot,
        round_store=bd.DEFAULT_ROUND_STORE,
        corpus_path=bd.DEFAULT_CORPUS_ARTIFACT,
    )
    assert report["lost_round_file_ids"] == []
    assert report["prompt_text_mismatch_ids"] == []
    assert set(report["drifted_fields"]) <= {
        "corpus_unresolved",
        "corpus_unresolved_size",
        "corpus_resolved_size",
        "no_round_file_ids",
        "no_round_file_size",
        "empty_prompt_ids",
        "empty_prompt_size",
    }
    assert report["recorded_corpus_size"] == snapshot["corpus_size"] == 5430
    assert report["live_unresolved_size"] == (
        report["recorded_unresolved_size"]
        - len(report["recovered_ids"])
        + len(report["lost_round_file_ids"])
    )
    assert report["recorded_unresolved_size"] == 30
    assert len(report["recovered_ids"]) <= 30
    assert sorted(report["recovered_ids"]) == report["recovered_ids"]
    assert set(report["recovered_ids"]) <= set(snapshot["corpus_unresolved"])


# --- first divergence naming (the byte gate's message) ------------------------


def test_first_divergent_field_handles_a_length_only_difference() -> None:
    """A row that only grew has no differing byte inside the common prefix.

    The byte gate calls this to describe its own refusal; an earlier version
    walked a zip that never broke and crashed on an unbound offset instead.
    """
    short = json.dumps({"id": "ab" * 16, "bin": "other"}, sort_keys=True).encode() + b"\n"
    long = short + b"extra\n"
    assert "byte" in bd._first_divergent_field(short, long)
    assert "byte" in bd._first_divergent_field(long, short)


# --- CLI wiring of the pinned build ------------------------------------------


def _cli_argv(fx: dict, monkeypatch=None) -> list[str]:
    """The pinned build's CLI over the fixture.

    ``assert_anchors`` is neutralised when a monkeypatch is supplied: the anchors
    are the live 5430/242/30 numbers, and a 12-id fixture corpus can only ever
    trip them. The anchor guard itself is asserted separately.
    """
    if monkeypatch is not None:
        monkeypatch.setattr(bd, "assert_anchors", lambda snapshot: None)
    return [
        "--source", str(fx["source"]),
        "--rounds-dir", str(fx["store"]),
        "--out-dir", str(fx["dataset"]),
        "--snapshot", str(fx["dataset"] / bd.SNAPSHOT_ARTIFACT),
    ]


def test_cli_pinned_build_reemits_the_same_bytes(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    fx = _pin_fixture(tmp_path)
    before = {
        name: path.read_bytes()
        for name, path in (
            ("gold", fx["gold_path"]),
            ("corpus", fx["corpus_path"]),
            ("snapshot", fx["snapshot_path"]),
        )
    }
    assert bd.main(_cli_argv(fx, monkeypatch)) == 0
    for name, path in (
        ("gold", fx["gold_path"]),
        ("corpus", fx["corpus_path"]),
        ("snapshot", fx["snapshot_path"]),
    ):
        assert path.read_bytes() == before[name], name
    out = capsys.readouterr().out
    assert "source=pinned snapshot" in out
    assert "store drift: none" in out


def test_cli_pinned_build_survives_a_moved_store(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """The U2 disease, in a fixture: rounds come back, the artifacts do not move."""
    fx = _pin_fixture(tmp_path)
    for rid in PIN_LEDGER:
        if rid not in set(PIN_UNUSABLE):
            _write_round(fx["store"], rid, f"a recovered round {rid}")
    for n in range(20):
        _write_round(fx["store"], f"unrelated{n:03d}", "store growth")
    before = {
        name: path.read_bytes()
        for name, path in (
            ("gold", fx["gold_path"]),
            ("corpus", fx["corpus_path"]),
            ("snapshot", fx["snapshot_path"]),
        )
    }
    assert bd.main(_cli_argv(fx, monkeypatch)) == 0
    for name, path in (
        ("gold", fx["gold_path"]),
        ("corpus", fx["corpus_path"]),
        ("snapshot", fx["snapshot_path"]),
    ):
        assert path.read_bytes() == before[name], name
    out = capsys.readouterr().out
    assert "the corpus was not re-drawn" in out
    assert "recovered=3" in out


def test_cli_refuses_to_redraw_a_pinned_snapshot(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    corpus_before = fx["corpus_path"].read_bytes()
    with pytest.raises(SystemExit) as excinfo:
        bd.main(_cli_argv(fx) + ["--redraw"])
    assert "refuses to overwrite the pinned snapshot" in str(excinfo.value)
    assert fx["corpus_path"].read_bytes() == corpus_before


def test_cli_refuses_a_scan_snapshot_while_a_build_is_pinned(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    scan = _write_jsonl(tmp_path / "scan.jsonl", [{"id": rid} for rid in PIN_IDS])
    with pytest.raises(SystemExit) as excinfo:
        bd.main(_cli_argv(fx) + ["--scan-results", str(scan)])
    assert "--scan-results" in str(excinfo.value)


def test_cli_store_that_rewrote_recorded_text_fails_before_writing(
    tmp_path: Path, monkeypatch
) -> None:
    """The drift guard is the first line of defence: a rewritten prompt stops it."""
    fx = _pin_fixture(tmp_path)
    victim = next(i for i in PIN_IDS if i not in set(PIN_LEDGER))
    _write_round(fx["store"], victim, f"prompt for {victim} (edited upstream)")
    corpus_before = fx["corpus_path"].read_bytes()
    gold_before = fx["gold_path"].read_bytes()
    with pytest.raises(bd.StoreDriftError) as excinfo:
        bd.main(_cli_argv(fx, monkeypatch))
    assert "invalidates the pinned snapshot" in str(excinfo.value)
    assert victim in str(excinfo.value)
    assert fx["corpus_path"].read_bytes() == corpus_before
    assert fx["gold_path"].read_bytes() == gold_before


def test_cli_hand_edited_artifact_is_refused_by_the_byte_gate(
    tmp_path: Path, monkeypatch
) -> None:
    """A column the drift report does not cover is caught by the byte gate.

    ``origin_id``/``origin_source`` come from store membership rather than from
    prompt text, so an edited gold artifact is invisible to the drift check. The
    gate is what refuses to overwrite it with the build's own rendering, naming
    the row it would have rewritten.
    """
    fx = _pin_fixture(tmp_path)
    corpus_before = fx["corpus_path"].read_bytes()
    _rewrite_record_artifact(
        fx["gold_path"],
        lambda rows: rows[0].update(origin_source=bd.ORIGIN_IN_SET),
    )
    gold_before = fx["gold_path"].read_bytes()
    with pytest.raises(SystemExit) as excinfo:
        bd.main(_cli_argv(fx, monkeypatch))
    message = str(excinfo.value)
    assert "would change" in message and bd.GOLD_ARTIFACT in message
    assert "the recorded snapshot no longer matches its own bytes" in message
    assert fx["corpus_path"].read_bytes() == corpus_before
    assert fx["gold_path"].read_bytes() == gold_before


def test_cli_pinned_build_needs_the_artifacts_it_records(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    empty = tmp_path / "empty-dataset"
    empty.mkdir()
    shutil.copy(fx["snapshot_path"], empty / bd.SNAPSHOT_ARTIFACT)
    with pytest.raises(SystemExit) as excinfo:
        bd.main(
            [
                "--source", str(fx["source"]),
                "--rounds-dir", str(fx["store"]),
                "--out-dir", str(empty),
                "--snapshot", str(empty / bd.SNAPSHOT_ARTIFACT),
            ]
        )
    assert "a pinned build reproduces them, it does not create them" in str(excinfo.value)


def test_cli_fresh_draw_records_a_snapshot(tmp_path: Path, monkeypatch) -> None:
    """No snapshot pinned: the store decides, and the build records what it saw."""
    fx = _pin_fixture(tmp_path)
    scan = _write_jsonl(tmp_path / "scan.jsonl", [{"id": rid} for rid in PIN_IDS])
    out = tmp_path / "fresh"
    out.mkdir()
    snapshot_path = out / bd.SNAPSHOT_ARTIFACT
    argv = [
        "--source", str(fx["source"]),
        "--scan-results", str(scan),
        "--rounds-dir", str(fx["store"]),
        "--out-dir", str(out),
        "--snapshot", str(snapshot_path),
    ]
    assert bd.main(argv) == 0
    recorded = bd.load_snapshot(snapshot_path)
    assert recorded["corpus_unresolved"] == PIN_LEDGER
    assert recorded["no_round_file_ids"] == PIN_NO_FILE
    assert recorded["empty_prompt_ids"] == PIN_UNUSABLE
    assert recorded["corpus_ids_sha256"] == bd.corpus_ids_sha256(PIN_IDS)
    # A second run is now the pinned run, and reproduces byte-for-byte.
    corpus_before = (out / bd.CORPUS_ARTIFACT).read_bytes()
    monkeypatch.setattr(bd, "assert_anchors", lambda snapshot: None)
    pinned_argv = _cli_argv(
        {"source": fx["source"], "store": fx["store"], "dataset": out}, monkeypatch
    )
    assert bd.main(pinned_argv) == 0
    assert (out / bd.CORPUS_ARTIFACT).read_bytes() == corpus_before


def test_cli_fresh_draw_without_an_id_source_fails_loudly(tmp_path: Path) -> None:
    fx = _pin_fixture(tmp_path)
    out = tmp_path / "fresh2"
    out.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        bd.main(
            [
                "--source", str(fx["source"]),
                "--scan-results", str(tmp_path / "no-scan.jsonl"),
                "--rounds-dir", str(fx["store"]),
                "--out-dir", str(out),
                "--snapshot", str(out / bd.SNAPSHOT_ARTIFACT),
            ]
        )
    assert "no id universe to build the corpus from" in str(excinfo.value)


def test_cli_gold_that_lost_a_round_file_fails_before_writing(
    tmp_path: Path, monkeypatch
) -> None:
    """A gold row the record says is resolved must stay re-derivable."""
    fx = _pin_fixture(tmp_path)
    (fx["store"] / f"{PIN_GOLD[0]}.json").unlink()
    gold_before = fx["gold_path"].read_bytes()
    with pytest.raises(bd.StoreDriftError):
        bd.main(_cli_argv(fx, monkeypatch))
    assert fx["gold_path"].read_bytes() == gold_before
