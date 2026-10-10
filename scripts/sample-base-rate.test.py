#!/usr/bin/env python3
"""Tests for scripts/sample-base-rate.py.

Two layers, mirroring C1's test style:
  * synthetic fixtures pin the frame arithmetic and the sample determinism
    without touching the live corpus or round store;
  * anchor tests assert issue #2's verified numbers against the committed
    artifacts when they are reachable (skipped otherwise).

Unit-001 selectors: ``frame`` and ``sample``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

# Import sample-base-rate.py despite the hyphen in its filename.
_MODULE_PATH = Path(__file__).with_name("sample-base-rate.py")
_spec = importlib.util.spec_from_file_location("sample_base_rate", _MODULE_PATH)
sbr = importlib.util.module_from_spec(_spec)
sys.modules["sample_base_rate"] = sbr
_spec.loader.exec_module(sbr)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _make_store(tmp_path: Path, ids: list[str]) -> Path:
    store = tmp_path / "rounds"
    store.mkdir()
    for rid in ids:
        (store / f"{rid}.json").write_text(json.dumps({"userPrompt": f"p-{rid}"}))
    return store


# --- frame arithmetic --------------------------------------------------------


def test_eligible_frame_subtracts_gold_and_missing() -> None:
    corpus = ["a", "b", "c", "d", "e"]
    frame = sbr.eligible_frame(corpus, gold={"b"}, missing=["c"])
    assert frame == ["a", "d", "e"]


def test_eligible_frame_preserves_corpus_order() -> None:
    corpus = ["e", "d", "c", "b", "a"]
    frame = sbr.eligible_frame(corpus, gold=set(), missing=set())
    assert frame == ["e", "d", "c", "b", "a"]


def test_eligible_frame_empty_when_all_excluded() -> None:
    corpus = ["a", "b"]
    frame = sbr.eligible_frame(corpus, gold={"a"}, missing=["b"])
    assert frame == []


def test_eligible_frame_size_is_exact_subtraction() -> None:
    """|frame| == |corpus| − |gold| − |missing| for disjoint gold/missing."""
    corpus = [f"id{i:03d}" for i in range(50)]
    gold = {f"id{i:03d}" for i in range(0, 10)}
    missing = [f"id{i:03d}" for i in range(10, 20)]
    frame = sbr.eligible_frame(corpus, gold, missing)
    assert len(frame) == 50 - 10 - 10
    assert set(frame).isdisjoint(gold)
    assert set(frame).isdisjoint(missing)


def test_gold_ids_includes_drop_ids(tmp_path: Path) -> None:
    """The 242 labelled rows plus C1's two dropped ids = the 244 gold set."""
    gold_file = _write_jsonl(
        tmp_path / "rounds-labeled.jsonl",
        [{"id": "a"}, {"id": "b"}],
    )
    got = sbr.gold_ids(gold_file, drop_ids={"x", "y"})
    assert got == {"a", "b", "x", "y"}


def test_gold_ids_re_adds_real_drop_ids() -> None:
    """The two DROP_IDS are unavoidable members of the 244 gold set."""
    assert sbr.DROP_IDS == {
        "83ebde302dd2fa4d00bed2a1a65b39f5",
        "91853eb0e0115771ceb582755d72a017",
    }


def test_missing_round_ids_detects_absent_files(tmp_path: Path) -> None:
    store = _make_store(tmp_path, ["a", "c"])
    missing = sbr.missing_round_ids(["a", "b", "c", "d"], store)
    assert missing == ["b", "d"]


def test_frame_arithmetic_end_to_end(tmp_path: Path) -> None:
    """corpus − gold − missing composes into the eligible frame."""
    corpus = [f"id{i:03d}" for i in range(30)]
    _write_jsonl(
        tmp_path / "corpus.jsonl",
        [{"id": rid} for rid in corpus],
    )
    _write_jsonl(
        tmp_path / "gold.jsonl",
        [{"id": rid} for rid in corpus[:5]],
    )
    store = _make_store(tmp_path, corpus[:5] + corpus[5:25])  # all but 25..29
    manifest = sbr.build_manifest(
        sample_size=4,
        corpus_path=tmp_path / "corpus.jsonl",
        gold_path=tmp_path / "gold.jsonl",
        round_store=store,
        drop_ids=set(),
    )
    assert manifest["corpus_size"] == 30
    assert manifest["gold_size"] == 5
    assert manifest["missing_size"] == 5
    assert manifest["frame_size"] == 20


def test_assert_anchors_rejects_frame_drift(tmp_path: Path, capsys) -> None:
    broken = {
        "corpus_size": sbr.EXPECTED_CORPUS,
        "gold_size": sbr.EXPECTED_GOLD,
        "missing_size": sbr.EXPECTED_MISSING,
        "frame_size": sbr.EXPECTED_FRAME + 1,
    }
    with pytest.raises(SystemExit):
        sbr.assert_anchors(broken)


# --- seeded uniform sample ---------------------------------------------------


def test_draw_sample_is_deterministic() -> None:
    frame = [f"id{i:03d}" for i in range(100)]
    a = sbr.draw_sample(frame, size=10, seed=42)
    b = sbr.draw_sample(frame, size=10, seed=42)
    assert a == b


def test_draw_sample_differs_across_seeds() -> None:
    frame = [f"id{i:03d}" for i in range(100)]
    a = sbr.draw_sample(frame, size=10, seed=1)
    b = sbr.draw_sample(frame, size=10, seed=2)
    assert a != b


def test_draw_sample_is_without_replacement() -> None:
    frame = [f"id{i:03d}" for i in range(50)]
    got = sbr.draw_sample(frame, size=20, seed=7)
    assert len(got) == 20
    assert len(set(got)) == 20


def test_draw_sample_ids_come_from_frame() -> None:
    frame = [f"id{i:03d}" for i in range(30)]
    got = sbr.draw_sample(frame, size=10, seed=3)
    assert set(got) <= set(frame)


def test_draw_sample_rejects_oversized_request() -> None:
    with pytest.raises(ValueError):
        sbr.draw_sample(["a", "b"], size=3, seed=1)


def test_draw_sample_is_approximately_uniform() -> None:
    """Every eligible id appears at roughly the expected rate.

    A weak uniformity check: over many seeds each id's inclusion frequency
    should sit near ``size/|frame|``. This catches an off-by-one or a biased
    partition far more cheaply than a statistical test would.
    """
    frame = [f"id{i:02d}" for i in range(40)]
    size = 10
    trials = 400
    counts = {rid: 0 for rid in frame}
    for seed in range(trials):
        for rid in sbr.draw_sample(frame, size=size, seed=seed):
            counts[rid] += 1
    expected = trials * size / len(frame)
    # generous band: 3σ of a binomial(trials, size/len(frame))
    import math

    p = size / len(frame)
    sigma = math.sqrt(trials * p * (1 - p))
    lo, hi = expected - 3 * sigma, expected + 3 * sigma
    for rid, c in counts.items():
        assert lo <= c <= hi, f"{rid}: {c} outside [{lo:.1f}, {hi:.1f}]"


def test_build_manifest_records_seed_and_ids(tmp_path: Path) -> None:
    corpus = [f"id{i:03d}" for i in range(40)]
    _write_jsonl(tmp_path / "corpus.jsonl", [{"id": rid} for rid in corpus])
    _write_jsonl(tmp_path / "gold.jsonl", [{"id": rid} for rid in corpus[:4]])
    store = _make_store(tmp_path, corpus[4:])
    manifest = sbr.build_manifest(
        seed=99,
        sample_size=8,
        corpus_path=tmp_path / "corpus.jsonl",
        gold_path=tmp_path / "gold.jsonl",
        round_store=store,
        drop_ids=set(),
    )
    assert manifest["seed"] == 99
    assert len(manifest["sampled_ids"]) == 8
    assert set(manifest["sampled_ids"]) <= set(corpus[4:])
    # never a gold id
    assert set(manifest["sampled_ids"]).isdisjoint(corpus[:4])


def test_manifest_round_trips_byte_identical(tmp_path: Path) -> None:
    manifest = {"seed": 1, "sampled_ids": ["b", "a"], "frame_size": 2}
    p1 = sbr.write_manifest(manifest, tmp_path / "m1.json")
    p2 = sbr.write_manifest(manifest, tmp_path / "m2.json")
    assert p1.read_bytes() == p2.read_bytes()


# --- S2 adjudication (offline, cache-backed) ---------------------------------


class _FakeS2:
    """Minimal stand-in exposing the S2 surface the adapter drives.

    It keeps the real adapter/resume/append logic under test without depending
    on a sibling checkout or a network. ``llm_request`` records calls so a test
    can prove the cache seam suppresses the second call.
    """

    def __init__(self, verdicts: dict[str, dict] | None = None) -> None:
        self.verdicts = verdicts or {}
        self.llm_calls = 0

    def llm_request(self, system: str, user: str, api_key: str, model: str, max_retries: int = 3) -> str:
        self.llm_calls += 1
        # The user payload embeds the target round's prompt text.
        rid = next((r for r in self.verdicts if r in user), None)
        body = self.verdicts.get(rid, {"verdict": "no_fault_within_round"})
        return json.dumps(body)

    def process_round(self, rid, rounds_dir, api_key, args, stages):
        assert stages == {"S2"}, f"adjudication must run S2 only, got {stages}"
        body = self.verdicts.get(rid, {"verdict": "no_fault_within_round"})
        # Embed the round id so ``llm_request`` can resolve the canned verdict.
        text = self.llm_request("sys", f"TARGET ROUND:\n{rid}", api_key, args.llm_model)
        return {"id": rid, "s2": json.loads(text), "bin": body.get("verdict")}


def _adjudicate(rid, module, **kwargs):
    return sbr.adjudicate_one(rid, s2_module=module, **kwargs)


def test_adjudicate_one_runs_s2_only() -> None:
    fake = _FakeS2()
    rec = _adjudicate("a", fake)
    assert rec["id"] == "a"
    assert rec["s2"]["verdict"] == "no_fault_within_round"


def test_adjudicate_one_carries_fault_type() -> None:
    fake = _FakeS2({"b": {"verdict": "fault_observed", "fault_type": "prompt-misread"}})
    rec = _adjudicate("b", fake)
    assert rec["s2"]["fault_type"] == "prompt-misread"
    # S0/S1 never run in this script.
    assert "s0_f" not in rec and "s1_fault_type" not in rec


def test_cache_seam_suppresses_second_llm_call() -> None:
    fake = _FakeS2()
    cache: dict[str, str] = {}
    with sbr.cached_llm(fake, cache):
        _adjudicate("a", fake)
        first_calls = fake.llm_calls
        _adjudicate("a", fake)  # identical request -> served from cache
    assert fake.llm_calls == first_calls, "cached request must not re-call the LLM"
    assert cache  # something was cached


def test_cache_seam_restores_original_callable() -> None:
    fake = _FakeS2()
    with sbr.cached_llm(fake, {}):
        # Inside the seam the attribute is the stand-in, not the class method.
        assert getattr(fake.llm_request, "__self__", None) is None
    # On exit the real bound method is restored.
    assert fake.llm_request.__self__ is fake
    assert fake.llm_request.__func__ is _FakeS2.llm_request


def test_llm_cache_round_trips(tmp_path: Path) -> None:
    cache = {"k1": "v1", "k2": "v2"}
    p = sbr.save_llm_cache(cache, tmp_path / "cache.json")
    assert sbr.load_llm_cache(p) == cache


def test_load_llm_cache_absent_is_empty(tmp_path: Path) -> None:
    assert sbr.load_llm_cache(tmp_path / "missing.json") == {}


# --- resume-on-union + append guard ------------------------------------------


def test_resume_plan_excludes_already_done_ids() -> None:
    manifest = {"sampled_ids": ["a", "b", "c"]}
    results = [{"id": "b"}]
    assert sbr.resume_plan(manifest, results) == ["a", "c"]


def test_resume_plan_keeps_manifest_order() -> None:
    manifest = {"sampled_ids": ["c", "a", "b"]}
    results = [{"id": "a"}]
    assert sbr.resume_plan(manifest, results) == ["c", "b"]


def test_resume_plan_ignores_results_outside_manifest() -> None:
    """A result id not in the manifest is done, not a new todo."""
    manifest = {"sampled_ids": ["a", "b"]}
    results = [{"id": "z"}]
    assert sbr.resume_plan(manifest, results) == ["a", "b"]


def test_resume_plan_is_empty_when_all_done() -> None:
    manifest = {"sampled_ids": ["a", "b"]}
    assert sbr.resume_plan(manifest, [{"id": "a"}, {"id": "b"}]) == []


def test_append_result_writes_once(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    assert sbr.append_result({"id": "a", "bin": "x"}, out) is True
    assert sbr.load_results(out) == [{"id": "a", "bin": "x"}]


def test_append_result_refuses_duplicate_id(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    sbr.append_result({"id": "a", "bin": "x"}, out)
    assert sbr.append_result({"id": "a", "bin": "y"}, out) is False
    rows = sbr.load_results(out)
    assert len(rows) == 1 and rows[0]["bin"] == "x"


def test_append_guard_survives_reprocessed_todo(tmp_path: Path) -> None:
    """Resume-on-union never writes a duplicate even if todo repeats."""
    out = tmp_path / "results.jsonl"
    manifest = {"sampled_ids": ["a", "b"]}
    for rid in sbr.resume_plan(manifest, sbr.load_results(out)):
        sbr.append_result({"id": rid}, out)
    # A restart re-derives the same todo; every append must be refused.
    for rid in sbr.resume_plan(manifest, sbr.load_results(out)):
        assert sbr.append_result({"id": rid}, out) is False
    ids = [r["id"] for r in sbr.load_results(out)]
    assert ids == ["a", "b"]


# --- live anchors (skipped when artifacts are unreachable) --------------------


def test_real_frame_anchor() -> None:
    """Issue #2's frame numbers are anchors of the *pinned snapshot*.

    U1: this test used to scan the live round store, so the suite was green only
    on a store frozen at C2's run date. The frame's missing-round-file term is
    now the manifest's recorded ``missing_ids`` ledger, which makes the same
    arithmetic checkable with no store present at all; a store that disagrees is
    reported by ``check_store_drift``, it no longer moves the frame.
    """
    if not sbr.DEFAULT_CORPUS.exists() or not sbr.DEFAULT_GOLD.exists():
        pytest.skip("committed corpus/gold artifacts not present")
    if not sbr.DEFAULT_MANIFEST.exists():
        pytest.skip("committed sample manifest not present")
    committed = json.loads(sbr.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    manifest = sbr.reproduce_snapshot(committed)
    assert manifest["corpus_size"] == sbr.EXPECTED_CORPUS
    assert manifest["gold_size"] == sbr.EXPECTED_GOLD
    assert manifest["missing_size"] == sbr.EXPECTED_MISSING == 29
    assert manifest["frame_size"] == sbr.EXPECTED_FRAME
    # The anchors are the same guard the stage runs, on the reproduced frame.
    sbr.assert_anchors(manifest)
    assert manifest["missing_ids"] == committed["missing_ids"]
    assert manifest["sampled_ids"] == committed["sampled_ids"]
    assert len(manifest["sampled_ids"]) == committed["sample_size"]


# --- unit-003: run report + backfill -----------------------------------------


def test_classify_bin_positive_set() -> None:
    for name in ("prompt-misread", "stale-context", "other", "retrieval-noise"):
        assert sbr.classify_bin(name) == "positive"


def test_classify_bin_negative_set() -> None:
    assert sbr.classify_bin("no-fault-within-round") == "negative"
    assert sbr.classify_bin("external") == "negative"


def test_classify_bin_excludes_ambiguous() -> None:
    assert sbr.classify_bin("ambiguous") == "excluded"


def test_classify_bin_unknown_is_excluded() -> None:
    # An unknown bin must not be silently counted as a fault.
    assert sbr.classify_bin("some-new-bin") == "excluded"


def test_bin_vocabulary_matches_c1_gold() -> None:
    """The report's positive/negative/excluded rule mirrors C1's gold bins."""
    if not sbr.DEFAULT_GOLD.exists():
        pytest.skip("committed gold artifact not present")
    vocabulary = {row["bin"] for row in sbr.read_jsonl(sbr.DEFAULT_GOLD)}
    assert vocabulary <= (
        sbr.POSITIVE_BINS | sbr.NEGATIVE_BINS | sbr.EXCLUDED_BINS
    ), f"gold carries bins the report does not classify: {vocabulary}"


def test_wilson_interval_brackets_point_estimate() -> None:
    low, high = sbr.wilson_interval(29, 100)
    assert low < 0.29 < high


def test_wilson_interval_matches_issue_anchor() -> None:
    """At n=100, p=0.29 the 95% interval is ≈ ±9 pp (issue #2)."""
    low, high = sbr.wilson_interval(29, 100)
    assert abs((high - low) / 2) == pytest.approx(0.088, abs=0.006)


def test_wilson_interval_stays_in_unit_range_at_zero() -> None:
    low, high = sbr.wilson_interval(0, 100)
    assert low == 0.0 and 0.0 < high < 0.1


def test_wilson_interval_stays_in_unit_range_at_full() -> None:
    low, high = sbr.wilson_interval(100, 100)
    assert 0.9 < low < 1.0 and high == 1.0


def test_wilson_interval_narrows_with_n() -> None:
    narrow = sbr.wilson_interval(290, 1000)
    wide = sbr.wilson_interval(29, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_rejects_empty_n() -> None:
    with pytest.raises(ValueError):
        sbr.wilson_interval(0, 0)


def test_wilson_interval_rejects_out_of_range_k() -> None:
    with pytest.raises(ValueError):
        sbr.wilson_interval(11, 10)


def test_summarize_excludes_ambiguous_from_denominator() -> None:
    rows = [
        {"id": "a", "bin": "prompt-misread"},
        {"id": "b", "bin": "no-fault-within-round"},
        {"id": "c", "bin": "ambiguous"},
    ]
    report = sbr.summarize(rows)
    assert report["n"] == 3
    assert report["denominator"] == 2
    assert report["k"] == 1
    assert report["fault_rate"] == pytest.approx(0.5)
    assert report["ambiguous"] == 1


def test_summarize_counts_external_as_negative() -> None:
    rows = [
        {"id": "a", "bin": "external"},
        {"id": "b", "bin": "stale-context"},
    ]
    report = sbr.summarize(rows)
    assert report["external"] == 1
    assert report["denominator"] == 2
    assert report["k"] == 1


def test_summarize_tallies_each_bin_name() -> None:
    rows = [
        {"id": "a", "bin": "prompt-misread"},
        {"id": "b", "bin": "prompt-misread"},
        {"id": "c", "bin": "other"},
    ]
    report = sbr.summarize(rows)
    assert report["bins"] == {"other": 1, "prompt-misread": 2}
    assert report["positive"] == 3


def test_summarize_treats_missing_bin_as_ambiguous() -> None:
    report = sbr.summarize([{"id": "a"}])
    assert report["ambiguous"] == 1
    assert report["denominator"] == 0
    assert report["fault_rate"] == 0.0


def test_summarize_records_unresolved_sorted() -> None:
    report = sbr.summarize([], unresolved=["z", "a"])
    assert report["unresolved"] == ["a", "z"]


def test_summarize_unresolved_is_deduplicated() -> None:
    """F2 regression: a repeatedly-failing id appears once, not once per pass.

    ``run_adjudication`` appends to the unresolved list on every failed
    adjudication, so a persisted failure (or a replacement that also fails)
    would repeat the same id. The report must list distinct ids.
    """
    report = sbr.summarize([], unresolved=["b", "a", "b", "b", "b"])
    assert report["unresolved"] == ["a", "b"]


def test_build_report_fails_loudly_below_min_rows() -> None:
    rows = [{"id": str(i), "bin": "no-fault-within-round"} for i in range(3)]
    with pytest.raises(SystemExit) as excinfo:
        sbr.build_report(rows, unresolved=["gone"], min_rows=100)
    message = str(excinfo.value)
    assert "3" in message and "100" in message and "gone" in message


def test_build_report_passes_at_min_rows() -> None:
    rows = [{"id": str(i), "bin": "prompt-misread"} for i in range(100)]
    report = sbr.build_report(rows, min_rows=100)
    assert report["n"] == 100
    assert report["k"] == 100


def test_build_report_carries_frame_arithmetic() -> None:
    manifest = {
        "corpus_size": 5430,
        "gold_size": 244,
        "missing_size": 29,
        "frame_size": 5159,
        "sample_size": 100,
        "seed": 20261009,
    }
    rows = [{"id": str(i), "bin": "no-fault-within-round"} for i in range(100)]
    report = sbr.build_report(rows, manifest=manifest, min_rows=100)
    assert report["frame"]["frame_size"] == 5159
    assert report["frame"]["seed"] == 20261009


def test_backfill_replaces_unresolved_ids(tmp_path: Path) -> None:
    corpus = [f"c{i}" for i in range(10)]
    corpus_path = _write_jsonl(
        tmp_path / "corpus.jsonl", [{"id": cid} for cid in corpus]
    )
    gold_path = _write_jsonl(tmp_path / "gold.jsonl", [])
    store = _make_store(tmp_path, corpus)  # every id has a file
    manifest = {
        "seed": 7,
        "missing_ids": ["c1", "c2"],
        "sampled_ids": ["c0", "c1", "c2"],
    }
    replacements = sbr.backfill_ids(
        manifest,
        results=[],
        corpus_path=corpus_path,
        gold_path=gold_path,
        round_store=store,
        drop_ids=(),
    )
    assert len(replacements) == 2
    # Replacements are drawn from the frame, not from the dropped ids.
    assert set(replacements).isdisjoint({"c1", "c2"})
    assert set(replacements) <= set(corpus)


def test_backfill_ids_counts_live_unresolved(tmp_path: Path) -> None:
    """F1 regression at the seam: a live-unresolved id gets a replacement.

    ``s0`` has no ``missing_ids`` entry but failed the pass (unavailable
    reply). Passing it as ``unresolved`` must yield one replacement, and that
    replacement must not be the failed id itself.
    """
    corpus = [f"c{i}" for i in range(10)]
    corpus_path = _write_jsonl(
        tmp_path / "corpus.jsonl", [{"id": cid} for cid in corpus]
    )
    gold_path = _write_jsonl(tmp_path / "gold.jsonl", [])
    store = _make_store(tmp_path, corpus)
    manifest = {"seed": 7, "missing_ids": [], "sampled_ids": ["c0", "c1"]}
    replacements = sbr.backfill_ids(
        manifest,
        results=[],
        unresolved=["c0"],
        corpus_path=corpus_path,
        gold_path=gold_path,
        round_store=store,
        drop_ids=(),
    )
    assert len(replacements) == 1
    assert replacements[0] not in {"c0", "c1"}


def test_backfill_returns_empty_when_nothing_missing(tmp_path: Path) -> None:
    corpus = ["c0", "c1"]
    corpus_path = _write_jsonl(
        tmp_path / "corpus.jsonl", [{"id": cid} for cid in corpus]
    )
    gold_path = _write_jsonl(tmp_path / "gold.jsonl", [])
    store = _make_store(tmp_path, corpus)
    manifest = {"seed": 7, "missing_ids": [], "sampled_ids": ["c0"]}
    assert (
        sbr.backfill_ids(
            manifest,
            results=[],
            corpus_path=corpus_path,
            gold_path=gold_path,
            round_store=store,
            drop_ids=(),
        )
        == []
    )


def test_backfill_excludes_already_adjudicated_ids(tmp_path: Path) -> None:
    corpus = [f"c{i}" for i in range(10)]
    corpus_path = _write_jsonl(
        tmp_path / "corpus.jsonl", [{"id": cid} for cid in corpus]
    )
    gold_path = _write_jsonl(tmp_path / "gold.jsonl", [])
    store = _make_store(tmp_path, corpus)
    manifest = {"seed": 7, "missing_ids": ["c1"], "sampled_ids": ["c0", "c1"]}
    results = [{"id": f"c{i}"} for i in range(2, 9)]
    replacements = sbr.backfill_ids(
        manifest,
        results=results,
        corpus_path=corpus_path,
        gold_path=gold_path,
        round_store=store,
        drop_ids=(),
    )
    assert replacements == ["c9"]


def test_backfill_is_deterministic(tmp_path: Path) -> None:
    corpus = [f"c{i}" for i in range(20)]
    corpus_path = _write_jsonl(
        tmp_path / "corpus.jsonl", [{"id": cid} for cid in corpus]
    )
    gold_path = _write_jsonl(tmp_path / "gold.jsonl", [])
    store = _make_store(tmp_path, corpus)
    manifest = {"seed": 7, "missing_ids": ["c1"], "sampled_ids": ["c0", "c1"]}
    kwargs = dict(
        results=[],
        corpus_path=corpus_path,
        gold_path=gold_path,
        round_store=store,
        drop_ids=(),
    )
    assert sbr.backfill_ids(manifest, **kwargs) == sbr.backfill_ids(manifest, **kwargs)


def _adjudicator(verdicts: dict[str, dict], missing: set[str] | None = None):
    """An offline adjudicator: canned bins, and a missing-round failure mode."""
    missing = missing or set()

    def adjudicate(rid: str) -> dict:
        if rid in missing:
            raise FileNotFoundError(rid)
        # ``bin`` is the derived label (hyphenated), as ``bin_for`` produces.
        body = verdicts.get(rid, {"verdict": "no_fault_within_round"})
        return {"id": rid, "s2": body, "bin": body.get("bin", "no-fault-within-round")}

    return adjudicate


def test_run_adjudication_reaches_min_rows() -> None:
    manifest = {"seed": 7, "sampled_ids": [f"s{i}" for i in range(100)]}
    report = sbr.run_adjudication(
        manifest,
        adjudicate=_adjudicator({}),
        append=lambda rec: True,
        backfill=lambda m, r, u: [],
        min_rows=100,
    )
    assert report["n"] == 100
    assert report["denominator"] == 100
    assert report["k"] == 0


def test_run_adjudication_backfills_past_missing() -> None:
    manifest = {"seed": 7, "sampled_ids": [f"s{i}" for i in range(100)]}
    missing = {"s0", "s1"}
    # Backfill hands back two replacement ids that resolve fine.
    calls_seen = []

    def backfill(m, rows, unresolved):
        calls_seen.append(sorted(row["id"] for row in rows))
        return ["r0", "r1"]

    report = sbr.run_adjudication(
        manifest,
        adjudicate=_adjudicator({}, missing=missing),
        append=lambda rec: True,
        backfill=backfill,
        min_rows=100,
    )
    assert report["n"] == 100
    assert report["unresolved"] == ["s0", "s1"]
    # Backfill ran once, seeing the 98 survivors that the first pass appended.
    assert calls_seen == [sorted(f"s{i}" for i in range(2, 100))]


def test_run_adjudication_skips_already_done_ids() -> None:
    manifest = {"seed": 7, "sampled_ids": [f"s{i}" for i in range(100)]}
    done = [{"id": f"s{i}", "bin": "no-fault-within-round"} for i in range(99)]
    adjudicated = []

    def adjudicate(rid):
        adjudicated.append(rid)
        return {"id": rid, "bin": "prompt-misread"}

    report = sbr.run_adjudication(
        manifest,
        results=done,
        adjudicate=adjudicate,
        append=lambda rec: True,
        backfill=lambda m, r, u: [],
        min_rows=100,
    )
    assert adjudicated == ["s99"]
    assert report["n"] == 100
    assert report["k"] == 1


def test_run_adjudication_fails_loudly_when_backfill_dries_up() -> None:
    manifest = {"seed": 7, "sampled_ids": [f"s{i}" for i in range(3)]}
    with pytest.raises(SystemExit) as excinfo:
        sbr.run_adjudication(
            manifest,
            adjudicate=_adjudicator({}, missing={"s0", "s1", "s2"}),
            append=lambda rec: True,
            backfill=lambda m, r, u: [],
            min_rows=100,
        )
    assert "3" in str(excinfo.value) or "0" in str(excinfo.value)
    assert "100" in str(excinfo.value)


def test_run_adjudication_respects_append_guard() -> None:
    """A rejected append (duplicate id) must not inflate the count."""
    manifest = {"seed": 7, "sampled_ids": ["s0", "s1"]}
    appended = []

    def append(rec):
        if rec["id"] in appended:
            return False
        appended.append(rec["id"])
        return True

    report = sbr.run_adjudication(
        manifest,
        adjudicate=_adjudicator({}),
        append=append,
        backfill=lambda m, r, u: [],
        min_rows=2,
    )
    assert report["n"] == 2
    assert appended == ["s0", "s1"]


# --- unit-004: determinism + frame/gold cross-check --------------------------


def _gold_bin_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in sbr.read_jsonl(sbr.DEFAULT_GOLD):
        counts[row["bin"]] = counts.get(row["bin"], 0) + 1
    return counts


def test_determinism_manifest_rerun_is_byte_identical(tmp_path: Path) -> None:
    """Same seed + same inputs ⇒ byte-identical manifest, twice over."""
    corpus = [f"id{i:03d}" for i in range(60)]
    _write_jsonl(tmp_path / "corpus.jsonl", [{"id": rid} for rid in corpus])
    _write_jsonl(tmp_path / "gold.jsonl", [{"id": rid} for rid in corpus[:5]])
    store = _make_store(tmp_path, corpus[5:])
    kwargs = dict(
        seed=4242,
        sample_size=25,
        corpus_path=tmp_path / "corpus.jsonl",
        gold_path=tmp_path / "gold.jsonl",
        round_store=store,
        drop_ids=set(),
    )
    a = sbr.build_manifest(**kwargs)
    b = sbr.build_manifest(**kwargs)
    assert a == b
    pa = sbr.write_manifest(a, tmp_path / "a.json")
    pb = sbr.write_manifest(b, tmp_path / "b.json")
    assert pa.read_bytes() == pb.read_bytes()


def test_determinism_different_seed_changes_manifest(tmp_path: Path) -> None:
    """Control: a different seed must actually move the draw."""
    corpus = [f"id{i:03d}" for i in range(60)]
    _write_jsonl(tmp_path / "corpus.jsonl", [{"id": rid} for rid in corpus])
    _write_jsonl(tmp_path / "gold.jsonl", [{"id": rid} for rid in corpus[:5]])
    store = _make_store(tmp_path, corpus[5:])
    base = dict(
        sample_size=25,
        corpus_path=tmp_path / "corpus.jsonl",
        gold_path=tmp_path / "gold.jsonl",
        round_store=store,
        drop_ids=set(),
    )
    a = sbr.build_manifest(seed=1, **base)
    b = sbr.build_manifest(seed=2, **base)
    assert a["sampled_ids"] != b["sampled_ids"]


def test_determinism_run_adjudication_is_reproducible() -> None:
    """A seeded backfill makes the offline pass reproducible end to end."""
    manifest = {"seed": 11, "sampled_ids": ["s0", "s1", "s2"]}

    def backfill(m, rows, unresolved):
        return ["s3", "s4"]

    kwargs = dict(
        adjudicate=_adjudicator({}, missing={"s2"}),
        append=lambda rec: True,
        backfill=backfill,
        min_rows=4,
    )
    report_a = sbr.run_adjudication(manifest, **kwargs)
    report_b = sbr.run_adjudication(manifest, **kwargs)
    assert report_a == report_b


def test_determinism_committed_manifest_reproduces_from_seed() -> None:
    """The committed manifest is byte-reproducible from seed + its own ledger.

    This is the determinism obligation anchored on a committed artifact. U1 moved
    the derivation from the live round store to the snapshot's recorded frame, so
    the obligation now holds on a moved store — and in a clean clone with no
    store at all — instead of only on the machine C2 ran on.
    """
    if not sbr.DEFAULT_CORPUS.exists() or not sbr.DEFAULT_GOLD.exists():
        pytest.skip("committed corpus/gold artifacts not present")
    if not sbr.DEFAULT_MANIFEST.exists():
        pytest.skip("committed sample manifest not present")
    committed_bytes = sbr.DEFAULT_MANIFEST.read_bytes()
    committed = json.loads(committed_bytes.decode("utf-8"))
    rebuilt = sbr.reproduce_snapshot(committed)
    assert rebuilt == committed
    assert rebuilt["sampled_ids"] == committed["sampled_ids"]
    assert rebuilt["missing_ids"] == committed["missing_ids"]
    assert rebuilt["frame_size"] == committed["frame_size"]
    # Byte-for-byte, not just field-for-field: the shipped file is the pin.
    assert sbr.manifest_bytes(rebuilt) == committed_bytes


def test_cross_check_frame_arithmetic_matches_fresh_computation(tmp_path: Path) -> None:
    """The manifest's recorded arithmetic equals a fresh eligible-frame pass."""
    corpus = [f"id{i:03d}" for i in range(50)]
    _write_jsonl(tmp_path / "corpus.jsonl", [{"id": rid} for rid in corpus])
    _write_jsonl(tmp_path / "gold.jsonl", [{"id": rid} for rid in corpus[:4]])
    # Gold ids are known rounds: their files exist. The exact subtraction
    # |corpus| - |gold| - |missing| holds only when gold and missing are
    # disjoint, which is the real-world invariant.
    store = _make_store(tmp_path, corpus)
    manifest = sbr.build_manifest(
        sample_size=10,
        corpus_path=tmp_path / "corpus.jsonl",
        gold_path=tmp_path / "gold.jsonl",
        round_store=store,
        drop_ids=set(),
    )
    gold = {row["id"] for row in sbr.read_jsonl(tmp_path / "gold.jsonl")}
    missing = set(manifest["missing_ids"])
    fresh = sbr.eligible_frame(corpus, gold, missing)
    assert manifest["frame_size"] == len(fresh)
    assert manifest["corpus_size"] - manifest["gold_size"] - manifest["missing_size"] == len(fresh)


def test_cross_check_sample_ids_avoid_gold_and_missing(tmp_path: Path) -> None:
    """No sampled id may come from the excluded gold or missing sets."""
    corpus = [f"id{i:03d}" for i in range(60)]
    _write_jsonl(tmp_path / "corpus.jsonl", [{"id": rid} for rid in corpus])
    _write_jsonl(tmp_path / "gold.jsonl", [{"id": rid} for rid in corpus[:5]])
    store = _make_store(tmp_path, corpus[5:])
    manifest = sbr.build_manifest(
        sample_size=20,
        corpus_path=tmp_path / "corpus.jsonl",
        gold_path=tmp_path / "gold.jsonl",
        round_store=store,
        drop_ids=set(),
    )
    sampled = set(manifest["sampled_ids"])
    assert sampled.isdisjoint(corpus[:5])
    assert sampled.isdisjoint(manifest["missing_ids"])
    assert len(sampled) == manifest["sample_size"]


def test_cross_check_gold_split_reconciles_with_c1() -> None:
    """C1's published split: 242 rows = 29 positive + 205 negative + 8 excluded."""
    if not sbr.DEFAULT_GOLD.exists():
        pytest.skip("committed gold artifact not present")
    counts = _gold_bin_counts()
    positive = sum(n for b, n in counts.items() if sbr.classify_bin(b) == "positive")
    negative = sum(n for b, n in counts.items() if sbr.classify_bin(b) == "negative")
    excluded = sum(n for b, n in counts.items() if sbr.classify_bin(b) == "excluded")
    assert positive == 29
    assert negative == 205
    assert excluded == 8
    assert positive + negative + excluded == 242


def test_cross_check_gold_ids_size_matches_issue_anchor() -> None:
    """The 244 deduped gold ids = 242 labelled rows + 2 C1-dropped ids."""
    if not sbr.DEFAULT_GOLD.exists():
        pytest.skip("committed gold artifact not present")
    labelled = {row["id"] for row in sbr.read_jsonl(sbr.DEFAULT_GOLD)}
    assert len(labelled) == 242
    gold = sbr.gold_ids()
    assert len(gold) == sbr.EXPECTED_GOLD == 244
    assert gold == labelled | set(sbr.DROP_IDS)


def test_cross_check_gold_never_reaches_the_frame(tmp_path: Path) -> None:
    """Gold ids are excluded from the frame by construction (drift detector)."""
    corpus = [f"id{i:03d}" for i in range(30)]
    _write_jsonl(tmp_path / "corpus.jsonl", [{"id": rid} for rid in corpus])
    _write_jsonl(
        tmp_path / "gold.jsonl",
        [{"id": rid, "bin": "ambiguous"} for rid in corpus[:3]],
    )
    # Gold ids are known rounds; their files exist, so they leave the frame by
    # construction and never re-appear as "missing".
    store = _make_store(tmp_path, corpus)
    manifest = sbr.build_manifest(
        sample_size=5,
        corpus_path=tmp_path / "corpus.jsonl",
        gold_path=tmp_path / "gold.jsonl",
        round_store=store,
        drop_ids=set(),
    )
    # Gold stays out of the frame, and the subtraction stays exact because
    # gold and missing are disjoint (the round files for gold ids exist).
    assert set(manifest["sampled_ids"]).isdisjoint(corpus[:3])
    assert set(manifest["missing_ids"]).isdisjoint(corpus[:3])
    assert manifest["frame_size"] == len(corpus) - manifest["gold_size"]


# --- unit-007: pinned frame snapshot, loud store drift (U1) ------------------


def _snapshot_fixture(
    tmp_path: Path,
    corpus_n: int = 60,
    gold_n: int = 5,
    missing_n: int = 4,
    sample_size: int = 10,
    seed: int = 7,
):
    """A synthetic corpus/gold/store plus the manifest drawn from that store.

    Returns ``(snapshot, store, corpus_path, gold_path)``. The last
    ``missing_n`` corpus ids have no round file, so the ledger is non-empty; gold
    ids all have files, so gold and missing stay disjoint and the frame is
    ``corpus_n - gold_n - missing_n``. ``drop_ids`` is left at the module default
    so the fixture behaves exactly like the production call path.
    """
    corpus = [f"id{i:03d}" for i in range(corpus_n)]
    corpus_path = _write_jsonl(
        tmp_path / "corpus.jsonl", [{"id": rid} for rid in corpus]
    )
    gold_path = _write_jsonl(
        tmp_path / "gold.jsonl", [{"id": rid} for rid in corpus[:gold_n]]
    )
    # The absent rounds are spread through the frame, not clustered at its tail:
    # recovering one must actually shift frame positions, or a redraw could
    # coincidentally reproduce the same sample and the drift tests would prove
    # nothing. Gold ids always have files, so gold and missing stay disjoint.
    stride = (corpus_n - gold_n) // missing_n
    missing = {corpus[gold_n + i * stride] for i in range(missing_n)}
    store = _make_store(tmp_path, [rid for rid in corpus if rid not in missing])
    snapshot = sbr.build_manifest(
        seed=seed,
        sample_size=sample_size,
        corpus_path=corpus_path,
        gold_path=gold_path,
        round_store=store,
    )
    return snapshot, store, corpus_path, gold_path


def _reproduce(snapshot: dict, corpus_path: Path, gold_path: Path) -> dict:
    return sbr.reproduce_snapshot(
        snapshot, corpus_path=corpus_path, gold_path=gold_path
    )


def _drift(snapshot: dict, store: Path, corpus_path: Path, gold_path: Path) -> dict:
    return sbr.check_store_drift(
        snapshot, round_store=store, corpus_path=corpus_path, gold_path=gold_path
    )


def _recover(store: Path, ids: list[str]) -> None:
    """Simulate an upstream recovery: the round files for ``ids`` appear."""
    for rid in ids:
        (store / f"{rid}.json").write_text(
            json.dumps({"userPrompt": f"recovered-{rid}"})
        )


def _frame_victim(
    snapshot: dict, corpus_path: Path, gold_path: Path, store: Path
) -> str:
    """A frame member whose round file exists and that the sample did not draw.

    Deleting one moves the frame without touching the recorded sample, which is
    the distinction the drift check has to get right.
    """
    excluded = (
        set(snapshot["missing_ids"])
        | set(snapshot["sampled_ids"])
        | sbr.gold_ids(gold_path)
    )
    for rid in sbr.corpus_ids(corpus_path):
        if rid not in excluded and (store / f"{rid}.json").exists():
            return rid
    raise AssertionError("fixture must leave an unsampled round inside the frame")


def _diverged_fields(message: str) -> list[str]:
    """The field names a ``SnapshotError`` blames, parsed out of its message."""
    tail = message.split("— ", 1)[1]
    return [part.split(":", 1)[0].strip() for part in tail.split("; ")]


# The four frame numbers as issue #2 states them. Deliberately literals rather
# than sbr.EXPECTED_*: a test that reads the code's own anchor table cannot
# notice an anchor being dropped from it — the F1 lesson, re-used here.
ISSUE_TWO_FRAME = {
    "corpus_size": 5430,
    "gold_size": 244,
    "missing_size": 29,
    "frame_size": 5159,
}


@pytest.mark.parametrize("field", sorted(ISSUE_TWO_FRAME))
def test_assert_anchors_names_the_field_that_drifted(field: str) -> None:
    """Every frame anchor fails with its own field named — U1's loud failure."""
    manifest = dict(ISSUE_TWO_FRAME)
    manifest[field] += 1
    with pytest.raises(SystemExit) as excinfo:
        sbr.assert_anchors(manifest)
    message = str(excinfo.value)
    assert message.startswith(f"{field} drift: "), message
    assert f"expected {ISSUE_TWO_FRAME[field]}, got {ISSUE_TWO_FRAME[field] + 1}" in message


def test_assert_anchors_accepts_the_issue_two_numbers() -> None:
    """Control: the pinned frame numbers satisfy the anchors untouched."""
    sbr.assert_anchors(dict(ISSUE_TWO_FRAME))


def test_ledger_is_the_frame_authority_not_the_store(tmp_path: Path) -> None:
    """With the ledger recorded, no store can move the frame or the sample.

    The control half of the test is what makes it discriminate: the same store
    loss *does* move a store-derived frame, so a regression back to scanning
    would fail here rather than silently redrawing the C2 sample.
    """
    snapshot, store, corpus_path, gold_path = _snapshot_fixture(tmp_path)
    ledger = snapshot["missing_ids"]
    assert ledger, "fixture must record a non-empty ledger"

    # The ledger path never reads the store: a nonexistent one reproduces it.
    rebuilt = sbr.build_manifest(
        seed=snapshot["seed"],
        sample_size=snapshot["sample_size"],
        corpus_path=corpus_path,
        gold_path=gold_path,
        round_store=tmp_path / "no-such-store",
        missing_ids=ledger,
    )
    assert rebuilt == snapshot

    # Control: lose one frame round, and only the store-derived frame moves.
    victim = _frame_victim(snapshot, corpus_path, gold_path, store)
    (store / f"{victim}.json").unlink()
    redrawn = sbr.build_manifest(
        seed=snapshot["seed"],
        sample_size=snapshot["sample_size"],
        corpus_path=corpus_path,
        gold_path=gold_path,
        round_store=store,
    )
    assert redrawn["missing_size"] == snapshot["missing_size"] + 1
    assert redrawn["frame_size"] == snapshot["frame_size"] - 1
    assert redrawn["sampled_ids"] != snapshot["sampled_ids"]
    assert _reproduce(snapshot, corpus_path, gold_path) == snapshot


def test_pinned_snapshot_survives_the_loss_of_the_whole_store(
    tmp_path: Path,
) -> None:
    """Re-deriving the sample needs no round store: the ledger is the record."""
    snapshot, store, corpus_path, gold_path = _snapshot_fixture(tmp_path)
    shutil.rmtree(store)
    assert _reproduce(snapshot, corpus_path, gold_path) == snapshot


@pytest.mark.parametrize(
    "field, expected_named",
    [
        ("corpus_size", ["corpus_size"]),
        ("gold_size", ["gold_size"]),
        ("missing_size", ["missing_size"]),
        ("frame_size", ["frame_size"]),
        ("sampled_ids", ["sampled_ids"]),
        # A shortened ledger is an *input*, so it cannot diverge from itself:
        # what the report blames are the fields computed from it — and a wider
        # frame draws a different sample, which is exactly the silent redraw
        # U1 removed.
        ("missing_ids", ["missing_size", "frame_size", "sampled_ids"]),
    ],
)
def test_reproduce_snapshot_names_the_field_that_diverges(
    tmp_path: Path, field: str, expected_named: list[str]
) -> None:
    """ext 7a's rule for the frame pin: a mismatch names the field it broke."""
    snapshot, store, corpus_path, gold_path = _snapshot_fixture(tmp_path)
    tampered = dict(snapshot)
    if field == "missing_ids":
        tampered[field] = snapshot[field][1:]
    elif field == "sampled_ids":
        tampered[field] = list(reversed(snapshot[field]))
    else:
        tampered[field] = snapshot[field] + 1
    with pytest.raises(sbr.SnapshotError) as excinfo:
        _reproduce(tampered, corpus_path, gold_path)
    message = str(excinfo.value)
    assert _diverged_fields(message) == expected_named, message
    assert "recorded" in message and "re-derived" in message


def test_reproduce_snapshot_requires_a_complete_record(tmp_path: Path) -> None:
    """A pre-pin manifest without a ledger fails naming what is absent."""
    snapshot, store, corpus_path, gold_path = _snapshot_fixture(tmp_path)
    old_format = {"seed": snapshot["seed"], "sampled_ids": snapshot["sampled_ids"]}
    with pytest.raises(sbr.SnapshotError) as excinfo:
        _reproduce(old_format, corpus_path, gold_path)
    message = str(excinfo.value)
    assert "missing fields" in message
    for field in ("corpus_size", "missing_ids", "frame_size"):
        assert field in message, message


def test_reproduce_snapshot_keeps_the_ledger_canonical(tmp_path: Path) -> None:
    """A ledger that is not sorted-and-distinct is not the pinned ledger.

    ``build_manifest`` re-canonicalises the recorded ids before comparing, so a
    hand-reordered snapshot fails naming ``missing_ids`` instead of reproducing
    quietly in a form the byte gate would later have to reject.
    """
    snapshot, store, corpus_path, gold_path = _snapshot_fixture(tmp_path)
    reversed_ledger = dict(snapshot)
    reversed_ledger["missing_ids"] = list(reversed(snapshot["missing_ids"]))
    with pytest.raises(sbr.SnapshotError) as excinfo:
        _reproduce(reversed_ledger, corpus_path, gold_path)
    assert _diverged_fields(str(excinfo.value)) == ["missing_ids"]

    duplicated = dict(snapshot)
    duplicated["missing_ids"] = sorted(
        snapshot["missing_ids"] + snapshot["missing_ids"][:1]
    )
    duplicated["missing_size"] = len(duplicated["missing_ids"])
    with pytest.raises(sbr.SnapshotError) as excinfo:
        _reproduce(duplicated, corpus_path, gold_path)
    # The ledger is a *set* of ids: a duplicated entry inflates the count and
    # the recorded ledger itself, while the frame it excludes stays the same.
    assert _diverged_fields(str(excinfo.value)) == ["missing_size", "missing_ids"]


def test_ledger_is_sorted_whatever_order_the_corpus_arrives_in(
    tmp_path: Path,
) -> None:
    """One canonical form for the ledger, whatever the corpus file looks like.

    The pin's bytes must not depend on the order the ids happen to be listed in:
    a corpus re-sorted upstream would otherwise re-draw the ledger's shape and the
    byte gate would fire on formatting rather than on a changed sample.
    """
    corpus = [f"id{i:03d}" for i in range(30)][::-1]
    corpus_path = _write_jsonl(
        tmp_path / "corpus.jsonl", [{"id": rid} for rid in corpus]
    )
    gold_path = _write_jsonl(
        tmp_path / "gold.jsonl", [{"id": rid} for rid in corpus[:3]]
    )
    missing = {"id004", "id005", "id006"}
    store = _make_store(tmp_path, [rid for rid in corpus if rid not in missing])
    snapshot = sbr.build_manifest(
        sample_size=5, corpus_path=corpus_path, gold_path=gold_path, round_store=store
    )
    assert snapshot["missing_ids"] == ["id004", "id005", "id006"]
    assert snapshot["frame_size"] == 30 - 3 - 3


def test_check_store_drift_is_quiet_on_a_matching_store(tmp_path: Path) -> None:
    snapshot, store, corpus_path, gold_path = _snapshot_fixture(tmp_path)
    report = _drift(snapshot, store, corpus_path, gold_path)
    assert report["drifted_fields"] == []
    assert report["recovered_ids"] == []
    assert report["newly_missing_ids"] == []
    assert report["sampled_missing_ids"] == []
    assert report["live_frame_size"] == report["pinned_frame_size"]
    assert report["live_missing_size"] == report["recorded_missing_size"]
    assert "none" in sbr.drift_summary(report)


def test_check_store_drift_names_the_fields_a_moved_store_would_move(
    tmp_path: Path,
) -> None:
    """The C2 case: recovered rounds are disclosed by field name, not redrawn."""
    snapshot, store, corpus_path, gold_path = _snapshot_fixture(tmp_path)
    recovered = sorted(snapshot["missing_ids"][:2])
    _recover(store, recovered)
    report = _drift(snapshot, store, corpus_path, gold_path)
    assert set(report["drifted_fields"]) == {"missing_size", "missing_ids", "frame_size"}
    assert report["recorded_missing_size"] == snapshot["missing_size"]
    assert report["live_missing_size"] == snapshot["missing_size"] - 2
    assert report["recovered_ids"] == recovered
    assert report["newly_missing_ids"] == []
    assert report["pinned_frame_size"] == snapshot["frame_size"]
    assert report["live_frame_size"] == snapshot["frame_size"] + 2
    summary = sbr.drift_summary(report)
    for field in report["drifted_fields"]:
        assert field in summary, summary
    assert "not redrawn" in summary, summary


def test_check_store_drift_reports_a_round_that_went_missing(
    tmp_path: Path,
) -> None:
    """Loss outside the recorded sample is drift, named, but not fatal."""
    snapshot, store, corpus_path, gold_path = _snapshot_fixture(tmp_path)
    victim = _frame_victim(snapshot, corpus_path, gold_path, store)
    (store / f"{victim}.json").unlink()
    report = _drift(snapshot, store, corpus_path, gold_path)
    assert report["newly_missing_ids"] == [victim]
    assert report["recovered_ids"] == []
    assert "missing_ids" in report["drifted_fields"]
    assert report["sampled_missing_ids"] == []


def test_check_store_drift_fails_loudly_when_a_sampled_round_is_gone(
    tmp_path: Path,
) -> None:
    """A sampled id without a round file invalidates the pinned sample."""
    snapshot, store, corpus_path, gold_path = _snapshot_fixture(tmp_path)
    victim = snapshot["sampled_ids"][0]
    (store / f"{victim}.json").unlink()
    with pytest.raises(sbr.StoreDriftError) as excinfo:
        _drift(snapshot, store, corpus_path, gold_path)
    message = str(excinfo.value)
    assert "sampled_missing_ids" in message
    assert victim in message


def test_drift_never_changes_the_recorded_sample(tmp_path: Path) -> None:
    """U1's core claim: seed + pinned snapshot beat a moved live store."""
    snapshot, store, corpus_path, gold_path = _snapshot_fixture(tmp_path)
    _recover(store, snapshot["missing_ids"][:3])
    assert _reproduce(snapshot, corpus_path, gold_path)["sampled_ids"] == (
        snapshot["sampled_ids"]
    )
    # And the live store would have drawn something else from the wider frame.
    widened = sbr.build_manifest(
        seed=snapshot["seed"],
        sample_size=snapshot["sample_size"],
        corpus_path=corpus_path,
        gold_path=gold_path,
        round_store=store,
    )
    assert widened["frame_size"] == snapshot["frame_size"] + 3
    assert widened["sampled_ids"] != snapshot["sampled_ids"]


def test_manifest_bytes_are_the_pinned_serialization(tmp_path: Path) -> None:
    """The shipped manifest equals its own canonical rendering."""
    if not sbr.DEFAULT_MANIFEST.exists():
        pytest.skip("committed sample manifest not present")
    committed_bytes = sbr.DEFAULT_MANIFEST.read_bytes()
    committed = json.loads(committed_bytes.decode("utf-8"))
    assert sbr.manifest_bytes(committed) == committed_bytes
    assert sbr.write_manifest(committed, tmp_path / "m.json").read_bytes() == (
        committed_bytes
    )


def test_committed_snapshot_is_a_complete_pinned_record() -> None:
    """Every field the pin needs is in the committed artifact, and consistent."""
    if not sbr.DEFAULT_MANIFEST.exists():
        pytest.skip("committed sample manifest not present")
    if not sbr.DEFAULT_CORPUS.exists():
        pytest.skip("committed corpus artifact not present")
    committed = json.loads(sbr.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert set(committed) == set(sbr.SNAPSHOT_FIELDS)
    assert committed["missing_size"] == sbr.EXPECTED_MISSING
    assert committed["missing_size"] == len(committed["missing_ids"])
    # The ledger is pinned in canonical form: sorted and free of duplicates.
    assert committed["missing_ids"] == sorted(set(committed["missing_ids"]))
    assert committed["frame_size"] == sbr.EXPECTED_FRAME
    assert committed["corpus_size"] == sbr.EXPECTED_CORPUS
    assert committed["gold_size"] == sbr.EXPECTED_GOLD
    assert len(committed["sampled_ids"]) == committed["sample_size"]
    assert len(set(committed["sampled_ids"])) == committed["sample_size"]
    # The ledger is a claim about corpus ids, so it must stay inside its domain.
    assert set(committed["missing_ids"]) <= set(sbr.corpus_ids())
    assert set(committed["sampled_ids"]).isdisjoint(committed["missing_ids"])


def test_live_store_drift_is_named_and_leaves_the_sample_reproducible() -> None:
    """The real moved store: 15 of the 29 recorded rounds came back upstream.

    Deliberately date-independent: no live *count* is asserted (that was U1's
    disease). What must hold on any store is that the pinned sample still
    reproduces, that every recorded round is still adjudicable, and that
    whatever differs is named by field.
    """
    if not sbr.DEFAULT_CORPUS.exists() or not sbr.DEFAULT_GOLD.exists():
        pytest.skip("committed corpus/gold artifacts not present")
    if not sbr.DEFAULT_MANIFEST.exists() or not sbr.DEFAULT_ROUND_STORE.exists():
        pytest.skip("pinned manifest or round store not present")
    committed = json.loads(sbr.DEFAULT_MANIFEST.read_bytes().decode("utf-8"))
    report = sbr.check_store_drift(committed, round_store=sbr.DEFAULT_ROUND_STORE)
    assert report["sampled_missing_ids"] == [], "a sampled round file is gone"
    assert set(report["drifted_fields"]) <= {
        "missing_size",
        "missing_ids",
        "frame_size",
    }
    assert report["recorded_missing_size"] == sbr.EXPECTED_MISSING
    assert report["pinned_frame_size"] == sbr.EXPECTED_FRAME
    # Set algebra over the ledger: recovered ids leave, newly missing ones join.
    assert report["live_missing_size"] == (
        report["recorded_missing_size"]
        - len(report["recovered_ids"])
        + len(report["newly_missing_ids"])
    )
    assert sbr.reproduce_snapshot(committed)["sampled_ids"] == committed["sampled_ids"]


# --- unit-005: CLI wiring (--out, run_adjudication, key + cache) -------------


def _cli_fixture(tmp_path: Path, monkeypatch=None):
    """A minimal corpus/gold/store for CLI paths that don't need the live frame.

    The frame is *not* the live one, so callers must use ``--manifest-only``
    for the no-key path or inject fakes for the live path. When ``monkeypatch``
    is supplied, the live-anchor guard is neutralised and the live ``DROP_IDS``
    are dropped: ``assert_anchors`` demands the live 5430/244/5159 numbers, and
    ``DROP_IDS`` are live ids absent from a synthetic corpus. Both stay
    unconditional in production.
    """
    if monkeypatch is not None:
        monkeypatch.setattr(sbr, "assert_anchors", lambda manifest: None)
        monkeypatch.setattr(sbr, "DROP_IDS", frozenset())
    corpus = [f"id{i:03d}" for i in range(150)]
    _write_jsonl(tmp_path / "corpus.jsonl", [{"id": rid} for rid in corpus])
    _write_jsonl(tmp_path / "gold.jsonl", [{"id": rid} for rid in corpus[:10]])
    store = _make_store(tmp_path, corpus[10:])
    return [
        "--sample-size", "100",
        "--corpus", str(tmp_path / "corpus.jsonl"),
        "--gold", str(tmp_path / "gold.jsonl"),
        "--rounds-dir", str(store),
        "--manifest", str(tmp_path / "manifest.json"),
    ]


def test_cli_manifest_only_needs_no_api_key(tmp_path: Path, monkeypatch, capsys) -> None:
    """The frame+sample step must stay runnable with no key and no network."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rc = sbr.main(_cli_fixture(tmp_path, monkeypatch) + ["--manifest-only"])
    assert rc == 0
    assert (tmp_path / "manifest.json").exists()


def test_cli_without_key_fails_loudly(tmp_path: Path, monkeypatch) -> None:
    """Adjudication with no key must abort, not silently do nothing."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        sbr.main(_cli_fixture(tmp_path, monkeypatch))
    assert "key" in str(excinfo.value).lower()


def test_cli_out_flag_routes_results(tmp_path: Path, monkeypatch) -> None:
    """``--out`` must be the path adjudicated rows are appended to."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    out = tmp_path / "results.jsonl"
    seen_paths: list[str] = []

    class _Stub:
        def llm_request(self, system, user, api_key, model, max_retries=3):
            return "{}"

        def process_round(self, rid, rounds_dir, api_key, args, stages):
            return {"id": rid, "bin": "no-fault-within-round"}

    def fake_load_s2(path=None):
        return _Stub()

    def fake_run(manifest, results=(), adjudicate=None, append=None, backfill=None,
                 min_rows=100, max_rounds=5):
        seen_paths.append("called")
        # Drive the real append closure so --out routing is genuinely tested.
        for rid in manifest["sampled_ids"][:min_rows]:
            append(adjudicate(rid))
        return {"n": min_rows, "positive": 0, "negative": min_rows,
                "ambiguous": 0, "fault_rate": 0.0, "ci_low": 0.0,
                "ci_high": 0.0, "unresolved": []}

    monkeypatch.setattr(sbr, "load_s2_module", fake_load_s2)
    monkeypatch.setattr(sbr, "run_adjudication", fake_run)
    rc = sbr.main(_cli_fixture(tmp_path, monkeypatch) + ["--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert len(sbr.load_results(out)) == 100
    assert seen_paths == ["called"]


def test_cli_cache_round_trips_when_supplied(tmp_path: Path, monkeypatch) -> None:
    """A supplied ``--cache`` file is read and written around the run."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cache_path = tmp_path / "cache.json"
    cache_path.write_text('{"pre": "existing"}\n', encoding="utf-8")

    class _Stub:
        def llm_request(self, system, user, api_key, model, max_retries=3):
            return "{}"

        def process_round(self, rid, rounds_dir, api_key, args, stages):
            return {"id": rid, "bin": "no-fault-within-round"}

    monkeypatch.setattr(sbr, "load_s2_module", lambda path=None: _Stub())

    def fake_run(manifest, results=(), adjudicate=None, append=None, backfill=None,
                 min_rows=100, max_rounds=5):
        return {"n": min_rows, "positive": 0, "negative": min_rows,
                "ambiguous": 0, "fault_rate": 0.0, "ci_low": 0.0,
                "ci_high": 0.0, "unresolved": []}

    monkeypatch.setattr(sbr, "run_adjudication", fake_run)
    rc = sbr.main(_cli_fixture(tmp_path, monkeypatch) + ["--cache", str(cache_path)])
    assert rc == 0
    # The pre-existing entry survived the save (load-then-save, no clobber).
    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    assert saved.get("pre") == "existing"


def test_resolve_api_key_prefers_explicit_then_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert sbr.resolve_api_key("explicit") == "explicit"
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    assert sbr.resolve_api_key(None) == "from-env"
    assert sbr.resolve_api_key("explicit") == "explicit"


def test_adjudicate_one_wraps_null_completion_as_unavailable() -> None:
    """A null LLM reply must surface as AdjudicationUnavailable, not AttributeError.

    The provider can return ``choices[0].message.content == null``; the
    imported parser then calls ``text.strip()`` on ``None``. That transient
    failure must be a typed, catchable "unresolved" signal.
    """

    class _NullS2:
        def process_round(self, rid, rounds_dir, api_key, args, stages):
            raise AttributeError("'NoneType' object has no attribute 'strip'")

    with pytest.raises(sbr.AdjudicationUnavailable):
        sbr.adjudicate_one("r1", s2_module=_NullS2())


def test_run_adjudication_survives_unavailable_replies() -> None:
    """An unavailable reply drops the id as unresolved and is backfilled.

    F1 regression: the id that raised :class:`AdjudicationUnavailable` must
    appear in the unresolved set handed to ``backfill``, so a replacement is
    drawn and the adjudicated count still reaches ``min_rows``. Stubbing
    backfill to ``[]`` hid this — the loop ended with the shortfall instead.
    """
    manifest = {"seed": 3, "sampled_ids": ["s0", "s1", "s2"]}
    seen_unresolved: list[list[str]] = []

    def adjudicate(rid):
        if rid == "s1":
            raise sbr.AdjudicationUnavailable(rid)
        return {"id": rid, "bin": "no-fault-within-round"}

    def backfill(m, results, unresolved):
        seen_unresolved.append(sorted(unresolved))
        return ["r0"]

    report = sbr.run_adjudication(
        manifest,
        adjudicate=adjudicate,
        append=lambda rec: True,
        backfill=backfill,
        min_rows=3,
    )
    # The unavailable id was visible to backfill and replaced.
    assert seen_unresolved == [["s1"]]
    assert report["n"] == 3
    assert report["unresolved"] == ["s1"]


def test_cli_cache_is_persisted_after_each_row(tmp_path: Path, monkeypatch) -> None:
    """The cache file must exist during the run, not only on clean exit.

    The live pass is paid; a crash after N adjudications must leave a cache on
    disk. This asserts the file is created and stays valid JSON even when the
    adjudication seam never reaches ``llm_request`` (a stubbed S2 still drives
    the per-row save).
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cache_path = tmp_path / "cache.json"
    out = tmp_path / "results.jsonl"

    class _Stub:
        def llm_request(self, system, user, api_key, model, max_retries=3):
            return "{}"

        def process_round(self, rid, rounds_dir, api_key, args, stages):
            return {"id": rid, "bin": "no-fault-within-round"}

    monkeypatch.setattr(sbr, "load_s2_module", lambda path=None: _Stub())

    rc = sbr.main(
        _cli_fixture(tmp_path, monkeypatch)
        + [
            "--out", str(out),
            "--cache", str(cache_path),
            "--min-rows", "3",
        ]
    )
    assert rc == 0
    assert cache_path.exists()
    assert isinstance(json.loads(cache_path.read_text(encoding="utf-8")), dict)
    # The synthetic run must land in tmp_path, never the committed artifact.
    # ``run_adjudication`` walks the whole resume plan before consulting
    # ``min_rows`` (which only gates backfill), so all 100 sampled ids land.
    assert len(sbr.load_results(out)) == 100


# --- unit-007: CLI wiring of the pinned snapshot (U1) ------------------------


def _cli_snapshot_fixture(tmp_path: Path, monkeypatch):
    """CLI args pointing at a *pinned* snapshot already written on disk.

    Returns ``(argv, snapshot, store, corpus_path, gold_path, manifest_path)``.
    ``assert_anchors`` is neutralised because the synthetic frame cannot match the
    live 5430/244/29/5159 anchors; the snapshot path itself is the real one.
    """
    snapshot, store, corpus_path, gold_path = _snapshot_fixture(tmp_path)
    monkeypatch.setattr(sbr, "assert_anchors", lambda manifest: None)
    manifest_path = tmp_path / "manifest.json"
    sbr.write_manifest(snapshot, manifest_path)
    argv = [
        "--seed", str(snapshot["seed"]),
        "--sample-size", str(snapshot["sample_size"]),
        "--corpus", str(corpus_path),
        "--gold", str(gold_path),
        "--rounds-dir", str(store),
        "--manifest", str(manifest_path),
    ]
    return argv, snapshot, store, corpus_path, gold_path, manifest_path


def test_cli_reproduces_the_pinned_snapshot_and_prints_the_drift(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A moved store on the pinned path: disclose by name, keep the sample."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    argv, snapshot, store, _, _, manifest_path = _cli_snapshot_fixture(
        tmp_path, monkeypatch
    )
    before = manifest_path.read_bytes()
    _recover(store, snapshot["missing_ids"][:2])
    rc = sbr.main(argv + ["--manifest-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "store drift" in out
    for field in ("missing_size", "missing_ids", "frame_size"):
        assert field in out, out
    assert "source=pinned snapshot" in out
    assert manifest_path.read_bytes() == before, "the pinned bytes moved"


def test_cli_fresh_manifest_path_still_draws_from_the_store(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Nothing pinned at the path: the frame is scanned, and the pin is written."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    argv, snapshot, store, _, _, manifest_path = _cli_snapshot_fixture(
        tmp_path, monkeypatch
    )
    manifest_path.unlink()
    rc = sbr.main(argv + ["--manifest-only"])
    assert rc == 0
    assert "source=live store scan" in capsys.readouterr().out
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written == snapshot


def test_cli_refuses_to_redraw_a_pinned_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    """The pin protects itself: a redraw needs a new manifest path."""
    argv, snapshot, store, _, _, manifest_path = _cli_snapshot_fixture(
        tmp_path, monkeypatch
    )
    before = manifest_path.read_bytes()
    with pytest.raises(SystemExit) as excinfo:
        sbr.main(argv + ["--manifest-only", "--redraw"])
    message = str(excinfo.value)
    assert "pinned snapshot" in message
    assert "new --manifest path" in message
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    "flag, field",
    [("--seed", "seed"), ("--sample-size", "sample_size")],
)
def test_cli_draw_request_conflicting_with_the_pin_names_the_field(
    tmp_path: Path, monkeypatch, flag: str, field: str
) -> None:
    """Asking for a different seed over a pinned snapshot is a loud error."""
    argv, snapshot, store, _, _, manifest_path = _cli_snapshot_fixture(
        tmp_path, monkeypatch
    )
    before = manifest_path.read_bytes()
    other = "999" if flag == "--seed" else "4"
    with pytest.raises(SystemExit) as excinfo:
        # argparse keeps the last occurrence, so this overrides the fixture's value.
        sbr.main(argv + [flag, other, "--manifest-only"])
    message = str(excinfo.value)
    assert "conflict with the pinned snapshot" in message
    assert field in message, message
    assert "--redraw" in message
    assert manifest_path.read_bytes() == before


def test_cli_snapshot_that_would_change_its_bytes_is_not_written(
    tmp_path: Path, monkeypatch
) -> None:
    """Same data, different bytes: refuse before writing, do not normalise it.

    A hand-reformatted snapshot still reproduces as a manifest, so only the
    byte gate keeps the committed file's identity — and it must fire *before* the
    file is touched, or the run would rewrite history under the name of pinning.
    """
    argv, snapshot, store, _, _, manifest_path = _cli_snapshot_fixture(
        tmp_path, monkeypatch
    )
    reformatted = json.dumps(snapshot, indent=4, ensure_ascii=False) + "\n"
    assert reformatted.encode() != sbr.manifest_bytes(snapshot)
    manifest_path.write_text(reformatted, encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        sbr.main(argv + ["--manifest-only"])
    assert "would change" in str(excinfo.value)
    assert manifest_path.read_text(encoding="utf-8") == reformatted


def test_cli_store_that_lost_a_sampled_round_fails_loudly(
    tmp_path: Path, monkeypatch
) -> None:
    """The one drift the pin cannot absorb: a sampled round file is gone."""
    argv, snapshot, store, _, _, manifest_path = _cli_snapshot_fixture(
        tmp_path, monkeypatch
    )
    victim = snapshot["sampled_ids"][0]
    (store / f"{victim}.json").unlink()
    with pytest.raises(sbr.StoreDriftError) as excinfo:
        sbr.main(argv + ["--manifest-only"])
    assert "sampled_missing_ids" in str(excinfo.value)
    assert victim in str(excinfo.value)
