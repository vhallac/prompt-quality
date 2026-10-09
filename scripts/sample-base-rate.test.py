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
    """Frame arithmetic against the committed corpus and live round store."""
    if not sbr.DEFAULT_CORPUS.exists() or not sbr.DEFAULT_GOLD.exists():
        pytest.skip("committed corpus/gold artifacts not present")
    if not sbr.DEFAULT_ROUND_STORE.exists():
        pytest.skip("round store not present")
    manifest = sbr.build_manifest(sample_size=1)
    assert manifest["corpus_size"] == sbr.EXPECTED_CORPUS
    assert manifest["gold_size"] == sbr.EXPECTED_GOLD
    assert manifest["missing_size"] == 29
    assert manifest["frame_size"] == sbr.EXPECTED_FRAME
