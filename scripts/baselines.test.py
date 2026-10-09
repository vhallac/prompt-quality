"""C3 baseline tests: metrics core + gold loader + lexical baselines."""

import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import pytest

# Import baselines.py despite a future hyphen-free name; keep the file-path
# import pattern used by the other C-stage test modules.
_MODULE_PATH = Path(__file__).with_name("baselines.py")
_spec = importlib.util.spec_from_file_location("baselines", _MODULE_PATH)
baselines = importlib.util.module_from_spec(_spec)
sys.modules["baselines"] = baselines
_spec.loader.exec_module(baselines)

from baselines import (
    FEATURE_NAMES,
    JEV_VARIANTS,
    RESPONSE_MARKER,
    assemble_baselines,
    auc,
    base_rate_threshold,
    baseline_metrics,
    bootstrap_ci,
    combined_lexical_scores,
    constant_prevalence_scores,
    deixis_density,
    emit_artifacts,
    fit_logistic,
    fnr,
    imperative_density,
    length_feature,
    lexical_features,
    load_corpus_base_rate,
    load_gold,
    load_gold_status_ids,
    logistic_score,
    operating_points,
    output_contract_absence,
    paired_auc_delta,
    parent_response_state,
    prompt_only_state,
    jev_flag_score,
    jev_cache_key,
    load_jev_cache,
    own_response_state,
    query_jev,
    rescore_rows,
    save_jev_cache,
    sensitivity,
    serialize_metrics,
    serialize_score_rows,
    specificity,
    youden_threshold,
)


class TestMetricsAuc:
    def test_perfect_separation(self):
        assert auc([1.0, 2.0], [0, 1]) == 1.0

    def test_inverted(self):
        assert auc([1.0, 2.0], [1, 0]) == 0.0

    def test_hand_computed_tie(self):
        # pos {0.8, 0.6} vs neg {0.6, 0.4}:
        # 0.8>0.6 ✓, 0.8>0.4 ✓, 0.6 vs 0.6 tie (0.5), 0.6>0.4 ✓ → 3.5/4
        assert auc([0.8, 0.6, 0.6, 0.4], [1, 1, 0, 0]) == pytest.approx(0.875)

    def test_all_ties_is_chance(self):
        assert auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.5)

    def test_empty_class_returns_none(self):
        assert auc([1.0, 2.0], [1, 1]) is None
        assert auc([1.0, 2.0], [0, 0]) is None

    def test_order_independent(self):
        a = auc([0.1, 0.4, 0.35, 0.8], [1, 0, 0, 1])
        b = auc([0.8, 0.35, 0.4, 0.1], [1, 0, 0, 1])
        assert a == b


class TestMetricsFnr:
    def test_at_extreme_low_threshold_zero_fnr(self):
        assert fnr([0.2, 0.9], [1, 1], 0.1) == 0.0

    def test_hand_computed(self):
        # pos [0.2, 0.9], threshold 0.5 → 1 of 2 missed
        assert fnr([0.2, 0.9], [1, 1], 0.5) == pytest.approx(0.5)

    def test_no_positives_raises(self):
        with pytest.raises(ValueError):
            fnr([0.1], [0], 0.5)


class TestMetricsOperatingPoints:
    def test_youden_picks_perfect_threshold(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        t = youden_threshold(scores, labels)
        assert t == pytest.approx(0.5)  # midpoint 0.2/0.8
        assert sensitivity(scores, labels, t) == 1.0
        assert specificity(scores, labels, t) == 1.0

    def test_youden_tie_resolves_low(self):
        # every candidate threshold yields J=0 → lowest (0.2) wins
        scores = [0.2, 0.4, 0.6, 0.8]
        labels = [1, 0, 1, 0]
        assert youden_threshold(scores, labels) == pytest.approx(0.2)

    def test_base_rate_threshold_hits_alarm_rate(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = [0] * 5 + [1] * 5
        t = base_rate_threshold(scores, labels, 0.5)
        alarmed = sum(1 for s in scores if s >= t) / len(scores)
        assert alarmed == pytest.approx(0.5)

    def test_base_rate_threshold_tie_resolves_low(self):
        scores = [0.1, 0.1, 0.2, 0.2]
        labels = [0, 1, 0, 1]
        # alarm rates: t=0.1 → 1.0, t=0.15 → 0.5, t=0.25 → 0.0
        assert base_rate_threshold(scores, labels, 0.5) == pytest.approx(0.15)

    def test_base_rate_threshold_can_alarm_nothing(self):
        # Constant score vector: the only candidate that alarms any row alarms
        # every row (rate 1.0). With a small target the nearest alarm rate is
        # 0.0, reachable only by the above-max threshold. Regression for the
        # floor baseline's corpus-base-rate operating point.
        scores = [0.1239] * 10
        labels = [1, 0] * 5
        t = base_rate_threshold(scores, labels, 0.1939)
        assert t > max(scores)
        assert sum(1 for s in scores if s >= t) == 0

    def test_operating_points_names_both(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        ops = operating_points(scores, labels, corpus_base_rate=0.25)
        assert set(ops) == {"youden_j", "corpus_base_rate"}


class TestMetricsBootstrapCi:
    def test_deterministic_same_seed(self):
        scores = [0.1 * i for i in range(20)]
        labels = [1 if i % 3 == 0 else 0 for i in range(20)]
        a = bootstrap_ci(scores, labels, auc, n_boot=200, seed=7)
        b = bootstrap_ci(scores, labels, auc, n_boot=200, seed=7)
        assert a == b

    def test_ci_brackets_point_and_ordered(self):
        scores = [0.1 * i for i in range(40)]
        labels = [1 if i % 5 == 0 else 0 for i in range(40)]
        ci = bootstrap_ci(scores, labels, auc, n_boot=300, seed=1)
        assert ci is not None
        lo, hi = ci["ci95"]
        assert lo <= ci["point"] <= hi
        assert lo < hi

    def test_perfectly_separated_scores_point_is_one(self):
        scores = [0.1, 0.2, 0.9, 1.0]
        labels = [0, 0, 1, 1]
        ci = bootstrap_ci(scores, labels, auc, n_boot=100, seed=3)
        assert ci is not None
        assert ci["point"] == 1.0
        assert ci["ci95"][0] >= 0.5  # resamples rarely flip with 2v2

    def test_degenerate_single_class_resamples_skipped(self):
        # 1 positive, many negatives: ~1/e of resamples lose the positive
        scores = [0.5] + [0.1 * (i + 1) for i in range(30)]
        labels = [1] + [0] * 30
        ci = bootstrap_ci(scores, labels, auc, n_boot=500, seed=11)
        assert ci is not None
        assert ci["degenerate_resamples"] > 0
        assert ci["degenerate_resamples"] + len([1]) <= 500

    def test_empty_input_returns_none(self):
        assert bootstrap_ci([], [], auc) is None

    def test_fnr_as_metric(self):
        scores = [0.1 * i for i in range(30)]
        labels = [1 if i % 4 == 0 else 0 for i in range(30)]
        t = youden_threshold(scores, labels)
        ci = bootstrap_ci(scores, labels, lambda s, l: fnr(s, l, t), n_boot=200, seed=5)
        assert ci is not None
        lo, hi = ci["ci95"]
        assert 0.0 <= lo <= ci["point"] <= hi <= 1.0


# ---------------------------------------------------------------------------
# unit-002 — gold loader, baseline (a), baseline (b) lexical features
# ---------------------------------------------------------------------------

def _write_gold(tmp_path, pos=29, neg=205, exc=8):
    rows = []
    for i in range(pos):
        rows.append({"id": f"p{i}", "label": "positive", "prompt": f"fix p{i}", "prompt_status": "resolved"})
    for i in range(neg):
        rows.append({"id": f"n{i}", "label": "negative", "prompt": f"hello n{i}", "prompt_status": "resolved"})
    for i in range(exc):
        rows.append({"id": f"e{i}", "label": "excluded", "prompt": "", "prompt_status": "resolved"})
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


class TestGoldLoader:
    """Loader contract: split arithmetic, 'run C1 first', excluded dropped."""

    def test_loader_split_and_excluded_dropped(self, tmp_path):
        path = _write_gold(tmp_path)
        rows, tally = load_gold(path)
        assert len(rows) == 234 and tally == {"positive": 29, "negative": 205, "excluded": 8}
        assert all(r["label"] in ("positive", "negative") for r in rows)

    def test_loader_missing_file_run_c1_first(self, tmp_path):
        with pytest.raises(RuntimeError, match="run C1 first"):
            load_gold(str(tmp_path / "absent.jsonl"))

    def test_loader_drift_fails_loudly(self, tmp_path):
        path = _write_gold(tmp_path, pos=28)
        with pytest.raises(ValueError, match="drifted"):
            load_gold(path)

    def test_loader_empty_prompt_in_scored_row_fails(self, tmp_path):
        path = _write_gold(tmp_path)
        lines = (tmp_path / "gold.jsonl").read_text().splitlines()
        bad = json.loads(lines[0]); bad["prompt"] = ""
        lines[0] = json.dumps(bad)
        (tmp_path / "gold.jsonl").write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="run C1 first"):
            load_gold(path)


class TestConstantPrevalence:
    """Baseline (a): constant score at gold prevalence; AUC = 0.5."""

    def test_scores_equal_prevalence_and_auc_half(self):
        rows = [{"label": "positive", "prompt": "x"}] * 3 + [{"label": "negative", "prompt": "y"}] * 7
        scores = constant_prevalence_scores(rows)
        assert scores == [0.3] * 10
        labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        assert auc(scores, labels) == 0.5


class TestLexicalFeatures:
    """Feature definitions from issue #3, on fixed synthetic prompts."""

    def test_length_tokens_and_chars(self):
        f = length_feature("fix the bug now")
        assert f == {"length": 4, "length_chars": 15}

    def test_imperative_density_fraction_of_clauses(self):
        # two clauses, first imperative, second declarative -> 0.5
        assert imperative_density("Fix the typo. The weather is nice") == 0.5
        assert imperative_density("The weather is nice. Run the tests") == 0.5
        assert imperative_density("") == 0.0
        # non-imperative first tokens never match
        assert imperative_density("Fixing it quickly") == 0.0

    def test_deixis_per_100_tokens(self):
        # 4 tokens, one deictic hit ("it") -> 100*1/4 = 25.0
        assert deixis_density("look at it now") == 25.0
        # phrase marker "the previous" counts once as phrase, not as "the"
        assert deixis_density("check the previous round again") == 100.0 * 2 / 5  # phrase + again
        # word "that" as pronoun, not inside "that's" only if word-bounded
        assert deixis_density("that is it") == 100.0 * 2 / 3
        assert deixis_density("") == 0.0

    def test_output_contract_absence(self):
        assert output_contract_absence("fix the bug") == 1.0
        assert output_contract_absence("fix the bug and write it to out.json") == 0.0
        assert output_contract_absence("update scripts/foo.py") == 0.0
        assert output_contract_absence("the output must be a table") == 0.0
        assert output_contract_absence("rename x to y") == 1.0

    def test_lexical_features_returns_all_four(self):
        # 'module.py' contains a sentence-splitting dot: two clauses, one imperative
        f = lexical_features("fix the bug in module.py")
        assert set(f) >= set(FEATURE_NAMES)
        assert f["output_contract_absence"] == 0.0
        assert f["imperative_density"] == 0.5


class TestLogisticCombiner:
    """Zero-dependency logistic fit: deterministic, separates synthetic data."""

    def test_fit_deterministic_and_separates(self):
        X = [[0.0], [0.1], [0.9], [1.0]]
        y = [0, 0, 1, 1]
        w1 = fit_logistic(X, y)
        w2 = fit_logistic(X, y)
        assert w1 == w2
        s = [logistic_score(w1, x) for x in X]
        # positives rank high
        assert auc(s, y) == 1.0

    def test_fit_logistic_is_not_seed_sensitive(self):
        # The fit never samples, so it must not expose a seed: a published
        # seed on the combiner would present the weights as seed-re-drawable.
        params = inspect.signature(fit_logistic).parameters
        assert "seed" not in params
        assert not hasattr(_b, "LOGISTIC_SEED")

    def test_standardisation_zero_std_feature(self):
        # a constant feature must not blow up: std 0 -> published as 0, and its
        # standardised value is 0 so it cannot contribute to any logit
        rows = ([{"label": "positive", "prompt": "fix it"}] * 3
                + [{"label": "negative", "prompt": "fine day"}] * 3)
        scores, combiner = combined_lexical_scores(rows)
        assert len(scores) == 6
        assert set(combiner) == {"intercept", "features"}
        assert set(combiner["features"]) == set(FEATURE_NAMES)
        assert math.isfinite(combiner["intercept"])
        assert all(math.isfinite(s) for s in scores)
        raw = [{n: lexical_features(r["prompt"])[n] for n in FEATURE_NAMES} for r in rows]
        for name in FEATURE_NAMES:
            values = {row[name] for row in raw}
            spec = combiner["features"][name]
            assert math.isfinite(spec["weight"])
            if len(values) == 1:
                assert spec["std"] == 0.0
                assert spec["mean"] == pytest.approx(next(iter(values)))


class TestGoldBaselineEndToEnd:
    """Loader + baselines over the synthetic gold fixture."""

    def test_constant_and_lexical_over_fixture(self, tmp_path):
        path = _write_gold(tmp_path)
        rows, _ = load_gold(path)
        scores = constant_prevalence_scores(rows)
        labels = [1 if r["label"] == "positive" else 0 for r in rows]
        assert auc(scores, labels) == 0.5
        lex, _ = combined_lexical_scores(rows)
        assert len(lex) == len(rows)


# ---------------------------------------------------------------------------
# unit-003: jev re-score harness
# ---------------------------------------------------------------------------

import baselines as _b  # module handle for constants/seams


def _gold_rows():
    return [
        {"id": "a" * 32, "label": "positive", "prompt": "Fix the failing test."},
        {"id": "b" * 32, "label": "negative", "prompt": "interesting. look at it now."},
    ]


def _round_file(rid, prompt, resp_seq="assistant text", segments=None, parent_id=None):
    data = {
        "id": rid,
        "userPrompt": prompt,
        "responseSequence": resp_seq,
        "responseSegments": segments if segments is not None else [
            {"type": "text", "text": resp_seq},
        ],
        "parentId": parent_id,
    }
    return data


def test_prompt_only_state_uses_constant_marker():
    s1 = prompt_only_state("Fix the bug.")
    s2 = prompt_only_state("Fix the bug very differently.")
    assert s1 == "USER PROMPT:\nFix the bug.\n\nASSISTANT RESPONSE:\n[not shown]"
    assert RESPONSE_MARKER in s1
    # the state is a function of the prompt alone: different rounds with the
    # same prompt but different responses produce the same state
    own = own_response_state("Fix the bug.", _round_file("a" * 32, "Fix the bug.", "response A"))
    assert own != prompt_only_state("Fix the bug.")
    assert "response A" not in prompt_only_state("Fix the bug.")


def test_own_response_state_matches_jev_round_scan_shape():
    data = _round_file("a" * 32, "Fix the bug.", "the fix is here")
    assert (
        own_response_state("Fix the bug.", data)
        == "USER PROMPT:\nFix the bug.\n\nASSISTANT RESPONSE:\nthe fix is here"
    )
    # string-typed responseSequence (the current store shape) is used verbatim
    data_str = dict(data, responseSequence="plain string response")
    assert own_response_state("q", data_str).endswith("plain string response")


def test_parent_response_state_redacts_tool_calls():
    parent = {
        "responseSegments": [
            {"type": "text", "text": "first text"},
            {"type": "toolCall", "tool": "bash"},
            {"type": "text", "text": "second text"},
        ],
    }
    s = parent_response_state("Fix the bug.", parent)
    assert s.startswith("USER PROMPT:\nFix the bug.\n\nPREVIOUS ROUND RESPONSE")
    assert "first text" in s and "second text" in s
    assert "[1 tool calls in the parent response were redacted]" in s


def test_parent_response_state_truncates_to_state_chars():
    parent = {"responseSegments": [{"type": "text", "text": "x" * 20000}]}
    s = parent_response_state("prompt", parent)
    assert len(s) == _b.JEV_STATE_CHARS


def test_jev_flag_score_is_monotone_max():
    assert jev_flag_score({"correction": {"noul": 0.2}, "frustration": {"score": 1.0}}) == 0.25
    assert jev_flag_score({"correction": {"noul": 0.9}, "frustration": {"score": 1.0}}) == 0.9
    # scan flag rule (fr>=3 or corr>=0.7) implies score >= 0.7
    assert jev_flag_score({"correction": {"noul": 0.0}, "frustration": {"score": 3.0}}) >= 0.7
    assert jev_flag_score({}) == 0.0


def test_rescore_prompt_only_never_touches_round_store(tmp_path):
    calls = []

    def fake_query(state, api_key, questions=None):
        calls.append(state)
        return {"correction": {"noul": 0.8}, "frustration": {"score": 0.0}}

    cache = {}
    # empty round store: prompt-only still scores
    recs = rescore_rows(_gold_rows(), "prompt_only", cache, "key",
                        query_fn=fake_query, round_store=tmp_path)
    assert len(calls) == 2
    assert [r["score"] for r in recs] == [0.8, 0.8]
    assert all(r["reason"] is None for r in recs)
    # both states were cached
    assert len(cache) == 2


def test_rescore_cache_hit_avoids_api_and_roundtrips(tmp_path):
    calls = []

    def fake_query(state, api_key, questions=None):
        calls.append(state)
        return {"correction": {"noul": 0.6}, "frustration": {"score": 2.0}}

    cache = {}
    first = rescore_rows(_gold_rows(), "prompt_only", cache, "key",
                         query_fn=fake_query, round_store=tmp_path)
    assert len(calls) == 2
    path = save_jev_cache(cache, tmp_path / "jev-cache.json")
    loaded = load_jev_cache(path)
    assert loaded == cache
    second = rescore_rows(_gold_rows(), "prompt_only", loaded, "key",
                          query_fn=fake_query, round_store=tmp_path)
    assert len(calls) == 2  # no additional calls
    assert [r["score"] for r in second] == [r["score"] for r in first]


def test_rescore_retries_then_succeeds(tmp_path):
    attempts = []

    def flaky(state, api_key, questions=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("boom")
        return {"correction": {"noul": 0.9}, "frustration": {"score": 0.0}}

    recs = rescore_rows(_gold_rows()[:1], "prompt_only", {}, "key",
                        query_fn=flaky, round_store=tmp_path)
    assert len(attempts) == 3
    assert recs[0]["score"] == 0.9


def test_rescore_api_failure_yields_null_score(tmp_path):
    attempts = []

    def dead(state, api_key, questions=None):
        attempts.append(1)
        raise OSError("network down")

    recs = rescore_rows(_gold_rows(), "prompt_only", {}, "key",
                        query_fn=dead, round_store=tmp_path)
    assert len(attempts) == 2 * _b.JEV_MAX_RETRIES
    assert all(r["score"] is None and r["reason"] == "api_error" for r in recs)


def test_rescore_own_response_missing_round_file(tmp_path):
    recs = rescore_rows(_gold_rows(), "own_response", {}, "key",
                        query_fn=lambda *a, **k: {}, round_store=tmp_path)
    assert all(r["score"] is None and r["reason"] == "missing_round" for r in recs)


def test_rescore_parent_response_missing_or_parentless(tmp_path):
    rows = [
        {"id": "p" * 32, "label": "negative", "prompt": "parentless prompt"},
        {"id": "c" * 32, "label": "positive", "prompt": "child prompt"},
    ]
    store = tmp_path / "rounds"
    store.mkdir()
    # child points at a parent that does not exist
    (store / ("c" * 32 + ".json")).write_text(json.dumps(
        _round_file("c" * 32, "child prompt", parent_id="p" * 32 + ".json")))
    recs = rescore_rows(rows, "parent_response", {}, "key",
                        query_fn=lambda *a, **k: {"correction": {"noul": 0.5}},
                        round_store=store)
    by_id = {r["id"]: r for r in recs}
    assert by_id["p" * 32]["reason"] == "missing_round"  # its own file is absent
    assert by_id["c" * 32]["reason"] == "missing_parent"

    # with the parent present it scores
    (store / ("p" * 32 + ".json")).write_text(json.dumps(
        _round_file("p" * 32, "parentless prompt", segments=[
            {"type": "text", "text": "parent prose"},
        ])))
    recs = rescore_rows(rows, "parent_response", {}, "key",
                        query_fn=lambda *a, **k: {"correction": {"noul": 0.5}},
                        round_store=store)
    by_id = {r["id"]: r for r in recs}
    assert by_id["c" * 32]["score"] == 0.5
    assert by_id["p" * 32]["reason"] == "missing_parent"  # empty parentId


def test_rescore_unknown_variant_raises(tmp_path):
    with pytest.raises(ValueError):
        rescore_rows(_gold_rows(), "full_chain", {}, "key", round_store=tmp_path)


def test_query_jev_parses_answers_and_rejects_bad_shape(monkeypatch):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"answers": {"correction": {"noul": 1.0}}}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: FakeResp())
    out = query_jev("state", "key")
    assert out == {"correction": {"noul": 1.0}}

    class BadResp(FakeResp):
        def read(self):
            return json.dumps({"unexpected": True}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: BadResp())
    with pytest.raises(ValueError):
        query_jev("state", "key")


def test_jev_cache_key_is_stable_and_separates_states(tmp_path):
    assert jev_cache_key("s1") == jev_cache_key("s1")
    assert jev_cache_key("s1") != jev_cache_key("s2")
    # model separates keys
    assert jev_cache_key("s1", "m1") != jev_cache_key("s1", "m2")


# ---------------------------------------------------------------------------
# unit-004: reporting + determinism
# ---------------------------------------------------------------------------


def _report_rows(n_pos: int = 6, n_neg: int = 6) -> list[dict]:
    rows = [
        {"id": f"p{i:02d}", "label": "positive", "prompt": f"fix the file p{i:02d}"}
        for i in range(n_pos)
    ]
    rows += [
        {"id": f"n{i:02d}", "label": "negative", "prompt": f"hello there n{i:02d}"}
        for i in range(n_neg)
    ]
    return rows


def _report_rounds(tmp_path: Path, rows: list[dict]) -> Path:
    store = tmp_path / "rounds"
    store.mkdir()
    for r in rows:
        (store / f"{r['id']}.json").write_text(
            json.dumps(
                {"responseSequence": f"response for {r['id']}", "parentId": "sharedparent"}
            )
        )
    (store / "sharedparent.json").write_text(
        json.dumps({"responseSequence": "parent response text"})
    )
    return store


def _report_query(state, api_key=None, questions=None):
    return {
        "correction": {"noul": 0.9 if "fix" in state else 0.1},
        "frustration": {"score": 1.0},
    }


class TestReporting:
    def test_report_metrics_has_all_baselines_and_deltas(self, tmp_path):
        rows = _report_rows()
        store = _report_rounds(tmp_path, rows)
        metrics, score_rows = assemble_baselines(
            rows, 0.2, {}, query_fn=_report_query, round_store=store
        )
        expected = {"base_rate", "lexical", *[f"jev_{v}" for v in JEV_VARIANTS]}
        assert set(metrics["baselines"]) == expected
        assert set(metrics["deltas"]) == {"leak", "session"}
        assert metrics["feature_definitions"]["imperative_density"]["imperative_verbs"]
        assert metrics["feature_definitions"]["deixis"]["markers"]
        assert set(metrics["lexical_weights"]) == {"intercept", "features"}
        assert set(metrics["lexical_weights"]["features"]) == set(FEATURE_NAMES)
        assert set(metrics["run_report"]["thresholds_used"]) == expected
        # per-row dump carries every score C4/C5/C6 reuse
        assert len(score_rows) == 12
        assert {
            "base_rate",
            "lexical_combined",
            "feat_deixis",
            "jev_prompt_only",
            "jev_own_response",
            "jev_parent_response",
        } <= set(score_rows[0])

    def test_feature_definitions_match_the_implementations(self):
        """F6: every declared score is the value the function really returns.

        Extension 3a's "an ambiguous feature that matches nothing scores 0" is
        true of the count/ratio features and false of output_contract_absence,
        which is an absence indicator. The block has to state the polarity the
        code implements, not the polarity the extension implies.
        """
        defs = _b._feature_definitions()
        assert set(defs) == {*FEATURE_NAMES, "ext_3a_scope"}
        for name in ("length", "imperative_density", "deixis"):
            declared = defs[name]["empty_prompt_score"]
            assert declared == 0.0
            assert lexical_features("")[name] == declared
        no_marker = "think about it more"
        assert output_contract_absence(no_marker) == 1.0
        assert defs["output_contract_absence"]["no_marker_score"] == 1.0
        assert (
            lexical_features(no_marker)["output_contract_absence"]
            == defs["output_contract_absence"]["no_marker_score"]
        )
        assert "unmatched_prompt_score" not in defs["output_contract_absence"]
        scope = defs["ext_3a_scope"]
        for name in ("length", "imperative_density", "deixis"):
            assert name in scope
        assert "scores 1" in scope

    def test_published_combiner_is_rederivable(self, tmp_path):
        """F1: the published combiner must be re-derivable, not just named.

        Rebuilds every row's logit from the artifact's own published fields
        (intercept, per-feature weight/mean/std) and the raw features in the
        per-row dump, and requires it to equal the published lexical_combined.
        A weight map keyed by feature name cannot carry the intercept, so the
        fit that produced these scores was not the one published: the bias was
        published as `length`'s weight and every other weight shifted one slot.
        The second half of the test pins that failure mode — the shifted
        assignment must reproduce no row.
        """
        rows = _report_rows()
        store = _report_rounds(tmp_path, rows)
        metrics, score_rows = assemble_baselines(
            rows, 0.2, {}, query_fn=_report_query, round_store=store
        )
        combiner = metrics["lexical_weights"]
        published = [combiner["features"][n]["weight"] for n in FEATURE_NAMES]
        assert len(published) == len(FEATURE_NAMES)
        assert len(set(published)) > 1, "degenerate fixture: weights all equal"

        def rebuild(weights_in_feature_order, rec):
            z = combiner["intercept"]
            for name, weight in zip(FEATURE_NAMES, weights_in_feature_order):
                spec = combiner["features"][name]
                x = rec[f"feat_{name}"]
                scaled = 0.0 if spec["std"] == 0 else (x - spec["mean"]) / spec["std"]
                z += weight * scaled
            return z

        # the defect F1 reported: bias in slot 0, every weight shifted one
        # feature to the right, the last weight dropped
        shifted = [combiner["intercept"], *published[:-1]]

        shifted_mismatches = 0
        for rec in score_rows:
            assert rebuild(published, rec) == pytest.approx(
                rec["lexical_combined"], abs=1e-9
            ), rec["id"]
            if rebuild(shifted, rec) != pytest.approx(
                rec["lexical_combined"], abs=1e-9
            ):
                shifted_mismatches += 1
        assert shifted_mismatches == len(score_rows), (
            "the off-by-one weight assignment reproduced some rows: the fixture "
            "cannot distinguish the published mapping from the shifted one"
        )

    def test_report_emits_per_feature_aucs(self, tmp_path):
        # issue #3 main success scenario step 3: baseline (b) reports each
        # feature's AUC, not only the combined score.
        rows = _report_rows()
        store = _report_rounds(tmp_path, rows)
        metrics, _ = assemble_baselines(
            rows, 0.2, {}, query_fn=_report_query, round_store=store
        )
        feature_aucs = metrics["feature_aucs"]
        assert set(feature_aucs) == set(FEATURE_NAMES)
        labels = [1 if r["label"] == "positive" else 0 for r in rows]
        for name in FEATURE_NAMES:
            scores = [lexical_features(r["prompt"])[name] for r in rows]
            assert feature_aucs[name]["point"] == auc(scores, labels)
            assert len(feature_aucs[name]["ci95"]) == 2
            assert feature_aucs[name]["seed"] == metrics["seed"]

    def test_report_lists_excluded_unresolved_and_null_ids(self, tmp_path):
        rows = _report_rows(3, 3)
        store = _report_rounds(tmp_path, rows)
        (store / f"{rows[-1]['id']}.json").unlink()

        def q(state, api_key=None, questions=None):
            if "p00" in state:
                raise OSError("boom")
            return _report_query(state)

        metrics, _ = assemble_baselines(
            rows,
            0.2,
            {},
            query_fn=q,
            round_store=store,
            excluded_ids=["ex1"],
            unresolved_ids=["un1"],
        )
        rr = metrics["run_report"]
        assert rr["excluded_ids"] == ["ex1"]
        assert rr["unresolved_ids"] == ["un1"]
        assert "p00" in rr["null_scores"]["prompt_only"]["api_error"]
        assert "missing_round" in rr["null_scores"]["own_response"]
        assert rows[-1]["id"] in rr["null_scores"]["own_response"]["missing_round"]

    def test_report_deltas_pair_only_valid_rows(self, tmp_path):
        rows = _report_rows(3, 3)
        store = _report_rounds(tmp_path, rows)
        (store / f"{rows[0]['id']}.json").unlink()
        metrics, _ = assemble_baselines(
            rows, 0.2, {}, query_fn=_report_query, round_store=store
        )
        assert metrics["deltas"]["leak"]["n_pairs"] == 5
        assert metrics["deltas"]["session"]["n_pairs"] == 5
        assert rows[0]["id"] in metrics["run_report"]["delta_exclusions"]["leak"]
        assert rows[0]["id"] in metrics["run_report"]["delta_exclusions"]["session"]

    def test_report_unresolved_prompt_excluded_from_scored_rows(self, tmp_path):
        rows = []
        for i in range(29):
            rows.append(
                {"id": f"p{i}", "label": "positive", "prompt": "fix", "prompt_status": "resolved"}
            )
        for i in range(205):
            rows.append(
                {"id": f"n{i}", "label": "negative", "prompt": "hello", "prompt_status": "resolved"}
            )
        rows[29]["prompt_status"] = "unresolved"
        for i in range(8):
            rows.append(
                {"id": f"e{i}", "label": "excluded", "prompt": "", "prompt_status": "resolved"}
            )
        path = tmp_path / "gold.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        scored, tally = load_gold(str(path))
        assert len(scored) == 233
        assert tally["negative"] == 205
        _, unresolved = load_gold_status_ids(str(path))
        assert unresolved == [rows[29]["id"]]


class TestReportDeterminism:
    def test_report_determinism_with_warm_cache(self, tmp_path):
        rows = _report_rows(3, 3)
        store = _report_rounds(tmp_path, rows)
        cache: dict = {}
        calls = {"n": 0}

        def q(state, api_key=None, questions=None):
            calls["n"] += 1
            return _report_query(state)

        m1, s1 = assemble_baselines(
            rows, 0.2, cache, query_fn=q, round_store=store
        )
        assert calls["n"] > 0  # cold cache hits the API
        first_calls = calls["n"]
        m2, s2 = assemble_baselines(
            rows, 0.2, cache, query_fn=q, round_store=store
        )
        assert calls["n"] == first_calls  # warm cache: fully offline
        assert serialize_metrics(m1) == serialize_metrics(m2)
        assert serialize_score_rows(s1) == serialize_score_rows(s2)

    def test_report_empty_cold_cache_emits_byte_identical(self, tmp_path):
        rows = _report_rows(3, 3)
        store = _report_rounds(tmp_path, rows)
        cache: dict = {}
        m1, s1 = assemble_baselines(rows, 0.2, cache, query_fn=_report_query, round_store=store)
        assert cache  # cold cache was populated
        m2, s2 = assemble_baselines(rows, 0.2, cache, query_fn=_report_query, round_store=store)
        mp, sp = tmp_path / "metrics.json", tmp_path / "scores.jsonl"
        p1 = emit_artifacts(m1, s1, mp, sp)
        # same content emitted again must not raise (extension 7a)
        emit_artifacts(m2, s2, mp, sp)
        assert p1["metrics"].read_bytes() == serialize_metrics(m1).encode()
        assert p1["scores"].read_bytes() == serialize_score_rows(s1).encode()

    def test_determinism_fails_naming_drifting_field(self, tmp_path):
        rows = _report_rows(3, 3)
        store = _report_rounds(tmp_path, rows)
        m, s = assemble_baselines(rows, 0.2, {}, query_fn=_report_query, round_store=store)
        mp, sp = tmp_path / "metrics.json", tmp_path / "scores.jsonl"
        emit_artifacts(m, s, mp, sp)
        drifted = json.loads(serialize_metrics(m))
        drifted["baselines"]["lexical"]["auc"]["point"] += 0.01
        with pytest.raises(RuntimeError, match=r"baselines\.lexical\.auc\.point"):
            emit_artifacts(drifted, s, mp, sp)

    def test_determinism_fails_naming_drifting_score_row(self, tmp_path):
        rows = _report_rows(3, 3)
        store = _report_rounds(tmp_path, rows)
        m, s = assemble_baselines(rows, 0.2, {}, query_fn=_report_query, round_store=store)
        mp, sp = tmp_path / "metrics.json", tmp_path / "scores.jsonl"
        emit_artifacts(m, s, mp, sp)
        drifted = [dict(r) for r in s]
        drifted[0]["lexical_combined"] = drifted[0]["lexical_combined"] + 0.5
        with pytest.raises(RuntimeError, match=drifted[0]["id"]):
            emit_artifacts(m, drifted, mp, sp)
