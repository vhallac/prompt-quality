"""C3 baseline tests: metrics core + gold loader + lexical baselines."""

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
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
    JEV_CACHE_LEGACY_QUESTIONS_SHA256,
    JEV_CACHE_QUESTIONS_KEY,
    JEV_QUESTIONS,
    JEV_QUESTIONS_SHA256,
    JEV_REFERENCE_PATH,
    JEV_REFERENCE_SHA256,
    JEV_VARIANTS,
    RESPONSE_MARKER,
    alarm_rate,
    alarm_rate_tolerance,
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
    gold_content_sha256,
    imperative_density,
    input_display_path,
    jev_cache_entry_count,
    jev_cache_key,
    jev_cache_key_legacy,
    jev_cache_questions_sha256,
    jev_flag_score,
    jev_legacy_adoption_allowed,
    jev_questions_sha256,
    jev_reference_identity,
    length_feature,
    lexical_features,
    load_corpus_base_rate,
    load_gold,
    load_gold_status_ids,
    load_jev_cache,
    load_jev_reference,
    logistic_score,
    main,
    operating_points,
    output_contract_absence,
    paired_auc_delta,
    parent_response_state,
    prompt_only_state,
    JevCacheError,
    JevReferenceError,
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


class TestAlarmRateDisclosure:
    """F4: each FNR cell states the alarm rate it achieved, not only the one it aimed at."""

    def test_alarm_rate_primitive_counts_all_rows(self):
        scores = [0.1, 0.2, 0.3, 0.4]
        assert alarm_rate(scores, 0.3) == pytest.approx(0.5)
        assert alarm_rate(scores, math.nextafter(0.4, math.inf)) == 0.0
        with pytest.raises(ValueError, match="no rows"):
            alarm_rate([], 0.5)

    def test_alarm_rate_tolerance_is_half_the_rate_grid_step(self):
        assert alarm_rate_tolerance(4) == pytest.approx(0.125)
        with pytest.raises(ValueError, match="empty vector"):
            alarm_rate_tolerance(0)

    def test_both_operating_points_disclose_achieved_alarm_rate(self):
        scores = [0.1 * i for i in range(1, 21)]
        labels = [1 if i % 3 == 0 else 0 for i in range(20)]
        m = baseline_metrics(scores, labels, corpus_base_rate=0.3, n_boot=50, seed=3)
        for name in ("youden_j", "corpus_base_rate"):
            cell = m["fnr"][name]
            assert cell["achieved_alarm_rate"] == pytest.approx(
                alarm_rate(scores, cell["threshold"])
            )
        # (i) targets Youden's J, not a rate; only (ii) declares a target.
        assert m["fnr"]["youden_j"]["target_alarm_rate"] is None
        assert "alarm_rate_tolerance" not in m["fnr"]["youden_j"]
        assert m["fnr"]["corpus_base_rate"]["target_alarm_rate"] == pytest.approx(0.3)
        assert m["fnr"]["corpus_base_rate"]["alarm_rate_tolerance"] == pytest.approx(
            alarm_rate_tolerance(20)
        )

    def test_alarm_rate_disclosure_does_not_move_threshold_or_fnr(self):
        # Disclosure must not change selection: the published threshold and FNR
        # point are exactly what the untouched selectors produce.
        scores = [0.05, 0.05, 0.2, 0.4, 0.6, 0.9]
        labels = [0, 1, 0, 1, 0, 1]
        m = baseline_metrics(scores, labels, corpus_base_rate=0.5, n_boot=20, seed=11)
        ty = youden_threshold(scores, labels)
        tb = base_rate_threshold(scores, labels, 0.5)
        assert m["fnr"]["youden_j"]["threshold"] == ty
        assert m["fnr"]["corpus_base_rate"]["threshold"] == tb
        assert m["fnr"]["youden_j"]["point"] == pytest.approx(fnr(scores, labels, ty))
        assert m["fnr"]["corpus_base_rate"]["point"] == pytest.approx(
            fnr(scores, labels, tb)
        )

    def test_reachable_target_alarm_rate_is_not_marked(self):
        # n=8 → tolerance 0.0625; 4/8 rows alarmed hits the 0.5 target exactly.
        scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        m = baseline_metrics(scores, labels, corpus_base_rate=0.5, n_boot=20, seed=2)
        cell = m["fnr"]["corpus_base_rate"]
        assert cell["achieved_alarm_rate"] == pytest.approx(0.5)
        assert "note" not in cell

    def test_unreachable_target_alarm_rate_is_marked_non_rate_matched(self):
        # The prevalence floor: one distinct score, so the only reachable rates
        # are 1.0 and 0.0 and the 0.1939 target cannot be met.
        scores = [0.12393162393162394] * 20
        labels = [1, 0] * 10
        m = baseline_metrics(
            scores, labels, corpus_base_rate=0.19387755102040816, n_boot=20, seed=4
        )
        cell = m["fnr"]["corpus_base_rate"]
        assert cell["target_alarm_rate"] == pytest.approx(0.19387755102040816)
        assert cell["achieved_alarm_rate"] == 0.0
        assert cell["point"] == 1.0
        note = cell["note"]
        assert "non-rate-matched" in note
        assert "alarms nothing" in note
        assert f"{cell['alarm_rate_tolerance']:.6g}" in note
        assert "1 distinct value" in note


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
    # prompt-only is the reference's own state with the response slot replaced:
    # same builder, marker instead of a response.
    assert s1 == REF.build_state(
        {"userPrompt": "Fix the bug.", "responseSequence": RESPONSE_MARKER},
        baselines.JEV_STATE_CHARS,
    )
    # the state is a function of the prompt alone: different rounds with the
    # same prompt but different responses produce the same state
    own = own_response_state("Fix the bug.", _round_file("a" * 32, "Fix the bug.", "response A"))
    assert own != prompt_only_state("Fix the bug.")
    assert "response A" not in prompt_only_state("Fix the bug.")


def test_own_response_state_matches_jev_round_scan_shape():
    """Parity with the pinned reference, not with a local string literal (F3).

    The shape sweep over response types lives in
    ``test_pinned_reference_build_state_parity``; what this test keeps is the
    original claim, now anchored on the reference's own output.
    """
    data = _round_file("a" * 32, "Fix the bug.", "the fix is here")
    assert own_response_state("Fix the bug.", data) == REF.build_state(
        data, baselines.JEV_STATE_CHARS
    )
    # string-typed responseSequence (the current store shape) is used verbatim
    data_str = dict(data, responseSequence="plain string response")
    assert own_response_state("Fix the bug.", data_str) == REF.build_state(
        data_str, baselines.JEV_STATE_CHARS
    )
    assert own_response_state("Fix the bug.", data_str).endswith("plain string response")


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
    # byte equality with the reference for this shape is asserted in
    # test_pinned_reference_build_refine_state_parity


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


# ---------------------------------------------------------------------------
# unit-004 — F5: the delta block must name its own paired population, and F7:
# the ext 5a single-class path and the zero-pairs path must be asserted.
# ---------------------------------------------------------------------------

# Per-row scores that overlap between classes: a constant-per-class fixture
# gives every arm AUC 1.0 in every subpopulation, so it cannot tell a paired
# difference from a marginal one. With these tables prompt_only AUC is 0.4375
# and own_response AUC is 0.6875 over the full eight rows, and both move when a
# row leaves the paired population.
_PROMPT_ONLY_BY_ROW = {
    "p00": 0.55, "p01": 0.45, "p02": 0.65, "p03": 0.25,
    "n00": 0.35, "n01": 0.75, "n02": 0.15, "n03": 0.85,
}
_OWN_RESPONSE_BY_ROW = {
    "p00": 0.90, "p01": 0.35, "p02": 0.80, "p03": 0.20,
    "n00": 0.60, "n01": 0.10, "n02": 0.50, "n03": 0.30,
}


def _row_varying_query(state, api_key=None, questions=None):
    """Stand-in jev scorer keyed on the row id, so an arm's AUC depends on
    which rows are present (issue #3 extensions 4a/4b drop rows per arm)."""
    rid = re.search(r"[pn]\d\d", state).group(0)
    table = _OWN_RESPONSE_BY_ROW if "response for" in state else _PROMPT_ONLY_BY_ROW
    return {"correction": {"noul": table[rid]}, "frustration": {"score": 0.0}}


def _pair_record(rid, score, label):
    return {"id": rid, "score": score, "label": label, "reason": None}


class TestPairedDeltaProvenance:
    def test_paired_delta_block_names_its_own_population(self):
        left = [
            _pair_record("p00", 0.90, "positive"),
            _pair_record("n00", 0.10, "negative"),
        ]
        right = [
            _pair_record("p00", 0.20, "positive"),
            _pair_record("n00", 0.80, "negative"),
        ]
        block = paired_auc_delta(left, right, n_boot=200, seed=7)
        assert block["n_pairs"] == 2
        assert block["n_reference_rows"] == 2
        assert "n_pairs = 2" in block["population"]
        assert block["auc_reference_paired"]["reference"] == auc([0.90, 0.10], [1, 0])
        assert block["auc_reference_paired"]["comparison"] == auc([0.20, 0.80], [1, 0])
        # the paired columns are what the published point is made of
        assert block["auc_reference_paired"]["difference"] == block["point"]

    @pytest.mark.parametrize(
        "reason, left, right",
        [
            (
                "no id in common",
                [_pair_record("p00", 0.9, "positive"), _pair_record("n00", 0.1, "negative")],
                [_pair_record("q00", 0.5, "positive"), _pair_record("q01", 0.5, "negative")],
            ),
            (
                "null score in the comparison arm",
                [_pair_record("p00", 0.9, "positive"), _pair_record("n00", 0.1, "negative")],
                [_pair_record("p00", None, "positive"), _pair_record("n00", None, "negative")],
            ),
            (
                "null score in the reference arm",
                [_pair_record("p00", None, "positive"), _pair_record("n00", None, "negative")],
                [_pair_record("p00", 0.7, "positive"), _pair_record("n00", 0.4, "negative")],
            ),
            (
                "no rows at all",
                [],
                [],
            ),
        ],
    )
    def test_paired_delta_without_shared_valid_rows_is_null_but_keeps_its_shape(
        self, reason, left, right
    ):
        block = paired_auc_delta(left, right, n_boot=200, seed=7)
        populated = paired_auc_delta(
            [_pair_record("p00", 0.9, "positive"), _pair_record("n00", 0.1, "negative")],
            [_pair_record("p00", 0.2, "positive"), _pair_record("n00", 0.8, "negative")],
            n_boot=200,
            seed=7,
        )
        assert reason  # every unpairable flavour returns the same block shape
        assert set(block) == set(populated), block
        assert block["point"] is None
        assert block["ci95"] is None
        assert block["n_pairs"] == 0
        assert block["auc_reference_paired"] == {
            "reference": None, "comparison": None, "difference": None,
        }
        assert "n_pairs = 0" in block["population"]

    def test_paired_delta_equals_marginal_difference_when_arms_share_every_row(self, tmp_path):
        rows = _report_rows(4, 4)
        store = _report_rounds(tmp_path, rows)
        metrics, _ = assemble_baselines(
            rows, 0.2, {}, query_fn=_row_varying_query, round_store=store
        )
        block = metrics["deltas"]["leak"]
        baselines = metrics["baselines"]
        assert block["arms"] == {"reference": "jev_prompt_only", "comparison": "jev_own_response"}
        assert block["n_pairs"] == block["n_reference_rows"] == 8
        assert block["auc_reference_paired"]["reference"] == baselines["jev_prompt_only"]["auc"]["point"]
        assert block["auc_reference_paired"]["comparison"] == baselines["jev_own_response"]["auc"]["point"]
        assert block["auc_marginal"]["difference"] == block["point"]
        # anchored arithmetic: 11/16 - 7/16
        assert block["point"] == pytest.approx(0.25)

    def test_paired_delta_discloses_the_population_it_differs_on(self, tmp_path):
        # F5: subtracting the two headline AUCs is not the paired delta once an
        # arm loses a row; the block has to show both numbers and why.
        rows = _report_rows(4, 4)
        store = _report_rounds(tmp_path, rows)
        (store / "p00.json").unlink()
        metrics, _ = assemble_baselines(
            rows, 0.2, {}, query_fn=_row_varying_query, round_store=store
        )
        block = metrics["deltas"]["leak"]
        assert block["n_pairs"] == 7
        assert block["n_reference_rows"] == 8
        assert "n_pairs = 7" in block["population"] and "8 reference-arm rows" in block["population"]
        assert block["auc_marginal"]["reference"] == metrics["baselines"]["jev_prompt_only"]["auc"]["point"]
        assert block["auc_marginal"]["comparison"] == metrics["baselines"]["jev_own_response"]["auc"]["point"]
        assert block["point"] == pytest.approx(block["auc_reference_paired"]["difference"])
        # the paired reference population is not the reference arm's own denominator
        assert block["auc_reference_paired"]["reference"] == pytest.approx(5 / 12)
        assert block["auc_reference_paired"]["reference"] != block["auc_marginal"]["reference"]
        assert block["point"] != pytest.approx(block["auc_marginal"]["difference"])
        assert block["point"] == pytest.approx(2 / 12)

        # and the session delta keeps its own (identical-score) arms honest: its
        # point is 0 while the marginal difference is not
        session = metrics["deltas"]["session"]
        assert session["point"] == 0.0
        assert session["auc_marginal"]["difference"] != 0.0
        assert session["auc_reference_paired"]["difference"] == session["point"]


class TestSingleClassStratum:
    @pytest.mark.parametrize(
        "scores, labels",
        [
            ([0.2, 0.5, 0.9], [1, 1, 1]),
            ([0.2, 0.5, 0.9], [0, 0, 0]),
            ([], []),
        ],
    )
    def test_single_class_stratum_yields_null_auc_and_null_fnrs_with_note(
        self, scores, labels
    ):
        out = baseline_metrics(scores, labels, 0.19)
        assert out["auc"] is None
        assert out["fnr"] is None
        assert "single-class stratum" in out["note"]
        assert "extension 5a" in out["note"]
        assert out["n"] == len(scores)
        assert out["positives"] == sum(labels)
        assert out["negatives"] == len(labels) - sum(labels)

    def test_single_class_stratum_is_absent_from_thresholds_used(self, tmp_path):
        # extension 5a at the reporting surface: a stratum with one class has no
        # operating points, so run_report must not claim any.
        rows = _report_rows(4, 0)
        store = _report_rounds(tmp_path, rows)
        metrics, _ = assemble_baselines(
            rows, 0.2, {}, query_fn=_row_varying_query, round_store=store
        )
        assert metrics["run_report"]["thresholds_used"] == {}
        for name, entry in metrics["baselines"].items():
            assert entry["auc"] is None, name
            assert entry["fnr"] is None, name
            assert "extension 5a" in entry["note"], name
        for name, block in metrics["deltas"].items():
            assert block["n_pairs"] == 4, name
            assert block["point"] is None, name
            assert block["auc_reference_paired"] == {
                "reference": None, "comparison": None, "difference": None,
            }, name
            assert block["auc_marginal"]["difference"] is None, name


# ---------------------------------------------------------------------------
# unit-005 — F2: the cache key must address the whole request payload (state,
# model, questions), so a rubric change is a miss and not the old answers.
# F8: an unreadable cache is cold or names the path, the save is atomic, and a
# run rejected by the determinism gate leaves the tracked cache untouched.
# ---------------------------------------------------------------------------

# Two rubrics that differ in text only. States are identical across them, so a
# cache that ignores the questions cannot tell the two apart — which is exactly
# the substitution review F2 found. The marker is what the stub reads to answer
# differently per payload, so a stale hit is visible in the scores.
_RUBRIC_A = {
    "marker": "a",
    "frustration": {"type": "score", "instructions": "Rate the joy.", "criteria": ["low", "high"]},
}
_RUBRIC_B = {
    "marker": "b",
    "frustration": {"type": "score", "instructions": "Rate the rage.", "criteria": ["low", "high"]},
}
_RUBRIC_ANSWERS = {"default": 0.9, "a": 0.8, "b": 0.05}


def _rubric_query(state, api_key=None, questions=None):
    """jev answers that depend on the question payload, not the state alone.

    Under rubric A the positives separate from the negatives; under B they do
    not, so a run that reused A's answers would be visible in the AUC as well
    as in the per-row dump.
    """
    marker = (questions or {}).get("marker", "default")
    return {
        "correction": {"noul": _RUBRIC_ANSWERS[marker] if "fix" in state else 0.1},
        "frustration": {"score": 1.0},
    }


def _never_queried(state, api_key=None, questions=None):
    raise AssertionError("the warm cache should have answered this request")


def _legacy_cache_for(rows, answers):
    """A cache file in the pre-F2 shape: keyed on (state, model), no digest."""
    payload = json.dumps(answers, sort_keys=True)
    return {
        jev_cache_key_legacy(prompt_only_state(r["prompt"])): payload
        for r in rows
    }


class TestJevCacheKeying:
    def test_jev_cache_key_binds_the_questions_payload(self):
        assert jev_cache_key("s") == jev_cache_key(
            "s", questions_sha256=JEV_QUESTIONS_SHA256
        )
        assert jev_cache_key("s") != jev_cache_key(
            "s", questions_sha256=jev_questions_sha256(_RUBRIC_B)
        )
        # the pre-F2 key is a different address, so a re-keyed file cannot be
        # read by accident and the adoption path is the only bridge
        assert jev_cache_key_legacy("s") != jev_cache_key("s")
        assert jev_cache_key_legacy("s1") != jev_cache_key_legacy("s2")

    def test_jev_questions_sha256_is_canonical_over_the_payload(self):
        one = {"q": {"type": "score", "instructions": "x", "criteria": ["a", "b"]}}
        other = {"q": {"criteria": ["a", "b"], "instructions": "x", "type": "score"}}
        assert jev_questions_sha256(one) == jev_questions_sha256(other)
        assert jev_questions_sha256(_RUBRIC_A) != jev_questions_sha256(_RUBRIC_B)
        # criteria order is payload, not formatting
        reordered = {"q": {"type": "score", "instructions": "x", "criteria": ["b", "a"]}}
        assert jev_questions_sha256(reordered) != jev_questions_sha256(one)
        assert jev_questions_sha256(None) == JEV_QUESTIONS_SHA256
        # an absent payload falls back to the shipped rubric exactly as
        # query_jev does, so the key always names what was actually sent
        assert jev_questions_sha256({}) == JEV_QUESTIONS_SHA256

    def test_legacy_cache_entries_are_adopted_and_not_refetched(self):
        rows = _gold_rows()
        answers = {"correction": {"noul": 0.4}, "frustration": {"score": 0.0}}
        cache = _legacy_cache_for(rows, answers)
        legacy_keys = set(cache)
        recs = rescore_rows(
            rows, "prompt_only", cache, "key", query_fn=_never_queried,
            round_store="/nonexistent",
        )
        assert [r["score"] for r in recs] == [0.4, 0.4]
        assert legacy_keys.isdisjoint(cache), "the legacy keys were left addressable"
        for r, row in zip(recs, rows):
            assert jev_cache_key(prompt_only_state(row["prompt"])) in cache

    def test_legacy_cache_is_not_adopted_for_a_different_rubric(self):
        # Review F2's counterexample, at the cache: an edited rubric must miss
        # instead of publishing the answers the old rubric produced.
        rows = _gold_rows()
        stale = {"correction": {"noul": 0.9}, "frustration": {"score": 0.0}}
        cache = _legacy_cache_for(rows, stale)
        seen = []

        def q(state, api_key=None, questions=None):
            seen.append(questions)
            return {"correction": {"noul": 0.1}, "frustration": {"score": 0.0}}

        recs = rescore_rows(
            rows, "prompt_only", cache, "key", query_fn=q,
            round_store="/nonexistent", questions=_RUBRIC_B,
        )
        assert len(seen) == len(rows), "the stale rubric's answers were reused"
        assert [r["score"] for r in recs] == [0.1, 0.1]
        # the legacy entries stay put: they belong to another payload
        assert set(cache) >= set(_legacy_cache_for(rows, stale))

    def test_cache_adoption_is_gated_on_the_recorded_digest(self, tmp_path):
        rows = _gold_rows()
        answers = {"correction": {"noul": 0.6}, "frustration": {"score": 0.0}}
        legacy = _legacy_cache_for(rows, answers)
        for recorded, expected_calls in (
            (jev_questions_sha256(_RUBRIC_A), 0),  # matches the payload in force
            (JEV_QUESTIONS_SHA256, len(rows)),  # names a different rubric
        ):
            cache = save_jev_cache(
                dict(legacy), tmp_path / f"cache-{recorded[:8]}.json",
                questions_sha256=recorded,
            )
            loaded = load_jev_cache(cache)
            assert jev_cache_questions_sha256(loaded) == recorded
            calls = []

            def q(state, api_key=None, questions=None):
                calls.append(1)
                return {"correction": {"noul": 0.2}, "frustration": {"score": 0.0}}

            recs = rescore_rows(
                rows, "prompt_only", loaded, "key", query_fn=q,
                round_store="/nonexistent", questions=_RUBRIC_A,
            )
            assert len(calls) == expected_calls, recorded
            assert [r["score"] for r in recs] == (
                [0.6] * len(rows) if expected_calls == 0 else [0.2] * len(rows)
            )

    def test_cache_miss_without_api_key_fails_loudly(self):
        # A miss is not a degraded run: no key means the answers cannot exist.
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            query_jev("state", "")
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            rescore_rows(
                _gold_rows(), "prompt_only", {}, None, round_store="/nonexistent"
            )

    def test_rubric_change_does_not_reproduce_byte_identical_scores(self, tmp_path):
        # The required regression test, end to end: the same states asked under
        # a different payload must not come back byte-identical.
        rows = _report_rows(4, 4)
        store = _report_rounds(tmp_path, rows)
        path = tmp_path / "jev-cache.json"

        warm = {}
        m_a, s_a = assemble_baselines(
            rows, 0.2, warm, query_fn=_rubric_query, round_store=store,
            questions=_RUBRIC_A,
        )
        save_jev_cache(warm, path, questions_sha256=jev_questions_sha256(_RUBRIC_A))

        m_b, s_b = assemble_baselines(
            rows, 0.2, load_jev_cache(path), query_fn=_rubric_query,
            round_store=store, questions=_RUBRIC_B,
        )
        assert m_b["jev"]["questions_sha256"] == jev_questions_sha256(_RUBRIC_B)
        assert m_a["jev"]["questions_sha256"] == jev_questions_sha256(_RUBRIC_A)
        assert serialize_score_rows(s_a) != serialize_score_rows(s_b)
        assert serialize_metrics(m_a) != serialize_metrics(m_b)

        # ...while the same payload on the warm cache is still offline and
        # byte-identical (decision 004's gate survives the re-keying).
        m_a2, s_a2 = assemble_baselines(
            rows, 0.2, load_jev_cache(path), query_fn=_never_queried,
            round_store=store, questions=_RUBRIC_A,
        )
        assert serialize_metrics(m_a2) == serialize_metrics(m_a)
        assert serialize_score_rows(s_a2) == serialize_score_rows(s_a)

    def test_pinned_legacy_digest_is_what_authorizes_the_shipped_cache(self):
        # The shipped cache records no digest, so the pin is its provenance: it
        # must be the payload in force. A rubric edit moves JEV_QUESTIONS_SHA256
        # and leaves the pin where it was, which is what makes the paid entries
        # unreachable instead of quietly answering new questions. Re-pin only
        # when re-stamping a pre-metadata cache file deliberately.
        assert JEV_CACHE_LEGACY_QUESTIONS_SHA256 == JEV_QUESTIONS_SHA256, (
            "the shipped pre-metadata cache was fetched under the pinned payload; "
            "a rubric edit means its answers are unreachable and must be re-fetched "
            "(or the file re-stamped by hand), not the pin quietly moved"
        )

    def test_metrics_record_the_questions_digest(self, tmp_path):
        rows = _report_rows(2, 2)
        store = _report_rounds(tmp_path, rows)
        metrics, _ = assemble_baselines(
            rows, 0.2, {}, query_fn=_rubric_query, round_store=store
        )
        assert metrics["jev"]["questions_sha256"] == jev_questions_sha256(
            JEV_QUESTIONS
        )


class TestJevCacheCrashSafety:
    @pytest.mark.parametrize("content", ["", "   ", "\n"])
    def test_empty_cache_file_is_cold(self, tmp_path, content):
        path = tmp_path / "jev-cache.json"
        path.write_text(content)
        assert load_jev_cache(path) == {}

    def test_absent_cache_file_is_cold(self, tmp_path):
        assert load_jev_cache(tmp_path / "nope.json") == {}

    @pytest.mark.parametrize(
        "content,match",
        [
            ("{ not json", "is not valid JSON"),
            ("[]", "must hold a JSON object"),
            (
                '{"not-a-digest": "{\\"correction\\": {}}"}',
                "not a cache digest",
            ),
            (
                '{"' + "0" * 64 + '": {"correction": {}}}',
                "not a JSON string",
            ),
            (
                '{"' + "0" * 64 + '": "{ broken"}',
                "not valid JSON",
            ),
            (
                '{"' + JEV_CACHE_QUESTIONS_KEY + '": "nope"}',
                "not a sha256 hex digest",
            ),
        ],
    )
    def test_corrupt_cache_raises_naming_the_path(self, tmp_path, content, match):
        path = tmp_path / "jev-cache.json"
        path.write_text(content)
        with pytest.raises(JevCacheError, match=match):
            load_jev_cache(path)
        # and the message always names the file the operator has to fix
        with pytest.raises(JevCacheError, match=str(path)):
            load_jev_cache(path)

    def test_save_jev_cache_is_deterministic_and_roundtrips(self, tmp_path):
        rows = _gold_rows()
        cache = {}
        rescore_rows(
            rows, "prompt_only", cache, "key", query_fn=_rubric_query,
            round_store="/nonexistent",
        )
        first = save_jev_cache(cache, tmp_path / "a.json")
        second = save_jev_cache(dict(reversed(list(cache.items()))), tmp_path / "b.json")
        assert first.read_bytes() == second.read_bytes()
        assert load_jev_cache(first) == cache
        assert jev_cache_questions_sha256(load_jev_cache(first)) is None

    def test_save_jev_cache_stamps_the_questions_digest(self, tmp_path):
        cache = {"0" * 64: '{"correction": {"noul": 0.5}}'}
        path = save_jev_cache(cache, tmp_path / "c.json", questions_sha256="f" * 64)
        loaded = load_jev_cache(path)
        assert jev_cache_questions_sha256(loaded) == "f" * 64
        assert jev_cache_entry_count(loaded) == 1
        assert dict(loaded) != cache  # the stamp is a member of the document
        assert jev_legacy_adoption_allowed(cache, "f" * 64) is False

    def test_failed_save_leaves_the_warm_cache_and_no_temp_file(
        self, tmp_path, monkeypatch
    ):
        cache = {"0" * 64: '{"correction": {"noul": 0.5}}'}
        path = save_jev_cache(cache, tmp_path / "jev-cache.json")
        before = path.read_bytes()

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            save_jev_cache({"1" * 64: "{}"}, path)
        assert path.read_bytes() == before
        assert list(tmp_path.glob("*tmp*")) == []

    def test_main_writes_no_cache_when_the_artifact_gate_rejects(
        self, tmp_path, monkeypatch
    ):
        # F8: the cache was saved before the determinism gate, so a run that
        # published nothing had already rewritten a tracked artifact.
        store, cache_path, _cache = self._warm_run(tmp_path, monkeypatch)
        metrics_path = tmp_path / "baseline-metrics.json"
        metrics_path.write_text('{"drift": 1}\n', encoding="utf-8")
        before = cache_path.read_bytes()
        with pytest.raises(RuntimeError, match="non-deterministic output"):
            main(
                ["--gold", "unused", "--base-rate-results", "unused",
                 "--cache", str(cache_path), "--metrics", str(metrics_path),
                 "--scores", str(tmp_path / "scores.jsonl"),
                 "--round-store", str(store), "--api-key", ""]
            )
        assert cache_path.read_bytes() == before
        assert not (tmp_path / "scores.jsonl").exists()

    def test_main_stamps_the_cache_after_a_successful_emit(
        self, tmp_path, monkeypatch
    ):
        store, cache_path, cache = self._warm_run(tmp_path, monkeypatch)
        assert jev_cache_questions_sha256(load_jev_cache(cache_path)) is None
        main(
            ["--gold", "unused", "--base-rate-results", "unused",
             "--cache", str(cache_path),
             "--metrics", str(tmp_path / "baseline-metrics.json"),
             "--scores", str(tmp_path / "baseline-scores.jsonl"),
             "--round-store", str(store), "--api-key", ""]
        )
        loaded = load_jev_cache(cache_path)
        assert jev_cache_questions_sha256(loaded) == JEV_QUESTIONS_SHA256
        assert jev_cache_entry_count(loaded) == jev_cache_entry_count(cache)

    @staticmethod
    def _warm_run(tmp_path, monkeypatch):
        """A tiny gold set with every jev state already answered offline."""
        rows = _report_rows(2, 2)
        store = _report_rounds(tmp_path, rows)
        answers = {"correction": {"noul": 0.7}, "frustration": {"score": 2.0}}
        cache = {}
        for row in rows:
            data = json.loads((store / f"{row['id']}.json").read_text())
            parent = json.loads((store / "sharedparent.json").read_text())
            for state in (
                prompt_only_state(row["prompt"]),
                own_response_state(row["prompt"], data),
                parent_response_state(row["prompt"], parent),
            ):
                cache[jev_cache_key(state)] = json.dumps(answers, sort_keys=True)
        path = save_jev_cache(cache, tmp_path / "jev-cache.json")
        monkeypatch.setattr(baselines, "load_gold", lambda p: (rows, {}))
        monkeypatch.setattr(baselines, "load_gold_status_ids", lambda p: ([], []))
        monkeypatch.setattr(baselines, "load_corpus_base_rate", lambda p=1: 0.25)
        return store, path, cache


# ---------------------------------------------------------------------------
# F3: the pinned reference scorer (scripts/reference/jev-round-scan.py)
# ---------------------------------------------------------------------------
#
# The C3 jev harness copies four things from the external scorer: the question
# set, the model id, the state-char limit, and the two state builders. Before
# F3 that parity was asserted with local string literals, so "re-runs the jev
# round-scan scorer" could only be checked by someone holding a sibling semblr
# checkout. Everything below reads the pinned copy instead, so the claim is
# checkable from a clean clone of this repository alone.

REF = load_jev_reference()

# The reference file as pinned: semblr scripts/jev-round-scan.py at ba10970,
# 375 lines, digest verified against that checkout on 2026-10-09.
REFERENCE_SHA256 = (
    "4e294839a341c953d38dba17dadfd87be53d36f536700d3274da16f387bf8704"
)


def _reference_state_chars_limit() -> int:
    """The --state-chars default the pinned reference parses from its own CLI."""
    tree = ast.parse(JEV_REFERENCE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
        ):
            continue
        if node.args and getattr(node.args[0], "value", None) == "--state-chars":
            for kw in node.keywords:
                if kw.arg == "default":
                    return kw.value.value
    raise AssertionError("--state-chars default not found in the pinned reference")


def test_pinned_reference_copy_matches_its_recorded_digest():
    """The checked-in copy is byte-identical to the reference it pins."""
    assert JEV_REFERENCE_PATH.is_file(), JEV_REFERENCE_PATH
    digest = hashlib.sha256(JEV_REFERENCE_PATH.read_bytes()).hexdigest()
    assert digest == REFERENCE_SHA256 == JEV_REFERENCE_SHA256


def test_pinned_reference_questions_are_the_jev_payload():
    assert REF.QUESTIONS == JEV_QUESTIONS
    assert jev_questions_sha256(REF.QUESTIONS) == JEV_QUESTIONS_SHA256


def test_pinned_reference_model_endpoint_and_state_limit():
    assert REF.MODEL == baselines.JEV_MODEL
    assert REF.DECISIONS_URL == baselines.JEV_DECISIONS_URL
    assert baselines.JEV_STATE_CHARS == _reference_state_chars_limit()


@pytest.mark.parametrize(
    "response_sequence",
    [
        "plain string response",
        [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        [{"type": "toolCall", "tool": "bash"}],
        [{"text": "no type key"}],
        ["bare string segment"],
        [],
        None,
    ],
)
def test_pinned_reference_build_state_parity(response_sequence):
    """own_response_state and the reference build_state yield the same state."""
    data = _round_file("a" * 32, "Fix the bug.", response_sequence)
    assert own_response_state("Fix the bug.", data) == REF.build_state(
        data, baselines.JEV_STATE_CHARS
    )


def test_pinned_reference_build_state_truncates_at_the_same_limit():
    long = "x" * (baselines.JEV_STATE_CHARS + 500)
    data = _round_file("a" * 32, "Fix the bug.", long)
    mine, theirs = (
        own_response_state("Fix the bug.", data),
        REF.build_state(data, baselines.JEV_STATE_CHARS),
    )
    assert mine == theirs
    assert len(mine) == len(theirs) == baselines.JEV_STATE_CHARS


@pytest.mark.parametrize(
    "segments",
    [
        [{"type": "text", "text": "only text"}],
        [
            {"type": "text", "text": "first text"},
            {"type": "toolCall", "tool": "bash"},
            {"type": "text", "text": "second text"},
        ],
        [{"type": "toolCall"}, {"type": "toolCall"}],
        [],
        [{"type": "text", "text": "x" * (baselines.JEV_STATE_CHARS + 500)}],
    ],
)
def test_pinned_reference_build_refine_state_parity(segments):
    """parent_response_state matches the reference build_refine_state."""
    parent = {"responseSegments": segments}
    child = _round_file("b" * 32, "continue", "child response")
    assert parent_response_state("continue", parent) == REF.build_refine_state(
        child, parent, baselines.JEV_STATE_CHARS
    )


def test_pinned_reference_absent_copy_fails_the_build_loudly(tmp_path):
    """No pinned copy is a build error, not an artifact with an unverifiable claim."""
    missing = tmp_path / "jev-round-scan.py"
    assert not missing.exists()
    # both seams fail with the path in the message, not a bare traceback
    with pytest.raises(JevReferenceError, match=re.escape(str(missing))):
        jev_reference_identity(missing)
    with pytest.raises(JevReferenceError, match="not found"):
        load_jev_reference(missing)


def test_pinned_reference_tampering_is_named_not_published(tmp_path):
    """A byte change to the pinned copy fails, naming path and both digests."""
    edited = tmp_path / "jev-round-scan.py"
    edited.write_bytes(JEV_REFERENCE_PATH.read_bytes() + b"\n# reworded\n")
    with pytest.raises(JevReferenceError) as excinfo:
        jev_reference_identity(edited)
    message = str(excinfo.value)
    assert str(edited) in message
    assert JEV_REFERENCE_SHA256 in message
    assert hashlib.sha256(edited.read_bytes()).hexdigest() in message


def test_metrics_inputs_records_the_pinned_reference(tmp_path):
    """inputs.jev_reference names the copy the parity claim rests on."""
    rows = _report_rows()
    store = _report_rounds(tmp_path, rows)
    metrics, _ = assemble_baselines(
        rows, 0.2, {}, query_fn=_report_query, round_store=store
    )
    ref = metrics["inputs"]["jev_reference"]
    assert ref["sha256"] == REFERENCE_SHA256
    assert ref["path"] == "scripts/reference/jev-round-scan.py"
    assert ref["upstream"] == baselines.JEV_REFERENCE_UPSTREAM
    # the questions digest is recorded once (under jev); the pin points at that
    # field, and the pointer has to resolve — a renamed field is a rotting
    # citation, which is what F3 is about
    node = metrics
    for part in ref["questions_sha256_field"].split("."):
        node = node[part]
    assert node == jev_questions_sha256(REF.QUESTIONS) == JEV_QUESTIONS_SHA256


def test_metrics_inputs_records_the_copy_it_was_given(tmp_path):
    """The reference seam records the bytes actually used, wherever they live."""
    rows = _report_rows()
    store = _report_rounds(tmp_path, rows)
    other = tmp_path / "copies" / "jev-round-scan.py"
    other.parent.mkdir()
    other.write_bytes(JEV_REFERENCE_PATH.read_bytes())
    metrics, _ = assemble_baselines(
        rows,
        0.2,
        {},
        query_fn=_report_query,
        round_store=store,
        reference_path=other,
    )
    recorded = metrics["inputs"]["jev_reference"]
    assert recorded["sha256"] == REFERENCE_SHA256
    assert recorded["path"] == str(other)


# ---------------------------------------------------------------------------
# inputs provenance (unit-009): a committed artifact has to re-emit somewhere else
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _strings_in(node):
    """Every string key and value inside a nested JSON structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _strings_in(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings_in(item)
    elif isinstance(node, str):
        yield node


def _canonical_gold_digest(rows):
    """The documented formula, re-derived here rather than called from the code."""
    payload = [[r["id"], r["label"], r["prompt"]] for r in sorted(rows, key=lambda r: r["id"])]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def test_metrics_inputs_key_table_is_literal(tmp_path):
    """The provenance block is a fixed set of fields, spelled out in the test.

    A table generated from the code's own dict loses the case where a field is
    deleted, which is exactly the drift this block exists to make visible.
    """
    rows = _report_rows()
    store = _report_rounds(tmp_path, rows)
    metrics, _ = assemble_baselines(
        rows, 0.2, {}, query_fn=_report_query, round_store=store
    )
    assert sorted(metrics["inputs"]) == [
        "base_rate_results",
        "gold",
        "gold_sha256",
        "jev_reference",
        "round_store",
    ]


def test_metrics_inputs_records_a_digest_of_the_rows_scored(tmp_path):
    """``gold_sha256`` hashes the gold content this run scored (review F3's rule).

    The path alone does not pin anything: C1 owns that file and can re-emit it
    with a different label split, moving every number here. The digest is what
    lets a reader tell which gold the published metrics were computed on.
    """
    rows = _report_rows()
    store = _report_rounds(tmp_path, rows)
    metrics, _ = assemble_baselines(
        rows, 0.2, {}, query_fn=_report_query, round_store=store
    )
    assert metrics["inputs"]["gold_sha256"] == _canonical_gold_digest(rows)


def test_gold_digest_tracks_content_not_order_or_path():
    """Same rows in another order → same digest; any content change → another."""
    rows = _report_rows()
    base = gold_content_sha256(rows)
    assert gold_content_sha256(list(reversed(rows))) == base
    relabelled = [dict(r) for r in rows]
    relabelled[0]["label"] = "negative"
    assert gold_content_sha256(relabelled) != base
    reworded = [dict(r) for r in rows]
    reworded[3]["prompt"] = "totally different ask"
    assert gold_content_sha256(reworded) != base
    assert gold_content_sha256(rows[:-1]) != base


def test_metrics_inputs_records_no_machine_absolute_path(tmp_path, monkeypatch):
    """A home-directory location is recorded as ``~``-relative, never absolute.

    ``inputs.round_store`` used to carry ``/home/<user>/…``, so the committed
    artifact could not re-emit byte-identically on another box: the determinism
    gate would name the path field and call a username non-determinism.
    """
    home = tmp_path / "fakehome"
    home.mkdir()
    monkeypatch.setattr(baselines.Path, "home", staticmethod(lambda: home))
    store = _report_rounds(home, _report_rows())
    metrics, _ = assemble_baselines(
        _report_rows(),
        0.2,
        {},
        query_fn=_report_query,
        round_store=store,
    )
    assert metrics["inputs"]["round_store"] == "~/rounds"
    assert str(home) not in serialize_metrics(metrics)
    assert not [s for s in _strings_in(metrics["inputs"]) if s.startswith("/")], (
        "inputs must not record an absolute path"
    )


def test_input_display_path_is_verbatim_when_it_is_not_the_convention(tmp_path, monkeypatch):
    """Only a location under home is rewritten; overrides are recorded as read."""
    home = tmp_path / "fakehome"
    monkeypatch.setattr(baselines.Path, "home", staticmethod(lambda: home))
    cases = [
        (home / ".pi" / "agent" / "semblr" / "rounds", "~/.pi/agent/semblr/rounds"),
        (home / "rounds", "~/rounds"),
        (home, "~"),
        (Path("/elsewhere/rounds"), "/elsewhere/rounds"),
        (Path("dataset/rounds-labeled.jsonl"), "dataset/rounds-labeled.jsonl"),
        (Path("~/rounds"), "~/rounds"),
    ]
    for raw, expected in cases:
        assert input_display_path(raw) == expected, raw
    # the same location as a str and as a Path must render alike
    assert input_display_path(str(home / "rounds")) == input_display_path(home / "rounds")


def test_metrics_inputs_records_the_paths_it_was_given(tmp_path):
    """--gold/--base-rate-results overrides must not be recorded as the defaults."""
    rows = _report_rows()
    store = _report_rounds(tmp_path, rows)
    gold = tmp_path / "other-gold.jsonl"
    results = tmp_path / "other-base-rate-results.jsonl"
    metrics, _ = assemble_baselines(
        rows,
        0.2,
        {},
        query_fn=_report_query,
        round_store=store,
        gold_path=gold,
        base_rate_results_path=results,
    )
    assert metrics["inputs"]["gold"] == str(gold)
    assert metrics["inputs"]["base_rate_results"] == str(results)


def test_committed_metrics_inputs_reemit_off_this_machine():
    """Live anchor: the committed artifact names its inputs portably and truly.

    Two claims, both checkable from the committed files alone: the gold digest
    it records is the digest of the gold that is committed beside it, and no
    field carries this machine's home directory.
    """
    metrics_path = Path(baselines.METRICS_PATH)
    if not metrics_path.is_absolute():
        metrics_path = REPO_ROOT / metrics_path
    gold_path = REPO_ROOT / baselines.GOLD_PATH
    if not metrics_path.is_file() or not gold_path.is_file():
        pytest.skip("committed metrics or gold artifact not present")
    committed = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows, _tally = load_gold(str(gold_path))
    inputs = committed["inputs"]
    assert inputs["gold_sha256"] == gold_content_sha256(rows)
    assert inputs["gold"] == str(baselines.GOLD_PATH)
    assert inputs["base_rate_results"] == str(baselines.BASE_RATE_RESULTS_PATH)
    assert inputs["round_store"].startswith("~/")
    assert not [s for s in _strings_in(inputs) if s.startswith("/")]
    assert str(Path.home()) not in json.dumps(inputs)
