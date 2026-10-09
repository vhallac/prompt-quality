"""C3 baseline tests: metrics core + gold loader + lexical baselines."""

import importlib.util
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
    auc,
    base_rate_threshold,
    bootstrap_ci,
    combined_lexical_scores,
    constant_prevalence_scores,
    deixis_density,
    fit_logistic,
    fnr,
    imperative_density,
    length_feature,
    lexical_features,
    load_gold,
    logistic_score,
    operating_points,
    output_contract_absence,
    sensitivity,
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

    def test_standardisation_zero_std_feature(self):
        # a constant feature must not blow up: std 0 -> 0
        rows = ([{"label": "positive", "prompt": "fix it"}] * 3
                + [{"label": "negative", "prompt": "fine day"}] * 3)
        scores, weights = combined_lexical_scores(rows)
        assert len(scores) == 6 and len(weights) == 5
        assert all(math.isfinite(s) for s in scores)


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
