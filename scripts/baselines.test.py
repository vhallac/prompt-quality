"""Unit-001 tests: metrics core (AUC, FNR/operating points, bootstrap CI)."""

import importlib.util
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
    auc,
    base_rate_threshold,
    bootstrap_ci,
    fnr,
    operating_points,
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
