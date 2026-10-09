"""C3 — Baselines (issue #3, E2).

Metrics core (unit-001): pure-Python AUC (Mann-Whitney, ties handled,
positives rank high), FNR at declared operating points, and a seeded
bootstrap 95% CI. No numpy/scipy/sklearn.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

# ---------------------------------------------------------------------------
# Metrics core
# ---------------------------------------------------------------------------


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Mann-Whitney AUC. labels: 1 = positive, 0 = negative.

    Positives rank high by convention; ties count 0.5.
    Returns None when a class has no members (extension 5a).
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def fnr(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    """False-negative rate at `threshold`. Alarm rule: score >= threshold."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    if not pos:
        raise ValueError("FNR undefined: no positive rows")
    missed = sum(1 for s in pos if s < threshold)
    return missed / len(pos)


def sensitivity(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    return 1.0 - fnr(scores, labels, threshold)


def specificity(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not neg:
        raise ValueError("specificity undefined: no negative rows")
    alarmed = sum(1 for s in neg if s >= threshold)
    return 1.0 - alarmed / len(neg)


def _candidate_thresholds(scores: Sequence[float]) -> list[float]:
    """Deterministic candidates: midpoints between sorted unique scores,
    plus an extreme low endpoint that alarms everything."""
    uniq = sorted(set(scores))
    cands = [uniq[0]]  # alarm everything (all scores >= uniq[0])
    for lo, hi in zip(uniq, uniq[1:]):
        cands.append((lo + hi) / 2.0)
    return cands


def youden_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Threshold maximising Youden's J = sens + spec - 1.

    Ties resolve to the lowest threshold (more conservative alarms).
    """
    best: tuple[float, float] | None = None  # (J, threshold)
    for t in _candidate_thresholds(scores):
        j = sensitivity(scores, labels, t) + specificity(scores, labels, t) - 1.0
        if best is None or j > best[0] or (j == best[0] and t < best[1]):
            best = (j, t)
    assert best is not None
    return best[1]


def base_rate_threshold(
    scores: Sequence[float], labels: Sequence[int], target_rate: float
) -> float:
    """Threshold whose alarm rate is closest to `target_rate` on gold.

    Ties resolve to the lowest threshold.
    """
    best: tuple[float, float] | None = None  # (|rate - target|, threshold)
    for t in _candidate_thresholds(scores):
        alarmed = sum(1 for s in scores if s >= t) / len(scores)
        d = abs(alarmed - target_rate)
        if best is None or d < best[0] or (d == best[0] and t < best[1]):
            best = (d, t)
    assert best is not None
    return best[1]


def operating_points(
    scores: Sequence[float], labels: Sequence[int], corpus_base_rate: float
) -> dict[str, float]:
    """The two declared operating points of issue #3."""
    return {
        "youden_j": youden_threshold(scores, labels),
        "corpus_base_rate": base_rate_threshold(scores, labels, corpus_base_rate),
    }


# ---------------------------------------------------------------------------
# Seeded bootstrap
# ---------------------------------------------------------------------------

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260109


def bootstrap_ci(
    scores: Sequence[float],
    labels: Sequence[int],
    metric: Callable[[Sequence[float], Sequence[int]], float | None],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict | None:
    """Percentile bootstrap 95% CI over paired row resamples.

    Degenerate resamples (no positives or no negatives, or a metric that
    returns None) are skipped and counted. Returns None only when fewer
    than half the resamples are usable.
    """
    rng = random.Random(seed)
    n = len(scores)
    if n == 0:
        return None
    values: list[float] = []
    degenerate = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bs = [scores[i] for i in idx]
        bl = [labels[i] for i in idx]
        v = metric(bs, bl)
        if v is None:
            degenerate += 1
        else:
            values.append(v)
    if len(values) < n_boot // 2:
        return None
    values.sort()
    lo = values[int(0.025 * len(values))]
    hi = values[min(len(values) - 1, int(0.975 * len(values)))]
    point = metric(scores, labels)
    return {
        "point": point,
        "ci95": [lo, hi],
        "n_boot": n_boot,
        "seed": seed,
        "degenerate_resamples": degenerate,
    }
