"""C3 — Baselines (issue #3, E2).

Metrics core (unit-001): pure-Python AUC (Mann-Whitney, ties handled,
positives rank high), FNR at declared operating points, and a seeded
bootstrap 95% CI. No numpy/scipy/sklearn.
"""

from __future__ import annotations

import json
import math
import random
import re
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


# ---------------------------------------------------------------------------
# Gold loader (unit-002)
# ---------------------------------------------------------------------------

GOLD_PATH = "dataset/rounds-labeled.jsonl"


def load_gold(path: str = GOLD_PATH) -> tuple[list[dict], dict]:
    """Load the C1 gold set. Returns (scored rows, split tally).

    Scored rows carry only label ('positive'/'negative') and 'prompt';
    'excluded' rows are dropped from the denominator but counted in the
    tally. Fails with 'run C1 first' when the artifact is absent, and
    loudly when the split drifts from the recorded gold arithmetic
    (242 = 29 positive + 205 negative + 8 excluded).
    """
    try:
        f = open(path)
    except FileNotFoundError:
        raise RuntimeError(
            f"gold set not found at {path} — run C1 first (build-dataset.py)"
        ) from None
    with f:
        rows = [json.loads(line) for line in f]
    tally = {"positive": 0, "negative": 0, "excluded": 0}
    for r in rows:
        label = r.get("label")
        if label not in tally:
            raise ValueError(f"unknown gold label {label!r} on id {r.get('id')}")
        tally[label] += 1
    EXPECTED = {"positive": 29, "negative": 205, "excluded": 8, "total": 242}
    if tally != {k: EXPECTED[k] for k in tally} or len(rows) != EXPECTED["total"]:
        raise ValueError(
            f"gold split drifted: {tally}, expected 29/205/8 over 242 rows"
        )
    scored = [
        {"id": r["id"], "label": r["label"], "prompt": r["prompt"]}
        for r in rows
        if r["label"] != "excluded"
    ]
    if any(not r["prompt"] for r in scored):
        raise ValueError("empty prompt in scored gold rows — run C1 first")
    return scored, tally


def constant_prevalence_scores(rows: list[dict]) -> list[float]:
    """Baseline (a): every row scored at the gold positive prevalence.

    AUC is 0.5 by construction; this is the floor.
    """
    prev = sum(1 for r in rows if r["label"] == "positive") / len(rows)
    return [prev] * len(rows)


# ---------------------------------------------------------------------------
# Baseline (b): lexical features
# ---------------------------------------------------------------------------

# Feature definitions — printed into the metrics file by unit-004.
IMPERATIVE_VERBS = frozenset(
    """fix add remove write rename run check update create delete implement
    change make use do try build install set enable disable move copy refactor
    replace revert merge commit push pull test verify deploy configure""".split()
)

DEICTIC_MARKERS = (
    "the previous", "as discussed",
    "it", "that", "this", "yes", "again", "he", "she", "they", "them",
)

FORMAT_WORDS = frozenset(
    """json yaml yml markdown md csv tsv xml table list bullet format
    schema template snippet diff patch""".split()
)

CONTRACT_PHRASES = ("must", "should", "acceptance", "criteria", "requirement", "deliverable")


def _clauses(text: str) -> list[str]:
    parts = re.split(r"[.!?\n;]+", text)
    return [p for p in (c.strip() for c in parts) if p]


def _tokens(text: str) -> list[str]:
    return text.split()


def _word_count(text: str, word: str) -> int:
    return len(re.findall(rf"\b{re.escape(word)}\b", text.lower()))


def length_feature(prompt: str) -> dict:
    """`length` — whitespace-token count (chars also recorded)."""
    return {"length": len(_tokens(prompt)), "length_chars": len(prompt)}


def imperative_density(prompt: str) -> float:
    """`imperative_density` — fraction of clauses whose first token is an
    imperative verb (IMPERATIVE_VERBS). Empty prompts score 0."""
    clauses = _clauses(prompt)
    if not clauses:
        return 0.0
    firsts = sum(
        1
        for c in clauses
        if re.sub(r"[^\w-]", "", c.split()[0].lower()) in IMPERATIVE_VERBS
    )
    return firsts / len(clauses)


def deixis_density(prompt: str) -> float:
    """`deixis` — deictic/back-reference marker count per 100 tokens."""
    toks = _tokens(prompt)
    if not toks:
        return 0.0
    low = prompt.lower()
    hits = sum(_word_count(low, m) if " " not in m else low.count(m) for m in DEICTIC_MARKERS)
    return 100.0 * hits / len(toks)


def output_contract_absence(prompt: str) -> float:
    """`output_contract_absence` — 1 when the prompt states no explicit
    deliverable (no file path, no format word, no acceptance/'must'
    phrasing), else 0."""
    low = prompt.lower()
    has_path = bool(re.search(r"[\w./-]+\.[a-z]{1,5}\b", low)) or "/" in prompt
    has_format = any(w in FORMAT_WORDS for w in (t.strip(".,:;!?()") for t in _tokens(low)))
    has_phrase = any(p in low for p in CONTRACT_PHRASES)
    return 0.0 if (has_path or has_format or has_phrase) else 1.0


def lexical_features(prompt: str) -> dict[str, float]:
    """The four named lexical features of issue #3."""
    feats = dict(length_feature(prompt))
    feats["imperative_density"] = imperative_density(prompt)
    feats["deixis"] = deixis_density(prompt)
    feats["output_contract_absence"] = output_contract_absence(prompt)
    return feats


FEATURE_NAMES = [
    "length", "imperative_density", "deixis", "output_contract_absence",
]

# ---------------------------------------------------------------------------
# Logistic combiner (zero-dependency, fixed seed)
# ---------------------------------------------------------------------------

LOGISTIC_SEED = 20260109
LOGISTIC_EPOCHS = 2000
LOGISTIC_LR = 0.1
LOGISTIC_L2 = 1e-3


def fit_logistic(
    X: list[list[float]],
    y: list[int],
    epochs: int = LOGISTIC_EPOCHS,
    lr: float = LOGISTIC_LR,
    l2: float = LOGISTIC_L2,
    seed: int = LOGISTIC_SEED,
) -> list[float]:
    """Plain gradient descent on L2-regularised logistic loss.

    Deterministic: weights init to zeros, fixed epoch/lr; the seed is
    recorded for provenance but the fit does not sample.
    Returns [w0, w1..wd] (bias first).
    """
    d = len(X[0])
    n = len(X)
    w = [0.0] * (d + 1)
    for _ in range(epochs):
        grad = [0.0] * (d + 1)
        for xi, yi in zip(X, y):
            z = w[0] + sum(wj * xj for wj, xj in zip(w[1:], xi))
            p = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
            err = p - yi
            grad[0] += err
            for j in range(d):
                grad[j + 1] += err * xi[j]
        for j in range(d + 1):
            g = grad[j] / n + (l2 * w[j] if j > 0 else 0.0)
            w[j] -= lr * g
    return w


def logistic_score(weights: list[float], x: list[float]) -> float:
    """Raw logit — monotone in probability, so it is a valid AUC score."""
    return weights[0] + sum(wj * xj for wj, xj in zip(weights[1:], x))


def combined_lexical_scores(rows: list[dict]) -> tuple[list[float], list[float]]:
    """Fit the logistic combiner on the gold rows' standardised features and
    return (scores, weights). Standardisation uses the corpus mean/std
    (std 0 features stay at 0)."""
    feats = [lexical_features(r["prompt"]) for r in rows]
    X_raw = [[f[name] for name in FEATURE_NAMES] for f in feats]
    d = len(FEATURE_NAMES)
    means = [sum(x[j] for x in X_raw) / len(X_raw) for j in range(d)]
    stds = [
        math.sqrt(sum((x[j] - means[j]) ** 2 for x in X_raw) / len(X_raw))
        for j in range(d)
    ]
    X = [
        [0.0 if stds[j] == 0 else (x[j] - means[j]) / stds[j] for j in range(d)]
        for x in X_raw
    ]
    y = [1 if r["label"] == "positive" else 0 for r in rows]
    weights = fit_logistic(X, y)
    return [logistic_score(weights, x) for x in X], weights
