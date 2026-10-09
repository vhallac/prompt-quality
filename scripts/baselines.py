"""C3 — Baselines (issue #3, E2).

Metrics core (unit-001): pure-Python AUC (Mann-Whitney, ties handled,
positives rank high), FNR at declared operating points, and a seeded
bootstrap 95% CI. No numpy/scipy/sklearn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib import error, request

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
    plus an extreme low endpoint that alarms everything and an extreme high
    endpoint that alarms nothing."""
    uniq = sorted(set(scores))
    cands = [uniq[0]]  # alarm everything (all scores >= uniq[0])
    for lo, hi in zip(uniq, uniq[1:]):
        cands.append((lo + hi) / 2.0)
    cands.append(math.nextafter(uniq[-1], math.inf))  # alarm nothing
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
    # Extension 1a: an unresolved prompt is excluded from every baseline's
    # denominator (never score a prompt_preview). Gold is frozen resolved, so
    # this is defensive and leaves the recorded arithmetic untouched.
    scored = [
        {"id": r["id"], "label": r["label"], "prompt": r["prompt"]}
        for r in rows
        if r["label"] != "excluded"
        and r.get("prompt_status", "resolved") != "unresolved"
    ]
    if any(not r["prompt"] for r in scored):
        raise ValueError("empty prompt in scored gold rows — run C1 first")
    return scored, tally


def load_gold_status_ids(path: str = GOLD_PATH) -> tuple[list[str], list[str]]:
    """Ids the run report must list: (excluded, unresolved).

    ``excluded`` rows carry the ``excluded`` label (ambiguous); ``unresolved``
    rows carry ``prompt_status: unresolved`` (extension 1a). Both are absent
    from ``load_gold``'s scored rows.
    """
    excluded: list[str] = []
    unresolved: list[str] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("label") == "excluded":
                excluded.append(r["id"])
            if r.get("prompt_status", "resolved") == "unresolved":
                unresolved.append(r["id"])
    return excluded, unresolved


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


# ---------------------------------------------------------------------------
# Baseline (c): jev re-score harness (unit-003)
# ---------------------------------------------------------------------------

JEV_MODEL = "typesafe/jev-1.13"
JEV_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_STATE_CHARS = 12000  # jev-round-scan.py default --state-chars
ROUND_STORE = Path(
    os.environ.get(
        "PROMPT_QUALITY_ROUND_STORE",
        Path.home() / ".pi" / "agent" / "semblr" / "rounds",
    )
)
JEV_CACHE_PATH = Path(
    os.environ.get("PROMPT_QUALITY_JEV_CACHE", "dataset/jev-cache.json")
)
JEV_MAX_RETRIES = 3

# Marker substituted into the assistant-response slot for the prompt-only
# variant. The slot is held constant so the jev score is a function of the
# prompt alone while the payload keeps the shape the rubric expects.
RESPONSE_MARKER = "[not shown]"

# The question set of jev-round-scan.py (verbatim: the old wording that
# separated clean vs positive in the phase-1 probes).
JEV_QUESTIONS = {
    "frustration": {
        "type": "score",
        "instructions": (
            "How much frustration, irritation, or annoyance does the user "
            "express in this prompt? Consider complaints, exasperation, "
            "sarcasm, urgency, or corrections delivered with heat. A calm "
            "neutral request scores 0."
        ),
        "criteria": [
            "Calm and neutral",
            "Slightly impatient",
            "Clearly annoyed",
            "Frustrated or upset",
            "Angry or at the end of their rope",
        ],
    },
    "correction": {
        "type": "noul",
        "instructions": (
            "Is the user correcting the assistant's failure to find, recall, "
            "or surface a relevant past conversation round? Signals: the user "
            "says 'you already did this', 'we discussed this before', 'you "
            "forgot', 'search again', 'that's not what I asked', or points "
            "out that the assistant lost/misremembered context from earlier "
            "rounds. General corrections of code or answers do NOT count - "
            "only corrections about failed recall/discovery of past rounds."
        ),
    },
}


def _own_response_text(round_data: dict) -> str:
    """Own-response text as jev-round-scan.py's build_state extracts it
    (responseSequence is a plain string in current round files, but the
    segment-list shape is mirrored for older files)."""
    seq = round_data.get("responseSequence")
    if isinstance(seq, str):
        return seq
    response = ""
    for seg in seq or []:
        text = seg.get("text") if isinstance(seg, dict) else None
        if text:
            response += text + "\n"
    return response


def prompt_only_state(prompt: str) -> str:
    """State with the assistant-response slot replaced by a constant marker."""
    combined = f"USER PROMPT:\n{prompt}\n\nASSISTANT RESPONSE:\n{RESPONSE_MARKER}"
    return combined[:JEV_STATE_CHARS]


def own_response_state(prompt: str, round_data: dict) -> str:
    """State identical in shape to jev-round-scan.py's build_state output
    (the leak the scan pass already carries)."""
    combined = (
        f"USER PROMPT:\n{prompt}\n\n"
        f"ASSISTANT RESPONSE:\n{_own_response_text(round_data)}"
    )
    return combined[:JEV_STATE_CHARS]


def parent_response_state(prompt: str, parent_data: dict) -> str:
    """State in the shape of jev-round-scan.py's build_refine_state: the
    prompt plus the parent round's response with tool calls redacted."""
    texts = []
    n_tools = 0
    for seg in parent_data.get("responseSegments") or []:
        if seg.get("type") == "toolCall":
            n_tools += 1
        elif seg.get("type") == "text" and seg.get("text"):
            texts.append(seg["text"])
    redacted = "\n".join(texts)
    if n_tools:
        redacted += f"\n\n[{n_tools} tool calls in the parent response were redacted]"
    combined = (
        f"USER PROMPT:\n{prompt}\n\n"
        f"PREVIOUS ROUND RESPONSE (tool calls redacted):\n{redacted}"
    )
    return combined[:JEV_STATE_CHARS]


def jev_flag_score(answers: dict) -> float:
    """Single jev score per round: max(correction_p, frustration/4).

    Monotone in each component and consistent with the scan's OR-flag rule
    (frustration >= 3.0 or correction >= 0.7 implies score >= 0.7); the
    definition is printed into the metrics file by unit-004.
    """
    correction_p = float(answers.get("correction", {}).get("noul", 0.0))
    frustration = float(answers.get("frustration", {}).get("score", 0.0))
    return max(correction_p, frustration / 4.0)


def query_jev(
    state: str,
    api_key: str,
    questions: dict | None = None,
    url: str = JEV_DECISIONS_URL,
) -> dict:
    """One decisions-API call, in the shape of jev-round-scan.py's
    query_jev. Raises on network/HTTP errors or an unexpected response
    shape (the caller applies the retry/exclusion policy)."""
    payload = json.dumps(
        {"model": JEV_MODEL, "state": state, "questions": questions or JEV_QUESTIONS}
    ).encode()
    req = request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    answers = body.get("answers") or body.get("results") or {}
    if not answers:
        raise ValueError(f"unexpected response shape: {list(body.keys())}")
    return answers


def jev_cache_key(state: str, model: str = JEV_MODEL) -> str:
    """Stable cache key for one jev request (sample-base-rate.py pattern)."""
    h = hashlib.sha256()
    for part in (state, model):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_jev_cache(path: str | Path = JEV_CACHE_PATH) -> dict[str, str]:
    """Load a ``{cache_key: answers-json}`` cache (empty if absent)."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_jev_cache(cache: dict[str, str], path: str | Path = JEV_CACHE_PATH) -> Path:
    """Persist the cache deterministically (sorted keys, trailing newline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_round(round_id: str, round_store: str | Path = ROUND_STORE) -> dict:
    """Read one round file; raises FileNotFoundError when absent."""
    return json.loads((Path(round_store) / f"{round_id}.json").read_text())


def rescore_rows(
    rows: list[dict],
    variant: str,
    cache: dict[str, str],
    api_key: str | None = None,
    query_fn: Callable[..., dict] = query_jev,
    round_store: str | Path = ROUND_STORE,
    questions: dict | None = None,
) -> list[dict]:
    """Re-score gold rows through jev in one of three variants.

    Variants: 'prompt_only' (marker response), 'own_response' (the scan's
    leak shape), 'parent_response' (the refine shape). API results are
    cached on disk (keyed like sample-base-rate.py); a cache hit makes the
    run offline and deterministic.

    Failure policy (issue #3 extensions 4a/4b): a round whose API call
    fails after JEV_MAX_RETRIES gets score null with reason 'api_error';
    a missing round or parent file gets score null with reason
    'missing_round' / 'missing_parent'. Nulls are excluded from the
    baseline's denominator by the caller (unit-004) and listed in the run
    report; parent-response nulls exclude from the session delta only.
    """
    if variant not in ("prompt_only", "own_response", "parent_response"):
        raise ValueError(f"unknown rescore variant {variant!r}")
    out: list[dict] = []
    for row in rows:
        rid = row["id"]
        prompt = row["prompt"]
        record = {
            "id": rid,
            "label": row["label"],
            "variant": variant,
            "score": None,
            "correction": None,
            "frustration": None,
            "reason": None,
        }
        state: str | None = None
        if variant == "prompt_only":
            state = prompt_only_state(prompt)
        else:
            try:
                data = load_round(rid, round_store)
            except FileNotFoundError:
                record["reason"] = "missing_round"
                out.append(record)
                continue
            if variant == "own_response":
                state = own_response_state(prompt, data)
            else:
                parent_id = (data.get("parentId") or "").removesuffix(".json")
                if not parent_id:
                    record["reason"] = "missing_parent"
                    out.append(record)
                    continue
                try:
                    parent = load_round(parent_id, round_store)
                except FileNotFoundError:
                    record["reason"] = "missing_parent"
                    out.append(record)
                    continue
                state = parent_response_state(prompt, parent)
        assert state is not None
        key = jev_cache_key(state)
        if key in cache:
            answers = json.loads(cache[key])
        else:
            answers = None
            for _ in range(JEV_MAX_RETRIES):
                try:
                    answers = query_fn(state, api_key, questions=questions)
                    break
                except (error.URLError, OSError, ValueError):
                    continue
            if answers is None:
                record["reason"] = "api_error"
                out.append(record)
                continue
            cache[key] = json.dumps(answers, sort_keys=True)
        record["correction"] = float(answers.get("correction", {}).get("noul", 0.0))
        record["frustration"] = float(answers.get("frustration", {}).get("score", 0.0))
        record["score"] = jev_flag_score(answers)
        out.append(record)
    return out


def combined_lexical_scores(rows: list[dict]) -> tuple[list[float], dict]:
    """Fit the logistic combiner on the gold rows' standardised features and
    return (scores, combiner).

    Standardisation uses the corpus mean/std (std 0 features stay at 0).

    ``combiner`` is the published combiner, not a bare weight list: the bias is
    named as ``intercept``, and each of ``FEATURE_NAMES`` carries its own weight
    together with the mean/std used to standardise it. That is everything a
    reader needs to re-derive every ``lexical_combined`` score from the artifact
    alone (review F1); a feature-keyed weight map cannot express the intercept
    and silently shifts one weight onto every feature."""
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
    assert len(weights) == len(FEATURE_NAMES) + 1, (
        f"logistic fit returned {len(weights)} values for "
        f"{len(FEATURE_NAMES)} features + intercept"
    )
    combiner = {
        "intercept": weights[0],
        "features": {
            name: {
                "weight": weights[i + 1],
                "mean": means[i],
                "std": stds[i],
            }
            for i, name in enumerate(FEATURE_NAMES)
        },
    }
    return [logistic_score(weights, x) for x in X], combiner


# ---------------------------------------------------------------------------
# Reporting (unit-004): metrics + per-row scores + run report + determinism
# ---------------------------------------------------------------------------

METRICS_PATH = Path(
    os.environ.get("PROMPT_QUALITY_BASELINE_METRICS", "dataset/baseline-metrics.json")
)
SCORES_PATH = Path(
    os.environ.get("PROMPT_QUALITY_BASELINE_SCORES", "dataset/baseline-scores.jsonl")
)
BASE_RATE_RESULTS_PATH = Path(
    os.environ.get(
        "PROMPT_QUALITY_BASE_RATE_RESULTS", "dataset/base-rate-results.jsonl"
    )
)

# The jev variants, in emission order.
JEV_VARIANTS = ("prompt_only", "own_response", "parent_response")


def load_corpus_base_rate(path: str | Path = BASE_RATE_RESULTS_PATH) -> float:
    """C2's corpus base-rate estimate: positive / (positive + negative) over
    the adjudicated random sample.

    Reuses C2's ``classify_bin`` (issue #2) rather than re-deriving the label
    rule here; ``ambiguous`` is excluded from the denominator.
    """
    import importlib.util
    import sys as _sys

    module_path = Path(__file__).with_name("sample-base-rate.py")
    name = "baselines_c2_sample"
    if name not in _sys.modules:
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        _sys.modules[name] = module
        spec.loader.exec_module(module)
    classify_bin = _sys.modules[name].classify_bin

    positive = negative = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            kind = classify_bin(row.get("bin") or "ambiguous")
            if kind == "positive":
                positive += 1
            elif kind == "negative":
                negative += 1
    if positive + negative == 0:
        raise ValueError(f"no adjudicated rows in {path} — run C2 first")
    return positive / (positive + negative)


def _labels_of(rows: list[dict]) -> list[int]:
    return [1 if r["label"] == "positive" else 0 for r in rows]


def _records_to_vectors(
    records: list[dict],
) -> tuple[list[float], list[int], dict[str, list[str]]]:
    """Split re-score records into (scores, labels, nulls-by-reason).

    Null scores (extensions 4a/4b) are excluded from the baseline denominator
    and returned keyed by their ``reason`` for the run report.
    """
    scores: list[float] = []
    labels: list[int] = []
    nulls: dict[str, list[str]] = {}
    for r in records:
        if r["score"] is None:
            nulls.setdefault(r["reason"] or "unknown", []).append(r["id"])
            continue
        scores.append(r["score"])
        labels.append(1 if r["label"] == "positive" else 0)
    for ids in nulls.values():
        ids.sort()
    return scores, labels, nulls


def baseline_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    corpus_base_rate: float,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """AUC + FNR at the two declared operating points, each with a seeded
    bootstrap 95% CI (issue #3 extension 2a).

    FNR CIs re-derive the threshold inside each bootstrap resample, so the
    interval carries threshold uncertainty as well as sampling noise. A
    single-class score vector yields ``auc: null`` and null FNRs with a note
    (extension 5a); only the full gold set is required to have both classes.
    """
    scores = list(scores)
    labels = list(labels)
    pos = sum(labels)
    neg = len(labels) - pos
    out: dict = {
        "n": len(scores),
        "positives": pos,
        "negatives": neg,
        "auc": None,
        "fnr": None,
    }
    if pos == 0 or neg == 0:
        out["note"] = "single-class stratum: AUC is null (extension 5a)"
        return out
    out["auc"] = bootstrap_ci(scores, labels, auc, n_boot, seed)

    def fnr_at_youden(s: Sequence[float], l: Sequence[int]) -> float | None:
        try:
            return fnr(s, l, youden_threshold(s, l))
        except ValueError:
            return None

    def fnr_at_base_rate(s: Sequence[float], l: Sequence[int]) -> float | None:
        try:
            return fnr(s, l, base_rate_threshold(s, l, corpus_base_rate))
        except ValueError:
            return None

    def _point_and_ci(metric: Callable) -> dict:
        ci = bootstrap_ci(scores, labels, metric, n_boot, seed)
        return {
            "point": metric(scores, labels),
            "ci95": ci["ci95"] if ci else None,
        }

    out["fnr"] = {
        "youden_j": {
            "threshold": youden_threshold(scores, labels),
            **_point_and_ci(fnr_at_youden),
        },
        "corpus_base_rate": {
            "threshold": base_rate_threshold(scores, labels, corpus_base_rate),
            **_point_and_ci(fnr_at_base_rate),
        },
    }
    return out


def paired_auc_delta(
    records_a: list[dict],
    records_b: list[dict],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """AUC(records_b) - AUC(records_a) over the rows valid in *both* variants.

    Unpairable rows (a null score or a missing round file, extensions 4a/4b)
    are excluded from the delta only and listed by the caller's run report;
    the absolute baselines keep their own denominators.
    """
    b_by_id = {r["id"]: r for r in records_b}
    left: list[float] = []
    right: list[float] = []
    labels: list[int] = []
    for r in records_a:
        other = b_by_id.get(r["id"])
        if other is None or r["score"] is None or other["score"] is None:
            continue
        left.append(r["score"])
        right.append(other["score"])
        labels.append(1 if r["label"] == "positive" else 0)

    def delta(sl: Sequence[float], sr: Sequence[float], lb: Sequence[int]) -> float | None:
        a = auc(sl, lb)
        c = auc(sr, lb)
        if a is None or c is None:
            return None
        return c - a

    if not left:
        return {"point": None, "ci95": None, "n_pairs": 0, "seed": seed}
    rng = random.Random(seed)
    n = len(left)
    values: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = delta(
            [left[i] for i in idx],
            [right[i] for i in idx],
            [labels[i] for i in idx],
        )
        if v is not None:
            values.append(v)
    ci95 = None
    if len(values) >= n_boot // 2:
        values.sort()
        ci95 = [
            values[int(0.025 * len(values))],
            values[min(len(values) - 1, int(0.975 * len(values)))],
        ]
    return {
        "point": delta(left, right, labels),
        "ci95": ci95,
        "n_pairs": n,
        "seed": seed,
        "n_boot": n_boot,
    }


def _feature_definitions() -> dict:
    """The feature definitions printed into the metrics file (extension 3a)."""
    return {
        "length": {
            "description": "whitespace-token count of the prompt (chars also recorded)",
            "units": "tokens",
        },
        "imperative_density": {
            "description": "fraction of clauses whose first token is an imperative verb",
            "imperative_verbs": sorted(IMPERATIVE_VERBS),
            "empty_prompt_score": 0.0,
        },
        "deixis": {
            "description": "deictic/back-reference marker count per 100 tokens",
            "markers": list(DEICTIC_MARKERS),
            "empty_prompt_score": 0.0,
        },
        "output_contract_absence": {
            "description": (
                "1 when the prompt states no explicit deliverable (no file path, "
                "no format word, no acceptance/'must' phrasing), else 0"
            ),
            "format_words": sorted(FORMAT_WORDS),
            "contract_phrases": list(CONTRACT_PHRASES),
            "unmatched_prompt_score": 0.0,
        },
    }


def per_feature_aucs(
    rows: list[dict],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """AUC with seeded bootstrap CI for each lexical feature, on its own.

    Issue #3 main success scenario step 3: baseline (b) reports each
    feature's AUC alongside the combined score. Same scored gold rows and
    same seed as every other AUC, so the intervals are comparable.
    """
    labels = _labels_of(rows)
    feats = [lexical_features(r["prompt"]) for r in rows]
    return {
        name: bootstrap_ci([f[name] for f in feats], labels, auc, n_boot, seed)
        for name in FEATURE_NAMES
    }


def _score_rows(
    rows: list[dict],
    lexical_scores: list[float],
    base_scores: list[float],
    jev: dict[str, list[dict]],
) -> list[dict]:
    """Per-row score dump for C4/C5/C6: gold identity, lexical features,
    combiner score, and every jev variant score (nulls kept, with reason)."""
    by_variant = {v: {r["id"]: r for r in jev[v]} for v in JEV_VARIANTS}
    out: list[dict] = []
    for row, lex, base in zip(rows, lexical_scores, base_scores):
        rid = row["id"]
        rec: dict = {
            "id": rid,
            "label": row["label"],
            "base_rate": base,
            "lexical_combined": lex,
        }
        for name, value in lexical_features(row["prompt"]).items():
            rec[f"feat_{name}"] = value
        for variant in JEV_VARIANTS:
            j = by_variant[variant][rid]
            rec[f"jev_{variant}"] = j["score"]
            rec[f"jev_{variant}_reason"] = j["reason"]
            if j["score"] is not None:
                rec[f"jev_{variant}_correction"] = j["correction"]
                rec[f"jev_{variant}_frustration"] = j["frustration"]
        out.append(rec)
    return out


def assemble_baselines(
    rows: list[dict],
    corpus_base_rate: float,
    cache: dict[str, str],
    api_key: str | None = None,
    query_fn: Callable[..., dict] = query_jev,
    round_store: str | Path = ROUND_STORE,
    questions: dict | None = None,
    excluded_ids: list[str] | None = None,
    unresolved_ids: list[str] | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict, list[dict]]:
    """Assemble the metrics document and per-row score dump.

    Pure with respect to disk except for ``cache``, which ``rescore_rows``
    fills in place; the caller persists it. With a warm cache the call is
    offline and deterministic (extension 7a).
    """
    labels = _labels_of(rows)
    lexical_scores, combiner = combined_lexical_scores(rows)
    base_scores = constant_prevalence_scores(rows)
    jev = {
        v: rescore_rows(
            rows, v, cache, api_key, query_fn, round_store, questions
        )
        for v in JEV_VARIANTS
    }

    jev_vectors = {v: _records_to_vectors(jev[v]) for v in JEV_VARIANTS}
    baselines_out = {
        "base_rate": baseline_metrics(base_scores, labels, corpus_base_rate, n_boot, seed),
        "lexical": baseline_metrics(lexical_scores, labels, corpus_base_rate, n_boot, seed),
    }
    for variant in JEV_VARIANTS:
        scores, labs, _ = jev_vectors[variant]
        baselines_out[f"jev_{variant}"] = baseline_metrics(
            scores, labs, corpus_base_rate, n_boot, seed
        )

    prompt_only = jev["prompt_only"]
    deltas = {
        "leak": paired_auc_delta(prompt_only, jev["own_response"], n_boot, seed),
        "session": paired_auc_delta(prompt_only, jev["parent_response"], n_boot, seed),
    }

    null_scores = {v: jev_vectors[v][2] for v in JEV_VARIANTS}
    delta_exclusions = {
        "leak": sorted(
            {
                r["id"]
                for r in jev["own_response"]
                if r["score"] is None
            }
            | {r["id"] for r in prompt_only if r["score"] is None}
        ),
        "session": sorted(
            {
                r["id"]
                for r in jev["parent_response"]
                if r["score"] is None
            }
            | {r["id"] for r in prompt_only if r["score"] is None}
        ),
    }

    metrics: dict = {
        "seed": seed,
        "n_boot": n_boot,
        "corpus_base_rate_target": corpus_base_rate,
        "inputs": {
            "gold": str(GOLD_PATH),
            "base_rate_results": str(BASE_RATE_RESULTS_PATH),
            "round_store": str(round_store),
        },
        "gold": {
            "scored_rows": len(rows),
            "positives": labels.count(1),
            "negatives": labels.count(0),
        },
        "feature_definitions": _feature_definitions(),
        "feature_aucs": per_feature_aucs(rows, n_boot, seed),
        "lexical_weights": combiner,
        "jev": {
            "model": JEV_MODEL,
            "state_chars": JEV_STATE_CHARS,
            "response_marker": RESPONSE_MARKER,
            "score_definition": "max(correction.noul, frustration.score / 4)",
        },
        "baselines": baselines_out,
        "deltas": deltas,
        "run_report": {
            "excluded_ids": sorted(excluded_ids or []),
            "unresolved_ids": sorted(unresolved_ids or []),
            "null_scores": null_scores,
            "delta_exclusions": delta_exclusions,
            "thresholds_used": {
                name: {
                    "youden_j": entry["fnr"]["youden_j"]["threshold"],
                    "corpus_base_rate": entry["fnr"]["corpus_base_rate"]["threshold"],
                }
                for name, entry in baselines_out.items()
                if entry.get("fnr")
            },
        },
    }
    return metrics, _score_rows(rows, lexical_scores, base_scores, jev)


def serialize_metrics(metrics: dict) -> str:
    """Deterministic JSON rendering (sorted keys, 2-space indent, newline)."""
    return json.dumps(metrics, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def serialize_score_rows(rows: list[dict]) -> str:
    """Deterministic JSONL rendering (sorted keys, one row per line)."""
    return "".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows
    )


def _first_divergent_path(a, b, prefix: str = "") -> str:
    """Name the first field where two parsed structures differ.

    Used so extension 7a's failure says *which* number/row drifted rather
    than just that bytes differ.
    """
    if type(a) is not type(b):
        return prefix or "$type"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{prefix}.{key}" if prefix else key
            if a[key] != b[key]:
                child = f"{prefix}.{key}" if prefix else key
                return _first_divergent_path(a[key], b[key], child)
        return prefix
    if isinstance(a, list):
        if len(a) != len(b):
            return prefix or "$length"
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                child = f"{prefix}[{i}]" if prefix else f"[{i}]"
                return _first_divergent_path(x, y, child)
        return prefix
    return prefix or "$value"


def emit_artifacts(
    metrics: dict,
    score_rows: list[dict],
    metrics_path: str | Path = METRICS_PATH,
    scores_path: str | Path = SCORES_PATH,
) -> dict[str, Path]:
    """Write both artifacts, failing loudly if a prior identical-input run
    would now differ (extension 7a names the drifting field).

    The comparison is against the existing file rather than an in-memory
    re-serialization: it catches cross-run drift (ordering, numeric noise)
    that an in-process double-serialize cannot.
    """
    metrics_path = Path(metrics_path)
    scores_path = Path(scores_path)
    new_metrics = serialize_metrics(metrics)
    new_scores = serialize_score_rows(score_rows)

    if metrics_path.exists():
        old = metrics_path.read_text(encoding="utf-8")
        if old != new_metrics:
            field = _first_divergent_path(json.loads(old), json.loads(new_metrics))
            raise RuntimeError(
                f"non-deterministic output in {metrics_path.name}: {field}"
            )
    if scores_path.exists():
        old = scores_path.read_text(encoding="utf-8")
        if old != new_scores:
            old_lines = old.splitlines()
            new_lines = new_scores.splitlines()
            for i, (x, y) in enumerate(zip(old_lines, new_lines)):
                if x != y:
                    rid = json.loads(x).get("id", "?")
                    raise RuntimeError(
                        f"non-deterministic output in {scores_path.name}: "
                        f"row {rid} (line {i})"
                    )
            raise RuntimeError(
                f"non-deterministic output in {scores_path.name}: row count"
            )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(new_metrics, encoding="utf-8")
    scores_path.write_text(new_scores, encoding="utf-8")
    return {"metrics": metrics_path, "scores": scores_path}


def main(argv: list[str] | None = None) -> int:
    """Run C3 end-to-end: score gold, emit metrics + per-row scores.

    A warm jev cache makes the run offline; newly fetched API results are
    persisted so the next run is deterministic (extension 7a).
    """
    parser = argparse.ArgumentParser(description="C3 — Baselines (E2)")
    parser.add_argument("--gold", default=GOLD_PATH)
    parser.add_argument("--base-rate-results", default=str(BASE_RATE_RESULTS_PATH))
    parser.add_argument("--cache", default=str(JEV_CACHE_PATH))
    parser.add_argument("--metrics", default=str(METRICS_PATH))
    parser.add_argument("--scores", default=str(SCORES_PATH))
    parser.add_argument("--round-store", default=str(ROUND_STORE))
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    args = parser.parse_args(argv)

    rows, _tally = load_gold(args.gold)
    excluded_ids, unresolved_ids = load_gold_status_ids(args.gold)
    corpus_base_rate = load_corpus_base_rate(args.base_rate_results)
    cache = load_jev_cache(args.cache)
    metrics, score_rows = assemble_baselines(
        rows,
        corpus_base_rate,
        cache,
        api_key=args.api_key,
        round_store=args.round_store,
        excluded_ids=excluded_ids,
        unresolved_ids=unresolved_ids,
    )
    save_jev_cache(cache, args.cache)
    paths = emit_artifacts(metrics, score_rows, args.metrics, args.scores)
    print(
        f"wrote {paths['metrics']} and {paths['scores']} "
        f"(cache {len(cache)} entries, base rate {corpus_base_rate:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
