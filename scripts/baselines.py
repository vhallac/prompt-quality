"""C3 — Baselines (issue #3, E2).

Metrics core (unit-001): pure-Python AUC (Mann-Whitney, ties handled,
positives rank high), FNR at declared operating points, and a seeded
bootstrap 95% CI. No numpy/scipy/sklearn.
"""

from __future__ import annotations

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
