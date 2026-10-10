"""C3 — Baselines (issue #3, E2).

Metrics core (unit-001): pure-Python AUC (Mann-Whitney, ties handled,
positives rank high), FNR at declared operating points, and a seeded
bootstrap 95% CI. No numpy/scipy/sklearn.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


def alarm_rate(scores: Sequence[float], threshold: float) -> float:
    """Fraction of rows alarmed at `threshold` (alarm rule: score >= threshold).

    The denominator is **all** rows, which is how operating point (ii) states its
    target (issue #3: "the threshold whose alarm rate equals the C2 corpus
    base-rate estimate"). ``specificity`` divides by the negative rows only; the
    two are not interchangeable.
    """
    if not scores:
        raise ValueError("alarm rate undefined: no rows")
    return sum(1 for s in scores if s >= threshold) / len(scores)


def alarm_rate_tolerance(n: int) -> float:
    """Stated rate-matching tolerance for operating point (ii): ``0.5 / n``.

    Candidate thresholds are midpoints between distinct scores, so the alarm
    rates an n-row vector can express are multiples of ``1/n`` at best; half that
    step is the smallest error any selection rule can guarantee. A target missed
    by more than this is not a rounding artefact: the score vector cannot reach
    that rate at all. Ties collapse the reachable set, and a constant-score
    vector (the prevalence floor baseline) can only alarm everything or nothing.
    """
    if n <= 0:
        raise ValueError("alarm rate tolerance undefined for an empty vector")
    return 0.5 / n


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
        # Same primitive the disclosure publishes, so the rate the selection was
        # scored on can never drift from the rate the cell reports.
        alarmed = alarm_rate(scores, t)
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
# Logistic combiner (zero-dependency, deterministic)
# ---------------------------------------------------------------------------

LOGISTIC_EPOCHS = 2000
LOGISTIC_LR = 0.1
LOGISTIC_L2 = 1e-3


def fit_logistic(
    X: list[list[float]],
    y: list[int],
    epochs: int = LOGISTIC_EPOCHS,
    lr: float = LOGISTIC_LR,
    l2: float = LOGISTIC_L2,
) -> list[float]:
    """Plain gradient descent on L2-regularised logistic loss.

    Fully deterministic: weights init to zeros and the fit never samples, so
    there is no seed to pass. Only the bootstrap CIs are seed-sensitive.
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
#
# The constants and state builders below are *copies* of the external jev
# round-scan scorer's, so they need an in-repo authority to be checked
# against. That authority is the pinned reference copy: byte-identical to
# ``semblr scripts/jev-round-scan.py`` at commit ``ba10970``, committed here
# so the parity claim resolves from a clean clone with no sibling checkout
# (review F3; the same convention C2 uses for its scan snapshot).
#
# Comments below cite that copy as ``reference:NN``. Line numbers are safe to
# cite here — the bytes are pinned, so a re-serialised reference changes its
# digest and ``jev_reference_identity`` fails the run naming the path.

JEV_REFERENCE_PATH = (
    Path(__file__).resolve().parent / "reference" / "jev-round-scan.py"
)
JEV_REFERENCE_SHA256 = (
    "4e294839a341c953d38dba17dadfd87be53d36f536700d3274da16f387bf8704"
)
JEV_REFERENCE_UPSTREAM = "semblr scripts/jev-round-scan.py at ba10970"

JEV_MODEL = "typesafe/jev-1.13"  # reference:44 (MODEL)
JEV_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"  # reference:43
JEV_STATE_CHARS = 12000  # reference:272 (--state-chars default)
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
# separated clean vs positive in the phase-1 probes) — reference:46-76
# (QUESTIONS), byte-compared by the parity tests.
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


def jev_questions_sha256(questions: dict | None = None) -> str:
    """Digest of the questions payload a jev request carries.

    Canonicalised with sorted keys, so the digest is stable across runs and
    across dict key insertion order, while any change to the rubric text, the
    criteria, or the set of questions changes it. Decision 008 requires the
    cache key to cover the whole request payload; review F2 found the questions
    were the one payload field the key left out.
    """
    payload = json.dumps(questions or JEV_QUESTIONS, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


JEV_QUESTIONS_SHA256 = jev_questions_sha256()


class JevReferenceError(RuntimeError):
    """The pinned jev reference scorer is missing or no longer the pinned bytes.

    Raised with the path in the message (review F3): the point of pinning is
    that an edit to the reference — or to this stage's copies of it — stops
    the run instead of silently leaving "re-runs the jev round-scan scorer"
    false.
    """


def load_jev_reference(path: str | Path = JEV_REFERENCE_PATH):
    """Import the pinned reference scorer as a module.

    The copy is stdlib-only at module level and its ``main()`` sits behind an
    ``if __name__ == "__main__"`` guard, so importing it has no side effects
    and no dependency on the sibling checkout. Used by the parity tests; the
    stage itself keeps its own copies of the constants (decision 013).
    """
    ref_path = Path(path)
    if not ref_path.is_file():
        raise JevReferenceError(f"pinned jev reference scorer not found: {ref_path}")
    spec = importlib.util.spec_from_file_location("jev_reference_scan", ref_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jev_reference_identity(path: str | Path = JEV_REFERENCE_PATH) -> dict:
    """The pinned reference scorer's identity, for ``metrics.inputs``.

    Hashes the pinned bytes and refuses to publish them unless they still
    match ``JEV_REFERENCE_SHA256``. The questions digest is *not* duplicated
    here: it is published once as ``jev.questions_sha256`` (unit-005) and also
    keys the cache (decision 008), so this block points at that field.
    """
    ref_path = Path(path)
    try:
        digest = hashlib.sha256(ref_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise JevReferenceError(
            f"cannot read the pinned jev reference scorer: {exc}"
        ) from exc
    if digest != JEV_REFERENCE_SHA256:
        raise JevReferenceError(
            f"pinned jev reference scorer changed: {ref_path} is {digest}, "
            f"expected {JEV_REFERENCE_SHA256} ({JEV_REFERENCE_UPSTREAM})"
        )
    try:
        recorded = str(
            ref_path.resolve().relative_to(Path(__file__).resolve().parent.parent)
        )
    except ValueError:
        recorded = str(ref_path)
    return {
        "path": recorded,
        "sha256": digest,
        "upstream": JEV_REFERENCE_UPSTREAM,
        "questions_sha256_field": "jev.questions_sha256",
    }


# The shipped jev cache predates questions-keyed entries, so its file records
# no digest of its own. The question set that produced it is a fact of git
# history: JEV_QUESTIONS was introduced at 6635d82 and has not been edited in
# any commit since, and dataset/jev-cache.json was first committed at 567c8be.
# That payload digest is therefore pinned here as the unrecorded cache's
# provenance. Editing the rubric changes JEV_QUESTIONS_SHA256 without changing
# this constant, and the adoption path below closes: the paid entries become
# unreachable (a miss, and without an API key a loud failure) instead of
# answering questions they were never asked. Re-derive it from git and change
# it only to re-stamp a pre-metadata cache file deliberately.
JEV_CACHE_LEGACY_QUESTIONS_SHA256 = (
    "0af0518d16be199ccb22ab4d60d159269dd5a555dc55c0cb15f2510cab2c6c69"
)

# Reserved member of the cache document. Every cache key is a 64-hex digest, so
# this name cannot collide with one; storing it beside the entries keeps the
# file a single flat JSON object and its diff to one added line.
JEV_CACHE_QUESTIONS_KEY = "__questions_sha256__"


class JevCacheError(RuntimeError):
    """The jev cache file exists but cannot be read as a cache.

    Raised with the path in the message (review F8): silently treating a
    half-written or hand-edited cache as cold would discard paid answers and
    re-bill for them without telling anyone.
    """


def _own_response_text(round_data: dict) -> str:
    """Own-response text as jev-round-scan.py's build_state extracts it
    (reference:80-90) — responseSequence is a plain string in current round
    files, but the segment-list shape is mirrored for older files."""
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
    (reference:78-92; asserted against it by the parity tests) — the leak the
    scan pass already carries."""
    combined = (
        f"USER PROMPT:\n{prompt}\n\n"
        f"ASSISTANT RESPONSE:\n{_own_response_text(round_data)}"
    )
    return combined[:JEV_STATE_CHARS]


def parent_response_state(prompt: str, parent_data: dict) -> str:
    """State in the shape of jev-round-scan.py's build_refine_state
    (reference:154-173): the prompt plus the parent round's response with tool
    calls redacted."""
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
    query_jev (reference:95-114). Raises on network/HTTP errors or an
    unexpected response shape (the caller applies the retry/exclusion policy).

    Raises without touching the network when no key is configured: a cache
    miss then says "no API key" once instead of recording every row as an
    api_error after a hundred futile requests (review F2's miss path). A
    RuntimeError, so it is not swallowed by the retry policy.
    """
    if not api_key:
        raise RuntimeError(
            "jev cache miss and no API key configured: set OPENROUTER_API_KEY "
            "(or pass --api-key), or warm the jev cache (--cache, "
            "PROMPT_QUALITY_JEV_CACHE)"
        )
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


def jev_cache_key(
    state: str,
    model: str = JEV_MODEL,
    questions_sha256: str = JEV_QUESTIONS_SHA256,
) -> str:
    """Cache key for one jev request: state, model, and the question payload.

    Payload-complete in the sense of decision 008 — every input that can change
    the answer is in the key, the questions as their digest so the key stays a
    fixed-width hex (review F2).
    """
    h = hashlib.sha256()
    for part in (state, model, questions_sha256):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def jev_cache_key_legacy(state: str, model: str = JEV_MODEL) -> str:
    """The pre-F2 ``(state, model)`` key.

    Read only by the adoption path in ``rescore_rows`` and never written, so a
    cache file written by this code addresses its entries by the full payload
    alone (unit-005's output contract).
    """
    h = hashlib.sha256()
    for part in (state, model):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def jev_cache_questions_sha256(cache: dict[str, str]) -> str | None:
    """The question-set digest recorded for this cache, if the file stated one."""
    value = cache.get(JEV_CACHE_QUESTIONS_KEY)
    return value if isinstance(value, str) else None


def jev_cache_entry_count(cache: dict[str, str]) -> int:
    """Answer count, ignoring the reserved questions-digest member."""
    return sum(1 for key in cache if key != JEV_CACHE_QUESTIONS_KEY)


def jev_legacy_adoption_allowed(
    cache: dict[str, str], questions_sha256: str
) -> bool:
    """May ``(state, model)``-keyed entries answer this question set?

    Only when the digest recorded for the cache is the digest in force. A file
    that recorded nothing is read through the legacy key only while the payload
    in force still digests to the pinned provenance of the shipped cache.
    """
    recorded = jev_cache_questions_sha256(cache)
    if recorded is None:
        recorded = JEV_CACHE_LEGACY_QUESTIONS_SHA256
    return recorded == questions_sha256


def _is_cache_digest(key: str) -> bool:
    """A cache key is a sha256 hex digest — nothing else belongs in the map."""
    return len(key) == 64 and all(c in "0123456789abcdef" for c in key)


def _jev_cache_entries(doc: dict, path: Path) -> dict[str, str]:
    """Validate the loaded document's shape and return it.

    Every entry value is parsed as JSON here rather than at lookup, so a
    corrupt answer names the file and the key instead of raising from the
    middle of a scored run.
    """
    for key, value in doc.items():
        if key == JEV_CACHE_QUESTIONS_KEY:
            if not isinstance(value, str) or not _is_cache_digest(value):
                raise JevCacheError(
                    f"jev cache {path} records {value!r} as its questions digest, "
                    "which is not a sha256 hex digest"
                )
            continue
        if not _is_cache_digest(key):
            raise JevCacheError(
                f"jev cache {path} holds a key that is not a cache digest: {key!r}"
            )
        if not isinstance(value, str):
            raise JevCacheError(f"jev cache {path} entry {key!r} is not a JSON string")
        try:
            answers = json.loads(value)
        except json.JSONDecodeError as exc:
            raise JevCacheError(
                f"jev cache {path} entry {key!r} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(answers, dict):
            raise JevCacheError(
                f"jev cache {path} entry {key!r} does not hold a JSON object"
            )
    return doc


def load_jev_cache(path: str | Path = JEV_CACHE_PATH) -> dict[str, str]:
    """Load a ``{cache_key: answers-json}`` cache; a cold start is an empty map.

    Absent, empty, or whitespace-only → ``{}``: such a file holds no paid work
    to lose. Non-empty but unreadable → ``JevCacheError`` naming the path: a
    truncated or hand-edited cache may hold every answer this project has paid
    for, and silently starting cold would re-bill for them (review F8).
    """
    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JevCacheError(f"jev cache {path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise JevCacheError(f"jev cache {path} must hold a JSON object")
    return _jev_cache_entries(doc, path)


def save_jev_cache(
    cache: dict[str, str],
    path: str | Path = JEV_CACHE_PATH,
    questions_sha256: str | None = None,
) -> Path:
    """Persist the cache deterministically (sorted keys, trailing newline).

    Written through a temp file in the same directory and moved with
    ``os.replace``, so a kill mid-save cannot leave the tracked cache truncated
    (review F8). ``questions_sha256`` records the payload these entries answer;
    it is what gates the legacy-key adoption path on the next load, so it is
    written only when the caller states it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(cache)
    if questions_sha256 is not None:
        doc[JEV_CACHE_QUESTIONS_KEY] = questions_sha256
    text = json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
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
    cached on disk keyed on the whole request payload — state, model, and
    the question set (decision 008, review F2); a cache hit makes the run
    offline and deterministic. Entries written before the question set
    joined the key are adopted under the payload-complete key, and only
    while the digest recorded for that cache still matches the payload in
    force: editing the rubric is a miss, never the old answers.

    Failure policy (issue #3 extensions 4a/4b): a round whose API call
    fails after JEV_MAX_RETRIES gets score null with reason 'api_error';
    a missing round or parent file gets score null with reason
    'missing_round' / 'missing_parent'. Nulls are excluded from the
    baseline's denominator by the caller (unit-004) and listed in the run
    report; parent-response nulls exclude from the session delta only.
    """
    if variant not in ("prompt_only", "own_response", "parent_response"):
        raise ValueError(f"unknown rescore variant {variant!r}")
    questions_sha256 = jev_questions_sha256(questions)
    adopt_legacy = jev_legacy_adoption_allowed(cache, questions_sha256)
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
        key = jev_cache_key(state, questions_sha256=questions_sha256)
        answers_json = cache.get(key)
        if answers_json is None and adopt_legacy:
            # Pre-F2 entry: same state, same model, and the gate above proved
            # the same rubric, so it answers this exact question. Adopt it
            # under the payload-complete key and drop the legacy key, leaving
            # the file addressable by the full payload alone.
            answers_json = cache.pop(jev_cache_key_legacy(state), None)
            if answers_json is not None:
                cache[key] = answers_json
        if answers_json is not None:
            answers = json.loads(answers_json)
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

    Every FNR cell discloses the alarm rate it actually achieved. Operating point
    (i) has no rate target (it targets Youden's J), so its ``target_alarm_rate``
    is null. Point (ii) does, and when the achieved rate misses it by more than
    the stated ``alarm_rate_tolerance`` the cell carries a note naming the
    degeneracy: nearest-candidate selection is then reporting the closest
    reachable rate, not the declared one. Threshold *selection* is unchanged, so
    no published threshold or FNR moves.
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

    def _rate_disclosure(threshold: float, target: float | None) -> dict:
        achieved = alarm_rate(scores, threshold)
        cell: dict = {
            "target_alarm_rate": target,
            "achieved_alarm_rate": achieved,
        }
        if target is None:
            return cell
        tolerance = alarm_rate_tolerance(len(scores))
        cell["alarm_rate_tolerance"] = tolerance
        if abs(achieved - target) > tolerance:
            if achieved == 0.0:
                reached = "the threshold alarms nothing, so FNR is 1.0 by construction"
            elif achieved == 1.0:
                reached = "the threshold alarms every row, so FNR is 0.0 by construction"
            else:
                reached = f"the threshold alarms {achieved:.6g} of rows"
            cell["note"] = (
                "non-rate-matched operating point (ii): achieved alarm rate "
                f"{achieved:.6g} vs target {target:.6g} misses by more than the "
                f"stated {tolerance:.6g} tolerance; the score vector has only "
                f"{len(set(scores))} distinct value(s) so the target rate is "
                f"unreachable and the nearest candidate was selected \u2014 {reached}"
            )
        return cell

    threshold_youden = youden_threshold(scores, labels)
    threshold_base_rate = base_rate_threshold(scores, labels, corpus_base_rate)
    out["fnr"] = {
        "youden_j": {
            "threshold": threshold_youden,
            **_rate_disclosure(threshold_youden, None),
            **_point_and_ci(fnr_at_youden),
        },
        "corpus_base_rate": {
            "threshold": threshold_base_rate,
            **_rate_disclosure(threshold_base_rate, corpus_base_rate),
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

    The block therefore names the population it was computed on. ``n_pairs``
    counts the rows valid in both arms, ``population`` says what that means and
    why it matters, and ``auc_reference_paired`` gives both arms' AUCs *on that
    subpopulation*, whose difference is exactly ``point``. The caller adds
    ``arms`` (the ``baselines`` entries the two arms are) and ``auc_marginal``
    (their published points). Subtracting the marginal AUCs is then visibly a
    different quantity from the paired delta whenever the populations differ.
    No delta value changes: ``point`` and ``ci95`` are the same arithmetic on
    the same rows in the same bootstrap order.
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

    n = len(left)
    ci95: list[float] | None = None
    if n:
        rng = random.Random(seed)
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
        if len(values) >= n_boot // 2:
            values.sort()
            ci95 = [
                values[int(0.025 * len(values))],
                values[min(len(values) - 1, int(0.975 * len(values)))],
            ]
    paired_reference = auc(left, labels)
    paired_comparison = auc(right, labels)
    paired_difference = (
        None
        if paired_reference is None or paired_comparison is None
        else paired_comparison - paired_reference
    )
    return {
        "point": delta(left, right, labels),
        "ci95": ci95,
        "n_pairs": n,
        "n_reference_rows": len(records_a),
        "population": (
            f"n_pairs = {n}: {n} of the {len(records_a)} reference-arm rows "
            "score non-null in both arms. Rows unpairable on either side "
            "(extensions 4a/4b) are excluded from this delta only, so the point "
            "is a paired difference on that subpopulation and need not equal the "
            "difference of the two marginal AUCs published under `baselines`; "
            "auc_reference_paired holds both arms' AUCs computed on these paired "
            "rows, and its difference is the point."
        ),
        "auc_reference_paired": {
            "reference": paired_reference,
            "comparison": paired_comparison,
            "difference": paired_difference,
        },
        "seed": seed,
        "n_boot": n_boot,
    }


def _feature_definitions() -> dict:
    """The feature definitions printed into the metrics file (extension 3a)."""
    return {
        "ext_3a_scope": (
            "'An ambiguous feature that matches nothing scores 0' applies to the "
            "count/ratio features only: length, imperative_density, deixis. "
            "output_contract_absence is an absence indicator, so a prompt that "
            "matches none of its markers scores 1."
        ),
        "length": {
            "description": "whitespace-token count of the prompt (chars also recorded)",
            "units": "tokens",
            "empty_prompt_score": 0.0,
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
            "no_marker_score": 1.0,
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


def input_display_path(path: str | Path) -> str:
    """How a committed artifact names one of its input locations.

    An absolute path under the home directory is recorded relative to it
    (``~/.pi/agent/semblr/rounds``), so the artifact says the same thing on every
    box that keeps the convention (decision 015) and a re-emit is byte-identical
    off this machine. Anything else — a path outside home, a relative path, a
    ``--round-store`` override — is recorded verbatim: it is not the convention,
    so the artifact must name what was actually read rather than pretend.
    Overriding the location is the ``PROMPT_QUALITY_ROUND_STORE`` variable's job,
    not this field's.
    """
    raw = str(path)
    resolved = Path(raw).expanduser()
    if not resolved.is_absolute():
        return raw
    home = Path.home()
    if resolved == home:
        return "~"
    try:
        return "~/" + str(resolved.relative_to(home))
    except ValueError:
        return raw


def gold_content_sha256(rows: Sequence[dict]) -> str:
    """Digest of the gold rows this run scored (id, label, prompt).

    The artifact names the file it read, but a file name does not pin its
    content: C1 owns the gold and can re-emit it with a different label split or
    different prompt text, and every number in this artifact would move with it.
    Hashing the rows actually loaded — not the bytes on disk — keeps the claim
    true for whatever the caller passed in, and needs no second read of a file
    that may not exist on a data-less clone.

    Rows are sorted by id and rendered with sorted keys, so the digest is a
    property of the set, not of load order.
    """
    payload = sorted(
        ((r["id"], r["label"], r["prompt"]) for r in rows),
        key=lambda entry: entry[0],
    )
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
    reference_path: str | Path = JEV_REFERENCE_PATH,
    gold_path: str | Path = GOLD_PATH,
    base_rate_results_path: str | Path = BASE_RATE_RESULTS_PATH,
) -> tuple[dict, list[dict]]:
    """Assemble the metrics document and per-row score dump.

    Reads only what its inputs name: the round store (for the jev states,
    which is why the run is deterministic once the cache holds those states),
    the warm ``cache`` — which ``rescore_rows`` fills in place and the caller
    persists — and the pinned jev reference scorer, whose bytes are hashed into
    ``inputs`` so the artifact names the reference it claims parity with (review
    F3); a tampered or missing copy raises ``JevReferenceError`` rather than
    emitting numbers under a false claim.

    ``inputs`` is the artifact's own provenance block, so every field in it is
    something a re-run can reproduce: the gold by a digest of the rows scored
    (``gold_sha256``), the round store by a home-relative display path
    (``input_display_path``). Neither the row contents nor a machine username
    leak into any metric, but both would otherwise make a committed artifact
    re-emit differently somewhere else.
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
    delta_arms = {
        "leak": ("prompt_only", "own_response"),
        "session": ("prompt_only", "parent_response"),
    }
    deltas = {
        name: paired_auc_delta(jev[ref], jev[other], n_boot, seed)
        for name, (ref, other) in delta_arms.items()
    }
    for name, (ref, other) in delta_arms.items():
        # The delta's own population is a subpopulation of the two headline
        # AUCs, so the block states both: auc_reference_paired (emitted by
        # paired_auc_delta) and the marginal points the reader would subtract.
        ref_auc = baselines_out[f"jev_{ref}"]["auc"]
        other_auc = baselines_out[f"jev_{other}"]["auc"]
        reference_point = ref_auc["point"] if ref_auc else None
        comparison_point = other_auc["point"] if other_auc else None
        deltas[name]["arms"] = {"reference": f"jev_{ref}", "comparison": f"jev_{other}"}
        deltas[name]["auc_marginal"] = {
            "reference": reference_point,
            "comparison": comparison_point,
            "difference": (
                None
                if reference_point is None or comparison_point is None
                else comparison_point - reference_point
            ),
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
            "gold": str(gold_path),
            "gold_sha256": gold_content_sha256(rows),
            "base_rate_results": str(base_rate_results_path),
            "round_store": input_display_path(round_store),
            "jev_reference": jev_reference_identity(reference_path),
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
            "questions_sha256": jev_questions_sha256(questions),
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
    persisted only after both artifacts pass the determinism gate, so a
    rejected run leaves every tracked file untouched (review F8).
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
        gold_path=args.gold,
        base_rate_results_path=args.base_rate_results,
    )
    # Artifacts first, cache second: a run the determinism gate rejects must
    # not have rewritten a tracked artifact already (review F8).
    paths = emit_artifacts(metrics, score_rows, args.metrics, args.scores)
    save_jev_cache(
        cache, args.cache, questions_sha256=metrics["jev"]["questions_sha256"]
    )
    print(
        f"wrote {paths['metrics']} and {paths['scores']} "
        f"(cache {jev_cache_entry_count(cache)} entries, "
        f"base rate {corpus_base_rate:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
