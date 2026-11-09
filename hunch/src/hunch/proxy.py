"""The hunch proxy: answers typed questions from EmbeddingGemma 2 once it has learned to agree with a slower teacher
model (an LLM judge or a dedicated decision model), forwards the rest, and learns from what it forwards.

A request carries one state (any JSON, or a tool call and its output) and named questions; each question has a type:
`noul` (yes/no, answered as P(yes)), `choice` (one of named options) or `score` (one of ordered levels).

    proxy = Proxy(Model(), backend=HTTPBackend("http://teacher/v1/answer"), db="proxy.db")
    proxy.answer({"state": {"call": "...", "output": "..."},
                  "questions": {"failed": {"type": "noul", "instructions": "Did the tool call fail?"}}})
    proxy.train(question, [{"state": ..., "answer": True, "kind": "labelled"}])

    python -m hunch serve --db proxy.db --backend http://teacher/v1/answer   # HTTP, see `serve`

How it decides (docs/question-design.md says which questions it can take over):
  - Questions are identified by their content (type, instructions, criteria): a *version*. Rewording one starts a new
    version, with no sampled answers and no graduation. A question may also name its `family` (e.g. "injection"):
    every version in a family shares the family's labelled examples, since ground truth doesn't depend on wording.
    Versions in a family must have the same type and options.
  - Each question has a student: logistic regression on the layer-`layer` mean of EG2 (an early exit), trained on its
    examples, one per state:
      labelled  ground truth, given through `train`. Kept per family, so a reworded question starts with all of
                them, up to the newest `labelled_cap` of each class; weighted double, and the target wherever
                present.
      sampled   the teacher's answers: every forwarded request, plus anything `train` gives as sampled. A rotating
                window per version (`sampled_cap`, oldest dropped first): an answer to one wording is not an answer
                to another.
  - Graduation, at each refit, by 5-fold CV on the examples: symmetric (on the CV answers confident enough to be
    answered locally, the 95% lower bounds of agreement >= `target` and of recall on the answers other than the most
    common one >= `minority`) or, for a yes/no question, one-sided (only a confident "no" is answered locally:
    recall of forwarded "yes" >= 0.85 and local miss rate <= 2%, both bounds at 95%); whichever covers more.
  - A state the store already has an answer for (a label, or the teacher's last answer) is answered from the store,
    without encoding it.
  - Otherwise a request is answered locally only if every question in it has graduated, is confident, isn't demoted,
    and the state is not novel (its most similar example is at least as similar as the 5th percentile of
    example-to-example nearest-neighbour similarities). Otherwise the whole request goes to the teacher (which reads
    the state once for all questions, so a partial forward saves little), and every answer is stored as sampled.
    When a question in the request hasn't graduated or is demoted, the request goes to the teacher whatever its
    features say, so the teacher is asked while the state is encoded, not after.
  - Audits: a share `audit` of local answers is also sent to the teacher, in the background, and every forwarded
    request audits the students that would have answered it. Over its last 100 audits, a symmetric question is
    demoted while its agreement is significantly below `target` (95% upper bound), a one-sided one while its miss
    rate is significantly above 2% (95% lower bound); it answers locally again when the audits recover. The audit
    window is kept in the store, so a demotion outlasts a restart; a refit that changes the mode empties it.
  - Each version keeps the teacher's answer for every labelled state, so the teacher can be measured against ground
    truth (`question_status` counts them as "labels_with_teacher"). An answer the proxy already has is kept when a
    label arrives, and a label's state later answered by the teacher is kept as it arrives. With `ask_for_labels`,
    the teacher is asked, in the background, about labelled states that `train` gives and that it hasn't answered:
    one request per state for every version of the family the proxy knows then.
  - Teacher tuning: at each refit, a version with at least 50 such pairs fits a recalibration of the teacher
    (logistic regression on its log-probabilities, which moves its cut points to where the labels put them). It is
    switched on only if 5-fold CV shows it significantly improves the teacher's balanced accuracy (each class
    weighted equally) without significantly lowering its accuracy (sign tests at 95%). While on, every answer that
    comes from the teacher, forwarded or replayed from the store, is recalibrated, with the teacher's own answer
    under "untuned". The report under "tuning" in a question's status says whether it is on and why.
  - Graduation tiers, while tuning is on: each refit fits two students. The full one learns the tuned teacher's
    answers (labels stay as they are); the partial one learns the teacher's own. Both are graded by CV against the
    tuned answers. The full student answers once it graduates; until then the partial one answers where it
    graduates, which keeps it to states where the teacher's own answer and the tuned one agree. Audits compare
    against the tuned answer. A local answer, and a question's status, say which tier answered ("tier").
  - Refits run on a worker thread once a version has `refit_every` new examples, a count checked whenever the
    teacher's answers are recorded (forwarded, audited); requests keep being answered by the previous fit meanwhile.
    `train` refits the versions it touches before returning, or with `refit=False` leaves them to that check. One
    version's refits run one at a time. Fits are kept in the store, so a restarted proxy answers at once. Examples,
    states' features, fits and audits live in a Store (store.py): SQLite, or Redis to share one between processes.
    Features are tied to the encoder that computed them (checkpoint revision, layer, precision, clipping);
    re-encoding them (`rebuild`) drops the kept fits, and every version refits.
  - Encoding runs on one thread that batches whatever requests are waiting, so concurrent requests share passes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import queue
import random
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .store import LABELLED, SAMPLED, Examples, FingerprintMismatch, Row, Store, open_store

QTYPES = ("noul", "choice", "score")
EMBEDDING_MODEL = "google/embeddinggemma-2"  # the model name /v1/embeddings answers to
CLIP_CHARS = 3000  # the encoder reads at most 1,024 tokens: longer states keep their head and tail (`encoder_text`)


# ---------------------------------------------------------------- questions, states, answers


def question_spec(q: dict) -> dict:
    """The parts of a question that define it."""
    if q.get("type") not in QTYPES:
        raise ValueError(f"question type must be one of {QTYPES}, got {q.get('type')!r}")
    if q["type"] != "noul" and not q.get("criteria"):
        raise ValueError(f"a {q['type']} question needs criteria")
    return {"type": q["type"], "instructions": q.get("instructions", ""), "criteria": q.get("criteria")}


def question_key(q: dict) -> str:
    """A question version: a hash of its type, instructions and criteria (not its family)."""
    return hashlib.sha1(json.dumps(question_spec(q), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def family_key(q: dict) -> str:
    """Where a question's labelled examples live: its named family, else its own version."""
    return f"family:{q['family']}" if q.get("family") else question_key(q)


def family_signature(q: dict) -> dict:
    """What every version in a family must share for its labels to mean the same thing."""
    return {"type": q["type"], "options": options(q)}


def options(q: dict) -> list[str]:
    """Answer classes: noul -> false/true; choice -> criteria keys (in order); score -> level indices."""
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "choice":
        return list(q["criteria"])
    return [str(i) for i in range(len(q["criteria"]))]


def state_text(state: Any) -> str:
    """What the encoder and the teacher read: a string as is; a tool call and its output as "call\noutput", followed
    by any other fields as "name: value" lines (sorted by name); anything else as JSON."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        call = state.get("call", state.get("tool_call"))
        if call is not None and "output" in state:
            text = f"{call}\n{state['output']}"
            for k in sorted(set(state) - {"call", "tool_call", "output"}):
                v = state[k]
                text += f"\n{k}: {v if isinstance(v, str) else json.dumps(v, sort_keys=True, ensure_ascii=False)}"
            return text
    return json.dumps(state, sort_keys=True, ensure_ascii=False)


def encoder_text(text: str) -> str:
    """The text the encoder reads for a state: its text, or for one over CLIP_CHARS its first two thirds and last
    third of that budget, so the end of a long output (where an error usually is) stays within the encoder's
    1,024 tokens. Hashes and the teacher use the full text."""
    if len(text) <= CLIP_CHARS:
        return text
    head = CLIP_CHARS * 2 // 3
    return text[:head] + "\n…\n" + text[-(CLIP_CHARS - head):]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def state_hash(state: Any) -> str:
    """A state's identity, for cached answers and stored features: a hash of the text the encoder (and the upstream
    model) reads."""
    return text_hash(state_text(state))


def _probability(x: Any) -> float:
    """A probability from an answer: a finite number in [0, 1], else ValueError."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"a probability must be a number, got {x!r}")
    p = float(x)
    if not 0.0 <= p <= 1.0:  # also rejects NaN
        raise ValueError(f"a probability must be in [0, 1], got {x!r}")
    return p


def parse_answer(q: dict, answer: Any) -> tuple[int, list[float] | None]:
    """(class index, probabilities or None) from an answer object, a probability list, or a plain label. For a noul,
    a bare float is P(yes) (a bool, or the integer 0 or 1, is a label), and an answer object's "noul" is read
    before its "probabilities". Probabilities must be finite and in [0, 1], one per option; anything else raises
    ValueError."""
    opts = options(q)
    if isinstance(answer, dict):
        if "noul" in answer and q["type"] == "noul":  # taken over any "probabilities", whose keys vary by teacher
            p1 = _probability(answer["noul"])
            return int(p1 >= 0.5), [1 - p1, p1]
        if "probabilities" in answer:
            probs = answer["probabilities"]
            if not isinstance(probs, dict) or set(probs) != set(opts):
                raise ValueError(f"probabilities must name exactly the options {opts}, got {probs!r}")
            p = [_probability(probs[o]) for o in opts]
            return int(np.argmax(p)), p
        raise ValueError(f"unrecognized answer object: {answer}")
    if isinstance(answer, (list, tuple)):
        p = [_probability(x) for x in answer]
        if q["type"] == "noul" and len(p) == 1:
            p = [1 - p[0], p[0]]
        if len(p) != len(opts):
            raise ValueError(f"{len(p)} probabilities for {len(opts)} options")
        return int(np.argmax(p)), p
    if q["type"] == "noul":
        if isinstance(answer, str):
            if answer.lower() not in ("yes", "no", "true", "false", "1", "0"):
                raise ValueError(f"noul label must be yes/no or true/false, got {answer!r}")
            return int(answer.lower() in ("yes", "true", "1")), None
        if isinstance(answer, bool) or (isinstance(answer, int) and answer in (0, 1)):
            return int(answer), None
        if isinstance(answer, float):
            p1 = _probability(answer)
            return int(p1 >= 0.5), [1 - p1, p1]
        raise ValueError(f"noul label must be a boolean, 0/1, P(yes) or yes/no, got {answer!r}")
    if isinstance(answer, str):
        if answer in opts:
            return opts.index(answer), None
        if q["type"] == "score" and answer in q["criteria"]:
            return list(q["criteria"]).index(answer), None
        raise ValueError(f"label {answer!r} is not one of {opts}")
    if isinstance(answer, int) and 0 <= answer < len(opts):
        return answer, None
    raise ValueError(f"unrecognized label {answer!r} for a {q['type']} question")


def answer_object(q: dict, p: np.ndarray) -> dict:
    """An answer object, as the teacher returns them, from class probabilities."""
    opts = options(q)
    if q["type"] == "noul":
        return {"type": "noul", "noul": round(float(p[1]), 4)}
    best = int(np.argmax(p))
    probs = {o: round(float(x), 4) for o, x in zip(opts, p)}
    if q["type"] == "choice":
        return {"type": "choice", "choice": opts[best], "confidence": round(float(p[best]), 4), "probabilities": probs}
    return {"type": "score", "score": round(float(np.dot(np.arange(len(p)), p)), 4), "confidence": round(float(p[best]), 4),
            "legend": dict(zip(opts, q["criteria"])), "probabilities": probs}


def wilson(k: int, m: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval (low, high) for k successes in m trials."""
    if m == 0:
        return 0.0, 1.0
    p = k / m
    c = (p + z * z / (2 * m)) / (1 + z * z / m)
    h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / (1 + z * z / m)
    return c - h, c + h


# ---------------------------------------------------------------- students


@dataclass
class Fit:
    """A student's fitted state, plain arrays only. A refit builds a new one and the student swaps it in whole."""

    coef: np.ndarray | None = None  # [classes seen (1 for two), d] logistic-regression weights on standardized features
    intercept: np.ndarray | None = None
    classes: np.ndarray | None = None  # the class indices the weights cover
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    mode: str | None = None  # "symmetric" | "one_sided" | None (not graduated)
    thr: float = 0.0
    report: dict = field(default_factory=dict)
    ref: np.ndarray | None = None  # normalized example features (float32), for the novelty check
    nn_floor: float = -1.0
    tune_coef: np.ndarray | None = None  # the teacher's recalibration, on its log-probabilities; None = off
    tune_intercept: np.ndarray | None = None
    tune_classes: np.ndarray | None = None
    partial: Fit | None = None  # with tuning on: the student trained on the teacher's own answers

    _ARRAYS = ("coef", "intercept", "classes", "mu", "sd", "ref", "tune_coef", "tune_intercept", "tune_classes")

    def _meta(self) -> dict:
        return {"mode": self.mode, "thr": self.thr, "report": self.report, "nn_floor": self.nn_floor}

    def to_bytes(self) -> bytes:
        """An .npz of the arrays plus the scalars as JSON (no pickles); the partial fit's under "partial." names."""
        meta = self._meta() | ({"partial": self.partial._meta()} if self.partial is not None else {})
        arrays = {k: getattr(self, k) for k in self._ARRAYS if getattr(self, k) is not None}
        if self.partial is not None:
            arrays |= {f"partial.{k}": getattr(self.partial, k) for k in self._ARRAYS if getattr(self.partial, k) is not None}
        buf = io.BytesIO()
        np.savez(buf, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, blob: bytes) -> Fit:
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            meta = json.loads(z["meta"].tobytes())
            part = meta.pop("partial", None)
            f = cls(**{k: z[k] for k in cls._ARRAYS if k in z.files}, **meta)
            if part is not None:
                f.partial = cls(**{k: z[f"partial.{k}"] for k in cls._ARRAYS if f"partial.{k}" in z.files}, **part)
            return f


def _lr(X, y, w):
    """Weighted logistic regression on standardized features as (coef, intercept, classes); None when `y` has a
    single class."""
    from sklearn.linear_model import LogisticRegression

    if len(set(y.tolist())) < 2:
        return None
    m = LogisticRegression(max_iter=3000, C=0.5).fit(X, y, sample_weight=w)
    return m.coef_, m.intercept_, m.classes_


def _proba(coef, intercept, classes, Z, k):
    """Probabilities over all k classes (zero for a class the classifier never saw): a sigmoid for two classes, a
    softmax for more, as scikit-learn's LogisticRegression computes them."""
    s = Z @ coef.T + intercept
    if coef.shape[0] == 1:
        p1 = 1 / (1 + np.exp(-s[:, 0]))
        q = np.stack([1 - p1, p1], 1)
    else:
        e = np.exp(s - s.max(1, keepdims=True))
        q = e / e.sum(1, keepdims=True)
    P = np.zeros((len(Z), k))
    P[:, classes] = q
    return P


def fit_student(spec: dict, ex: Examples, target: float, minority: float, min_examples: int,
                novelty_refs: int = 1000, y: np.ndarray | None = None, conf: np.ndarray | None = None,
                grade: np.ndarray | None = None) -> Fit:
    """Fit a question's classifier on its examples and decide its graduation by 5-fold CV. `y` and `conf` replace the
    examples' answers and confidences as targets (the tuned teacher's); `grade` gives the answers its CV
    predictions are graded against (default: the targets)."""
    from sklearn.model_selection import KFold

    k = len(options(spec))
    f = Fit(report={"n": len(ex), "labelled": ex.kinds.count(LABELLED), "sampled": ex.kinds.count(SAMPLED)})
    if len(ex) == 0:
        return f
    X = ex.X.astype(np.float64)
    f.mu, f.sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - f.mu) / f.sd
    y = ex.y if y is None else y
    w = np.where(np.array(ex.kinds) == LABELLED, 2.0, ex.conf if conf is None else conf)
    m = _lr(Z, y, w)
    if m is not None:
        f.coef, f.intercept, f.classes = m
    ref = ex.X[-novelty_refs:]
    f.ref = (ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    if len(f.ref) > 1:
        S = f.ref @ f.ref.T
        np.fill_diagonal(S, -1)
        f.nn_floor = float(np.quantile(S.max(1), 0.05))
    if m is None or len(y) < min_examples:
        f.report["why_not"] = "too few examples" if len(y) < min_examples else "only one answer seen"
        return f
    P = np.zeros((len(y), k))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Z):
        c = _lr(Z[tr], y[tr], w[tr])
        if c is not None:
            P[te] = _proba(*c, Z[te], k)
    g = y if grade is None else grade
    pred, cf = P.argmax(1), P.max(1)
    agree = pred == g
    maj = int(np.bincount(g, minlength=k).argmax())
    minor = g != maj
    f.report |= {"cv_agreement": float(agree.mean()), "cv_minority_recall": float(agree[minor].mean()) if minor.any() else None,
                 "majority_share": float(np.mean(~minor))}
    best = (0.0, None, 0.0)  # coverage, mode, threshold
    for t in np.arange(0.5, 0.995, 0.01):
        sel = cf >= t
        ms = sel & minor
        if (sel.sum() >= 30 and wilson(int(agree[sel].sum()), int(sel.sum()))[0] >= target
                and ms.sum() >= 15 and wilson(int(agree[ms].sum()), int(ms.sum()))[0] >= minority):
            best = (float(sel.mean()), "symmetric", float(t))
            break
    if spec["type"] == "noul" and int((g == 1).sum()) >= 15:
        p_yes = P[:, 1]
        for tau in np.unique(np.round(p_yes, 4))[::-1]:
            local = p_yes < tau
            if (local.sum() >= 30 and wilson(int((g[~local] == 1).sum()), int((g == 1).sum()))[0] >= 0.85
                    and wilson(int((g[local] == 1).sum()), int(local.sum()))[1] <= MISS_RATE):
                if local.mean() > best[0]:
                    best = (float(local.mean()), "one_sided", float(tau))
                break
    cov, f.mode, f.thr = best
    f.report |= {"mode": f.mode, "threshold": f.thr, "cv_coverage": cov}
    if f.mode is None:
        f.report["why_not"] = "CV agreement / minority recall below the graduation bars"
    return f


def _log(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(P, 1e-6, 1.0))


def fit_tuning(spec: dict, pairs: list[tuple[str, int, list[float]]], min_pairs: int = 50) -> tuple[tuple | None, dict]:
    """A recalibration of the teacher from (state, label, teacher's probabilities) pairs: logistic regression on the
    log-probabilities, classes weighted equally, which moves the teacher's cut points to where the labels put them.
    Returns ((coef, intercept, classes) or None, report). Two sign tests at 95% on the answers 5-fold CV changes
    switch it on: it must fix significantly more than it breaks with each answer weighted by the inverse of its
    class's frequency (balanced accuracy, so it can't win by answering the common class at the rare classes'
    expense), and must not break significantly more than it fixes unweighted (accuracy, so a class seen only a few
    times can't carry it)."""
    from sklearn.model_selection import KFold

    k = len(options(spec))
    rep = {"pairs": len(pairs), "on": False}
    if len(pairs) < min_pairs:
        rep["why_not"] = f"fewer than {min_pairs} labels with a teacher answer"
        return None, rep
    y = np.array([lab for _, lab, _ in pairs])
    Z = _log(np.array([p for _, _, p in pairs], dtype=float))
    raw = Z.argmax(1)
    counts = np.bincount(y, minlength=k)
    w = len(y) / (np.count_nonzero(counts) * counts[y])  # 1 on average; a rare class's answers weigh more
    tuned = np.zeros((len(y), k))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Z):
        c = _lr(Z[tr], y[tr], w[tr])
        tuned[te] = _proba(*c, Z[te], k) if c is not None else np.eye(k)[raw[te]]
    pred = tuned.argmax(1)
    fixed, broken = (pred == y) & (raw != y), (pred != y) & (raw == y)

    def balanced(right):
        return float(np.mean([right[y == c].mean() for c in np.flatnonzero(counts)]))

    rep |= {"teacher_accuracy": float(np.mean(raw == y)), "cv_accuracy": float(np.mean(pred == y)),
            "teacher_balanced_accuracy": balanced(raw == y), "cv_balanced_accuracy": balanced(pred == y),
            "fixed": int(fixed.sum()), "broken": int(broken.sum())}
    if w[fixed].sum() - w[broken].sum() <= 1.96 * math.sqrt((w[fixed | broken] ** 2).sum()):
        rep["why_not"] = "recalibrating doesn't significantly improve the teacher's balanced accuracy in CV"
        return None, rep
    if broken.sum() - fixed.sum() > 1.96 * math.sqrt((fixed | broken).sum()):
        rep["why_not"] = "recalibrating significantly lowers the teacher's accuracy in CV"
        return None, rep
    m = _lr(Z, y, w)
    rep["on"] = m is not None
    return m, rep


def apply_tuning(fit: Fit, P: np.ndarray, k: int) -> np.ndarray:
    """The teacher's probabilities [n, k] recalibrated by a fit's tuning."""
    return _proba(fit.tune_coef, fit.tune_intercept, fit.tune_classes, _log(P), k)


def tuned_targets(fit: Fit, ex: Examples, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(classes, confidences) of the examples as the tuned teacher answers them; labels stay as they are."""
    y, conf = ex.y.copy(), ex.conf.copy()
    rows = [i for i, kind in enumerate(ex.kinds) if kind == SAMPLED]
    if rows:
        P = np.array([ex.probs[i] if ex.probs[i] is not None else np.eye(k)[ex.y[i]] for i in rows], dtype=float)
        T = apply_tuning(fit, P, k)
        y[rows], conf[rows] = T.argmax(1), T.max(1)
    return y, conf



def find_blind_spots(spec: dict, ex: Examples, pairs: list[tuple[str, int, list[float]]], fit: Fit,
                     max_spots: int = 8, examples: int = 3, suggest: int = 5) -> dict:
    """Where the teacher (tuned, when tuning is on) disagrees with the labels, from the labelled states that have a
    teacher answer. Returns a summary (accuracy raw and tuned, a confusion matrix and per-label recall) and spots:
    the errors grouped by (label, teacher's answer) and clustered by cosine k-means on the examples' features,
    largest and most confidently wrong first (a score question's summary includes its legend). Each spot has the
    share of its states where an out-of-fold student agrees with the label (high: a likely blind spot; low: possibly
    label noise), the states nearest its centre, and the unlabelled states the teacher answered the same way within
    the spot's radius: how many there are, and the nearest few, which are what to label next. States are by hash;
    texts are added by the caller."""
    from sklearn.cluster import KMeans
    from sklearn.model_selection import KFold

    opts, k = options(spec), len(options(spec))
    row = {s: i for i, s in enumerate(ex.states)}
    pairs = [p for p in pairs if p[0] in row]
    if not pairs:
        return {"summary": {"labels_with_teacher": 0}, "spots": []}
    y = np.array([lab for _, lab, _ in pairs])
    P = np.array([p for _, _, p in pairs], dtype=float)
    tuned = fit.tune_coef is not None
    Q = apply_tuning(fit, P, k) if tuned else P  # the answers callers get
    served = Q.argmax(1)
    conf = np.zeros((k, k), dtype=int)
    np.add.at(conf, (y, served), 1)
    summary = {"labels_with_teacher": len(y), "teacher_accuracy": float(np.mean(P.argmax(1) == y)),
               "tuned": tuned, "served_accuracy": float(np.mean(served == y)),
               "confusion": {opts[a]: {opts[b]: int(conf[a, b]) for b in range(k) if conf[a, b]} for a in range(k)},
               "recall": {opts[a]: round(float(conf[a, a] / conf[a].sum()), 4) for a in range(k) if conf[a].sum()}}
    if spec["type"] == "score":
        summary["legend"] = dict(zip(opts, spec["criteria"]))

    # an out-of-fold student, trained as the full student is, says whether each label is learnable from the features
    X = ex.X.astype(np.float64)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    t, c = tuned_targets(fit, ex, k) if tuned else (ex.y, ex.conf)
    w = np.where(np.array(ex.kinds) == LABELLED, 2.0, c)
    S = np.full((len(t), k), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Z) if len(t) >= 10 else []:
        m = _lr(Z[tr], t[tr], w[tr])
        if m is not None:
            S[te] = _proba(*m, Z[te], k)

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    idx = np.array([row[s] for s, _, _ in pairs])
    unl = np.array([i for i, kind in enumerate(ex.kinds) if kind == SAMPLED], dtype=int)
    if len(unl):
        U = np.array([ex.probs[i] if ex.probs[i] is not None else np.eye(k)[ex.y[i]] for i in unl], dtype=float)
        unl_answer = (apply_tuning(fit, U, k) if tuned else U).argmax(1)
    spots = []
    wrong = np.flatnonzero(served != y)
    for a, b in sorted({(int(y[i]), int(served[i])) for i in wrong}):
        group = wrong[(y[wrong] == a) & (served[wrong] == b)]
        n_c = max(1, int(round(math.sqrt(len(group) / 2))))
        labels = (KMeans(n_c, n_init=4, random_state=0).fit_predict(Xn[idx[group]]) if n_c > 1
                  else np.zeros(len(group), int))
        for c_id in range(n_c):
            members = group[labels == c_id]
            V = Xn[idx[members]]
            centre = V.mean(0)
            centre /= np.linalg.norm(centre) + 1e-12
            sims = V @ centre
            radius = float(sims.min())
            p_label = Q[members, a]
            oof = S[idx[members]]
            known = ~np.isnan(oof).any(1)
            spot = {"label": opts[a], "teacher": opts[b], "n": len(members),
                    "teacher_p_label": [round(float(x), 4) for x in np.quantile(p_label, [0, 0.5, 1])],
                    "student_agrees_with_label": (round(float(np.mean(oof[known].argmax(1) == a)), 4)
                                                  if known.any() else None),
                    "states": [{"state": pairs[members[j]][0], "teacher_p_label": round(float(p_label[j]), 4)}
                               for j in np.argsort(-sims)[:examples]],
                    "_score": len(members) * float(np.mean(1 - p_label))}
            if len(unl):
                near = unl[unl_answer == b]
                ns = Xn[near] @ centre
                inside = np.flatnonzero(ns >= radius)
                spot["unlabelled_nearby"] = len(inside)
                spot["label_next"] = [{"state": ex.states[near[j]], "similarity": round(float(ns[j]), 4)}
                                      for j in inside[np.argsort(-ns[inside])][:suggest]]
            spots.append(spot)
    spots.sort(key=lambda d: -d["_score"])
    for d in spots:
        del d["_score"]
    return {"summary": summary, "spots": spots[:max_spots]}

MISS_RATE = 0.02  # a one-sided question's bar: the share of local "no" answers the teacher calls "yes"
AUDIT_WINDOW, AUDIT_MIN = 100, 20


class Student:
    """One question version: its current Fit, its recent audits and whether they demote it. `predict` reads the fit
    without locking; `install` swaps in a new one. With a `store`, the audit window is kept there too, so a demotion
    outlasts a restart."""

    def __init__(self, key: str, spec: dict, family: str | None = None, target: float = 0.90,
                 store: Store | None = None):
        self.key, self.spec, self.family, self.target, self.store = key, spec, family or key, target, store
        self.k = len(options(spec))
        self.fit = Fit()
        self.lock = threading.Lock()  # audits and demotion
        self.refit_lock = threading.Lock()  # one refit at a time, so the last to finish read the newest examples
        # (agreed, missed a "yes", the mode the student answered in)
        self.audits: deque[tuple[bool, bool, str]] = deque(
            (tuple(a) for a in (store.audits(key) if store is not None else [])), maxlen=AUDIT_WINDOW)
        self.demoted = False
        self.since_refit = 0

    def _serving(self) -> tuple[Fit | None, str | None]:
        """The fit that answers and its tier: "full" when the student trained on the answers callers get (the tuned
        teacher's, when tuning is on) has graduated; else "partial" when the student trained on the teacher's own
        answers has graduated against the tuned ones; else none."""
        f = self.fit
        if f.mode is not None:
            return f, "full"
        if f.partial is not None and f.partial.mode is not None:
            return f.partial, "partial"
        return None, None

    @property
    def mode(self) -> str | None:
        fit, _ = self._serving()
        return fit.mode if fit is not None else None

    @property
    def tier(self) -> str | None:
        return self._serving()[1]

    @property
    def report(self) -> dict:
        return self.fit.report

    def install(self, fit: Fit) -> None:
        """Swap in a new fit. The audit window carries over while the mode stays the same: it is recent evidence, and
        audits keep arriving while a question is demoted, so a demotion lifts once the new fit's audits recover. A fit
        that answers in another mode (symmetric <-> one-sided) starts an empty window, since the old audits measured
        something else."""
        with self.lock:
            self.fit = fit
            mode = self.mode
            if mode is not None and any(m != mode for _, _, m in self.audits):
                self.audits.clear()
                if self.store is not None:
                    self.store.clear_audits(self.key)
            self.demoted = self._failing()

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, bool, bool, str | None]:
        """(class probabilities, eligible, local, tier): eligible when a fit has graduated (`_serving`), is confident
        on x and x isn't novel; local when it is also not demoted."""
        f, tier = self._serving()
        if f is None:
            f = self.fit
            if f.coef is None:
                return np.full(self.k, 1 / self.k), False, False, None
            return _proba(f.coef, f.intercept, f.classes, ((x - f.mu) / f.sd)[None], self.k)[0], False, False, None
        p = _proba(f.coef, f.intercept, f.classes, ((x - f.mu) / f.sd)[None], self.k)[0]
        xn = (x / (np.linalg.norm(x) + 1e-12)).astype(np.float32)
        if f.ref is not None and float((f.ref @ xn).max()) < f.nn_floor:
            return p, False, False, tier
        eligible = bool(p[1] < f.thr) if f.mode == "one_sided" else bool(p.max() >= f.thr)
        return p, eligible, eligible and not self.demoted, tier

    def tune(self, p: np.ndarray) -> np.ndarray | None:
        """The teacher's probabilities recalibrated on the labels, or None when the version has no tuning on."""
        f = self.fit
        if f.tune_coef is None:
            return None
        return apply_tuning(f, np.asarray(p, dtype=float)[None], self.k)[0]

    def teacher_answer(self, answer: dict, lab: int, probs: list | None) -> dict:
        """A teacher's answer as the caller gets it: recalibrated when the version has tuning on, with the teacher's
        own answer under "untuned"."""
        t = self.tune(probs if probs is not None else np.eye(self.k)[lab])
        return dict(answer) if t is None else answer_object(self.spec, t) | {"untuned": dict(answer)}

    def audit(self, predicted: int, answer: int) -> None:
        """Record the teacher's answer (tuned, when tuning is on) to a state this student would have answered
        locally."""
        with self.lock:
            mode = self.mode
            if mode is None:
                return
            entry = (predicted == answer, mode == "one_sided" and answer == 1, mode)
            self.audits.append(entry)
            if self.store is not None:
                self.store.add_audit(self.key, list(entry), AUDIT_WINDOW)
            self.demoted = self._failing()

    def _window(self) -> list[tuple[bool, bool, str]]:
        """The audits taken in the mode the student answers in now."""
        mode = self.mode
        return [a for a in self.audits if a[2] == mode]

    def _failing(self) -> bool:
        a = self._window()
        mode = self.mode
        if mode is None or len(a) < AUDIT_MIN:
            return False
        if mode == "one_sided":
            return wilson(sum(m for _, m, _ in a), len(a))[0] > MISS_RATE
        return wilson(sum(g for g, _, _ in a), len(a))[1] < self.target

    def audit_summary(self) -> dict:
        a = self._window()
        return {"n": len(a), "agreement": sum(g for g, _, _ in a) / len(a) if a else None,
                "misses": sum(m for _, m, _ in a) if self.mode == "one_sided" else None}


# ---------------------------------------------------------------- backends


class ModelNotFound(LookupError):
    """An embeddings request for a model the proxy doesn't serve."""


class BackendError(RuntimeError):
    """The teacher failed: unreachable, timed out, or answered without every question."""


class HTTPBackend:
    """A teacher behind HTTP: POSTs the request as JSON, expects {"answers": {question id: answer object}}."""

    def __init__(self, url: str, headers: dict | None = None, timeout: float = 30):
        self.url, self.headers, self.timeout = url, headers or {}, timeout

    def __call__(self, request: dict) -> dict:
        import urllib.request

        req = urllib.request.Request(self.url, json.dumps(request).encode(), {"Content-Type": "application/json", **self.headers})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)


class AnswerCache:
    """Upstream answers keyed by hashes, so they can be shipped and replayed without the states they answer: one
    JSONL line per answer, {"question": question_key, "state": state_hash, "answer": <answer object>, "model": ...}.
    Reads every *.jsonl under the given paths (files or folders); `put` appends to `write_to`."""

    def __init__(self, *paths: str | Path, write_to: str | Path | None = None):
        self.answers: dict[tuple[str, str], dict] = {}
        self.write_to = Path(write_to) if write_to else None
        self.lock = threading.Lock()
        for p in map(Path, paths):
            for f in sorted(p.rglob("*.jsonl")) if p.is_dir() else [p] if p.exists() else []:
                for line in open(f):
                    if line.strip():
                        r = json.loads(line)
                        self.answers[(r["question"], r["state"])] = r["answer"]

    def get(self, question: dict, state: Any) -> dict | None:
        return self.answers.get((question_key(question), state_hash(state)))

    def put(self, question: dict, state: Any, answer: dict, model: str = "") -> None:
        key = (question_key(question), state_hash(state))
        with self.lock:
            self.answers[key] = answer
            if self.write_to:
                self.write_to.parent.mkdir(parents=True, exist_ok=True)
                with open(self.write_to, "a") as f:
                    f.write(json.dumps({"question": key[0], "state": key[1], "answer": answer, "model": model}) + "\n")

    def __len__(self) -> int:
        return len(self.answers)


class OfflineBackend:
    """Answers requests from an AnswerCache. A miss goes to `fallback` (and is cached) if there is one;
    otherwise it raises KeyError."""

    def __init__(self, cache: AnswerCache, fallback: Callable[[dict], dict] | None = None, model: str = "offline"):
        self.cache, self.fallback, self.model = cache, fallback, model
        self.hits = self.misses = 0

    def __call__(self, request: dict) -> dict:
        answers, missing = {}, {}
        for qid, q in request["questions"].items():
            a = self.cache.get(q, request["state"])
            if a is None:
                missing[qid] = q
            else:
                answers[qid] = a
        self.hits += len(answers)
        self.misses += len(missing)
        if missing:
            if self.fallback is None:
                raise KeyError(f"no cached answer for {sorted(missing)} on state {state_hash(request['state'])}")
            got = self.fallback(dict(request, questions=missing))
            for qid, q in missing.items():
                answers[qid] = got["answers"][qid]
                self.cache.put(q, request["state"], got["answers"][qid], got.get("model", ""))
        return {"model": self.model, "answers": answers}


# ---------------------------------------------------------------- the encoder


def encoder_fingerprint(model, layer: int) -> dict:
    """What the stored features depend on: the checkpoint revision, the layer read, the matmul precision (with
    eg2cpu's 8-bit weights, if on) and the clipping of long states."""
    bf16 = os.environ.get("EG2_BF16", "off").strip().lower()
    precision = "fp32" if bf16 in ("", "off", "0", "none") else f"bf16:{bf16}"
    weights = os.environ.get("EG2_WEIGHTS", "")
    if weights in ("int8", "fp8"):  # the values eg2cpu acts on
        precision += f"+{weights}"
    return {"revision": getattr(model, "revision", type(model).__name__), "layer": layer, "precision": precision,
            "clip": CLIP_CHARS}


class Encoder:
    """Layer-`layer` features for state texts: a cache of `cache_size` states, and one thread that encodes whatever
    texts are waiting as one batch (up to `max_batch`), so concurrent requests share passes and the encoder never
    runs several passes against each other. A text already being encoded is waited for, not encoded twice. Final
    embeddings (`embed`) go through the same thread."""

    def __init__(self, model, layer: int, max_batch: int = 64, cache_size: int = 4096):
        self.model, self.layer, self.max_batch, self.cache_size = model, layer, max_batch, cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._inflight: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None

    def __call__(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        """(state hashes, float32 features [len(texts), d])."""
        return self.finish(*self.start(texts))

    def start(self, texts: list[str]) -> tuple[list[str], dict[str, Future]]:
        """Queue the texts not already cached or in flight, without waiting; `finish` waits for them. Lets a caller
        do something else (ask the teacher) while they are encoded. A cached text's vector is taken now, as a finished
        future, so evicting it before `finish` doesn't lose it."""
        hashes = [text_hash(t) for t in texts]
        waits: dict[str, Future] = {}
        with self._lock:
            for h, t in zip(hashes, texts):
                if h in waits:
                    continue
                v = self._cache.get(h)
                if v is not None:
                    self._cache.move_to_end(h)
                    waits[h] = f = Future()
                    f.set_result(v)
                    continue
                f = self._inflight.get(h)
                if f is None:
                    f = self._inflight[h] = Future()
                    self._queue.put((h, t, f))
                waits[h] = f
            if waits:
                self._start_thread()
        return hashes, waits

    def finish(self, hashes: list[str], waits: dict[str, Future]) -> tuple[list[str], np.ndarray]:
        """(state hashes, float32 features) for what `start` queued."""
        got = {h: f.result() for h, f in waits.items()}
        return hashes, np.stack([got[h] for h in hashes])

    def embed(self, texts: list[str]) -> np.ndarray:
        """The final embeddings of the texts as given (not clipped: the model truncates at 1,024 tokens), float32
        [len(texts), d] with unit norm. Not cached."""
        futures = [Future() for _ in texts]
        with self._lock:
            for t, f in zip(texts, futures):
                self._queue.put((None, t, f))
            if futures:
                self._start_thread()
        return np.stack([f.result() for f in futures])

    def _start_thread(self) -> None:  # with the lock held
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="hunch-encode", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            batch = [self._queue.get()]
            while len(batch) < self.max_batch:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if feats := [b for b in batch if b[0] is not None]:
                self._features(feats)
            if full := [(t, f) for h, t, f in batch if h is None]:
                try:
                    vals = [np.asarray(e, dtype=np.float32) for e, _ in self.model.embed_batch([t for t, _ in full])]
                except Exception as e:
                    for _, f in full:
                        f.set_exception(e)
                    continue
                for (_, f), v in zip(full, vals):
                    f.set_result(v)

    def _features(self, batch: list[tuple[str, str, Future]]) -> None:
        try:
            parts = self.model.embed_batch_partial([encoder_text(t) for _, t, _ in batch], self.layer)
            vals = [np.asarray(p.layer_means[self.layer], dtype=np.float32) for p in parts]
        except Exception as e:  # the waiting requests fail; the encoder keeps running
            with self._lock:
                for h, _, f in batch:
                    self._inflight.pop(h, None)
                    f.set_exception(e)
            return
        with self._lock:
            for (h, _, _), v in zip(batch, vals):
                self._cache[h] = v
                self._inflight.pop(h, None)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        for (_, _, f), v in zip(batch, vals):
            f.set_result(v)


# ---------------------------------------------------------------- the proxy


class Proxy:
    """Routes requests between the students and the teacher (`backend`) by the policy in the module docstring: answers
    locally when every question in a request may, forwards the rest, and learns from what it forwards. Examples,
    features and fits live in a Store."""

    def __init__(self, model=None, backend: Callable[[dict], dict] | None = None, db: str | Path | Store = "proxy.db", *,
                 layer: int = 8, audit: float = 0.10, target: float = 0.90, minority: float = 0.80,
                 sampled_cap: int = 2000, labelled_cap: int = 5000, refit_every: int = 100, min_examples: int = 60,
                 store_text: bool = False, rebuild: bool = False, background: bool = True, memo: bool = True,
                 max_batch: int = 64, audit_queue: int = 256, ask_for_labels: bool = True, seed: int = 0):
        """`db` is a SQLite path, a `redis://` URL or a Store. `backend` is the teacher; without one, questions that
        haven't graduated get no answer. With `background`, refits run on a worker thread and the request that
        triggers one doesn't wait for it (`wait` blocks until none are pending). `memo` answers a state the store
        already has an answer for from the store. `store_text` keeps each state's text, which lets `rebuild`
        re-encode the stored states when the encoder fingerprint changes (dropping the kept fits, so every version
        refits); without it a changed encoder needs a new store. `audit_queue` bounds the background audits waiting
        for the teacher; beyond it, audits are skipped. `ask_for_labels` asks the teacher about labelled states it
        hasn't answered (one call per state)."""
        if model is None:
            from eg2cpu import Model

            model = Model()
        self.model, self.backend, self.layer = model, backend, layer
        self.audit_rate, self.target, self.minority = audit, target, minority
        self.refit_every, self.min_examples, self.memo = refit_every, min_examples, memo
        self.ask_for_labels = ask_for_labels
        self.store = open_store(db, sampled_cap=sampled_cap, labelled_cap=labelled_cap, store_text=store_text)
        self.fingerprint = encoder_fingerprint(model, layer)
        self.encoder = Encoder(model, layer, max_batch=max_batch)
        try:
            self.store.check_fingerprint(self.fingerprint)
        except FingerprintMismatch:
            if not rebuild:
                raise
            self._reencode()  # drops the kept fits: every version refits below, on the new features
        self.rng = random.Random(seed)
        self.lock = threading.RLock()  # the students table and the stats
        self.students: dict[str, Student] = {}
        self.stats = {"requests": 0, "local": 0, "cached": 0, "forwarded": 0, "audited": 0, "audits_skipped": 0,
                      "label_asks": 0, "backend_errors": 0, "embedded": 0}
        self._latency: deque[tuple[str, float]] = deque(maxlen=2000)
        self.background = background
        self._pending: dict[str, None] = {}  # versions waiting for a refit, in order
        self._restoring: set[str] = set()  # versions with no kept fit, refitting after a restart
        self._busy = False
        self._cv = threading.Condition()
        self._worker: threading.Thread | None = None
        self._audits: queue.Queue = queue.Queue(maxsize=audit_queue)
        self._auditor: threading.Thread | None = None
        self._label_asks: queue.Queue = queue.Queue()
        self._label_asker: threading.Thread | None = None
        for key, (spec, family) in self.store.questions().items():  # restore from the store
            st = self.students[key] = Student(key, spec, family, target, self.store)
            try:
                blob = self.store.get_fit(key)
                fit = Fit.from_bytes(blob) if blob is not None else None
            except Exception as e:  # an unreadable fit is refitted, like a missing one
                print(f"hunch: refitting {key}: its kept fit is unreadable ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
                fit = None
            if fit is not None:
                st.install(fit)
            else:
                self._restoring.add(key)
                self._schedule(key)

    # -------- public API

    def answer(self, request: dict) -> dict:
        """Answer a request: from the store if it has every answer, else locally if every question may, else from the
        teacher (and learn from it)."""
        t0 = time.perf_counter()
        out = self._answer(request)
        with self.lock:
            self._latency.append((out["answered_by"], (time.perf_counter() - t0) * 1000))
        return out

    def train(self, question: dict, cases: list[dict], refit: bool = True) -> dict:
        """Add training cases for one question. Each case: {"state": ..., "answer": <label, probabilities or an
        answer object>, "kind": "labelled" | "sampled"}. Labelled = ground truth, shared by every version in the
        question's family, the newest `labelled_cap` of each class kept; sampled = as if answered by the teacher,
        kept in a rotating window for this version only. A state given twice keeps its latest answer. Every answer
        is checked (`parse_answer`) before any is recorded. With `refit`, the affected versions are refitted before
        this returns (after any refit of theirs already running); without it, the new examples count toward
        `refit_every`, which is checked when the teacher's answers are next recorded."""
        st = self._student(question)
        texts, parsed, kinds = [], [], []
        for i, c in enumerate(cases):
            kind = c.get("kind", LABELLED)
            if kind not in (LABELLED, SAMPLED):
                raise ValueError(f"case {i}: kind must be 'labelled' or 'sampled'")
            texts.append(state_text(c["state"]))
            parsed.append(parse_answer(st.spec, c["answer"]))
            kinds.append(kind)
        if texts:
            hashes, F = self.features(texts)
            feats, by_hash = dict(zip(hashes, F)), dict(zip(hashes, texts))
            states = {h: c["state"] for h, c in zip(hashes, cases)}
            for kind in (LABELLED, SAMPLED):
                rows = list({h: Row(h, lab, p) for h, (lab, p), k in zip(hashes, parsed, kinds) if k == kind}.values())
                if rows:
                    self._record(st, kind, rows, feats, by_hash, states)
        if refit:
            with self.lock:
                todo = [s.key for s in self.students.values() if s.family == st.family] if LABELLED in kinds else [st.key]
            for key in todo:  # labels reach every version in the family
                with self._cv:
                    self._pending.pop(key, None)
                self._refit(key)
        return self.question_status(st.key)

    def predict(self, question: dict, states: list) -> list[dict]:
        """The student's answers to one question for many states, without routing, forwarding or learning. Each
        answer says whether the proxy would answer it locally ("local")."""
        st = self._student(question)
        _, F = self.features([state_text(s) for s in states])
        out = []
        for x in F:
            p, _, ok, tier = st.predict(x)
            out.append(answer_object(st.spec, p) | {"local": ok, "tier": tier})
        return out

    def features(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        """(state hashes, layer-`layer` means as float32), through the encoder's cache and batching thread."""
        return self.encoder(texts)

    def embeddings(self, request: dict) -> dict:
        """An embeddings request and response in the OpenAI shape that vLLM serves. The request: "input" (a string or
        a list of strings), "model" (optional; EMBEDDING_MODEL), "encoding_format" ("float", the default, or "base64":
        little-endian float32) and "dimensions" (optional; only the model's own, since the checkpoint declares no
        Matryoshka sizes). Other fields are ignored. Each input gets the model's final embedding (unit norm), not the
        layer features the students read; inputs over 1,024 tokens are truncated, and "usage" counts the tokens read,
        <bos> and <eos> included."""
        model = request.get("model")
        if model is not None and model != EMBEDDING_MODEL:
            raise ModelNotFound(f"The model `{model}` does not exist.")
        inp = request.get("input")
        texts = [inp] if isinstance(inp, str) else inp
        if not isinstance(texts, list) or not texts:
            raise ValueError("input must be a string or a non-empty list of strings")
        if not all(isinstance(t, str) for t in texts):
            raise ValueError("input must be text: token ids are not supported")
        fmt = request.get("encoding_format") or "float"
        if fmt not in ("float", "base64"):
            raise ValueError(f"encoding_format must be 'float' or 'base64', not {fmt!r}")
        dims = request.get("dimensions")
        if dims is not None and (isinstance(dims, bool) or not isinstance(dims, int)):
            raise ValueError("dimensions must be an integer")
        E = self.encoder.embed(texts)
        if dims is not None and dims != E.shape[1]:
            raise ValueError(f"{EMBEDDING_MODEL} has no Matryoshka sizes: dimensions must be {E.shape[1]} or left out")
        tokens = sum(len(self.model.tokenize(t)) for t in texts)
        with self.lock:
            self.stats["embedded"] += len(texts)
        data = [{"index": i, "object": "embedding",
                 "embedding": base64.b64encode(v.astype("<f4").tobytes()).decode() if fmt == "base64" else v.tolist()}
                for i, v in enumerate(E)]
        return {"id": f"embd-{uuid.uuid4().hex}", "object": "list", "created": int(time.time()),
                "model": EMBEDDING_MODEL, "data": data,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens, "completion_tokens": 0}}

    def blind_spots(self, question: dict | str, **kw) -> dict:
        """Where the teacher disagrees with the labels for one question version (a question or its key): see
        `find_blind_spots`. States come with their text where the store keeps texts (`store_text`)."""
        key = question if isinstance(question, str) else self._student(question).key
        st = self.students.get(key)
        if st is None:
            raise KeyError(f"no question {key}")
        out = find_blind_spots(st.spec, self.store.examples(key, st.family), self.store.teacher_pairs(key, st.family),
                               st.fit, **kw)
        refs = [d for spot in out["spots"] for d in spot["states"] + spot.get("label_next", [])]
        texts = self.store.texts(list({d["state"] for d in refs}))
        for d in refs:
            d["text"] = texts.get(d["state"])
        return out | {"key": key}

    def question_status(self, key: str) -> dict:
        st = self.students[key]
        family = st.family[len("family:"):] if st.family.startswith("family:") else None
        with self._cv:
            pending = key in self._pending
        return {"key": key, "family": family, "spec": st.spec, "examples": self.store.counts(key, st.family),
                "labels_with_teacher": len(self.store.teacher_pairs(key, st.family)),
                "graduated": st.mode is not None and not st.demoted, "tier": st.tier, "mode": st.mode,
                "demoted": st.demoted,
                "report": st.report, "refit_pending": pending, "audits": st.audit_summary()}

    def status(self) -> dict:
        with self.lock:
            keys, stats, lat = list(self.students), dict(self.stats), list(self._latency)
        latency = {}
        for by in sorted({b for b, _ in lat}):
            ms = np.array([m for b, m in lat if b == by])
            latency[by] = {"n": len(ms), "p50_ms": round(float(np.percentile(ms, 50)), 1),
                           "p95_ms": round(float(np.percentile(ms, 95)), 1)}
        return {"ready": self.ready(), "stats": stats, "latency": latency, "layer": self.layer,
                "encoder": self.fingerprint, "states": self.store.num_states(),
                "questions": [self.question_status(k) for k in keys]}

    def ready(self) -> bool:
        """Whether every question restored from the store has a fit (kept, or refitted after the restart)."""
        with self._cv:
            return not self._restoring

    def wait(self, timeout: float | None = None) -> bool:
        """Block until no refit is pending or running and every background audit and label ask is done; False on
        timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            if not self._cv.wait_for(lambda: not self._pending and not self._busy, timeout):
                return False
        while self._audits.unfinished_tasks or self._label_asks.unfinished_tasks:
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.005)
        with self._cv:
            return self._cv.wait_for(lambda: not self._pending and not self._busy,
                                     None if deadline is None else max(0.0, deadline - time.monotonic()))

    def close(self) -> None:
        self.store.close()

    # -------- internals: routing and learning

    def _answer(self, request: dict) -> dict:
        questions = request.get("questions") or {}
        if not questions or "state" not in request:
            raise ValueError("a request needs a state and at least one question")
        text = state_text(request["state"])
        h = text_hash(text)  # what the encoder keys the state by, so the store can be read before it is encoded
        sts = {qid: self._student(q) for qid, q in questions.items()}
        with self.lock:
            self.stats["requests"] += 1
        if self.memo:
            known = {qid: self.store.lookup(st.key, st.family, h) for qid, st in sts.items()}
            if all(v is not None for v in known.values()):
                with self.lock:
                    self.stats["cached"] += 1
                answers = {}
                for qid, (kind, lab, probs) in known.items():
                    st = sts[qid]
                    a = answer_object(st.spec, np.asarray(probs) if probs is not None else np.eye(st.k)[lab])
                    answers[qid] = (st.teacher_answer(a, lab, probs) if kind == SAMPLED else a) | {"answered_by": "cache"}
                return {"model": "hunch", "answers": answers, "answered_by": "cache"}
        encoding = self.encoder.start([text])
        if self.backend is not None and any(st.mode is None or st.demoted for st in sts.values()):
            # A question that hasn't graduated, or is demoted, sends the request to the teacher whatever its features
            # say: ask the teacher now, while the state is encoded for learning and audits, rather than after.
            upstream, parsed = self._ask(request)
            _, F = self.encoder.finish(*encoding)
            x = F[0]
            preds = {qid: st.predict(x) for qid, st in sts.items()}
            return self._forwarded(request, sts, preds, upstream, parsed, h, x, text)
        _, F = self.encoder.finish(*encoding)
        x = F[0]
        preds = {qid: st.predict(x) for qid, st in sts.items()}
        if all(local for _, _, local, _ in preds.values()):
            with self.lock:
                self.stats["local"] += 1
            if self.backend is not None and self.rng.random() < self.audit_rate:
                self._audit_later(request, sts, preds, h, x, text)
            answers = {qid: answer_object(sts[qid].spec, p) | {"answered_by": "proxy", "tier": tier}
                       for qid, (p, _, _, tier) in preds.items()}
            return {"model": "hunch", "answers": answers, "answered_by": "proxy"}
        if self.backend is None:
            # no teacher: answer what may be answered, mark the rest unavailable
            answers = {qid: (answer_object(sts[qid].spec, p) | {"answered_by": "proxy", "tier": tier} if ok
                             else {"type": sts[qid].spec["type"], "answered_by": None, "error": "not graduated; no backend"})
                       for qid, (p, _, ok, tier) in preds.items()}
            return {"model": "hunch", "answers": answers, "answered_by": "partial"}
        upstream, parsed = self._ask(request)
        return self._forwarded(request, sts, preds, upstream, parsed, h, x, text)

    def _forwarded(self, request, sts, preds, upstream, parsed, h, x, text) -> dict:
        """Learn from the teacher's answers to a forwarded request, and answer with them."""
        questions = request["questions"]
        with self.lock:
            self.stats["forwarded"] += 1
        self._learn(sts, preds, parsed, h, x, text)
        answers = {qid: sts[qid].teacher_answer(upstream["answers"][qid], *parsed[qid]) | {"answered_by": "backend"}
                   for qid in questions}
        return {"model": upstream.get("model", "upstream"), "answers": answers, "answered_by": "backend",
                **({"usage": upstream["usage"]} if "usage" in upstream else {})}

    def _ask(self, request: dict) -> tuple[dict, dict[str, tuple[int, list[float] | None]]]:
        """The teacher's response to a request and every answer in it parsed (`parse_answer`), or BackendError if it
        failed or any answer is missing or malformed, so nothing is learned from a partly bad response."""
        try:
            got = self.backend(request)
            answers = got.get("answers") if isinstance(got, dict) else None
            if not isinstance(answers, dict):
                raise ValueError("the response has no answers object")
        except Exception as e:
            with self.lock:
                self.stats["backend_errors"] += 1
            raise BackendError(f"the teacher failed: {type(e).__name__}: {e}") from e
        missing = [qid for qid in request["questions"] if qid not in answers]
        if missing:
            with self.lock:
                self.stats["backend_errors"] += 1
            raise BackendError(f"the teacher's response has no answer for {missing}")
        parsed = {}
        for qid, q in request["questions"].items():
            try:
                parsed[qid] = parse_answer(q, answers[qid])
            except (ValueError, KeyError, TypeError) as e:
                with self.lock:
                    self.stats["backend_errors"] += 1
                raise BackendError(f"the teacher's answer for {qid!r} is malformed: {e}") from e
        return got, parsed

    def _learn(self, sts: dict, preds: dict, parsed: dict, h: str, x: np.ndarray, text: str) -> None:
        """Store the teacher's (parsed) answers as sampled examples and audit every student that would have
        answered."""
        for qid, st in sts.items():
            lab, probs = parsed[qid]
            self._record(st, SAMPLED, [Row(h, lab, probs)], {h: x}, {h: text})
            p, eligible, _, _ = preds[qid]
            if eligible:
                t = st.tune(self._probs(st, Row(h, lab, probs)))
                st.audit(int(np.argmax(p)), int(np.argmax(t)) if t is not None else lab)
            if st.since_refit >= self.refit_every:
                self._schedule(st.key)

    def _audit_later(self, request, sts, preds, h, x, text) -> None:
        try:
            self._audits.put_nowait((request, sts, preds, h, x, text))
        except queue.Full:
            with self.lock:
                self.stats["audits_skipped"] += 1
            return
        if self._auditor is None:
            with self.lock:
                if self._auditor is None:
                    self._auditor = threading.Thread(target=self._audit_loop, name="hunch-audit", daemon=True)
                    self._auditor.start()

    def _audit_loop(self) -> None:
        while True:
            request, sts, preds, h, x, text = self._audits.get()
            try:
                _, parsed = self._ask(request)
                with self.lock:
                    self.stats["audited"] += 1
                self._learn(sts, preds, parsed, h, x, text)
            except Exception as e:  # a failed audit is skipped; serving goes on
                print(f"hunch: audit failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            finally:
                self._audits.task_done()

    # -------- internals: students, refits and re-encoding

    def _student(self, q: dict) -> Student:
        spec, key, family = question_spec(q), question_key(q), family_key(q)
        with self.lock:
            if key not in self.students:
                self.store.add_question(key, spec, family, family_signature(spec))
                self.students[key] = Student(key, spec, family, self.target, self.store)
                if family != key:
                    self._schedule(key)  # a new version of a known family starts from the family's labels
            elif q.get("family") and self.students[key].family != family:
                st = self.students[key]
                if st.family != key:
                    raise ValueError(f"this question is already in {st.family}")
                self.store.join_family(key, spec, family, family_signature(spec))  # brings its labels along
                st.family = family
                for other in self.students.values():
                    if other.family == family:
                        self._schedule(other.key)
            return self.students[key]

    def _record(self, st: Student, kind: str, rows: list[Row], features: dict[str, np.ndarray],
                texts: dict[str, str], states: dict[str, Any] | None = None) -> None:
        """Labelled rows go to the family (every version learns from them), sampled rows to this version. Teacher
        answers for labelled states are kept per version; `states` (by hash) lets the teacher be asked for the
        missing ones."""
        self.store.add_examples(st.family if kind == LABELLED else st.key, kind, rows, features, texts)
        with self.lock:
            versions = [s for s in self.students.values() if s.family == st.family]
            for other in (versions if kind == LABELLED else [st]):
                other.since_refit += len(rows)
        if kind == SAMPLED:
            labelled = {r.state: self._probs(st, r) for r in rows
                        if (got := self.store.lookup(st.key, st.family, r.state)) and got[0] == LABELLED}
            self.store.put_teacher(st.key, labelled)
            return
        missing: dict[str, dict] = {}
        for v in versions:
            kept = {h for h, _, _ in self.store.teacher_pairs(v.key, v.family)}
            have = {r.state: (lab, p) for r in rows if r.state not in kept
                    and (got := self.store.get(v.key, SAMPLED, r.state)) for lab, p in [got]}
            self.store.put_teacher(v.key, {h: self._probs(v, Row(h, lab, p)) for h, (lab, p) in have.items()})
            for r in rows:
                if r.state not in have and r.state not in kept:
                    missing.setdefault(r.state, {})[v.key] = v.spec
        if missing and states and self.ask_for_labels and self.backend is not None:
            for h, questions in missing.items():
                self._label_asks.put((h, states[h], questions))
            with self.lock:
                if self._label_asker is None:
                    self._label_asker = threading.Thread(target=self._label_loop, name="hunch-labels", daemon=True)
                    self._label_asker.start()

    @staticmethod
    def _probs(st: Student, r: Row) -> list[float]:
        """A teacher answer's probabilities (one-hot when it gave only a class)."""
        return list(r.probs) if r.probs is not None else np.eye(st.k)[r.label].tolist()

    def _label_loop(self) -> None:
        touched: set[str] = set()
        while True:
            h, state, questions = self._label_asks.get()
            try:
                _, parsed = self._ask({"state": state, "questions": questions})
                with self.lock:
                    self.stats["label_asks"] += 1
                    sts = {key: self.students[key] for key in questions}
                for key, st in sts.items():
                    lab, probs = parsed[key]
                    self.store.put_teacher(key, {h: self._probs(st, Row(h, lab, probs))})
                    touched.add(key)
            except Exception as e:  # a failed ask is skipped; that label just has no teacher answer
                print(f"hunch: asking the teacher about a label failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            finally:
                if self._label_asks.empty():  # refit, so the teacher's tuning sees the new pairs
                    for key in touched:
                        self._schedule(key)
                    touched.clear()
                self._label_asks.task_done()

    def _refit(self, key: str) -> None:
        """Refit one version now, in this thread, swap the new fit in and keep it in the store. Refits of one version
        run one at a time (a request's `train` waits for the worker's), so the fit installed last was trained on the
        newest examples."""
        st = self.students[key]
        with st.refit_lock:
            self._refit_locked(st)

    def _refit_locked(self, st: Student) -> None:
        key = st.key
        st.since_refit = 0
        ex = self.store.examples(key, st.family)
        tuning, report = fit_tuning(st.spec, self.store.teacher_pairs(key, st.family))
        args = (st.spec, ex, self.target, self.minority, self.min_examples)
        if tuning is None:
            fit = fit_student(*args)
        else:  # the full student learns the tuned teacher; the partial one, the teacher's own answers graded on tuned
            tuned = Fit(tune_coef=tuning[0], tune_intercept=tuning[1], tune_classes=tuning[2])
            y, conf = tuned_targets(tuned, ex, st.k)
            fit = fit_student(*args, y=y, conf=conf)
            fit.partial = fit_student(*args, grade=y)
            fit.tune_coef, fit.tune_intercept, fit.tune_classes = tuning
            fit.report["partial"] = fit.partial.report
        fit.report["tuning"] = report
        st.install(fit)
        self.store.put_fit(key, fit.to_bytes())
        with self._cv:
            self._restoring.discard(key)

    def _schedule(self, key: str) -> None:
        """Refit a version on the worker thread (or now, without `background`)."""
        self.students[key].since_refit = 0
        if not self.background:
            return self._refit(key)
        with self._cv:
            self._pending[key] = None
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="hunch-refit", daemon=True)
                self._worker.start()
            self._cv.notify_all()

    def _work(self) -> None:
        while True:
            with self._cv:
                while not self._pending:
                    self._busy = False
                    self._cv.notify_all()
                    self._cv.wait()
                key = next(iter(self._pending))
                del self._pending[key]
                self._busy = True
            try:
                self._refit(key)
            except Exception as e:  # keep serving; the version keeps its previous fit
                print(f"hunch: refit of {key} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    def _reencode(self) -> None:
        texts = list(self.store.state_texts())
        if any(t is None for _, t in texts):
            raise FingerprintMismatch("the encoder changed and the store doesn't keep state texts to re-encode them "
                                      "(start it with store_text); use a new store")
        feats = {}
        for i in range(0, len(texts), 256):
            part = texts[i: i + 256]
            for (h, _), p in zip(part, self.model.embed_batch_partial([encoder_text(t) for _, t in part], self.layer)):
                feats[h] = np.asarray(p.layer_means[self.layer], dtype=np.float32)
        self.store.replace_features(feats, self.fingerprint)


# ---------------------------------------------------------------- HTTP


def make_handler(proxy: Proxy, token: str | None = None, train_token: str | None = None, max_body: int = 1 << 20,
                 access_log: bool = False, timeout: float | None = 30):
    import hmac
    import traceback
    from http import HTTPStatus

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive; every response carries Content-Length

        def setup(self):  # a client that sends nothing for `timeout` seconds, idle or mid-request, is disconnected
            self.timeout = timeout
            super().setup()

        def _send(self, code: int, body: dict) -> None:
            if self.path == "/v1/embeddings" and code >= 400:  # OpenAI's error shape, as vLLM sends it
                error = {"message": body.pop("error"), "type": HTTPStatus(code).phrase.replace(" ", "") + "Error",
                         "param": None, "code": code}
                body = {"error": error} | body
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            if access_log:
                print(f"{self.command} {self.path} {code} {(time.perf_counter() - self._t0) * 1000:.1f} ms"
                      + (f" {body.get('answered_by')}" if isinstance(body, dict) and "answered_by" in body else ""),
                      file=sys.stderr, flush=True)

        def log_message(self, fmt, *args):  # the access log, when on, is written by _send
            pass

        def _authorized(self, need: str | None) -> bool:
            if need is None:
                return True
            got = self.headers.get("Authorization", "")
            if hmac.compare_digest(got.encode(), f"Bearer {need}".encode()):
                return True
            self._send(401, {"error": "missing or wrong bearer token"})
            return False

        def do_GET(self):
            self._t0 = time.perf_counter()
            if self.path == "/healthz":
                return self._send(200, {"ok": True})
            if self.path == "/readyz":
                ready = proxy.ready()
                return self._send(200 if ready else 503, {"ready": ready})
            parts = self.path.split("/")
            blind = len(parts) == 5 and parts[:3] == ["", "v1", "questions"] and parts[4] == "blind_spots"
            if not self._authorized(train_token or token if blind else token):  # blind spots can return stored texts
                return None
            if self.path == "/v1/questions":
                return self._send(200, proxy.status())
            if blind:
                try:
                    return self._send(200, proxy.blind_spots(parts[3]))
                except KeyError:
                    return self._send(404, {"error": f"no question {parts[3]}"})
            self._send(404, {"error": "not found"})

        def _refuse(self, code: int, error: str) -> None:
            """Answer without reading the body, then close: the unread body can't be taken for the next request."""
            self.close_connection = True
            self._send(code, {"error": error})

        def do_POST(self):
            self._t0 = time.perf_counter()
            if self.path not in ("/v1/answer", "/v1/train", "/v1/embeddings"):
                return self._refuse(404, "not found")
            if not self._authorized(train_token or token if self.path == "/v1/train" else token):
                self.close_connection = True
                return None
            length = self.headers.get("Content-Length")
            if length is None:
                return self._refuse(411, "a request body needs a Content-Length")
            if not (length.strip().isascii() and length.strip().isdigit()):
                return self._refuse(400, f"invalid Content-Length {length!r}")
            n = int(length)
            if n > max_body:
                return self._refuse(413, f"request body over {max_body} bytes")
            try:
                raw = self.rfile.read(n)
            except TimeoutError:
                self.close_connection = True
                return None
            if len(raw) < n:  # the client closed the connection mid-body
                self.close_connection = True
                return None
            try:
                body = json.loads(raw or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("the request body must be a JSON object")
                if self.path == "/v1/answer":
                    return self._send(200, proxy.answer(body))
                if self.path == "/v1/embeddings":
                    return self._send(200, proxy.embeddings(body))
                if "question" not in body:
                    raise ValueError("a train request needs a question")
                return self._send(200, proxy.train(body["question"], body.get("cases", []), body.get("refit", True)))
            except BackendError as e:
                self._send(502, {"error": str(e)})
            except ModelNotFound as e:
                self._send(404, {"error": str(e)})
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})
            except Exception:  # the detail goes to the log, which the client can quote by its id
                error_id = uuid.uuid4().hex[:12]
                print(f"hunch: error {error_id} on {self.command} {self.path}:\n{traceback.format_exc()}",
                      file=sys.stderr, flush=True)
                self._send(500, {"error": "internal error", "error_id": error_id})

    return Handler


class Server(ThreadingHTTPServer):
    """A thread per connection, at most `max_connections` at once: a connection beyond that gets 503 and is
    closed."""

    def __init__(self, address, handler, max_connections: int = 64):
        super().__init__(address, handler)
        self.slots = threading.BoundedSemaphore(max_connections)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            body = json.dumps({"error": "too many connections"}).encode()
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                                b"Retry-After: 1\r\nConnection: close\r\n"
                                + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def serve(proxy: Proxy, host: str = "127.0.0.1", port: int = 8765, *, token: str | None = None,
          train_token: str | None = None, max_body: int = 1 << 20, access_log: bool = False,
          timeout: float | None = 30, max_connections: int = 64) -> Server:
    """Endpoints (HTTP/1.1, keep-alive):
      POST /v1/answer     a request; answered from the store, locally or by the teacher ("answered_by" says which)
      POST /v1/train      {"question": {...}, "cases": [{"state", "answer", "kind": "labelled"|"sampled"}], "refit": true}
      POST /v1/embeddings {"input": "text" | ["text", ...]}: the model's embeddings, as vLLM and OpenAI serve them
                          (`Proxy.embeddings`); its errors come as {"error": {"message", "type", "param", "code"}}
      GET  /v1/questions  {"ready", "stats", "latency", "layer", "encoder", "states", "questions": [...]}: request
                          counts, latency by who answered, the encoder fingerprint, the stored states, and every
                          question's examples, graduation, CV report and audits
      GET  /healthz       liveness
      GET  /readyz        200 once every restored question has a fit, else 503
    With `token`, every endpoint but /healthz and /readyz needs "Authorization: Bearer <token>"; `train_token`, if
    given, is the one /v1/train needs instead; the token is checked before the body is read. A POST without a
    Content-Length gets 411, an invalid one 400, and a body over `max_body` bytes 413. A failing teacher gives 502;
    any other failure 500 with only an "error_id", under which the detail is logged to stderr. A connection that
    sends nothing for `timeout` seconds is closed, and at most `max_connections` are served at once (each holds a
    thread while open); one more gets 503."""
    return Server((host, port), make_handler(proxy, token, train_token, max_body, access_log, timeout), max_connections)


def redact_url(url: str) -> str:
    """A URL without its password, for logs."""
    from urllib.parse import urlsplit

    u = urlsplit(url)
    return u._replace(netloc=u.netloc.replace(f":{u.password}@", ":***@")).geturl() if u.password else url


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m hunch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the proxy over HTTP")
    s.add_argument("--backend", required=True, help="the teacher's URL, or offline:<answer cache file or folder>")
    s.add_argument("--backend-header", action="append", default=[], metavar="'NAME: VALUE'",
                   help="a header for requests to the teacher (repeatable), e.g. an Authorization header")
    s.add_argument("--backend-timeout", type=float, default=30, help="seconds to wait for the teacher")
    s.add_argument("--fallback", help="with an offline backend: a teacher URL to ask on a miss")
    s.add_argument("--record", help="with an offline backend: append the fallback's answers to this file")
    s.add_argument("--db", default="proxy.db", help="a SQLite path, or redis://host:port/db[?prefix=name]")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--token", default=os.environ.get("HUNCH_TOKEN"),
                   help="require 'Authorization: Bearer <token>' (default: $HUNCH_TOKEN)")
    s.add_argument("--train-token", default=os.environ.get("HUNCH_TRAIN_TOKEN"),
                   help="a separate token for /v1/train (default: $HUNCH_TRAIN_TOKEN, else --token)")
    s.add_argument("--max-body", type=int, default=1 << 20, help="largest request body, in bytes")
    s.add_argument("--client-timeout", type=float, default=30,
                   help="seconds a client may send nothing (idle or mid-request) before it is disconnected")
    s.add_argument("--max-connections", type=int, default=64,
                   help="connections served at once (a thread each); more get 503")
    s.add_argument("--access-log", action="store_true", help="log every request to stderr")
    s.add_argument("--threads", type=int, help="threads for the encoder (sets RAYON_NUM_THREADS)")
    s.add_argument("--layer", type=int, default=8)
    s.add_argument("--audit", type=float, default=0.10, help="share of local answers audited by the teacher")
    s.add_argument("--target", type=float, default=0.90)
    s.add_argument("--minority", type=float, default=0.80)
    s.add_argument("--sampled-cap", type=int, default=2000, help="sampled examples kept per question version")
    s.add_argument("--labelled-cap", type=int, default=5000, help="labelled examples kept per family and class")
    s.add_argument("--refit-every", type=int, default=100)
    s.add_argument("--min-examples", type=int, default=60)
    s.add_argument("--no-memo", action="store_true", help="don't answer known states from the store")
    s.add_argument("--no-ask-for-labels", action="store_true",
                   help="don't ask the teacher about labelled states it hasn't answered")
    s.add_argument("--store-text", action="store_true", help="keep each state's text, so --rebuild can re-encode it")
    s.add_argument("--rebuild", action="store_true",
                   help="if the encoder (revision, layer, precision) differs from the store's, re-encode the stored states")
    args = ap.parse_args(argv)
    if args.threads:
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)  # read when the encoder's thread pool starts
    headers = {}
    for hdr in args.backend_header:
        name, sep, value = hdr.partition(":")
        if not sep:
            ap.error(f"--backend-header {hdr!r}: expected 'NAME: VALUE'")
        headers[name.strip()] = value.strip()
    if args.backend.startswith("offline:"):
        cache = AnswerCache(args.backend[len("offline:"):], write_to=args.record)
        fallback = HTTPBackend(args.fallback, headers, args.backend_timeout) if args.fallback else None
        backend = OfflineBackend(cache, fallback=fallback)
    else:
        backend = HTTPBackend(args.backend, headers, args.backend_timeout)
    proxy = Proxy(backend=backend, db=args.db, layer=args.layer, audit=args.audit, target=args.target,
                  minority=args.minority, sampled_cap=args.sampled_cap, labelled_cap=args.labelled_cap,
                  refit_every=args.refit_every, min_examples=args.min_examples, store_text=args.store_text,
                  rebuild=args.rebuild, memo=not args.no_memo, ask_for_labels=not args.no_ask_for_labels)
    server = serve(proxy, args.host, args.port, token=args.token, train_token=args.train_token,
                   max_body=args.max_body, access_log=args.access_log, timeout=args.client_timeout,
                   max_connections=args.max_connections)
    auth = "token required" if args.token else "no token"
    print(f"hunch on http://{args.host}:{args.port} (layer {args.layer}, backend {redact_url(args.backend)}, "
          f"db {redact_url(args.db)}, {auth})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
