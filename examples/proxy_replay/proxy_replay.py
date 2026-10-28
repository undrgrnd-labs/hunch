"""Replay the proxy policy over the teacher-answered stream (teacher_stream.py) and measure how much it
could take off the teacher, at what agreement with the teacher.

The proxy, per question (tool_failed noul, output_kind choice, severity score):
  student     logistic regression on EG2 features (final embedding + layer-8 mean), trained on the teacher's answers for
              the items it actually sent to the teacher: the warm-up, forwarded items and audits. Nothing else. Features
              are standardized with the mean and spread of the student's own training items, so nothing about later
              items is used.
  refit       every REFIT new teacher answers, after a WARMUP of all-forwarded items.
  graduation  at each refit, 5-fold CV on the labelled items picks the lowest confidence threshold whose agreement
              with the teacher has a 95% lower bound >= TARGET (at least 30 items above it) AND whose recall on the teacher's
              non-majority answers has a 95% lower bound >= MINORITY_TARGET (at least MINORITY_MIN of them above it).
              No such threshold: not graduated. (The minority condition matters: without it, severity would graduate on the base rate, with
              agreement 0.95 against a 0.91 majority baseline but recall 0.47 on results the teacher thought had a problem.)
  demotion    a graduated question whose rolling audit agreement (last 40 audited items) is significantly below
              TARGET: the upper 95% bound under TARGET, so a question is demoted only on evidence that it is worse
              than the graduation bar, not on the noise of 40 audits.
Routing, per request (the teacher reads the state once for all questions, so a partial forward saves little): served
locally only if every question is graduated and above its threshold, and the item is not novel (its nearest
labelled neighbour is less similar than the 5th percentile of labelled-to-labelled nearest neighbours, on features
standardized with the warm-up items' statistics). A random
AUDIT share of locally served items also goes to the teacher. Agreement of every local answer with the teacher is measured
offline, which production could only estimate from the audits.

    uv run proxy_replay/proxy_replay.py
"""

import json
import random
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import eg2cpu

from hunch_examples import OUT, kappa, stream_rows
from hunch.proxy import wilson
QS = ["tool_failed", "output_kind", "severity"]
WARMUP, REFIT, AUDIT_WINDOW = 150, 100, 40
MINORITY_TARGET, MINORITY_MIN = 0.80, 15  # graduation also needs recall on the teacher's non-majority answers


def agree_cells(agreements):
    """Agreement columns: a dash where nothing was answered locally (no local answers to agree with)."""
    return [f"{x:>22.3f}" if x == x else f"{'-':>22}" for x in agreements]

rows = stream_rows()  # the teacher's answers from data/teacher_stream.jsonl
n = len(rows)


def teacher(r, q):
    p = r["teacher"][q]
    return int(p[0] >= 0.5) if q == "tool_failed" else int(np.argmax(p))


Y = {q: np.array([teacher(r, q) for r in rows]) for q in QS}
MAJ = {q: int(np.bincount(Y[q]).argmax()) for q in QS}
W = {q: np.array([max(r["teacher"][q][0], 1 - r["teacher"][q][0]) if q == "tool_failed" else max(r["teacher"][q]) for r in rows])
     for q in QS}

# ---- features: one eg2cpu pass per item, all questions share it
model = eg2cpu.Model()
texts = [f"{r['call']}\n{r['output']}" for r in rows]
t0 = time.perf_counter()
outs = model.embed_batch(texts)
eg2_ms = (time.perf_counter() - t0) / n * 1000
X = np.array([np.concatenate([e, m[8]]) for e, m in outs], dtype=np.float64)
Xw = (X - X[:WARMUP].mean(0)) / (X[:WARMUP].std(0) + 1e-6)  # the warm-up comes before any routing decision
Xn = Xw / np.linalg.norm(Xw, axis=1, keepdims=True)
FEATS = {q: X for q in QS}  # what each question's student fits on (a warm start swaps this, see the end)
FIXED = {q: 0 for q in QS}  # leading feature columns kept as they are (a warm start's handle readout)
ONE_SIDED = set()  # noul questions routed one-sided: answer only a confident "no" locally (see the end)
ONE_SIDED_RECALL, ONE_SIDED_MISS = 0.85, 0.02
teacher_s = float(np.median([r["seconds"] for r in rows]))
print(f"{n} stream items; teacher median {teacher_s * 1000:.0f} ms/item (all 3 questions, one pass); "
      f"eg2cpu {eg2_ms:.1f} ms/item (batched, all cores)")
print("teacher answer mix: " + "; ".join(f"{q}: {np.bincount(Y[q]).tolist()}" for q in QS))


class Student:
    """A logistic regression on features standardized with its training items' mean and spread."""

    def __init__(self, idx, q):
        F = FEATS[q][idx]
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        self.mu[:FIXED[q]], self.sd[:FIXED[q]] = 0, 1
        self.clf = LogisticRegression(max_iter=2000, C=0.5).fit((F - self.mu) / self.sd, Y[q][idx], sample_weight=W[q][idx])
        self.classes_ = self.clf.classes_

    def predict_proba(self, F):
        return self.clf.predict_proba((F - self.mu) / self.sd)


def fit(idx, q):
    return Student(idx, q) if len(set(Y[q][idx].tolist())) >= 2 else None


def conf_of(clf, rows_idx, q):
    P = clf.predict_proba(FEATS[q][rows_idx])
    return clf.classes_[np.argmax(P, 1)], P.max(1)


def pick_one_sided(idx, q):
    """For a rare "yes": the largest tau such that, in 5-fold CV, items with p(yes) >= tau (forwarded) hold the teacher's
    "yes" answers with recall lower bound >= ONE_SIDED_RECALL, and local "no" answers (p < tau) are wrong at most
    ONE_SIDED_MISS (upper bound). Returned as a negative number so the routing test `conf >= thr` reads p < tau."""
    idx = np.array(idx)
    yq = Y[q][idx]
    if len(idx) < 60 or yq.sum() < MINORITY_MIN:
        return None
    prob = np.zeros(len(idx))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(idx):
        clf = fit(idx[tr], q)
        if clf is None:
            return None
        prob[te] = clf.predict_proba(FEATS[q][idx[te]])[:, list(clf.classes_).index(1)]
    best = None
    for tau in np.unique(np.round(prob, 4)):
        fwd, local = prob >= tau, prob < tau
        k_yes = int((yq[fwd] == 1).sum())
        miss = int((yq[local] == 1).sum())
        if (local.sum() >= 30 and wilson(k_yes, int(yq.sum()))[0] >= ONE_SIDED_RECALL
                and wilson(miss, int(local.sum()))[1] <= ONE_SIDED_MISS):
            best = float(tau)
    return None if best is None else -best


def pick_threshold(idx, q, target):
    """5-fold CV on the labelled items: the lowest threshold with lower-bound agreement >= target."""
    if q in ONE_SIDED:
        return pick_one_sided(idx, q)
    idx = np.array(idx)
    if len(idx) < 60:
        return None
    pred, conf = np.zeros(len(idx), dtype=int), np.zeros(len(idx))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(idx):
        clf = fit(idx[tr], q)
        if clf is None:
            return None
        pr, cf = conf_of(clf, idx[te], q)
        pred[te], conf[te] = pr, cf
    agree = pred == Y[q][idx]
    minority = Y[q][idx] != MAJ[q]
    for t in np.arange(0.5, 0.995, 0.01):
        sel = conf >= t
        ms = sel & minority
        # Imbalance: answering the majority agrees ~90-95% of the time on tool_failed / severity, so plain agreement
        # can pass while the teacher's rarer answers are mostly missed. Require recall on them too.
        if (sel.sum() >= 30 and wilson(int(agree[sel].sum()), int(sel.sum()))[0] >= target
                and ms.sum() >= MINORITY_MIN and wilson(int(agree[ms].sum()), int(ms.sum()))[0] >= MINORITY_TARGET):
            return float(t)
    return None


def replay(target=0.90, audit=0.10, seed=0, log_every=200):
    rng = random.Random(seed)
    labelled = []  # indices the teacher answered
    students, thr, grad = {}, {q: None for q in QS}, {q: False for q in QS}
    audits = {q: [] for q in QS}
    events, since_refit, nn_floor = [], 0, None
    served_local = np.zeros(n, dtype=bool)
    called = np.zeros(n, dtype=bool)
    local_ans = {q: np.full(n, -1) for q in QS}

    def refit(i):
        nonlocal nn_floor
        idx = np.array(labelled)
        for q in QS:
            students[q] = fit(idx, q)
            t = pick_threshold(idx, q, target)
            was = grad[q]
            thr[q], grad[q] = t, t is not None
            if grad[q] != was:
                events.append((i, q, "graduated" if grad[q] else "not graduated", t))
        S = Xn[idx] @ Xn[idx].T
        np.fill_diagonal(S, -1)
        nn_floor = float(np.quantile(S.max(1), 0.05))

    for i in range(n):
        local = False
        if i >= WARMUP and all(grad.values()):
            novel = float((Xn[np.array(labelled)] @ Xn[i]).max()) < nn_floor
            if not novel:
                ok = True
                for q in QS:
                    if q in ONE_SIDED:  # local only as a confident "no": p(yes) below tau
                        clf = students[q]
                        p_yes = clf.predict_proba(FEATS[q][[i]])[0, list(clf.classes_).index(1)]
                        local_ans[q][i] = 0
                        ok &= p_yes < -thr[q]
                        continue
                    pr, cf = conf_of(students[q], [i], q)
                    local_ans[q][i] = pr[0]
                    ok &= cf[0] >= thr[q]
                local = ok
        audited = local and rng.random() < audit
        served_local[i] = local
        if not local or audited:
            called[i] = True
            labelled.append(i)
            since_refit += 1
        if audited:
            for q in QS:
                audits[q].append(int(local_ans[q][i] == Y[q][i]))
                a = audits[q][-AUDIT_WINDOW:]
                # demote only on evidence of failure: even the optimistic bound is below target
                if grad[q] and len(a) >= 20 and wilson(sum(a), len(a))[1] < target:
                    grad[q] = False
                    events.append((i, q, "demoted by audit", None))
        if (i + 1 == WARMUP) or (i >= WARMUP and since_refit >= REFIT):
            refit(i)
            since_refit = 0
    return served_local, called, local_ans, events


def report(served, called, local_ans, events, label, windows=True):
    print(f"\n== {label}")
    for i, q, what, t in events:
        print(f"   item {i:>5}: {q} {what}" + (f" (threshold {t:.2f})" if t is not None else ""))
    if windows:
        print(f"   {'items':>11}{'served locally':>16}{'teacher calls/item':>20}" + "".join(f"{'agree ' + q:>22}" for q in QS))
        for a in range(0, n, 250):
            b = min(n, a + 250)
            s = served[a:b]
            ag = [np.mean(local_ans[q][a:b][s] == Y[q][a:b][s]) if s.any() else float("nan") for q in QS]
            print(f"   {a:>5}-{b:<5}{s.mean():>16.0%}{called[a:b].mean():>20.2f}" + "".join(agree_cells(ag)))
    half = slice(n // 2, n)
    s = served[half]
    out = {"served_local_second_half": float(s.mean()), "teacher_calls_per_item_second_half": float(called[half].mean()),
           "agreement_local_second_half": {q: float(np.mean(local_ans[q][half][s] == Y[q][half][s])) if s.any() else None for q in QS},
           "served_local_all": float(served.mean()), "teacher_calls_per_item_all": float(called.mean()),
           "events": [(int(i), q, w, t) for i, q, w, t in events]}
    # The teacher's answers are imbalanced (tool_failed is "no" ~95% of the time, severity "none" ~90%), so plain agreement
    # flatters those questions. Report, on locally served items: majority-answer baseline, Cohen's kappa, and recall
    # of the teacher's minority answers (tool_failed = yes; severity != none; output_kind: every non-majority class).
    out["quality_local_second_half"] = {}
    for q in QS:
        if not s.any():
            continue
        yq, aq = Y[q][half][s], local_ans[q][half][s]
        maj = int(np.bincount(Y[q]).argmax())
        minority = yq != maj
        out["quality_local_second_half"][q] = {
            "n": int(s.sum()), "agreement": float(np.mean(aq == yq)), "majority_baseline": float(np.mean(yq == maj)),
            "kappa": kappa(yq, aq), "teacher_minority_n": int(minority.sum()),
            "minority_recall": float(np.mean(aq[minority] == yq[minority])) if minority.any() else None}
        m = out["quality_local_second_half"][q]
        print(f"   {q:<12} served {m['n']:>4}  agreement {m['agreement']:.3f} (majority baseline {m['majority_baseline']:.3f})  "
              f"kappa {m['kappa']:.3f}  recall on the teacher's minority answers {m['minority_recall'] if m['minority_recall'] is None else round(m['minority_recall'], 3)} "
              f"(n={m['teacher_minority_n']})")
    # blended: served answers are the teacher's own when forwarded
    out["agreement_served_second_half"] = {q: float(np.mean(np.where(s, local_ans[q][half] == Y[q][half], True))) for q in QS}
    return out


results = {"teacher_ms": teacher_s * 1000, "eg2cpu_ms": eg2_ms, "n": n}
served, called, la, ev = replay()
results["main (target 0.90, audit 10%)"] = report(served, called, la, ev, "all three questions, request-level routing, target 0.90, audit 10%")

# Without severity: the two surface-readable questions alone
QS_FULL = QS
QS = ["tool_failed", "output_kind"]
served, called, la, ev = replay()
results["without severity"] = report(served, called, la, ev, "tool_failed + output_kind only (severity left to the teacher entirely)")
QS = QS_FULL

# One question per request: each question routed on its own (a teacher call then answers just that question; its
# cost is mostly reading the state, so about the same per call).
for q_only in QS_FULL:
    QS = [q_only]
    served, called, la, ev = replay()
    results[f"only {q_only}"] = report(served, called, la, ev, f"{q_only} alone (its own requests)", windows=False)
    c = results[f"only {q_only}"]["teacher_calls_per_item_second_half"]
    print(f"   served locally {results[f'only {q_only}']['served_local_second_half']:.0%}, teacher calls/item {c:.2f}, "
          f"compute {eg2_ms + c * teacher_s * 1000:.0f} ms vs {teacher_s * 1000:.0f} ms")
QS = QS_FULL

print("\n== sensitivity (second half of the stream; all three questions)")
print("   (agreement is over the answers served locally; - where none were)")
print(f"   {'target':>7}{'audit':>7}{'served locally':>16}{'teacher calls/item':>20}" + "".join(f"{'agree ' + q:>22}" for q in QS))
sens = []
for target in (0.85, 0.90, 0.95):
    for audit in (0.05, 0.10):
        served, called, la, ev = replay(target, audit)
        half = slice(n // 2, n)
        s = served[half]
        ag = [float(np.mean(la[q][half][s] == Y[q][half][s])) if s.any() else float("nan") for q in QS]
        sens.append({"target": target, "audit": audit, "served_local": float(s.mean()), "teacher_calls": float(called[half].mean()),
                     "agreement": dict(zip(QS, ag))})
        print(f"   {target:>7.2f}{audit:>7.0%}{s.mean():>16.0%}{called[half].mean():>20.2f}" + "".join(agree_cells(ag)))
results["sensitivity"] = sens

for key in ("main (target 0.90, audit 10%)", "without severity"):
    c = results[key]["teacher_calls_per_item_second_half"]
    cost = eg2_ms + c * teacher_s * 1000
    print(f"\ncompute per item, {key}, second half: {eg2_ms:.0f} ms eg2cpu + {c:.2f} x {teacher_s * 1000:.0f} ms teacher = "
          f"{cost:.0f} ms vs {teacher_s * 1000:.0f} ms teacher-only ({teacher_s * 1000 / cost:.2f}x)")
    results[key]["ms_per_item_second_half"] = cost
(OUT / "proxy_replay.json").write_text(json.dumps(results, indent=1))
print(f"\nwrote {OUT / 'proxy_replay.json'}")


# ---------------------------------------------------------------- warm start for tool_failed
# The compiled tool_call_failed handle (results/tool_call_failed.eg2h, from tool_call_failed.py: final layer, LDA direction from 56
# synthetic contrast pairs, never trained on the teacher or this stream) gives one standardized readout z per item.
#   handle only        the student is a 1-D logistic regression on z: only the cut-off is learned from the teacher
#   handle + residual  z (scaled x10, so its weight is cheap under the L2 penalty) next to the full EG2 features
HANDLE = OUT / "tool_call_failed.eg2h"
if not HANDLE.exists():
    sys.exit(f"\nno {HANDLE}: run tool_call_failed.py to compile it, then re-run for the warm-start comparison")
handle = eg2cpu.Handle(str(HANDLE))
[vs] = zip(*model.apply([handle], [{"kind": "tool_result", "call": r["call"], "output": r["output"]} for r in rows]))
Zh = np.array([v["z"][0] for v in vs])

print(f"\n== warm start: the compiled handle's readout against the teacher's tool_failed answers: AUC "
      f"{roc_auc_score(Y['tool_failed'], Zh):.3f} (no teacher labels used)")
warm = {}
for mode, feats, fixed in (("from scratch", X, 0), ("handle only", Zh.reshape(-1, 1), 1),
                           ("handle + residual", np.hstack([10 * Zh.reshape(-1, 1), X]), 1)):
    FEATS["tool_failed"], FIXED["tool_failed"] = feats, fixed
    QS = ["tool_failed"]
    served, called, la, ev = replay()
    warm[f"tool_failed alone, {mode}"] = report(served, called, la, ev, f"tool_failed alone, student {mode}", windows=False)
    QS = ["tool_failed", "output_kind"]
    served, called, la, ev = replay()
    warm[f"tool_failed + output_kind, {mode}"] = report(served, called, la, ev,
                                                        f"tool_failed + output_kind per request, tool_failed student {mode}",
                                                        windows=False)
    for k in (f"tool_failed alone, {mode}", f"tool_failed + output_kind, {mode}"):
        c = warm[k]["teacher_calls_per_item_second_half"]
        print(f"   [{k}] served locally {warm[k]['served_local_second_half']:.0%}, teacher calls/item {c:.2f}, "
              f"compute {eg2_ms + c * teacher_s * 1000:.0f} ms vs {teacher_s * 1000:.0f} ms teacher-only")
QS, FEATS["tool_failed"], FIXED["tool_failed"] = QS_FULL, X, 0
results["warm_start"] = warm
(OUT / "proxy_replay.json").write_text(json.dumps(results, indent=1))


# ---------------------------------------------------------------- one-sided routing for tool_failed
print("\n== one-sided routing for tool_failed: answer only a confident 'not failed' locally, forward the rest")
ONE_SIDED.add("tool_failed")
one = {}
for mode, feats, fixed in (("from scratch", X, 0), ("handle + residual", np.hstack([10 * Zh.reshape(-1, 1), X]), 1)):
    FEATS["tool_failed"], FIXED["tool_failed"] = feats, fixed
    for qs in (["tool_failed"], ["tool_failed", "output_kind"]):
        QS = qs
        served, called, la, ev = replay()
        key = f"{' + '.join(qs)}, tool_failed one-sided, {mode}"
        one[key] = report(served, called, la, ev, key, windows=False)
        half = slice(n // 2, n)
        s_ = served[half]
        yes = Y["tool_failed"][half] == 1
        c = one[key]["teacher_calls_per_item_second_half"]
        one[key]["teacher_failures_answered_locally"] = int((s_ & yes).sum())
        print(f"   served locally {s_.mean():.0%}, teacher calls/item {c:.2f}, compute {eg2_ms + c * teacher_s * 1000:.0f} ms vs "
              f"{teacher_s * 1000:.0f} ms; teacher 'failed' answered 'no' locally: {(s_ & yes).sum()} of {yes.sum()} in the second half")
QS, FEATS["tool_failed"], FIXED["tool_failed"] = QS_FULL, X, 0
ONE_SIDED.clear()
results["one_sided"] = one
(OUT / "proxy_replay.json").write_text(json.dumps(results, indent=1))
