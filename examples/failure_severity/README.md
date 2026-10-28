# failure_severity: grading severity with `kind: levels`

Can a `kind: levels` decision ([`../decisions/failure_severity.yaml`](../decisions/failure_severity.yaml)) grade
coding-agent tool results as none, minor, recoverable or blocking?

From `examples/`:

```bash
uv run failure_severity/failure_severity.py
```

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), the pi traces (the labels ship as hashes; the text is rebuilt from the download), plus
`results/tool_call_failed.eg2h` from `tool_call_failed` for one baseline.
**Writes:** `results/failure_severity_validation.json`. **Time:** about 1 minute.

**Teacher answers:** [`teacher.jsonl`](teacher.jsonl), from Clef: question hash, state hash and answer; no text.

## What it does

- **Labels**, and who wrote them:
  - [`../data/failure_severity_labels.jsonl`](../data/failure_severity_labels.jsonl): 320 pi results, labelled by
    an LLM whose model and prompt were not recorded. Half are flagged as errors by the harness and half are not, so
    the flag alone separates most of them.
  - [`../data/failure_severity_keyword_set.jsonl`](../data/failure_severity_keyword_set.jsonl): 69 results that
    match blocking-type keywords (rate limit, 403, credentials ...) but are mostly source code that mentions them;
    labelled the same way as the 320, with the same gap in the record.
  - [`../data/failure_severity_stream_labels.jsonl`](../data/failure_severity_stream_labels.jsonl): all 2,000 results
    of the pi stream, labelled by Claude Opus 5.5 subagents with the four-level rubric of
    `TEACHER_QUESTIONS["severity"]`, plus an instruction to judge silent failures (a clean exit, but the output shows
    the operation failed) by what happened.
- Calibrates on 60 of the 320 and tests on the other 260, split by session: no session has results on both sides.
- With additional labels: a student on final-layer features, trained on the 60 calibration rows plus up to 200
  labelled rows from the other folds of the 260 (5-fold cross-fitting, folds grouped by session), alone and combined
  with the teacher.
- Baselines: the harness's error flag (flagged → recoverable, the rest → none), the `tool_call_failed` handle's p(yes),
  the same contrasts compiled as an unordered 4-way `choice`, and the teacher recalibrated on the same labels.
- On the stream, the 120 exact repeats of an earlier state are dropped, so each of the 1,880 distinct results is
  scored once; folds are grouped by session. Each model is also scored on the unflagged results, where the flag can't
  help.

## Findings

QWK and macro-F1 with 95% bootstrap intervals (resampling sessions); differences are paired, on the same resampled
sessions. On the 260 labelled results (half of them flagged by the harness):

| | accuracy | QWK | macro-F1 | Brier |
|---|---|---|---|---|
| levels (spec) | 0.635 | 0.45 [0.33, 0.55] | 0.34 | 0.52 |
| levels, layer and method auto | 0.612 | 0.39 | 0.33 | 0.51 |
| harness flag only | 0.858 | 0.91 | 0.46 | 0.29 |
| teacher (Clef), zero-shot | 0.715 | 0.73 [0.65, 0.80] | 0.53 | 0.42 |
| teacher, calibrated on 200 labels | 0.842 | 0.89 [0.85, 0.93] | 0.65 | 0.23 |
| EG2 + 200 labels | 0.877 | 0.84 [0.76, 0.90] | 0.73 | 0.20 |
| teacher + EG2, combined on 200 labels | 0.912 | 0.90 [0.85, 0.94] | 0.76 | 0.14 |

From synthetic contrasts alone, severity levels barely separate, and blocking errors read as recoverable. This set is
half flagged, and flagged results are mostly `recoverable` and unflagged ones mostly `none`, so the flag alone scores
QWK 0.91, and no model's QWK is distinguishable from it: the combined model's lead is +0.010 [−0.061, +0.083]. The
flag never predicts `minor` or `blocking`, which macro-F1 shows: on F1 the labelled student (+0.27 [+0.06, +0.43]),
the calibrated teacher and the combination all beat it. Between the models, EG2 with 200 labels and the calibrated
teacher are level on both QWK (−0.054 [−0.131, +0.020]) and F1 (+0.086 [−0.047, +0.173]).

On the pi stream as it arrives (1,880 distinct results, 91 flagged; labels 1,679 none, 113 minor, 87 recoverable,
1 blocking). 111 of the 201 problems are unflagged: mostly searches that found nothing and warnings, and 24 failures
the flag misses, where the command exits cleanly but the output shows it failed (a rebase refused behind `|| true`,
a test run that never ran, a GitHub fetch that returned only an error page):

| | accuracy | QWK | macro-F1 | unflagged problems found | unflagged failures found (24) | precision on unflagged |
|---|---|---|---|---|---|---|
| harness flag only | 0.927 | 0.72 | 0.42 | 0% | 0% | — |
| levels (spec) | 0.718 | 0.24 | 0.27 | 58% | 71% | 0.14 |
| teacher (Clef), zero-shot | 0.895 | 0.66 | 0.45 | 36% | 75% | 0.36 |
| teacher, calibrated on 200 labels | 0.926 | 0.74 | 0.45 | 17% | 62% | 0.63 |
| EG2 + 200 labels | 0.928 | 0.70 | 0.53 | 23% | 38% | 0.44 |
| teacher + EG2, combined on 200 labels | 0.937 | 0.77 | 0.54 | 32% | 62% | 0.62 |

On QWK the flag, the calibrated teacher, the labelled student and the combination are all within each other's
intervals (combined − flag: +0.046 [−0.024, +0.108]). On macro-F1 the labelled student beats the calibrated teacher
(+0.080 [+0.045, +0.143]) and the flag (+0.108 [+0.068, +0.185]). Silent failures are where the zero-shot teacher is
strongest: it catches 18 of the 24, at the cost of flagging many results that are fine; the uncalibrated spec decision
catches 17, at precision 0.14. Calibrating on stream labels, where problems are rare, makes every model more
conservative. 24 failures is a small sample, so read those columns as a direction, not a measurement.
