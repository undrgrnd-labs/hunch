# Examples

Each folder here is one experiment with eg2cpu or the hunch proxy on public datasets, with a README that says what
it measures and what it found. Every experiment also compares EG2 with a **teacher**: an LLM asked the same typed
question zero-shot. The teacher here is Clef (Clef-Flash, a 9B model run locally in 4-bit MLX, about 500 ms per item on
an Apple-silicon laptop); its decoding settings were not recorded with its answers. The teacher's answers ship with
each example, keyed by hashes, so every comparison replays without running it.

## Setup

You need a Rust toolchain (1.88 or later), because `uv sync` builds eg2cpu from `../eg2cpu` (about a minute the first
time), and [uv](https://docs.astral.sh/uv/), which also fetches Python 3.12. Every example needs the EmbeddingGemma 2
checkpoint (1.4 GB), which is Apache-2.0 and not gated on the Hub (as of October 2026). Run every command here from
`examples/`.

```bash
cd examples
uv sync
uv run download.py                  # what's available and what's already here
uv run download.py embeddinggemma-2 sst5 InjecAgent   # asks before each download (--yes to skip the question, and
                                                      # without a terminal to answer in)
```

Datasets go to `data/hf/` (git-ignored), or to `EG2_DATA` if you set it. Results go to `results/` (git-ignored): each
example's own files, its `<example>_teacher.json` comparison, and caches of rebuilt rows and embeddings. The
model goes to the Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`); `EG2_MODEL_DIR` points eg2cpu at a
checkpoint you keep somewhere else instead (`download.py` doesn't write there). Every download is pinned to the
revision the shipped hashes refer to.

| dataset | size | licence |
|---|---|---|
| `embeddinggemma-2` (model) | 1.4 GB | Apache-2.0 |
| `real-pi-coding-agent-traces-sessions` | 750 MB | "other": a compilation of 21 sources, per-session licence in `manifest.jsonl` |
| `tau2-bench-trajectories` (4 of 31 files) | 37 MB | none stated |
| `sst5` | 1.8 MB | see the dataset card |
| `InjecAgent` (4 test-case files, GitHub) | 2.9 MB | MIT |
| `synthetic_pii_finance_multilingual` (English test) | 2.7 MB | Apache-2.0 |

## What ships

No dataset text ships with the examples. What does:
- **Hashed, labelled rows** in `data/`: the proxy stream (`pi_stream.jsonl`: 2,000 pi tool results in time order)
  and the severity labels (`failure_severity_labels.jsonl`, `failure_severity_keyword_set.jsonl`: 320 + 69 pi
  results labelled by an LLM whose model and prompt were not recorded; `failure_severity_stream_labels.jsonl`: every
  result of the stream, labelled by Claude Opus 5.5 with the severity question's rubric). Each row is a state hash with its labels; the text is rebuilt from the downloaded
  traces and checked against the hash (`hunch_examples.shipped_rows`).
- **The teacher's (Clef's) answers**, hashed: `<example>/teacher.jsonl`, and `data/teacher_stream.jsonl` for the stream.
- **Synthetic labelled data written for these examples:** the decision specs in `decisions/` (matched contrasts and
  hard negatives), the benign twins for InjecAgent (`hunch_examples/injection_fillers.py`), the PII hard negatives
  (`hunch_examples/pii_data.py`) and synthetic coding-tool pairs (`hunch_examples.synthetic`).
- **The question pilot's record:** its candidate questions, selections and dev and test logs
  (`question_pilot/recorded/`).

`hunch_examples/` is the examples' shared package: paths, dataset loaders, the rebuilt rows and the teacher
comparison.

## The examples

Run any one from this folder with `uv run <example>/<example>.py`. Times are from an Apple-silicon laptop.

| example | question | needs | time |
|---|---|---|---|
| [`tool_call_failed`](tool_call_failed/) | Does a compiled `tool_call_failed` decision work on real coding-agent traces, and does it abstain on support-bot outputs? | pi, tau2 | 3 min |
| [`layer_auto`](layer_auto/) | Does `layer: auto` pick the layer that transfers best? | pi, gretel | 6 min |
| [`failure_severity`](failure_severity/) | Can a `kind: levels` decision grade failure severity? | pi | 1 min |
| [`failure_severity_hybrid`](failure_severity_hybrid/) | The harness flag first, then a graded decision on flagged errors | pi | 3 min |
| [`levels_sst5`](levels_sst5/) | `kind: levels` on a public ordinal dataset (SST-5 sentiment) | sst5 | 1 min |
| [`prompt_injection`](prompt_injection/) | A compiled injection decision on unseen InjecAgent tools and attacks, and on real coding traces | InjecAgent, pi | 3 min |
| [`injection_teacher`](injection_teacher/) | Is the teacher worth distilling for injection? | InjecAgent, pi | 15 s |
| [`pii`](pii/) | Can embeddings tell that a text holds PII, which kind, and from which layer? | gretel | 4 min |
| [`proxy_replay`](proxy_replay/) | The proxy policy over 2,000 teacher answers: what graduates, at what agreement and cost | pi | 3 min |
| [`proxy_early_exit`](proxy_early_exit/) | Layer-8 students vs full-pass students, and cascades | pi | 1 min |
| [`proxy_service`](proxy_service/) | The proxy as an HTTP service, end to end | pi | 1 min |
| [`question_pilot`](question_pilot/) | Teach the teacher: reword its question from where it's wrong and EG2 is right | pi, InjecAgent, gretel | minutes offline |
| [`preflight`](preflight/) | Will a question graduate? A check on 150–300 teacher answers | your answers file | 1 min |
| [`teacher_stream`](teacher_stream/) | Rebuild the stream with your own teacher | pi, a teacher endpoint | ~1 s per item |
| [`objects`](objects/) | `hunch.live` on a webcam: name what's held up, out loud, and learn new things from you and from Clef | a webcam | live |
| [`gestures`](gestures/) | `hunch.live` on a webcam: hand signs, faces and one-shot detectors, taught by holding keys | a webcam | live |

## Results

Against ground truth. Each cell is accuracy, or QWK for graded questions, then macro-F1. Each column follows one rule,
the same in every row, fixed before the results were read:
- **EG2:** the example's compiled decision, exactly as its spec is written (its readout, its 60 calibration rows).
- **EG2, 200 labels (†):** a student on EG2 features (logistic regression) trained on the calibration rows plus up
  to 200 labelled rows from the test set's own distribution, cross-fitted in 5 folds, so no row is scored by a model
  trained on it.
- **teacher:** Clef zero-shot, at its default cut.
- **teacher, tuned:** Clef's probabilities recalibrated (logistic regression) on the same 200 labels.
- **EG2 + teacher:** a logistic regression on both models' probabilities, fitted on the same 200 labels.

Folds are grouped where rows are related: by session on the pi traces, by document type on gretel, and by attack text
and filler string on InjecAgent (so no attack or benign filler is on both sides of a fold). The clean pi set has no "yes" to learn from, so it
is scored by the models fitted on the unseen-tools injection labels, which never saw a pi output.

| example | test set (n) | metric | EG2 | EG2, 200 labels † | teacher | teacher, tuned | EG2 + teacher |
|---|---|---|---|---|---|---|---|
| tool_call_failed | pi results, harness flag as truth (580) | accuracy · F1 | 0.822 · 0.822 | 0.962 · 0.962 | 0.802 · 0.795 | 0.888 · 0.888 | 0.964 · 0.964 |
| failure_severity | labelled pi results (260) | QWK · F1 | 0.446 · 0.337 | 0.835 · 0.731 | 0.732 · 0.534 | 0.889 · 0.646 | 0.898 · 0.762 |
| | the pi stream, every distinct result labelled (1,880) | QWK · F1 | 0.235 · 0.270 | 0.696 · 0.530 | 0.657 · 0.453 | 0.736 · 0.450 | 0.766 · 0.542 |
| | unflagged failures in the stream (24) | recall | 71% | 38% | 75% | 62% | 62% |
| levels_sst5 | SST-5 test sentences (2,210) | QWK · F1 | 0.625 · 0.377 | 0.582 · 0.377 | 0.839 · 0.564 | 0.839 · 0.545 | 0.828 · 0.515 |
| prompt_injection | unseen tools and attacks (436) | accuracy · F1 | 0.803 · 0.802 | 0.989 · 0.989 | 0.617 · 0.541 | 0.908 · 0.907 | 0.945 · 0.945 |
| | enhanced wording (496) | accuracy · F1 | 0.841 · 0.841 | 0.990 · 0.990 | 1.000 · 1.000 | 1.000 · 1.000 | 1.000 · 1.000 |
| | seen tools, unseen attacks (558) | accuracy · F1 | 0.821 · 0.815 | 0.944 · 0.944 | 0.608 · 0.536 | 0.882 · 0.881 | 0.953 · 0.953 |
| | real pi outputs, assumed clean (480) | flagged (lower is better) | 5.2% | 46.7% | 0% | 0.2% | 56.2% |
| pii | gretel windows, unseen document types (1,935) | accuracy · F1 | 0.791 · 0.790 | 0.885 · 0.882 | 0.875 · 0.875 | 0.929 · 0.928 | 0.941 · 0.940 |

Paired differences in the main metric, with 95% bootstrap intervals: each of 1,000 resamples (of the fold groups where
there are groups, else of rows) scores both columns on the same rows. A difference counts as a win only when its
interval excludes zero (in bold).

| test set | EG2 † − teacher, tuned | EG2 + teacher − teacher, tuned | EG2 − teacher |
|---|---|---|---|
| tool failure | **+0.074** [+0.045, +0.103] | **+0.076** [+0.050, +0.102] | +0.021 [−0.017, +0.059] |
| severity, 260 labelled (QWK) | −0.054 [−0.131, +0.020] | +0.009 [−0.038, +0.054] | **−0.286** [−0.412, −0.158] |
| severity, stream (QWK) | −0.040 [−0.106, +0.022] | +0.030 [−0.027, +0.082] | **−0.421** [−0.479, −0.355] |
| SST-5 (QWK) | **−0.257** [−0.285, −0.228] | **−0.011** [−0.020, −0.003] | **−0.214** [−0.241, −0.187] |
| injection, unseen tools and attacks | **+0.080** [+0.034, +0.128] | +0.037 [−0.050, +0.113] | **+0.186** [+0.033, +0.325] |
| injection, enhanced wording | −0.010 [−0.031, +0.000] | 0.000 [0.000, 0.000] | **−0.159** [−0.238, −0.096] |
| injection, seen tools | +0.063 [−0.014, +0.140] | +0.072 [−0.004, +0.149] | **+0.213** [+0.126, +0.307] |
| clean pi outputs (share flagged) | **+0.465** [+0.419, +0.517] | **+0.560** [+0.513, +0.602] | **+0.052** [+0.033, +0.073] |
| PII, gretel | **−0.044** [−0.070, −0.021] | +0.012 [−0.003, +0.029] | **−0.085** [−0.135, −0.038] |

On the clean set a positive difference means more false alarms.

- **The compiled decision alone (EG2) against the zero-shot teacher:** better on injection in unseen and seen tool
  formats; level on tool failure; behind on severity, sentiment, enhanced-wording
  injections and gretel PII, and it flags 5% of clean pi outputs where the teacher flags none.
- **Tuning the teacher's cut points on the same labels moves it a long way:** 0.80 → 0.89 on `tool_call_failed`,
  QWK 0.73 → 0.89 on the 260 severity results, 0.62 → 0.91 on unseen injections. Compare the † column with the tuned
  one.
- **With 200 labels, EG2 beats the tuned teacher on 2 of the 9 test sets that have both:** tool failure (where the
  truth is the harness flag, which labels teach directly) and injection on unseen tools. It is level on 4 (both
  severity sets, enhanced wording, seen-tool injections) and behind on 3: SST-5 (sentiment intensity is barely in the
  embedding), gretel PII, and the clean pi outputs, where a student fitted on InjecAgent's formats flags 47% of real
  tool results. EG2 answers in about 1 ms on a CPU instead of a model call.
- **Combining both beats the tuned teacher on 1 of the 9 (tool failure)** and is level on 6; it is behind on SST-5
  (by 0.011) and on the clean set.
- **QWK and F1 disagree on severity.** On the stream, the † student and the tuned teacher are level on QWK, but the
  student's macro-F1 is higher (+0.080 [+0.045, +0.143]): the tuned teacher almost never says `minor`.
- **The harness's error flag is a strong severity baseline.** Mapping flagged to `recoverable` and the rest to `none`
  scores QWK 0.909 on the 260-result set (which is half flagged) and 0.720 on the stream; no model is significantly
  above it on QWK. Its F1 is 0.46 and 0.42, below the † student on both, and it misses every unflagged failure. In the
  stream, where 5% of results are flagged, 111 of the 201 problems are unflagged: mostly empty searches and warnings,
  and 24 silent failures (a clean exit, but the output shows the operation failed). The zero-shot teacher catches 18
  of those 24; EG2 with 200 labels catches 9. [`failure_severity`](failure_severity/) has the details.
- **Tool failure is rare in real traffic.** The `tool_call_failed` test is balanced; in the pi stream 4.7% of results
  fail. Reweighted to that rate, a "failed" answer would be right 19% of the time for the compiled decision, 60% for
  the † student and 64% for the zero-shot teacher.

Brier score and ECE for every model are in each example's `results/<example>_teacher.json`.

The proxy examples train students to *agree with* the teacher, so their results are agreement, not accuracy. Over the
last 500 items of `proxy_service`, the student answers 75% of `output_kind` requests with no teacher call at all
(audited answers and store hits not counted), agreeing with the teacher on 0.967 of them, where always giving the
teacher's most common answer would score 0.310. For `tool_failed`, graduated one-sided, it answers 69%, all of them
"no", at 0.995 agreement, which is exactly what always answering "no" scores on those items: what it saves is teacher
calls, not judgement. Each example's README has its full results.

## The teacher answers

`<example>/teacher.jsonl` holds everything that example needs, one answer per line:

```json
{"question": "ca598a2fb543eb50", "state": "7ba0993146ea25bb", "answer": {"type": "noul", "noul": 0.0882}, "model": "clef-flash"}
```

- `question` is `hunch.question_key`: a hash of the question's type, instructions and criteria (the questions are in
  `hunch_examples.TEACHER_QUESTIONS`). Rewording a question means new answers.
- `state` is `hunch.state_hash`: a hash of the text the teacher read. A tool result is `"{call}\n{output}"`; a plain
  text is itself.
- `model` names the teacher that answered: `clef-flash` (Clef-Flash, 9B, 4-bit MLX, run locally) for every shipped
  answer. Its prompt template and decoding settings are not recorded in these files.

If an example's units have no answer in its file, it copies one from another example's file with the same question
and state; otherwise, with `TEACHER_URL` set, it asks that endpoint (`{"state", "questions"}` in, `{"answers"}` out)
and appends the answer; otherwise it skips the comparison and says how many answers are missing.

```bash
TEACHER_URL=http://localhost:8000/v1/answer uv run tool_call_failed/tool_call_failed.py
```

The proxy reads the same files: `python -m hunch serve --backend offline:data/teacher_stream.jsonl` replays the
teacher without running it, which is how `proxy_service` works.

### Other teachers

Answers from two frontier judges ship next to Clef's, as `teacher.gpt-6-luna.jsonl` and
`teacher.deepseek-v4.1-flash.jsonl` in each example that compares against a teacher, plus Luna's on the stream
(`data/teacher_stream.gpt-6-luna.jsonl`). `TEACHER=<name>` makes an example, or `proxy_service`, use that judge's
files instead, and writes its results to `results/<example>_teacher.<name>.json`:

```bash
TEACHER=gpt-6-luna uv run tool_call_failed/tool_call_failed.py
```

Both were asked through OpenRouter by [`teacher_openrouter.py`](teacher_openrouter.py), which serves any OpenRouter
model as a teacher endpoint. They state their probabilities as JSON, since a hosted model may not return
log-probabilities, and Luna's are almost all exactly 0 or 1. That leaves the tuned teacher and graduation little to
work with. To add another judge (`TEACHER_WORKERS` asks in parallel):

```bash
uv run teacher_openrouter.py <openrouter model> --port 8101
TEACHER=<name> TEACHER_URL=http://localhost:8101/v1/answer TEACHER_WORKERS=64 uv run <example>/<example>.py
```

`layer_auto` and `question_pilot` have no frontier answers yet, and DeepSeek has none on the stream. The Results table
above uses Clef.
