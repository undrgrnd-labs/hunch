# proxy_replay: how much can the proxy take off the teacher?

Replays hunch's routing policy over a stream of 2,000 real coding-agent tool results, each answered by the teacher on
three questions, and measures how much the students could answer and how often they agree with the teacher. It is a
standalone copy of the policy, so it can try variants quickly; `proxy_service` runs the proxy itself.

From `examples/`:

```bash
uv run tool_call_failed/tool_call_failed.py   # first: compiles the handle the warm-start section reads
uv run proxy_replay/proxy_replay.py
```

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), the pi traces, public coding-agent sessions (`uv run download.py real-pi-coding-agent-traces-sessions`;
the stream, [`../data/pi_stream.jsonl`](../data/pi_stream.jsonl), ships as hashes, and the teacher's answers are in
[`../data/teacher_stream.jsonl`](../data/teacher_stream.jsonl)). Without `results/tool_call_failed.eg2h`, the script
writes its main results and stops before the warm-start section; `tool_call_failed`, which writes it, also needs
tau2 (`uv run download.py tau2-bench-trajectories`). **Writes:** `results/proxy_replay.json`. **Time:** about 3
minutes, plus 3 for `tool_call_failed`.

## What it does

- Questions: `tool_failed` (yes/no), `output_kind` (7 options) and `severity` (4 levels).
- Each question's student learns only from the items it sent to the teacher: a warm-up, the forwarded items and
  the audits.
- A student graduates when its cross-validated agreement with the teacher and its recall on the teacher's less common
  answers both clear their bars. The agreement bar is the *target*, 0.90 by default.
- A request is answered locally only if every question is confident and the state isn't novel; 10% of local answers
  are audited, and a student whose audits fall significantly below the target is demoted.
- Variants: one question per request, a grid over the target and the audit rate, `tool_failed` routed one-sided, and
  a warm start from the compiled `tool_call_failed` handle.

## Findings

The numbers are agreement with the teacher (Clef), not accuracy. Each refit standardizes the features with its own
training items' statistics, so the replay uses nothing about items it hasn't reached.

- `output_kind` graduates at item 149. On its own requests the student answers 92% of the second half locally, with
  kappa 0.94 against the teacher (agreement 0.958, where always giving the most common answer scores 0.350).
  `severity` never graduates.
- With `tool_failed` routed one-sided (only a confident "not failed" is answered locally), `tool_failed` and
  `output_kind` together are answered locally for 72% of the second half's requests, and compute is 213 ms per item
  against 494 for the teacher alone (eg2cpu measured at 31 ms per item here). `tool_failed`'s local answers are all
  "no", so their agreement (0.994) equals the always-"no" baseline; 4 of the teacher's 61 "failed" answers in the
  second half were answered "no" locally.

`proxy_service` reports different local shares on the same stream: it sends one request per question, so a question
that never graduates doesn't hold the others back, and it answers repeated states from the store.
