# proxy_service: the proxy as an HTTP service

The hunch proxy end to end over HTTP, on 2,000 real coding-agent tool results, with the teacher's recorded answers
standing in for the teacher, so the run needs no teacher.

From `examples/`:

```bash
uv run proxy_service/proxy_service.py                           # replay the recorded teacher
uv run proxy_service/proxy_service.py --no-seed                 # without the labelled seed for tool_failed
TEACHER_URL=... uv run proxy_service/proxy_service.py --live    # ask a live teacher instead
```

A live teacher must speak hunch's teacher protocol ([../../hunch/README.md](../../hunch/README.md#the-teacher)).

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), the pi traces, public coding-agent sessions (`uv run download.py real-pi-coding-agent-traces-sessions`;
the stream ships as hashes, and its text is rebuilt from the download). **Writes:** `results/proxy_service.{json,db}`.
**Time:** about 1 minute.

The stream's three questions (`hunch_examples.STREAM_QUESTIONS`):
- `tool_failed` (`noul`): did the tool call itself fail?
- `output_kind` (`choice`): source, test run, build, search, version control, edit, or other;
- `severity` (`score`): none, minor, recoverable or blocking.

## What it does

1. Starts a proxy on a free port (`hunch.serve`), backed by an `OfflineBackend` over the recorded answers in
   [`../data/teacher_stream.jsonl`](../data/teacher_stream.jsonl).
2. Trains `tool_failed` through `POST /v1/train` on the first 150 results, labelled with the agent harness's own error
   flag. The recorded teacher calls 20% of flagged errors "not failed"; labelled cases are how hunch learns past a
   teacher's blind spot.
3. Sends the other 1,850 results through `POST /v1/answer` in order, one request per question, so a question that never
   graduates doesn't send the others to the teacher.
4. Waits for the background audits, then reads each question's examples, graduation and audits from
   `GET /v1/questions`.

## Findings

Over the last 500 results (each a request per question). "Student" counts only answers that made no teacher call:
an audited answer did call the teacher, and a store hit reuses an answer the teacher gave before. Agreement is
measured on the student's answers in the same 500, beside what always giving the teacher's most common answer would
score on them:

| question | student, no teacher call | audited | from the store | forwarded | teacher calls per request | student agreement with the teacher | always the most common answer |
|---|---|---|---|---|---|---|---|
| `output_kind` | 75% | 9% | 3% | 13% | 0.22 | 0.967 | 0.310 |
| `tool_failed` (one-sided) | 69% | 6% | 4% | 21% | 0.27 | 0.995 | 0.995 |
| `severity` | 0% | 0% | 6% | 94% | 0.94 | — | — |

- `output_kind` agrees with the teacher far above its baseline.
- `tool_failed` graduates one-sided: the student answers only a confident "not failed" and forwards the rest, so
  its local answers are all "no", and their agreement equals the always-"no" baseline. What it saves is teacher
  calls; whether it misses failures shows in the audits (84 audits, no misses over the run) and in `proxy_replay`.
- `severity` never graduates; the 6% of its requests about a state seen before are answered from the store.

Agreement is with the teacher, not with ground truth. `proxy_replay` replays the same stream with its own copy of the
policy, all at once rather than one request per question, so its shares differ.
