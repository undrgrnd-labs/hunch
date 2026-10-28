# teacher_stream: rebuild the stream with your own teacher

Builds the stream the proxy examples replay and asks a teacher about every item: 2,000 real pi tool results in time
order, each answered once on the three stream questions (`hunch_examples.STREAM_QUESTIONS`: `tool_failed`,
`output_kind`, `severity`).

From `examples/`:

```bash
TEACHER_URL=http://localhost:8000/v1/answer uv run teacher_stream/teacher_stream.py
```

**Needs:** the pi traces and a teacher endpoint that takes `{"state", "questions"}` and returns `{"answers"}`.
**Writes:** `results/pi_stream.jsonl` and `results/teacher/stream.jsonl`; copy them over
[`../data/pi_stream.jsonl`](../data/pi_stream.jsonl) and [`../data/teacher_stream.jsonl`](../data/teacher_stream.jsonl)
to replay your teacher. **Time:** about a second per item, depending on the teacher; it resumes where it stopped.

The stream takes pi sessions in time order, up to 8 tool results per session, and clips outputs to 1,200 characters.
