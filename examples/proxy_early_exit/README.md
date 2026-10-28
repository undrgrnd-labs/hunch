# proxy_early_exit: which layers do the students need?

Which encoder layers do the proxy's students need, and can easy items stop at an early layer?

From `examples/`:

```bash
uv run proxy_early_exit/proxy_early_exit.py
```

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), the pi traces (the stream ships as hashes). Encoder cost per item comes from
[`early_exit_timing.json`](early_exit_timing.json): eg2cpu's batched exits after layers 4, 8 and 12, the full pass,
and resuming from layer 8. **Writes:** `results/proxy_early_exit.json`. **Time:** about 1 minute.

## What it does

- The same stream, teacher answers and policy as `proxy_replay`, with `tool_failed` routed one-sided and `severity`
  left out.
- Each question's student on different features, each costed at the exit it needs: final + layer 8, final only, or
  layer 4, 8 or 12 alone.
- A cascade: layer-8 students first; when they aren't confident, resume the pass to the end and use full-feature
  students; still not confident, the teacher.

## Findings

Layer-8 students answer 77% locally, 2.64× cheaper than the teacher alone, against 1.71× for final-layer students.
The cascade answers 81% locally (74% at layer 8, 6% after resuming), at 2.70×.
