# layer_auto: does `layer: auto` pick the layer that transfers?

A retrospective check on two decisions. Every candidate layer is compiled explicitly and scored on held-out target
data; then `readout.layer: auto` runs without seeing that data, using grouped cross-validation over the contrasts
(with and without contrast groups) and, for `contains_pii`, a small transfer set. It passes if it lands on the best
layer or on one whose AUC is not significantly lower (a paired bootstrap interval that includes zero).

From `examples/`:

```bash
uv run layer_auto/layer_auto.py
```

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), the pi traces and gretel (`download.py`). **Writes:** `results/layer_auto.json`. **Time:**
about 6 minutes.

**Teacher answers:** [`teacher.jsonl`](teacher.jsonl), from Clef: question hash, state hash and answer; no text.

## What it does

- `tool_call_failed`: 56 synthetic pairs; calibrated on 60 pi results; truth is the 580 other pi results.
- `contains_pii`: 100 natural and 100 matched gretel contrasts; calibrated on 60 gretel windows from unseen
  document types; truth is the other gretel windows of those types. The transfer set is 50 gretel windows of the
  unseen types with their typed twins, kept out of the truth.
- For each pick, a paired bootstrap of the AUC difference between the best layer on the truth and the picked layer
  (resampling results, or documents for PII). A pick is level with the best when the 95% interval includes zero.

## Findings

- Tool failure on pi: layer 20 is best (AUC 0.919). Auto picks the final layer (0.902) without contrast groups and
  layer 16 (0.896) with them. The final layer is level with layer 20 (difference +0.017 [−0.000, +0.036]); layer 16
  is not (+0.023 [+0.005, +0.039]).
- PII: auto picks the final layer every time, with or without groups and with the gretel transfer set. On gretel's
  unseen document types that is the best layer (0.884).

The teacher (Clef) is scored on the same units, for reference: AUC 0.962 on tool failure and 0.971 on gretel.
