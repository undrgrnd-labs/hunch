# levels_sst5: `kind: levels` on a public ordinal dataset

`kind: levels` on SST-5, five levels of sentence sentiment from very negative to very positive.

From `examples/`:

```bash
uv run levels_sst5/levels_sst5.py
```

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), sst5 (`download.py`). **Writes:** `results/levels_sst5.json`. **Time:** about 1 minute.

**Teacher answers:** [`teacher.jsonl`](teacher.jsonl), from Clef: question hash, state hash and answer; no text.

## What it does

- Contrasts: 200 sets of 5 training sentences, one per level. They aren't matched, so each set is centred, not
  paired.
- Calibration: 60 dev sentences, 12 per level. Test: the 2,210 test sentences.
- Compared with the same contrasts as an unordered 5-way `choice`, with `layer: auto`, and, as a reference for what
  the embedding can do with far more labels, multinomial logistic regression on all 8,544 training sentences.

## Findings

- From 1,000 contrast sentences, the spec's levels (final layer, ridge) reach QWK 0.625 [0.599, 0.649] (95% bootstrap
  interval); with `layer: auto` (final layer, diff of means) QWK 0.70 and within-one 0.89, the same as logistic
  regression trained on all 8,544 labelled sentences. The Results table reports the spec, as for every example.
- The teacher (Clef) zero-shot reaches accuracy 0.592 and QWK 0.839 [0.824, 0.851], ahead of the spec by 0.214
  [0.187, 0.241]. The spec's cascade reaches 0.501 and QWK 0.78 with EG2 answering 44%.
- Additional labels don't close the gap: a student on 60 calibration plus 200 labelled test-distribution sentences
  (5-fold cross-fitting) reaches QWK 0.58, below the spec; the teacher recalibrated on the same labels stays at 0.839,
  and combining the two gives 0.828, slightly below the teacher (−0.011 [−0.021, −0.001]). Sentiment intensity is
  barely in the embedding.
