# pii: can embeddings tell that a text holds PII?

Can EmbeddingGemma 2 tell whether a window of text contains a personal data value, which kind, and from which layer?
eg2cpu returns mean-pooled readouts, not per-token states, so this asks *whether* a window holds PII, not where.

![held-out gretel windows streamed through the compiled contains_pii decision, each answer with its time and whether it was right](pii.gif)

*60 held-out gretel windows (document types the decision never saw) through `contains_pii`, live on the CPU: each answer appears after the time it took (median 35 ms) and holds for 100 ms*

From `examples/`:

```bash
uv run pii/pii.py
```

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), gretel (`download.py`). **Writes:** `results/pii_probe.json` and the compiled
`results/contains_pii.{yaml,eg2h}`. **Time:** about 4 minutes.

**Teacher answers:** [`teacher.jsonl`](teacher.jsonl), from Clef: question hash, state hash and answer; no text.

## What it does

- Units are windows of whole lines or sentences (40–320 characters) from gretel's synthetic finance documents; the
  loader is in `hunch_examples/pii_data.py`.
- **natural:** windows with a PII value against windows from the same documents without one.
- **pairs:** the same window with each value replaced by a phrase naming its type ("the customer's email address"):
  topic and wording kept, value gone.
- **choice:** windows with one PII category (person, contact, address, financial, government ID, credential): which
  one, with a masked-value control.
- Probes (diff of means, LDA, logistic regression) on the final embedding and layers 4–23, with gretel folds grouped
  by document type. Then compiled decision handles from contrasts.

## Findings

- A probe trained on labelled gretel windows (logistic regression on layer 8, held out by document type) reaches
  AUC 0.97 for contains-PII. The compiled decision from 200 contrasts reaches 0.88 on gretel's unseen document types.
  A probe trained on matched pairs alone scores 0.64 on natural windows.
- Against the teacher (Clef), with 95% bootstrap intervals (resampling document types; differences paired):
  - On gretel's unseen document types the decision (the spec: layer 8, natural and matched contrasts) scores 0.791
    [0.755, 0.816] and the teacher 0.875 [0.835, 0.913]: the teacher is better by 0.085 [0.038, 0.135].
- With additional labels: a student on layer-8 features, trained on the 60 calibration windows plus up
  to 200 labelled windows from other document types (5-fold cross-fitting, grouped by document type), reaches 0.885;
  the teacher recalibrated on the same windows, 0.929, better than the student by 0.044 [0.021, 0.070]; the two
  combined, 0.941, level with the calibrated teacher (+0.012 [−0.003, +0.029]).
