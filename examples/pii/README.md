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

What's compared, all on gretel document types the model never saw:

| | what it learns from | result |
|---|---|---|
| probe (layer 8, logistic regression) | thousands of labelled windows, 5-fold, grouped by document type | AUC 0.97 |
| compiled decision (`contains_pii`, the **EG2** column) | 100 natural and 100 matched contrasts picked from half the document types, and 60 labelled windows from the other half | accuracy 0.791, AUC 0.88 |
| student on 200 labels (**EG2, 200 labels †**) | those 60 windows and up to 200 more, from the test half | accuracy 0.885 |
| teacher (Clef), zero-shot | nothing | accuracy 0.875 |
| teacher, tuned on the same 200 | its cut point, from the 200 labels | accuracy 0.929 |
| the two combined | both models' probabilities, on the 200 labels | accuracy 0.941 |

The decision and the models after it are tested on the same 1,935 windows: the half of the document types the
contrasts didn't come from, less the 60 calibration windows.

- A probe trained on labelled gretel windows (logistic regression on layer 8, held out by document type) reaches
  AUC 0.97 for contains-PII. The compiled decision from 200 contrasts reaches AUC 0.88 on gretel's unseen document
  types (accuracy 0.791, below). A probe trained on matched pairs alone scores AUC 0.64 on natural windows: a value
  against the name of its type is a different contrast from PII against no PII, which is why the decision uses
  both kinds of contrast.
- **Which layer:** with LDA or logistic regression, every readout from layer 4 to the final embedding is within 0.02
  AUC of the best (0.95–0.97; layer 8 is best, 0.973). Diff of means falls off at later layers (0.88 at layer 8,
  0.71 at layer 23).
- **Which kind:** the six categories are told apart with accuracy about 0.80 from layer 4 on (LDA, 0.806 at layer 8;
  chance is 0.17). With the values masked as `***` and the field labels kept, the same probe drops to 0.51 at layer
  8: much of what tells the kinds apart is the values themselves, though some of the drop may be the masked windows
  simply looking unlike anything the probe was fitted on.
- Against the teacher (Clef), with 95% bootstrap intervals (resampling document types; differences paired):
  - On gretel's unseen document types the decision (the spec: layer 8, natural and matched contrasts) scores 0.791
    [0.755, 0.816] and the teacher 0.875 [0.835, 0.913]: the teacher is better by 0.085 [0.038, 0.135]. The decision
    escalates 26% of the windows (it is unsure of them) and abstains on 2%; on the rest it is right 85% of the time.
- With additional labels: a student on layer-8 features, trained on the 60 calibration windows plus up
  to 200 labelled windows from other document types (5-fold cross-fitting, grouped by document type), reaches 0.885;
  the teacher recalibrated on the same windows, 0.929, better than the student by 0.044 [0.021, 0.070]; the two
  combined, 0.941, level with the calibrated teacher (+0.012 [−0.003, +0.029]).
