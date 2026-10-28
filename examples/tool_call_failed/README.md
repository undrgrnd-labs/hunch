# tool_call_failed: a compiled decision on real traces

Does the compiled `tool_call_failed` decision ([`../decisions/tool_call_failed.yaml`](../decisions/tool_call_failed.yaml):
56 synthetic contrast pairs and 4 hard negatives) work on real coding-agent tool results, and does it abstain on a
domain it wasn't built for?

From `examples/`:

```bash
uv run tool_call_failed/tool_call_failed.py
```

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), the pi traces and tau2 (`download.py`). **Writes:** `results/tool_call_failed.eg2h`, which `proxy_replay`
and `failure_severity` read. **Time:** about 3 minutes.

**Teacher answers:** [`teacher.jsonl`](teacher.jsonl), from Clef: question hash, state hash and answer; no text.

## What it does

- Calibrates on 60 pi tool results, with the harness's `isError` flag as the label, and tests on the other 580.
- Reports AUC, calibrated accuracy, how much the uncertainty band escalates and how accurate the rest is, and how
  often the decision abstains in domain and on tau2 customer-support tool outputs (a different domain).
- Ablations: the layer-4 readout and diff of means. Then the cost of applying several handles at once.
- Compares with the teacher on the same 580 results, and on 200 tau2 outputs.

## Findings

| readout | AUC | accuracy | accuracy when answered | escalated | abstains on tau2 |
|---|---|---|---|---|---|
| spec: final layer, LDA (auto) | 0.902 | 0.822 | 0.881 | 21% | 1% |
| layer 4, diff of means | 0.891 | 0.797 | 0.888 | 23% | 82% |
| final layer, diff of means | 0.899 | 0.838 | 0.902 | 30% | 1% |

The spec's readout separates failures well on real traces, but only the layer-4 readout recognises support-bot
outputs as out of domain.

Against the harness flag, on the 580 results (95% bootstrap intervals; differences are paired, on the same resampled
results):

| | accuracy | recall | precision at 4.7% failures | Brier | ECE |
|---|---|---|---|---|---|
| decision (spec) | 0.822 [0.791, 0.852] | 0.817 | 0.19 | 0.254 | 0.027 |
| teacher (Clef), zero-shot | 0.802 [0.769, 0.835] | 0.619 | 0.64 | 0.297 | 0.079 |
| teacher, calibrated on 200 labels | 0.888 [0.862, 0.912] | 0.851 | 0.36 | 0.171 | 0.033 |
| EG2 + 200 labels | 0.962 [0.947, 0.978] | 0.955 | 0.60 | 0.070 | 0.014 |
| teacher + EG2, combined on 200 labels | 0.964 [0.948, 0.979] | 0.972 | 0.52 | 0.052 | 0.014 |

- The decision and the zero-shot teacher are level: +0.021 [−0.017, +0.059]. The cascade (EG2 where it answers,
  the teacher where it escalates or abstains) scores 0.852 with EG2 answering 78% of results.
- With 200 more labelled results (5-fold cross-fitting), the student beats the calibrated teacher by +0.074
  [+0.045, +0.103]; combining the two adds nothing over the student (+0.002 [−0.014, +0.016]). The truth here is
  the flag itself, which those labels teach directly.
- The test is balanced: half of each tool's results failed. In the pi stream 4.7% do, and at that rate a
  false-positive rate that looks small costs most of the precision: the decision's "failed" would be right 19% of the
  time, the labelled student's 60% and the zero-shot teacher's 64%. The precision column reweights each model's
  false-positive rate on this test to that prevalence.
