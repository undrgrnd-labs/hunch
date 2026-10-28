# failure_severity_hybrid: code first, then a graded decision

Failure severity as a small program instead of one decision: a result the harness didn't flag is `none`; a flagged
one is graded minor, recoverable or blocking by
[`../decisions/failure_severity_graded.yaml`](../decisions/failure_severity_graded.yaml). Compared with single
handles over 5 random calibration and test splits.

From `examples/`:

```bash
uv run failure_severity_hybrid/failure_severity_hybrid.py
```

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), the pi traces (the labels ship as hashes; the text is rebuilt from the download). **Writes:** `results/failure_severity_hybrid.json`. **Time:** about 3 minutes.

**Teacher answers:** [`teacher.jsonl`](teacher.jsonl), from Clef: question hash, state hash and answer; no text.

## What it does

- **hybrid:** the harness flag, then the graded levels handle calibrated on 60 flagged errors.
- **single:** [`../decisions/failure_severity.yaml`](../decisions/failure_severity.yaml) over all four levels,
  calibrated on 60 random rows; with and without its hard negatives, and with layer and method auto.
- **flag only:** not flagged is `none`, flagged is `recoverable`.
- The teacher answers both a 4-level question on every result and the 3-level question on flagged results.
- Each split tests on the roughly 210 rows its calibration didn't use, plus the 69-result keyword set.

## Findings

| method | QWK | accuracy |
|---|---|---|
| hybrid: flag + graded handle | 0.909 | 0.883 |
| flag only | 0.913 | 0.881 |
| single levels handle | 0.33–0.36 | about 0.61 |
| teacher, one 4-level question | 0.725 | 0.737 |
| teacher hybrid: flag + teacher graded | 0.902 | 0.864 |
| cascade: EG2 hybrid, the teacher where it passes up | 0.909 | 0.881 |

The flag alone scores at least as well as the hybrid: almost all of the hybrid's quality comes from the harness flag,
and the graded step adds nothing measurable on top.
