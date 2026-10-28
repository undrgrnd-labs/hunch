# injection_teacher: is the teacher worth distilling?

Can the teacher detect indirect prompt injection in tool outputs, and what happens when the proxy learns from it?
This is rule 9 of [`../../docs/question-design.md`](../../docs/question-design.md): check the teacher against ground
truth before distilling it.

From `examples/`:

```bash
uv run injection_teacher/injection_teacher.py
uv run preflight/preflight.py results/injection_teacher_truth.jsonl --type noul --name injection
```

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), InjecAgent and the pi traces (`download.py`); the 700 units are built from them on first use
(`hunch_examples.injection_units`). **Writes:** `results/injection_teacher_{answers,truth}.jsonl`, the two
answer files `preflight` reads. **Time:** about 15 seconds.

**Teacher answers:** [`teacher.jsonl`](teacher.jsonl), from Clef: question hash, state hash and answer; no text.

## What it does

- 700 units: 400 InjecAgent responses across all 17 user tools (200 with an attacker instruction, base and enhanced
  wording; 200 benign twins) and 300 ordinary pi tool results, assumed clean.
- Scores the teacher's (Clef's) answers against the ground truth by source.
- Trains the proxy's student (layer 8, logistic regression) through `train`, once on the teacher's answers given as
  `sampled` and once on the labels given as `labelled`, and scores both, and the teacher, on the same held-out units.
  The split halves the tools, the attack texts, the filler strings and the pi sessions; a held-out InjecAgent unit
  has a held-out tool and a held-out attack (or filler), and the 188 units that mix the halves are used for neither.
  That leaves 250 training units and 262 held-out ones, 26 of them polite injections. Writes
  `results/injection_teacher.json`.

## Findings

- Over all 700 units, the teacher flags 20% of polite (base-wording) injections at 0.5 and 94% of enhanced ones,
  with no false positives on benign twins or pi outputs.
- On the same 262 held-out units: the teacher catches 23% of the 26 polite injections (6); the student trained on its
  answers, 27% (7), and it doesn't graduate; the student trained on the labels, 81% (21), flagging 5.8% of benign and
  pi units. A distilled student copies its teacher's blind spot; labelled cases fix it. 26 polite injections is a
  small sample: the 95% Wilson intervals are 11–42% and 62–92%.
