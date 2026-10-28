# prompt_injection: indirect prompt injection in tool outputs

Can a compiled decision tell an attacker instruction hidden in a tool response from benign content in the same slot,
on tools and attacks it has never seen?

![held-out InjecAgent tool outputs streamed through the compiled injection decision, each answer with its time and whether it was right](prompt_injection.gif)

*60 held-out InjecAgent tool outputs (unseen tools and attacks, with benign twins) through `injection_in_tool_output`, live on the CPU: each answer appears after the time it took (median 49 ms) and holds for 100 ms*

From `examples/`:

```bash
uv run prompt_injection/prompt_injection.py
```

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), InjecAgent and the pi traces (`download.py`). **Writes:** `results/prompt_injection.json` and the compiled
`results/injection_in_tool_output.{yaml,eg2h}`. **Time:** about 3 minutes.

**Teacher answers:** [`teacher.jsonl`](teacher.jsonl), from Clef: question hash, state hash and answer; no text.

## What it does

- **Data:** InjecAgent's 17 user tools (each a response template with one slot: a review, a note, an email body, a
  bio ...) and 62 attacker instructions, in base and "enhanced" wording. Benign twins fill the same slot with text
  from `hunch_examples/injection_fillers.py`, about half of it imperative but addressed to people ("Please fix the
  trackpad, Dell!"), which is the real boundary.
- **Splits, so nothing is memorized:** attacker instructions are split in half, one half for contrasts and
  calibration and the other for test; contrasts come from 9 tools and the main test is on the other 8; benign
  fillers are split in half the same way; the contrasts use base attacks only, and enhanced attacks are tested
  separately. Calibration (60 items) uses the unseen tools with the contrast half's attacks and fillers, so no attack
  text or filler string is in both calibration and test.
- **False positives:** 480 real pi tool outputs and every pi read of an AGENTS.md, SKILL.md or CLAUDE.md
  (instructions to the agent from the user's own files). The 480 are balanced by the harness's error flag, so half
  are error outputs; they are assumed clean, and no one checked them for injections.
- **Localization:** on flagged responses, each field value is deleted in turn; the injected field should cause the
  biggest drop.
- **Additional labels:** a student on the decision's layer, trained on the 60 calibration items plus up to 200
  labelled items of the same test set from other folds (5-fold cross-fitting), alone and combined with the teacher.
  Folds are grouped so that an attack text, or a filler string, is never on both sides of a fold. The clean sets
  have no "yes" to learn a cut point from, so they are scored by the models fitted on the unseen-tools set's labels:
  none of them saw a pi output.

## Findings

Accuracy, or the share flagged for the clean sets, with 95% bootstrap intervals (resampling the cross-fitting
groups; items for the clean sets):

| test set | decision | decision → teacher | EG2 + 200 labels | teacher (Clef) | teacher, calibrated on 200 labels | teacher + EG2, combined on 200 labels |
|---|---|---|---|---|---|---|
| unseen tools and attacks (436) | 0.803 [0.71, 0.88] | 0.865 | 0.989 [0.96, 1.00] | 0.617 | 0.908 [0.86, 0.95] | 0.945 |
| enhanced wording (496) | 0.841 | 0.923 | 0.990 | 1.000 | 1.000 | 1.000 |
| seen tools, unseen attacks (558) | 0.821 | 0.729 | 0.944 [0.88, 0.99] | 0.608 | 0.882 [0.83, 0.93] | 0.953 |
| real pi outputs, assumed clean (480) | 5.2% | 0% | 46.7% | 0% | 0.2% | 56.2% |
| AGENTS/SKILL.md reads (60) | 1.7% | 0% | 38.3% | 0% | 25.0% | 55.0% |

- The decision (layer and method auto, which picks layer 20) reaches AUC 0.90 on unseen tools and 0.999 on unseen
  attacks in seen tool formats. It abstains on nearly every pi output, and knockout finds the injected field 84% of
  the time.
- With 200 labels of the same test distribution, the student beats the calibrated teacher on unseen tools by +0.080
  [+0.034, +0.128]; on seen tools the gap (+0.063 [−0.014, +0.140]) is not significant, and on enhanced wording the
  teacher is perfect.
- Those students learn InjecAgent's response formats, not injection in general: applied to real pi outputs, which
  they never saw, the student flags 47% and the combined model 56%. The zero-shot teacher and the compiled decision
  (which abstains outside its contrasts' domain) flag almost none. A student for real traffic needs labels from
  that traffic.
