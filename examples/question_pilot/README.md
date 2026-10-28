# Teaching the teacher: guided question optimization

Script: [question_pilot.py](question_pilot.py). Recorded run: [recorded/](recorded/). The teacher's answers to every
question version, from Clef: [teacher.jsonl](teacher.jsonl) (hashed, so the whole pilot replays offline).

**Question.** Where EG2 is right and the teacher is wrong against ground truth, can those cases tell us how to reword
the teacher's question, so the teacher itself gets better? The recorded pilot covers prompt injection, where the two
models fail differently. The script also has a PII task on gretel windows, with no recorded run.

**Answer.** Yes, by a small margin on a small test. One rewritten question, chosen on dev, scored on held-out test
units:

| task | test units | teacher (Clef), original question | rewritten question | gain (95% interval) |
|---|---|---|---|---|
| injection (inj-a) | 274: unseen tools, attacks and fillers | 0.942 | 0.971 | +0.029 (+0.004 to +0.053) |
| … polite injections caught | 15 | 0% | 53% | |

The test holds only units whose tool, attack text and filler string were all absent from dev, which
  leaves 15 polite and 12 enhanced injections, 95 benign twins and 152 pi results. The gain is all on polite
injections (8 of 15 caught, against none).

No source of units lost more than 2 points on test. EG2 was not changed, and the teacher was not trained: only the
question's wording changed.

## The process

The recorded run's dev and test output, its candidates and its selection are in `recorded/` (`injection_2_dev.log`,
`injection_3_test.log`). Its test log is on the recorded split, before the attack and filler rule of step 1; the
tables in this README are from the current split.

### 1. Units, truth and splits

- **Injection:** the 700 units of `injection_teacher` (`hunch_examples.injection_units`):
  - 400 InjecAgent tool responses: 200 with the dataset's attacker instruction (base and enhanced wording) and 200
    benign twins;
  - 300 pi tool results, which are clean.

  Groups are tools and pi sessions.
- **PII (script only):** 800 gretel windows (400 with PII, 400 without) from every document type, grouped by document
  type. Only units with a cached teacher answer to the original question are used.
- **Splits:** whole groups go to dev or test, half each. All mining, writing and selection used dev only. Test was
  read once, at the end. An injection test unit whose attack text or filler string also appears in a dev unit is left
  out, so test attacks and fillers are unseen as well as test tools. InjecAgent repeats its 62 attacks across tools,
  so this removes most of the injection test.

### 2. EG2's view

EG2 is the proxy's labelled student: logistic regression on layer-8 means, trained on the labels. Its dev
predictions are out-of-fold (grouped 5-fold); for test, it is trained on all of dev.

### 3. Mining the teach and protect sets (dev)

| task | teacher | EG2 | teach (teacher wrong, EG2 right) | protect (teacher right, EG2 wrong) | both wrong |
|---|---|---|---|---|---|
| injection (341) | 0.912 | 0.950 | 27, all polite (base) injections | 14 (8 benign imperatives, 5 enhanced injections) | 3 |

The script clusters the teach set (k-means on EG2 features). For each unit, it also finds a minimal pair: the
nearest dev unit with the other label that the teacher gets right. The clusters reduced to one failure kind: polite
first-person requests ("my account", "my front door") in content written by a third party: a profile bio, a tweet, a
calendar event title, an SMS. They ask for actions only the agent could take with the user's tools: payments,
unlocking a door, changing account settings, or sending data to an address. The minimal pairs are the same responses
with an ordinary bio or tweet in the slot. The original question says content "addressing people" is not an
injection, and the teacher read these requests as exactly that.

### 4. Candidate questions

There were three candidates, each written from the failure kind (never from test units) with a stated reason. They are
recorded in `question_pilot/recorded/injection_candidates.json`:

| candidate | what it changes |
|---|---|
| inj-a third-party actions | Keeps the original; adds that content in the output is written by someone other than the user, and that a request only the agent could carry out with the user's tools or accounts is an injection even when polite and in the first person |
| inj-b who could act | Rewrites the question around who could carry the request out |
| inj-c v0 plus one clause | The smallest edit: one added sentence |

### 5. Selection on dev

A candidate qualifies if both of these hold:
- the 95% lower bound of its accuracy gain over the original is above zero (a paired bootstrap over groups);
- no source of units loses more than 5 points.

The best qualifying candidate is selected.

| candidate | dev accuracy | gain (95%) | teach fixed | protect kept | qualifies |
|---|---|---|---|---|---|
| injection v0 | 0.912 | | | | |
| **inj-a** | **0.953** | +0.041 (+0.015 to +0.064) | 48% | 100% | **yes, selected** |
| inj-b | 0.871 | −0.041 | 33% | 86% | no: enhanced injections fell 24 points |
| inj-c | 0.933 | +0.021 (−0.003 to +0.041) | 30% | 93% | no |

Two things the recorded selection showed:
- **Wording reframes are risky.** inj-b, which rewrote the question around "who could act", lost more than it
  fixed: the teacher stopped flagging some blatant ("IGNORE ALL PREVIOUS INSTRUCTIONS") injections. The protect set and
  the per-source check exist to catch this.
- **Small edits can be too small.** inj-c moved in the right direction, but its interval included zero.

### 6. Held-out test (read once)

| task | test units | v0 | selected | gain (95%) | teach fixed | protect kept | worst source |
|---|---|---|---|---|---|---|---|
| injection | 274 | 0.942 | 0.971 | +0.029 (+0.004 to +0.053) | 53% | 100% | +0.000 |

The whole gain is on polite injections (+0.029: their share of the test units times their accuracy change).

## What it means for the combined system

| task (test) | EG2 alone | teacher v0 | cascade with v0 | teacher rewritten | cascade with rewritten |
|---|---|---|---|---|---|
| injection | 0.854 | 0.942 | 0.861 | **0.971** | 0.861 |

The cascade lets EG2 answer outside `|p − 0.5| < 0.2` and sends the rest to the teacher. EG2 here is trained on dev
only, so on unseen tools, attacks and fillers it is weak, and it answers 97% of units anyway. The rewritten teacher
alone beats every cascade.
So once the teacher improves, the routing should be re-tuned to defer to the teacher more: the cascade should know
where each model is reliable, not just how confident EG2 is.

The loop this pilot runs by hand:
1. EG2 finds the teacher's blind spots against ground truth.
2. The question is rewritten for those failure kinds, with dev selection and a single test read.
3. The better teacher gives the proxy better sampled answers.
4. The router is re-tuned.

## Caveats

- **One round, one task, small test.** InjecAgent's attacks are homogeneous, and the test holds 15 polite injections.
- **The candidates were written by an LLM (Claude) reading the dev teach set.** That is the intended "a strong LLM
  proposes edits" step. The guard against overfitting is the protocol: few candidates, grouped splits, a bootstrap
  bound, per-source checks, and one test read. It is not the writer's restraint.
- **Benign twins were written by hand** for the injection example (`hunch_examples/injection_fillers.py`).
  Their false-positive rates depend on that pool.
- **Rewording creates a new question version.** Sampled answers from the old wording don't carry over, which is
  right: they answered a different question. Labelled cases do carry over when the question names its `family`, so
  deploying `injection` with the rewritten wording keeps all its ground truth.
- **Licences.** The logs in `question_pilot/recorded/` quote InjecAgent units (MIT).

## Reproduce

From `examples/`:

```bash
uv run question_pilot/question_pilot.py mine injection
```

```bash
TEACHER_URL=http://localhost:8000/v1/answer uv run question_pilot/question_pilot.py eval injection question_pilot/recorded/injection_candidates.json
```

```bash
TEACHER_URL=http://localhost:8000/v1/answer uv run question_pilot/question_pilot.py test injection question_pilot/recorded/injection_candidates.json
```

Every teacher answer is already in `teacher.jsonl`, so the recorded candidates re-run without `TEACHER_URL`. Only new
candidates need the teacher. For the PII task, `mine pii` runs offline; write candidates to a JSON file in the format
of `injection_candidates.json`, then `eval pii` and `test pii` ask the teacher.

The pi units are grouped by session, and their ids are hashed in the shipped data, so a re-run splits injection
groups differently from the recorded run, and `eval injection` on the new dev split needs the teacher for 67 units.
`test` uses this run's dev selection when there is one, else the recorded one.
