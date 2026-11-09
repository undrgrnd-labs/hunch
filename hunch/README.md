# hunch

hunch is a proxy: it sits in front of a /v1/systemone model you might already be using to answer questions about textual data (like
"did this tool call fail in this agent trace?", "is this log entry from a hacker?"). It learns to answer the easy questions itself, but way faster,
making your whole system more responsive (and sometimes smarter too!).

- **The teacher** either an LLM judge or a dedicated decision model. This is what hunch learns from.
- **The student** is hunch's own classifier model, one per question: Once cross-validation on the teacher's answers
  shows the student agrees with the teacher often enough, the question *graduates* and the student answers it.
- **A state** what we ask questions about: e.g. a tool call and its output, a message (or really any JSON).

A question has a type:
- `noul`: yes or no (the question protocol's name for it), answered as P(yes);
- `choice`: one of named options, given as `criteria` (option name -> description);
- `score`: one of ordered levels, given as `criteria` (a list, lowest first).

## Quick start

```bash
python -m hunch serve --db proxy.db --backend http://teacher.example/v1/answer   # listens on 127.0.0.1:8765
```

```bash
curl -s localhost:8765/v1/answer -d '{"state": {"call": "bash: ls config/", "output": "ls: config/: No such file or directory"},
  "questions": {"failed": {"type": "noul", "instructions": "Did the tool call fail?"}}}'
```

```json
{"answers": {"failed": {"type": "noul", "noul": 0.97, "answered_by": "backend"}}, "answered_by": "backend", "model": "teacher"}
```

`answered_by` lets you know who got it: `backend` (the teacher), `proxy` (the student) or `cache` (previously answered, reused).

A question with options looks like this:

```json
{"type": "choice", "instructions": "What kind of output is this?",
 "criteria": {"test_run": "Output of running tests", "build": "Output of a build or compile", "other": "Anything else"}}
```

## The teacher

hunch sends a `POST` request to the teacher, `{"state": ..., "questions": {id: question}}`, and expects
`{"answers": {id: answer}}` back, with an answer for every question:
- `noul`: `{"type": "noul", "noul": 0.97}`, P(yes);
- `choice`: `{"type": "choice", "probabilities": {"test_run": 0.9, "build": 0.05, "other": 0.05}}`;
- `score`: `{"type": "score", "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}}`, by level index.

hunch's returns the same response, plus `choice` or `score` and `confidence` fields.

 - `--backend`: is the teacher's URL. 
 - `--backend-header 'Authorization: Bearer …'`: adds headers to the teacher's requests, and
 - `--backend-timeout` (default 30 s) sets a timout.

Teacher problems turn into a 502 for the caller: an error, a timeout, a missing question or an answer hunch can't
parse.

## How it decides

For each request, in order:
1. **The store.** If the store already holds an answer for this exact state and every question, hunch returns it. The
   answer is a label or the teacher's last answer.
2. **The students.** The students answer if all of these are true: every question has graduated, its student is
   confident, and the state is not novel. A novel state is unlike the examples the student learned from.
3. **The teacher.** Otherwise the whole request goes to the teacher. The teacher reads the state once for all
   questions. Every answer becomes a training example.

The store is read by the hash of the state's text, so a stored answer needs no encoding. And when a question in the
request hasn't graduated, or is demoted, the request is bound for the teacher whatever its features say: hunch asks
the teacher at once and encodes the state at the same time, for learning and audits. Until a question graduates, a
request then takes as long as the slower of the two, not both one after the other.

**Graduation.** A refit retrains a question's student and cross-validates it on its examples. The question
graduates in one of two modes:
- **symmetric:** on the states it is confident enough to answer, the student agrees with the teacher at least 90% of
  the time, and on at least 80% of the answers other than the most common one (both the 95% lower bound);
- **one-sided** (`noul` only): the student answers only when it is confident the answer is "no", and passes every
  likely "yes" to the teacher. This needs at most 2% of its "no" answers to be wrong, and at least 85% of the teacher's
  "yes" answers to be passed on. It suits rare, important answers, such as "is this an injection?".

A question refits on a worker thread once it has 100 new examples (`--refit-every`). The system checks the count each
time it records the answers of the teacher. Requests keep the previous fit until the new fit is swapped in. `train`
with `"refit": true` (the default) refits before it responds. With `"refit": false`, its examples count toward the
next check. The refit then waits for the next answer from the teacher. The refits of one question run one at a time.
The fit in use is therefore always from the newest examples. Fits are kept in the store, so a restarted proxy answers
at once.

**Audits.** Graduated questions keep being checked against the teacher:
- 10% of the students' answers (`--audit`) are also sent to the teacher, in the background: the client gets the
  student's answer without waiting. If the teacher is down, the audit fails, not the request.
- Every request that goes to the teacher anyway also checks each student that would have answered it.

A question is demoted while its last 100 audits are significantly below its bar. The bar is: agreement below 90%
(symmetric), or more than 2% of "no" answers wrong (one-sided). While a question is demoted, its requests go to the
teacher. The question answers again when its audits recover. The store keeps the audits, so a demoted question stays
demoted after a restart. A refit that changes its mode starts a new window, because the old audits measured the other
mode. The thresholds and their statistics are in [../docs/glossary.md](../docs/glossary.md#the-proxy-hunch).

Which questions can graduate at all is the subject of [../docs/question-design.md](../docs/question-design.md).

## Training: `POST /v1/train`

You can also give hunch answers directly:

```json
{"question": {"type": "noul", "instructions": "Did the tool call fail?"},
 "cases": [
   {"state": {"call": "bash: ls config/", "output": "ls: config/: No such file or directory"}, "answer": true, "kind": "labelled"},
   {"state": "added 214 packages in 6s", "answer": {"type": "noul", "noul": 0.04}, "kind": "sampled"}],
 "refit": true}
```

- **`sampled`** answers are treated as the teacher's, for example answers collected from its logs. hunch keeps the
  newest 2,000 per question (`--sampled-cap`), together with the answers it forwards, and weights each by its
  confidence.
- **`labelled`** answers are ground truth: benchmark labels, harness flags, human review. hunch keeps the newest
  5,000 of each class per family (`--labelled-cap`); they weigh twice as much as a sampled answer, and win over a
  sampled answer for the same state. They are how a student learns
  past a teacher's blind spot (`../examples/injection_teacher/`).
- **`answer`** can be a label (`true`/`"yes"`/`1` for `noul`, an option name for `choice`, a level index or legend
  entry for `score`), P(yes) as a bare number for `noul` (`0.04`), a list of probabilities in option order
  (`[p_yes]` for `noul`), or an answer object. Probabilities must be in [0, 1]; a request with any invalid answer
  gets 400, and none of its cases are kept.

Each state counts once per question. A new answer for a state replaces the old answer. The response is the status of
the question after the refit.

For each labelled state, hunch also keeps the teacher's answer. This lets hunch measure the teacher against ground
truth. If the teacher has already answered the state, hunch keeps that answer. If not, hunch asks the teacher in the
background. It makes one call per state, which covers every version of the question it knows. `--no-ask-for-labels`
turns those calls off. Answers that arrive later through traffic are still kept. The question's status reports how
many labels have a teacher answer (`labels_with_teacher`).

### Teacher tuning

Those pairs let hunch tune the teacher. An LLM judge often ranks well but puts its cut points in the wrong place. In
the examples, a judge that separated prompt injections with AUC 0.96 scored 0.62 accuracy at its default 0.5. At each
refit, a question version with at least 50 labels that have a teacher answer fits a recalibration of the teacher. The
recalibration is a logistic regression on the log-probabilities of the teacher, with each class weighted equally.
5-fold cross-validation then compares it with the raw teacher, answer by answer. Tuning switches on only if it
significantly improves balanced accuracy, which counts each class equally. It must also not significantly lower plain
accuracy. The balanced test stops the tuning from winning by never predicting a rare class. The plain test stops a
class seen only a few times from carrying the tuning.

When tuning is on, every answer from the teacher is recalibrated. This includes an answer that was just forwarded and
an answer replayed from the store. The original answer of the teacher stays under `untuned`. To see whether tuning is
on, and why, read the status of the question under `report.tuning`.

Students then aim at the tuned answers, in two tiers. At each refit, hunch fits two students:
- **Full:** Learns the answers of the tuned teacher. Once it graduates, it answers.
- **Partial:** Learns the own answers of the teacher, but is graded against the tuned answers. It answers only until
  the full student graduates.

Grading the partial student against tuned answers keeps it to the states where tuning didn't change the answer.
Those are usually most of them: clear successes and clear failures, far from either cut point. States near the
moved cut point go to the teacher.

Audits compare against the tuned answer. Every local answer says which tier answered (`tier`). The question's status
does the same.

Tuning only moves the cut points. It does not make the teacher see things it misses. A student trained on labels with
the features of EG2 can learn past a blind spot. A recalibrated teacher cannot.

### Blind spots

`GET /v1/questions/{key}/blind_spots` (or `proxy.blind_spots(question)`) shows where the teacher disagrees with your
labels. It uses the tuned teacher when tuning is on. It reads only what the store already has, so it makes no teacher
calls:
- **Summary:** the accuracy of the teacher on the labels, raw and as served, a confusion matrix of label against the
  answer of the teacher, and recall for each label. A label that the teacher never predicts shows up here.
- **Spots:** the disagreements, grouped by (label, answer of the teacher) and clustered by EG2 features. The largest
  and most confidently wrong are first. Each spot gives its size, the probability of the teacher for the label, and
  the share where an out-of-fold student agrees with the label. A high share suggests a real blind spot. A low share
  suggests label noise, or a question that the labels and the teacher read differently.
- **What to label next:** for each spot, the number of unlabelled states from traffic that the teacher answered the
  same way within the radius of the spot, and the nearest few. They show whether a spot is a handful of cases or a
  large part of your traffic.

States come with their text when hunch keeps texts (`--store-text`). Otherwise, they come as hashes. The endpoint can
return stored texts. So it needs the training token when a token is set.

On the pi stream, the largest spot is searches that find nothing. The stream uses Clef's severity answers, 400 of our
labels, and the other 1,600 results as traffic. The rubric calls these searches minor. Clef calls them none. Several
hundred traffic results look alike. That is a wording problem, not a model problem. The clearest true blind spot is
GitHub issue text that describes errors. Clef reads it as a failure. The student backs the labels that say it is not
a failure.

## Rewording questions: versions and families

A question's *version* is its type, instructions and criteria, so a reworded question is a new version: it starts with
no sampled answers and no graduation, because the teacher's answers to one wording aren't answers to another. To keep
ground truth across rewordings, name the question's family:

```json
{"type": "noul", "family": "injection", "instructions": "…reworded…"}
```

- Every version in a family uses the labelled examples of the family. A new wording can therefore graduate before it
  has seen a single teacher answer.
- A version trained without a family joins one, with its labels, the first time a question names it.
- Versions in a family must have the same type and options. Otherwise their labels mean different things.
- A request may leave the family out once a version is known.

## Running it

| endpoint | |
|---|---|
| `POST /v1/answer` | a request (above) |
| `POST /v1/train` | training cases (above) |
| `POST /v1/embeddings` | EmbeddingGemma 2 embeddings, as vLLM and OpenAI serve them (below) |
| `GET /v1/questions` | `{"ready", "stats", "latency", "layer", "encoder", "states", "questions": [...]}`: request counts, latency by who answered, and each question's examples, graduation, cross-validation report and audits |
| `GET /healthz` | 200 while the process is up |
| `GET /readyz` | 200 once every question restored from the store has a fit, else 503 |

- **Access:** `--token` (or `$HUNCH_TOKEN`) makes every endpoint but `/healthz` and `/readyz` require
  `Authorization: Bearer <token>`. `--train-token` (or `$HUNCH_TRAIN_TOKEN`) gives `/v1/train` its own token, since
  training writes ground truth. Without a token the server is open, so keep it on `127.0.0.1` (the default `--host`).
- **Limits:** request bodies over `--max-body` (1 MiB) get 413, and a POST without a valid `Content-Length` gets 411
  or 400; the token is checked before the body is read. A client that sends nothing for `--client-timeout` seconds
  (30), idle or mid-request, is disconnected. At most `--max-connections` (64) connections are served at once, each
  on its own thread; one more gets 503.
- **Errors:** an unexpected failure gets 500 with only an `error_id`; the detail is logged to stderr under that id.
- **CPU:** `--threads` caps the threads the encoder uses (the default is every core).
- **Logs:** `--access-log` writes one line per request to stderr: method, path, status, milliseconds, who answered.

**Embeddings.** `/v1/embeddings` takes the request OpenAI's and vLLM's clients send, so they can use the proxy's
encoder directly:

```bash
curl -s localhost:8765/v1/embeddings -d '{"model": "google/embeddinggemma-2", "input": ["bash: git status", "ls: config/: No such file"]}'
# {"id": "embd-…", "object": "list", "created": …, "model": "google/embeddinggemma-2",
#  "data": [{"index": 0, "object": "embedding", "embedding": [0.0123, …]}, …],
#  "usage": {"prompt_tokens": 15, "total_tokens": 15, "completion_tokens": 0}}
```

- Each input gets the model's final embedding: 768 floats with unit norm. These are not the layer-8 features the
  students read.
- `input` is a string or a list of strings; token ids are refused. `model` may be left out, and any other name gets
  404.
- `encoding_format` is `float` or `base64` (little-endian float32).
- `dimensions` may only be 768, since the checkpoint declares no Matryoshka sizes.
- Inputs over 1,024 tokens are truncated, where vLLM would refuse them. `usage` counts the tokens read, `<bos>` and
  `<eos>` included.
- Embeddings run on the encoder's thread, batched with the states waiting to be encoded, and aren't cached.
- Errors on this endpoint use OpenAI's shape: `{"error": {"message", "type", "param", "code"}}`.
- `--max-body` limits how many inputs fit in one request.

**Storage.** `--db` takes a SQLite path, or a Redis URL (`redis://host:port/db?prefix=name`, with
`pip install hunch[redis]`). The store holds:
- each state once, by the hash of its text: its layer-8 features (2 KB) and, with `--store-text`, the text;
- each question's examples (one per state), its latest fit and its last 100 audits;
- the encoder fingerprint: the checkpoint revision, layer, matmul precision and clipping that computed the features.
  A proxy with a different encoder refuses the store; with `--store-text`, `--rebuild` re-encodes the stored states
  and refits every question on the new features.

Several processes can share a Redis store, but each keeps its own students and refit counters. They share examples,
fits and the audits they record. Each process reads the audits only when it starts, so their demotions can differ.

**State text.** The encoder and the teacher read a state as its text: a tool call and its output as `call\noutput`,
followed by any other fields as `name: value` lines, and any other JSON as sorted JSON. The encoder reads at most 1,024
tokens, so a state over 3,000 characters is clipped to its head and tail, keeping the end of a long output, where an
error usually is.

## Performance

Measured on an Apple M5 Pro with the real encoder, on real tool results of about 190 tokens (one question per
request, the teacher replayed with a 500 ms delay):

| | p50 | p95 | p99 |
|---|---|---|---|
| answered by the student, one request at a time | 22 ms | 51 ms | 76 ms |
| answered by the teacher | 531 ms | 579 ms | 583 ms |

Audited answers return as fast as other student answers. The audit happens afterwards.

Concurrent requests share the encoder: one thread encodes whatever is waiting as one batch. Throughput rises with load,
and latency under load is mostly the wait for the current batch:

| concurrent callers | requests/s | p50 | p95 | p99 |
|---|---|---|---|---|
| 1 | 41 | 21 ms | 47 ms | 71 ms |
| 4 | 51 | 83 ms | 122 ms | 137 ms |
| 16 | 73 | 226 ms | 294 ms | 301 ms |
| 32 | 86 | 382 ms | 451 ms | 466 ms |

- **CPU and memory:** a single caller keeps about 7 cores busy while a request is encoded. `--threads` caps that. The
  process holds about 1.5 GB, the model included.
- **Restart:** a restarted proxy is ready at once. It answers from the kept fits.
- **Cost per request** is mostly the encoder pass to layer 8. It grows with the length of the state, up to the
  1,024-token limit. The encoder's own numbers, batched and per hardware, are in
  [../docs/performance.md](../docs/performance.md).

## From Python

```python
from hunch import HTTPBackend, Proxy

proxy = Proxy(backend=HTTPBackend("http://teacher.example/v1/answer"), db="proxy.db")
proxy.answer({"state": {"call": "...", "output": "..."}, "questions": {"failed": {"type": "noul", "instructions": "..."}}})
proxy.train(question, cases)       # as POST /v1/train
proxy.predict(question, states)    # the student's answers for many states, without routing or learning
proxy.wait()                       # until pending refits and background audits are done
```

Custom stores implement `hunch.Store`.

## Live images: `hunch.live`

The proxy answers questions about text states behind a teacher. `hunch.live` answers questions about a stream, from
one EG2 embedding per item. It learns from the items that a person or a teacher labels as it runs. An item is an
image or a text. An image is, for example, a webcam frame (about 110 ms on an Apple M5 Pro's CPU at 70 soft tokens).
A text takes about 12 ms for a short tool result, by the full pass, so it shares the space of the prompts.

```python
from hunch.live import Choice, Detector, Eyes, Watcher

eyes = Eyes(tokens=70)
seen = Choice(eyes, {"nothing": ["a person with empty hands"], "mug": ["a person holding up a mug"]},
              quiet={"nothing"}, empty="nothing", new=lambda name: [f"a person holding up a {name}"])
fire = Detector(eyes, yes=["a photo of fire"], no=["a normal indoor photo"], threshold=-0.04)

x, ms = eyes.see(jpeg)            # or eyes.see("bash('npm test') -> 'Error: Timeout of 2000ms exceeded'")
seen.ask(x)                     # {"name", "p", "state": quiet | sure | unsure, "probs", "zero", ...}
seen.teach(x, "seashell")       # a new answer, with new("seashell")'s prompts; the student refits in the background
seen.forget()                   # back to the answers it started with
watcher = Watcher(seen, teacher)  # teacher(jpeg, names) -> {name or hunch.live.ELSE: probability}
watcher.look(jpeg, x, seen.ask(x))
```

- Every answer starts out zero-shot, from its text prompts. When you label frames, a logistic regression learns from
  them. The trust in it depends on how much a new frame looks like the frames it learned from (measured against the
  frames labelled with the `empty` answer).
- A `Watcher` asks the teacher only when it is worth it. It looks at the newest frame, one at a time. It labels the
  frames it is sure about.
- Answers graduate the same way proxy questions do, except one answer at a time, because new answers keep appearing.
  Whenever the truth for a frame comes in (a confident look from the teacher, or a label from a person), the system
  checks the sure answer of the student against it. An answer graduates when the 95% lower bound on its agreement
  reaches `target` (0.9). This takes at least 20 checks, at most one per second, so in practice about 35 clean ones.
  If the upper bound drops below the target, the answer is demoted. `ask()` tells you if an answer has graduated.
  `report()` shows the checks and bounds for every answer.
- Until an answer graduates, the `Watcher` looks at it whenever it can. After that, it spot-checks the answer at
  random times, about every 10 seconds.
- An `Album` saves labelled frames (images or text) to disk, so they can be taught again after a restart.
- If you work with text and already have a teacher, use the proxy instead. It reads layer 8 and waits for a question
  to graduate before it answers locally. It audits, and it forwards anything it is not sure about. `hunch.live`
  always answers. This is correct when someone watches the stream and teaches it.

[`examples/objects`](../examples/objects/) and [`examples/gestures`](../examples/gestures/) are built on it, and the
module's docstring has the details.

### In the browser

[`web/live.ts`](web/live.ts) is the same `Choice`, `Detector` and `Watcher` in TypeScript, with `Eyes` running EG2's
image path on the GPU through WebGPU ([`web/eg2/`](web/eg2/), about 60 ms a frame on an Apple M5 Pro in Chrome). The
browser runs no text path, so prompts arrive as vectors embedded beforehand with eg2cpu. `web/dist/live.js` is the
bundle; `npm run build` in `web/` makes it again, and `npm run check` type-checks.

```js
import { Choice, Eyes } from './hunch/web/dist/live.js';

const eyes = await Eyes.open({ weights: 'weights', tokens: 70 });   // examples/browser/convert.py writes the weights
const things = new Choice({ nothing: nothingVector, mug: mugVector }, { quiet: ['nothing'], empty: 'nothing' });
const { x, ms } = await eyes.see(video);
things.ask(x); things.teach(x, 'seashell'); things.forget();
```

[`examples/browser`](../examples/browser/) is built on it.

## Tests

```bash
uv run --package hunch pytest -q hunch/tests
```

The tests use a fake encoder, so they don't need the model.
