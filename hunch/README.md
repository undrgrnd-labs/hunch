# hunch

hunch sits in front of a /v1/systemone model you already use to answer questions about textual data, such as 
"did this tool call fail in this agent trace?", and learns to answer the easy ones itself.

- **The teacher** is that model: an LLM judge or a dedicated decision model behind HTTP. hunch sends it every
  question it can't answer yet and keeps the answers.
- **The student** is hunch's own classifier, one per question: logistic regression on layer 8 of EmbeddingGemma 2
  (run on CPU by `eg2cpu`). Once cross-validation on the teacher's answers shows the student agrees with the teacher
  often enough, the question *graduates* and the student answers it.
- **A state** is what a question is asked about: a tool call and its output, a message, or any JSON.

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

`answered_by` says who answered: `backend` (the teacher), `proxy` (the student) or `cache` (the store, for a state it
already has an answer for). A question with options looks like this:

```json
{"type": "choice", "instructions": "What kind of output is this?",
 "criteria": {"test_run": "Output of running tests", "build": "Output of a build or compile", "other": "Anything else"}}
```

## The teacher

hunch POSTs the teacher the request it received, `{"state": ..., "questions": {id: question}}`, and expects
`{"answers": {id: answer}}` back, with an answer for every question:
- `noul`: `{"type": "noul", "noul": 0.97}`, P(yes);
- `choice`: `{"type": "choice", "probabilities": {"test_run": 0.9, "build": 0.05, "other": 0.05}}`;
- `score`: `{"type": "score", "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}}`, by level index.

hunch's response is the same, plus `choice` or `score` and `confidence` fields.

`--backend` is the teacher's URL, or `offline:<file or folder>` to replay recorded answers (as the examples do; a
missing answer is then an error, unless `--fallback <url>` names a teacher to ask, and `--record <file>` saves its
answers). `--backend-header 'Authorization: Bearer …'` adds headers to the teacher's requests, and
`--backend-timeout` (default 30 s) bounds them. A teacher that fails, times out, leaves a question out, or gives a
malformed answer (probabilities that aren't finite numbers in [0, 1], one per option) gets the client a 502, and
nothing from that response is learned.

## How it decides

For each request, in order:
1. **The store.** If the store already holds an answer for this exact state and every question (a label, or the
   teacher's last answer), hunch returns it.
2. **The students.** If every question has graduated, its student is confident, and the state isn't novel (unlike the
   examples the student learned from), the students answer.
3. **The teacher.** Otherwise the whole request goes to the teacher, which reads the state once for all questions,
   and every answer becomes a training example.

**Graduation.** A refit retrains a question's student and cross-validates it on its examples. The question
graduates in one of two modes:
- **symmetric:** on the states it is confident enough to answer, the student agrees with the teacher at least 90% of
  the time, and on at least 80% of the answers other than the most common one (both the 95% lower bound);
- **one-sided** (`noul` only): the student answers only when it is confident the answer is "no", and passes every
  likely "yes" to the teacher. This needs at most 2% of its "no" answers to be wrong, and at least 85% of the teacher's
  "yes" answers to be passed on. It suits rare, important answers, such as "is this an injection?".

A question refits on a worker thread once it has 100 new examples (`--refit-every`), a count checked each time the
teacher's answers are recorded; requests keep using the previous fit until the new one is swapped in. `train` with
`"refit": true` (the default) refits before it responds; with `"refit": false` its examples count toward the next
check, so the refit waits for the next answer from the teacher. One question's refits run one at a time, so the fit
in use is always from the newest examples. Fits are kept in the store, so a restarted proxy answers at once.

**Audits.** Graduated questions keep being checked against the teacher:
- 10% of the students' answers (`--audit`) are also sent to the teacher, in the background: the client gets the
  student's answer without waiting. If the teacher is down, the audit fails, not the request.
- Every request that goes to the teacher anyway also checks each student that would have answered it.

A question is demoted, and its requests go to the teacher, while its last 100 audits are significantly below its bar:
agreement below 90% (symmetric), or more than 2% of "no" answers wrong (one-sided). It answers again once its audits
recover. The audits are kept in the store, so a demoted question stays demoted after a restart; a refit that changes
its mode starts a new window, since the old audits measured the other mode. The thresholds and their statistics are in [../docs/glossary.md](../docs/glossary.md#the-proxy-hunch).

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

Each state counts once per question: a new answer for it replaces the old one. The response is the question's status
after the refit.

For each labelled state, hunch also keeps the teacher's answer, so the teacher can be measured against ground truth.
If the teacher has already answered the state, that answer is kept. If not, hunch asks the teacher in the
background, one call per state covering every version of the question it knows. `--no-ask-for-labels` turns those
calls off; answers that arrive later through traffic are still kept. The question's status reports how many labels
have a teacher answer (`labels_with_teacher`).

### Teacher tuning

Those pairs let hunch tune the teacher. An LLM judge often ranks well but puts its cut points in the wrong place:
in the examples, a judge that separated prompt injections with AUC 0.96 scored 0.62 accuracy at its default 0.5.
At each refit, a question version with at least 50 labels that have a teacher answer fits a recalibration of the
teacher: a logistic regression on the teacher's log-probabilities, with each class weighted equally. 5-fold
cross-validation then compares it with the raw teacher, answer by answer. Tuning switches on only if it
significantly improves balanced accuracy, counting each class equally, without significantly lowering plain
accuracy. The balanced test stops it winning by never predicting a rare class; the plain one stops a class seen
only a few times from carrying it.

While tuning is on, every answer that comes from the teacher, whether forwarded or replayed from the store, is
recalibrated, and the teacher's own answer is kept under `untuned`. The question's status says whether tuning is
on and why, under `report.tuning`.

Students then aim at the tuned answers, in two tiers. At each refit hunch fits two students:
- **Full:** learns the tuned teacher's answers. Once it graduates, it answers.
- **Partial:** learns the teacher's own answers, but is graded against the tuned ones. It answers only until the
  full student graduates.

Grading the partial student against tuned answers keeps it to the states where tuning didn't change the answer.
Those are usually most of them: clear successes and clear failures, far from either cut point. States near the
moved cut point go to the teacher.

Audits compare against the tuned answer, and every local answer, like the question's status, says which tier
answered (`tier`).

Tuning moves cut points; it can't make the teacher see what it misses. A labelled student on EG2's features can
learn past a blind spot; a recalibrated teacher can't.

### Blind spots

`GET /v1/questions/{key}/blind_spots` (or `proxy.blind_spots(question)`) shows where the teacher, tuned when tuning
is on, disagrees with your labels. It reads only what the store already has, so it makes no teacher calls:
- **Summary:** the teacher's accuracy on the labels, raw and as served, a confusion matrix of label against the
  teacher's answer, and recall per label. A label the teacher never predicts shows up here.
- **Spots:** the disagreements, grouped by (label, teacher's answer) and clustered by EG2 features, largest and most
  confidently wrong first. Each spot gives its size, the teacher's probability for the label, and the share where an
  out-of-fold student agrees with the label. A high share suggests a real blind spot; a low one suggests label noise
  or a question the labels and the teacher read differently.
- **What to label next:** for each spot, how many unlabelled states from traffic the teacher answered the same way
  within the spot's radius, and the nearest few. They show whether a spot is a handful of cases or a large part of
  your traffic.

States come with their text when hunch keeps texts (`--store-text`); otherwise as hashes. Because it can return
stored texts, the endpoint needs the training token when one is set.

On the pi stream (Clef's severity answers, 400 of our labels, the other 1,600 results as traffic), the largest spot is
searches that find nothing: the rubric calls them minor, Clef calls them none, and several hundred traffic results look
alike. That is a wording problem, not a model one. The clearest true blind spot is GitHub issue text that describes
errors: Clef reads it as a failure, and the student backs the labels that say it isn't.

## Rewording questions: versions and families

A question's *version* is its type, instructions and criteria, so a reworded question is a new version: it starts with
no sampled answers and no graduation, because the teacher's answers to one wording aren't answers to another. To keep
ground truth across rewordings, name the question's family:

```json
{"type": "noul", "family": "injection", "instructions": "…reworded…"}
```

- Every version in a family uses the family's labelled examples, so a new wording can graduate before it has seen a
  single teacher answer.
- A version trained without a family joins one, with its labels, the first time a question names it.
- Versions in a family must have the same type and options; otherwise their labels would mean different things.
- Requests may leave the family out once a version is known.

## Running it

| endpoint | |
|---|---|
| `POST /v1/answer` | a request (above) |
| `POST /v1/train` | training cases (above) |
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

**Storage.** `--db` takes a SQLite path, or a Redis URL (`redis://host:port/db?prefix=name`, with
`pip install hunch[redis]`). The store holds:
- each state once, by the hash of its text: its layer-8 features (2 KB) and, with `--store-text`, the text;
- each question's examples (one per state), its latest fit and its last 100 audits;
- the encoder fingerprint: the checkpoint revision, layer, matmul precision and clipping that computed the features.
  A proxy with a different encoder refuses the store; with `--store-text`, `--rebuild` re-encodes the stored states
  and refits every question on the new features.

Several processes can share a Redis store, but each keeps its own students and refit counters: they share examples,
fits and the audits they record, but each reads the audits only when it starts, so their demotions can differ.

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

Audited answers return as fast as other student answers; the audit happens afterwards.

Concurrent requests share the encoder: one thread encodes whatever is waiting as one batch. Throughput rises with load,
and latency under load is mostly the wait for the current batch:

| concurrent callers | requests/s | p50 | p95 | p99 |
|---|---|---|---|---|
| 1 | 41 | 21 ms | 47 ms | 71 ms |
| 4 | 51 | 83 ms | 122 ms | 137 ms |
| 16 | 73 | 226 ms | 294 ms | 301 ms |
| 32 | 86 | 382 ms | 451 ms | 466 ms |

- **CPU and memory:** a single caller keeps about 7 cores busy while a request is encoded; `--threads` caps that. The
  process holds about 1.5 GB, the model included.
- **Restart:** a restarted proxy is ready at once and answers from the kept fits.
- **Cost per request** is mostly the encoder pass to layer 8, so it grows with the state's length up to the 1,024-token
  limit. The encoder's own numbers, batched and per hardware, are in [../docs/performance.md](../docs/performance.md).

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
one EG2 embedding per item, and learns from the items a person or a teacher labels as it runs. An item is an image,
such as a webcam frame (about 110 ms on an Apple M5 Pro's CPU at 70 soft tokens), or a text (about 12 ms for a short
tool result, by the full pass, so it shares the prompts' space).

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

- Every answer starts zero-shot from its text prompts. Labelled frames train a logistic regression, which answers as
  far as a frame resembles the frames it was taught, measured from the `empty` answer's frames.
- A `Watcher` asks the teacher about the newest frame only when that's worth it, one look at a time, and labels the
  frames it's sure about.
- Answers graduate as the proxy's questions do, but one answer at a time, since answers grow while it runs. A
  sure answer is checked whenever the truth arrives for the same frame: a teacher's sure look, or a label someone
  gives. An answer graduates once the 95% lower bound of its agreement reaches `target` (0.9; at least 20 checks,
  at most one per second, so about 35 clean ones). It is demoted while the upper bound is below that. `ask()` says
  whether the answer has graduated, and `report()` gives every answer's checks and bounds.
- The `Watcher` looks whenever it can at answers that haven't graduated. It audits graduated ones at random moments,
  every 10 s on average.
- An `Album` keeps labelled frames (images or texts) on disk to teach again after a restart.
- For text behind a teacher, the proxy is the better fit: it reads layer 8, graduates a question before answering it
  locally, audits, and forwards whatever it isn't sure of. `hunch.live` always answers, so it suits streams someone
  is watching and teaching.

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
