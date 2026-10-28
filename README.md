# hunch

hunch is a "System One"-style server that answers questions (e.g. "did this tool call fail?" or 
"does this output contain a prompt injection?"), on _CPUs_ in milliseconds. It learns and then graduates from a slower, more expensive "teacher" model and
keeps itself up-to-date by checking in, even after it graduates.

If LLMs are about _thinking hard_, and decision models are about fast _intuition_, hunch is like the fast _reflex_
system you can use for repetitive tasks.

hunch is designed to fit into systems built around tiers of reasoning for answering questions. Frontier LLMs will get
the answer right more often, but are cost and speed prohibitive to run on thousands of records or near-real time. "System One"-style models
like Jev, Clef, or djev are faster, but often larger models with much large memory requirements than hunch and 100s of
milliseconds of latency vs 10s.

![tool calls graded by hunch, clef, luna and the costs (hunch is faster and cheaper, luna the slowest and most expensive)](docs/demo/hunch-demo.gif)

*synthetic tool calls, hunch vs clef vs luna (serialized)*

### But not just that!

hunch is designed for fast decisions on high-volume data. Agent traces are one source, but it can be anything: log files,
user comments, transcripts... anything you like! (multi-modal support in the hunch proxy is coming soon)

hunch also ships with `hunch.live` - a Python and TypeScript library allowing you to build dynamic `choice`-style questions
in real-time on both text and images, learning from both a teacher model and user feedback.

<table border=0>

<tr><td>
  
![the objects demo naming an orange held up to a webcam, with Clef teaching](docs/demo/objects.png) 

</td><td>

![the gestures demo reading a thumbs up and a smile from a webcam](docs/demo/gestures.png)

</td></tr>

</table>

*webcam demos built on `hunch.live`: [objects](examples/objects/) (left) and [gestures](examples/gestures/) (right)*

### Caveats

hunch works for a lot of questions, but not all questions. hunch is fast because it doesn't think, so that means if your
questions need a lot of indirect reasoning, like "how bad is this failure, and who can fix it?", the other judges will be a better fit (we have some [question design guidance for this](docs/question-design.md), with a pre-flight check that tells you which of your questions hunch can take over). 
But don't worry - if hunch never graduates from learning, it keeps passing the questions up to the teacher indefinitely
instead of trying to answer something it isn't capable of.

## Ok, I'm hooked!

hunch is built on top of EmbeddingGemma2 - a model that takes multimodal input (text, images, audio) and spits out a
giant vector representing semantic concepts from that input.

Before we tell you more about hunch, some useful things to know:

- **Structured decisions:** a question with a fixed kind of answer: yes or no (`noul`), one of named options (`choice`), or
  one of ordered levels (`score`). If you're curious, `noul` is Jev's name for a yes/no question (from Ber`noul`li distribution), meaning "probability of yes".
- **Teacher:** the model you would otherwise call for every question, such as a decision model (Clef, Jev, etc) or an LLM-as-judge (frontier model or otherwise).
- **Student:** hunch's small classifier for one question, which reads an embedding of the state from EmbeddingGemma 2.
- **State:** what you're asking the question about.
- **Graduation:** where hunch starts answering on its own (cross-validation on the
  teacher's answers shows it agrees with the teacher often enough)
- **Audit:** after graduation, some of the student's answers also go to the teacher to check up.

hunch trains the student on the teacher's judgements and then takes over once it is confident enough.

## Why

(... and why we built it!)

One _very_ useful area for decision models is agent trace analysis. Agent loops generate hundreds of lines of prose,
code, tool calls. We need to analyze all of these traces, in real-time, for correctness, failure modes, safety or other
metrics.

These feedback systems ask the same small questions over and over, on every single streaming record. The usual answer is an LLM
or decision model judge: hundreds of milliseconds in the best case, and a model call (which adds up when you bring in smarter models) per question.

_Most_ of those questions are easy _most_ of the time. hunch sits in front of your judge (its teacher) and learns from its answers. hunch
only starts answering only once cross-validation shows it agrees with the teacher often enough. It's continually audited
against the teacher afterwards to make sure it's still answering well. If it sees something that's too far outside
of what it learned, it sends it to the teacher to help it tune itself.

You keep your judge in the loop, but you call it less!

- **Cheaper and faster.** hunch can take a lot of the load off the more expensive judge. Depending on the
  difficulty of the question, it might even be nearly 100%.
- **Meets or beats the teacher alone.** hunch tries to distill the teacher's answer to the question, but the combination
  of hunch + teacher often beats a teacher or EG2 alone (this depends on the question - see caveats).
- **No GPU.** EmbeddingGemma 2 is run using a highly-tuned Rust engine for commodity CPUs.

## What's here

- **[hunch](hunch/):** the proxy (including [hunch.live](hunch/src/hunch/live.py)).
- **[eg2cpu](eg2cpu/):** the engine hunch runs on: Google's EmbeddingGemma 2 embedding model, with an optimized CPU
  engine. Can be used on its own if hunch doesn't meet your needs.
- **[examples](examples/):** experiments on public datasets (coding-agent traces, InjecAgent, SST-5, PII) that
  you can test offline.
- **[docs](docs/):** [designing questions hunch can take over](docs/question-design.md),
  [performance](docs/performance.md) and a [glossary](docs/glossary.md).

## Performance

- **hunch:** text decisions in 22 ms (p50) on an Apple M5 Pro, for a tool result of about 190
  tokens. Under load, concurrent requests: 86 requests/s at 32 concurrent callers, p99
  466 ms. See [hunch/README.md](hunch/README.md#performance).
- **eg2cpu:** encoding a 32-token text to layer 8 (an optimized path hunch often takes), takes about 0.6 ms per text in batches on a
  24-core Sapphire Rapids server with AMX. Per-hardware numbers are in [docs/performance.md](docs/performance.md).

## Setup

You need Rust (1.88 or later) and [uv](https://docs.astral.sh/uv/). `uv` and `maturin` are used to build eg2cpu's Python extension.

```bash
uv sync --all-packages
```

Download the model:

```bash
cd examples && uv run download.py embeddinggemma-2 --yes
```

## Tests

```bash
(cd eg2cpu && cargo test --release)
uv run --package eg2cpu pytest -q eg2cpu/tests
uv run --package hunch pytest -q hunch/tests
```

## Licence

hunch, eg2cpu and the examples are licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option. Unless you say otherwise, any contribution you submit for inclusion is licensed the same way.

That covers this repository's code and data files. The examples ship hashes and labels, not dataset text; the
datasets they download, listed in [examples/README.md](examples/README.md), keep their own licences. The EmbeddingGemma
2 checkpoint is Apache-2.0 and is downloaded, not included.
