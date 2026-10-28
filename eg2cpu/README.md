# eg2cpu

CPU inference for the EmbeddingGemma 2 encoder (`google/embeddinggemma-2`), in Rust, with no GPU or PyTorch: text,
and images through the checkpoint's vision tower. It uses bf16 weights and fp32 math. Kernels are picked at runtime:
AVX-512, AVX2+FMA, NEON, or a scalar fallback. On Apple M4 and later the matmuls run on the SME matrix unit, still in
fp32. On Intel CPUs with AMX, `EG2_BF16=amx` runs the matmuls and attention in bf16 on AMX tiles instead. On other Arm
CPUs with BF16 (Neoverse V2 / Graviton 4), `EG2_BF16=bfmmla` runs the weight matmuls in bf16 with `BFMMLA`.

The text tower is loaded up front. The vision tower is loaded by the first image; the audio tower is never read. A
forward pass returns:
- the mean-pooled, L2-normalized 768-d embedding;
- the mean-pooled 512-d residual stream after each of the 24 layers.

```rust
let model = eg2cpu::Model::load_default()?;   // $EG2_MODEL_DIR, else the HF cache snapshot
let out = model.embed("bash: git status")?;   // tokenizes (bos/eos, truncate at 1024) and runs the encoder
out.embedding;                                 // Vec<f32>, 768, unit norm
out.layer_means;                               // Vec<Vec<f32>>, 24 x 512
let outs = model.embed_batch(&texts)?;         // texts run concurrently, unpadded
```

```bash
cargo run --release --bin eg2 -- embed "hello world"
cargo run --release --bin eg2 -- bench               # single texts and batches
cargo run --release --bin eg2 -- exit 8              # early exits vs the full pass
cargo run --release --bin eg2 -- callers 8 60 32     # 8 concurrent callers, 60 requests each, 32 tokens
cargo run --release --bin eg2 -- image photo.jpg     # image latency at 70, 140 and 280 soft tokens
```

The checkpoint, `google/embeddinggemma-2`, is Apache-2.0 and not gated on the Hugging Face Hub (as of October 2026),
so no login is needed; `../examples/download.py embeddinggemma-2` downloads it.

## Images

```rust
let out = model.embed_image(&std::fs::read("photo.jpg")?)?;  // PNG, JPEG or WebP; 280 soft tokens
let out = model.embed_image_with(&bytes, 70)?;                 // a smaller budget: about 5x faster
out.embedding;                                                 // Vec<f32>, 768, in the same space as text
```

An image goes through the same pipeline as `AutoProcessor` plus `EmbeddingGemma2Model` in transformers:
1. **Preprocessing.** The image is decoded, EXIF-rotated and converted to RGB. It is then resized, keeping its aspect
   ratio, to the largest size with at most `9 x max_soft_tokens` patches of 16 x 16 px and sides divisible by 48. Small
   images are scaled up. The resize reproduces torchvision's antialiased bicubic on uint8 bit for bit.
2. **Vision tower.** A 16-layer, 768-wide ViT with 2-D RoPE encodes the patches. 3 x 3 average pooling turns them into
   soft tokens, which are RMS-normalized and projected to 512 dimensions.
3. **Text tower.** It encodes `<bos> <boi> soft tokens <eoi> <eos>`. The output is the usual mean-pooled, normalized
   embedding, plus the 24 layer means.

Images and text share one embedding space. In the test fixtures, each of four photos is nearest to its own caption,
at both 70 and 280 soft tokens: cosine 0.67-0.73, against 0.48-0.61 for the other captions.

The soft-token budget (70, 140, 280, 560 or 1120) sets the resolution, and the vision tower's cost grows with it.
`embed_image` uses the processor's default of 280.

On an Apple M5 Pro with SME, one image takes about 98 ms at 70 soft tokens and 496 ms at the default 280, most of it
in the vision tower; without SME (`EG2_SME=off`) 248 ms and 1.2 s. The first image also loads the vision tower, about
170 ms. Per-step timings are in [../docs/performance.md](../docs/performance.md).

**Not supported:**
- audio and video;
- several images, or text and images interleaved, in one input;
- formats other than PNG, JPEG and WebP (no GIF, HEIC or TIFF).

## Python

The `python` cargo feature builds a PyO3 extension (via maturin) that needs only numpy, not torch or transformers.
The forward pass releases the GIL.

```python
import eg2cpu

model = eg2cpu.Model()                       # or Model("/path/to/snapshot"); same lookup as load_default
emb, means = model.embed("bash: git status") # np.float32 arrays, shapes (768,) and (24, 512)
pairs = model.embed_batch(texts)             # list of (emb, means), texts run concurrently on all cores
emb, means = model.embed_image("photo.jpg")  # a path or bytes; max_soft_tokens=70 for speed
eg2cpu.isa()                                 # "neon", "avx2", "avx512" or "scalar"
```

In a uv workspace or script, depend on `eg2cpu` with a path source; uv builds the wheel with maturin on first use
and rebuilds it when the Rust or Python sources change (`cache-keys` in `pyproject.toml`).

### Early exit

`model.embed_batch_partial(texts, upto=k)` runs each text through layers 0..k only and returns `Partial` objects
(`.layer_means` is [k+1, 512]). `model.resume_batch(partials)` finishes them without recomputing those layers,
giving the same output as `embed_batch`, bit for bit (`tests/concurrency.rs`). A layer-8 exit costs about a third
of a full pass. A decision that reads layer ≤ k only needs the exit at k; the hunch proxy's students read layer 8.

### Batching

Embedding texts one at a time leaves most cores idle. `eg2cpu.Embedder` batches in the background: a worker thread
embeds every text queued while the previous batch ran, all together, and concurrent requests for the same text share
one forward pass. Results (24 layer means, then the embedding) are cached by text, and calls are thread-safe.

A plain `for` loop can't be batched, because each call waits for its result before the next one is made. Two ways
to give the batcher work:

```python
embed = eg2cpu.Embedder()                    # or Embedder(path=..., model=..., max_batch=256)
rows = embed.map(features, items)            # like [features(it) for it in items], with fn run on a thread pool
embed.prefetch(texts)                        # when the texts are known up front: one batch, later calls hit the cache
```

`fn` must be safe to run concurrently, which read-only feature code is.

## Concurrency

**Inside one forward pass,** the 24 layers run in order. Within each layer the work is split across rayon's pool:
matmuls into tiles, attention by (head, 32-query block), and norms, RoPE and GELU by rows.

**Requests from outside the pool go through a process-wide scheduler** (`src/sched.rs`):
- **A lone request runs at once,** on a pool sized to its length (`EG2_POOLS`). Below a few hundred tokens the
  barriers between ops cost more than extra threads gain.
- **Requests that arrive while one is running queue per caller.** A waiting caller becomes the dispatcher. It waits
  up to `EG2_BATCH_WINDOW_US` (only while fewer than a full batch are queued), then runs `max_in_flight()` workers
  on the global pool. Each worker takes the next job round-robin across callers and publishes the result as soon as
  it's done.
- **One global bound.** At most `max_in_flight()` forward passes are in flight in the whole process. This covers
  single calls, `forward_batch`, `embed_batch` and `Model::apply` from any number of threads alike. A big batch from
  one caller delays another caller's request by about one pass, not the whole batch.
- **Workers linger** for `EG2_LINGER_US` when the queue is empty. Callers that resubmit as soon as their result
  arrives would otherwise find the workers gone each round, and throughput collapses once there are as many callers
  as threads.
- **Calls from inside the pool** (nested use) run directly. `forward_batch_with` runs a batch directly with an
  explicit bound, bypassing the scheduler.

Images also bypass the request scheduler: each call runs at once, on a pool of the performance cores plus two
(`EG2_VISION_THREADS`), so `EG2_MAX_IN_FLIGHT` doesn't bound images.

`tests/concurrency.rs` checks that concurrent single and batch callers get results bit-identical to direct passes,
and that a bad request panics only its own caller.

## Memory

Once loaded, the text encoder's bf16 weights take 261 MB and the tokenizer about 136 MB. The 256 MB embedding table
stays memory-mapped, so only the rows of tokens actually seen are read. The vision tower adds about 300 MB of packed
weights, loaded on the first image; its 31 MB position table stays memory-mapped. Activations add about 30 MB per
image at 70 soft tokens and 130 MB at 280. Peak footprint is 419 MB for text only, 762 MB after one image at 70 soft
tokens, and 850 MB at 280. A forward pass in flight costs about 30 MB
at 512 tokens; `EG2_MAX_IN_FLIGHT` bounds how many are in flight. Measured footprints are in
[../docs/performance.md](../docs/performance.md).

## Decisions: compiled contrastive readouts

> Writing a new question, for a handle or for a proxy that learns from a teacher? Read
> [../docs/question-design.md](../docs/question-design.md) first. [../docs/glossary.md](../docs/glossary.md)
> defines the methods and statistical terms used below.

A decision spec (YAML) states a question about a span of a trace and the matched contrasts that define it.
`eg2cpu.decisions.compile` turns the spec into a **handle**, a small safetensors file holding:
- one direction in the encoder's space (one per option for `choice`);
- the constants to standardize and calibrate the readout;
- a whitening matrix for the abstain check;
- the escalation rule and prompt.

Handles are applied in Rust. Each distinct unit text is encoded once, so applying 1 or 3 handles to the same
spans costs one encoder pass per text; each handle adds a dot product per direction and, when it can abstain, a
dim × dim whitening product (dim is the readout's width).

```python
import eg2cpu
from eg2cpu import decisions

model = eg2cpu.Model()
d = decisions.compile("tool_call_failed.yaml", model, calibration_rows)             # rows: {"span": {...}, "label": ...}
d.save("tool_call_failed.eg2h"); d.report                                           # AUCs, accuracy, hard negatives
h = eg2cpu.Handle("tool_call_failed.eg2h")
verdicts = model.apply([h, other_handle], spans)   # spans: dicts of str; result is [span][handle]
# {"decision": "tool_call_failed", "status": "answer" | "escalate" | "abstain" | "not_applicable",
#  "answer": "yes", "p": 0.94, "value": None, "z": [...], "distance": 31.2, "prompt": None}
decisions.escalate(verdict, h, "http://127.0.0.1:8080/v1/chat/completions")        # resolve an "escalate"
```

```bash
eg2cpu-decisions compile tool_call_failed.yaml --calibration cal.jsonl -o tcf.eg2h
eg2cpu-decisions apply tcf.eg2h --spans spans.jsonl
```

Three example specs are in [`../examples/decisions/`](../examples/decisions/): `tool_call_failed` (boolean),
`failure_severity` (levels) and `failure_severity_graded` (levels, for flagged errors only). The examples compile
and test them on real traces.

**Spec fields:**
- `kind`: `boolean`, `choice` (needs `options`), `score`, or `levels` (needs `levels`, lowest first).
- `unit`: `tool_result` (`{call}\n{output}`), `tool_call`, `text`, or `{template, varies}`. It decides what
  gets embedded. Contrasts and calibration spans are rendered through the same template.
- `applies_when`: a gate over span fields, using `==`, `!=`, `in [...]`, `not in`, `contains`, `and`, `or`,
  `not` and parentheses. Spans that fail the gate get `not_applicable`. On the right, `span.<field>` is that
  field's value (`span.expected == span.actual`) and other words are text; `true` and `false` match in any case,
  so a span field `"True"` passes `span.is_error == true`.
- `encoder: google/embeddinggemma-2@<rev>`: compile records the loaded snapshot's revision, and `apply` refuses
  a handle compiled against a different checkpoint.
- `readout`:
  - `layer`: `final` (normalized embedding) or `0..23` (mean-pooled residual after that layer);
  - `method`: `diff_of_means` or `lda_shrinkage` (Ledoit–Wolf), or for score `ridge` or `diff_of_means`, or
    `auto`.

  Compile fits **every** method and reports them side by side in `report["methods"]`. For each one the report
  gives the hard negatives it misreads and the calibration AUC (or accuracy, or correlation) with a 95%
  interval, plus the escalation rate. Calibration scores are never used to choose a method: at n = 60 the
  intervals span about ±0.08 AUC, so picking the higher number mostly picks noise, and it would also inflate the
  numbers reported.

  `auto` follows a fixed rule. It uses the default (`diff_of_means`) unless that misreads a hard negative and
  the other method doesn't; failing that, it picks the method that misreads the fewest. The handle records the
  chosen method, the requested one and the reason.
- `readout.layer: auto` (optional `readout.layers`, default `final, 4, 8, 12, 16, 20, 23`) compiles every
  candidate layer on a shared embedding cache. One forward pass yields all layers, so nothing is encoded twice. It
  then picks by a fixed rule:
  1. **Guard:** drop layers whose calibration interval lies wholly below the best layer's. The 60-item sample can
     rule a bad layer out but can't choose between close ones.
  2. **Rank:** order the rest by the `transfer` set's score when there is one, otherwise by grouped
     cross-validation on the contrasts. Whole groups are held out: a contrast's optional `group` (for example
     a document type or template), or else the contrast itself.
  3. **Tie-break:** keep `final` if it's within one standard error of the best.

  The report has the per-layer table (score, its standard error, calibration interval, CV and transfer
  metrics), the reason, and a note that the chosen score is selected, so it's optimistic. Every compile also
  reports `contrast_cv`, plus `transfer` when a transfer set is given.
- `transfer`: optional labelled rows from a second domain, given as `compile(..., transfer=)` or
  `transfer.data` (jsonl, same format as calibration). They're only scored, never fitted.
- `contrasts` (matched sets, so only the property differs):
  - boolean: `{call, yes, no}`;
  - choice: `{call, <option>: ...}`, with at least two options per set;
  - score: `{call, levels: [{output, value}]}`;
  - levels: `{call, <level>: ...}`, with at least two levels per set.

  Unquoted YAML `yes:` / `no:` keys, which YAML parses as booleans, are handled.
- `hard_negatives`: text or field dicts that must read "no". For choice and levels give a `label`, for score a
  `value`. Score hard negatives anchor the axis but aren't checked (a value has no wrong side), and the report
  gives `null` for their violations.
  Each one is checked **held out**: the method is refit without it, then it is read. Hard negatives are also
  training points, and in 768 dimensions a method can carve nearly any training point onto its labelled side.
  For example, a copy of a "yes" span listed as a hard negative reads "no" in-sample under LDA. The report gives
  both counts.
- `hard_negative_check`: `warn` (default), `error` or `off`. This is what compile does when the chosen method
  misreads a hard negative held out; the message names the misread examples and any method that reads them all.
- `calibration`: required, since rankings transfer but cut-offs don't. The method must match the kind:
  - boolean: `platt` (Platt scaling on standardized scores);
  - choice: `multinomial`;
  - score: `linear`;
  - levels: `ordinal`.

  Rows come from `compile(..., calibration=)` or `calibration.data` (jsonl); `calibration.data` and
  `transfer.data` resolve relative to the spec's folder. `allow_uncalibrated=True` calibrates on the contrasts
  instead and marks the handle. A choice sample needs rows of every option. Labels, options, levels and contrast
  keys are all compared as text: unquoted YAML `no` is the label `"no"`, and a level `3` is the label `"3"` (not
  an index). Metrics that can't be computed, such as an AUC on one class, are `null` in the report.
- `abstain.mahalanobis_quantile`: the distance threshold is that quantile of **held-out** distances, because
  in-sample distances with few rows in 768 dimensions make everything look far away. Folds hold out whole
  groups: a contrast set, or the calibration rows that share every field but the unit's varying one (for
  `tool_result`, every row with the same call and other fields).
- `monotonicity_check`: `warn` (default), `error` or `off`. For `levels`, what compile does when the held-out mean
  readout of the levels isn't in order.
- `escalate.when`:
  - boolean: `|p - c| < w`;
  - score: `|value - c| < w`;
  - choice: `p_max < t` or `margin < t`;
  - levels: `|value - c| < w` (on the expected level), `confidence < t`, `p_max < t` or `margin < t`.

  A score verdict can read `escalate`, but `decisions.escalate` resolves only boolean, choice and levels decisions.

  `prompt` is a template. It may name `{unit}` (the embedded text) and the unit template's fields, which every
  span the decision reads has; compile refuses any other field. Without `{unit}`, the unit text comes first and
  the prompt follows it, so the escalation model always sees what the encoder saw. `decisions.escalate` asks
  an OpenAI-compatible endpoint for a single token (yes/no, or an option letter). With `compile(..., llm=URL)`
  a boolean decision's LLM answers are also Platt-calibrated on the same sample.

## Settings

| variable | default | what |
|---|---|---|
| `EG2_MODEL_DIR` | the HF cache snapshot | where the checkpoint is |
| `EG2_ISA` | detected | `scalar`, `avx2`, `avx512` or `neon`: caps the ISA, to test the fallback paths |
| `EG2_BF16` | `off` | `amx`: bf16 matmuls on AMX tiles. `dot`: AVX-512 BF16 dot products (slower than fp32 on current CPUs). `bfmmla`: bf16 weight matmuls with Arm `BFMMLA`, where SME is not in use (slower than fp32 NEON on Apple silicon, see [performance](../docs/performance.md)) |
| `EG2_BF16_ATTN` | `all` | with `EG2_BF16=amx`: which attention products run on AMX (`all`, `pv` or `off`) |
| `EG2_SME` | on where available | `off`: NEON matmuls on CPUs with SME (Apple M4 and later) |
| `EG2_POOLS` | with SME `128:4,320:8,640:10`; else from the core count | thread pools for single texts by length (tokens:threads); longer texts use the whole pool. Without SME, texts up to 40 tokens get half the performance cores and up to 160 all of them, at least 8 and 12 threads: `40:8,160:12` on an M5 Pro with `EG2_SME=off`, `40:8,160:16` (so 160 tokens use the whole pool) on 16 Neoverse V2 cores |
| `EG2_VISION_THREADS` | with SME, performance cores + 2; else all | threads for the vision tower |
| `EG2_MAX_IN_FLIGHT` | thread count | forward passes in flight at once, process-wide; lower it to trade speed for memory |
| `EG2_BATCH_WINDOW_US` | 1000 | how long a waiting caller gathers concurrent requests into one dispatch |
| `EG2_LINGER_US` | 2000 | how long an idle dispatch worker waits for the next request before leaving |
| `EG2_TRACE` | off | `1`: `eg2 bench` prints the wall time of each op in the forward pass |

`RAYON_NUM_THREADS` sets the pool size and turns the per-length pool sizing off.

## Correctness

`tests/reference.rs` checks token ids, all 24 layer means, and the final embedding against PyTorch fp32 CPU
outputs. The cases are 4, 18, 36, 687 (inside the 512 sliding window) and 1024 (truncated) tokens. The largest
difference is about 6e-7 on the embedding and 1.4e-6 relative on layer means, and cosine is 1.0000000. With
`EG2_BF16=amx` or `bfmmla` the activations are rounded to bf16 too, and the embeddings agree to cosine ≥ 0.99999.

`tests/image_reference.rs` does the same for images, against PyTorch fp32 through `AutoProcessor`: the resized
pixels, the soft tokens, the layer means and the embedding, for four photos and a synthetic RGBA gradient at 280 and
70 soft tokens. It also checks the text-image similarities of four captions.
- PNG images: the resized pixels are identical; soft tokens are within 5e-6 relative; embeddings have cosine
  ≥ 0.9999994.
- JPEG: `jpeg-decoder` and PIL's libjpeg-turbo disagree on about 5% of decoded values, by a few levels, which gives
  cosine 0.99988 on the test photo.

```bash
cargo test --release
uv run scripts/make_reference.py tests/fixtures/reference.json   # regenerate the fixtures (needs torch)
uv run scripts/make_image_reference.py tests/fixtures            # regenerate the image fixtures
uv run scripts/check_python.py                                   # the Python binding against an inline torch reference
uv run --package eg2cpu pytest -q eg2cpu/tests                   # the decisions compiler (from the repository root)
```

## Layout

- `src/simd.rs`: the `Simd` trait and its ISA implementations. Kernels are written once and instantiated per ISA
  inside `#[target_feature]` functions:
  - the packed bf16 GEMM, with a broadcast or lane-FMA micro-kernel over weights repacked into column panels at load;
  - the AMX GEMM over pair-packed weights;
  - the Arm `BFMMLA` GEMM over weights packed in k groups of four;
  - SME matmuls (`src/sme.rs`): an inline-assembly strip kernel that accumulates fp32 outer products in the ZA tiles,
    used for the packed GEMM and the vision tower's attention;
  - a dot-form GEMM for attention;
  - SIMD exp, GELU-tanh and softmax.
- `src/model.rs`: config, safetensors loading and the forward pass:
  - projection-only per-layer embeddings;
  - q/k/v RMS norms with attention scaling 1.0;
  - GQA with per-layer head dims;
  - bidirectional sliding window, |q - k| <= 512;
  - final norm, mean pool, 512 -> 768 projection.
- `src/vision.rs`: image decoding, the bit-exact resize, the vision tower (fused q/k/v and gate/up matmuls, 2-D
  RoPE), pooling and the projection into the text tower.
- `src/sched.rs`: the process-wide request scheduler.
- `src/decision.rs`: decision handles (loading, `applies_when` gates, unit templates, readout, calibration,
  abstain, escalate) and `Model::apply`. `python/eg2cpu/decisions.py` is the spec compiler.
- `src/python.rs` (`python` feature) and `python/eg2cpu/`: the PyO3 module and the `Embedder` wrapper.
- `src/bin/eg2.rs`: the `eg2` command-line tool (embed and benchmarks, `image` for image latency).
- `scripts/`: reference fixtures for text and images, the Python binding check, and a PyTorch benchmark for
  comparison.
