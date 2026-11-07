# eg2cpu performance

Measured with `eg2 bench`, `eg2 exit` and `eg2 callers` (`eg2cpu/src/bin/eg2.rs`). Absolute numbers vary by about
15% between cloud hosts of the same type, so each comparison below is within one machine.

## Apple M5 Pro (6 P + 12 E cores, NEON and SME)

| | eg2cpu, SME (default) | eg2cpu, NEON (`EG2_SME=off`) | PyTorch 2.14 fp32 CPU |
|---|---|---|---|
| 1 text × 16 tokens | 7.4 ms | 16 ms | 20 ms |
| 1 text × 128 tokens | 30 ms | 67 ms | 63 ms |
| 1 text × 512 tokens | 126 ms | 229 ms | 169 ms |
| 1 text × 1,024 tokens | 263 ms | 429 ms | 498 ms |
| 256 texts × 32 tokens | 9,200 tokens/s | 4,315 tokens/s | 2,484 tokens/s |
| 256 texts × 128 tokens | 10,000 tokens/s | 4,107 tokens/s | 3,499 tokens/s |

The NEON GEMM reaches 90–95 GFLOP/s per P-core, about 70% of the core's fp32 FMA rate (16 G `FMLA`/s, 128 GFLOP/s,
with 16 independent chains). The M5's SME matrix unit, which
PyTorch's sgemm also reaches through Accelerate, does fp32 outer products at about 1.5 TFLOP/s from one core and 3
TFLOP/s for the chip; eg2cpu drives it directly (`src/sme.rs`), in fp32, so results stay within rounding of the
reference (cosine ≥ 0.9999998). Its matmuls run at 2–2.7 TFLOP/s on the vision tower's shapes. The matrix unit is
shared by the cores of a cluster, so with SME fewer threads per text pay off (`EG2_POOLS`).

**Images** (`eg2 image`), one at a time, ms per image:

| soft tokens | patches | SME | NEON (`EG2_SME=off`) |
|---|---|---|---|
| 70 | 540 | 98 | 248 |
| 140 | 1,134 | 213 | |
| 280 (default) | 2,340 | 496 | 1,196 |
| 560 | 4,788 | 1,342 | |

At 70 soft tokens: preprocessing 1 ms (a 300 × 200 PNG; a 12 MP JPEG takes 26 ms to decode and resize), vision tower
77 ms, text tower 17–20 ms. The vision tower runs on 8 threads: on more, matmul tasks land on the efficiency cores'
slower matrix unit and straggle.

The early-exit, concurrent-caller and memory tables below are for NEON (`EG2_SME=off`).

**Pools for single texts on NEON** (`EG2_SME=off`). Without SME the default pools come from the performance-core
count (`hw.perflevel0.physicalcpu`, 6 here) with floors of 8 and 12 threads: `40:8,160:12`. Against
`40:4,80:8,160:12`, alternating runs:

| | `40:4,80:8,160:12` | default (`40:8,160:12`) |
|---|---|---|
| 1 text × 16 tokens | 16.6–18.4 ms | 16.4–17.1 ms |
| 1 text × 32 tokens | 32.1–35.5 ms | 25.6–26.8 ms |
| 1 text × 128 tokens | 66.8–68.9 ms | 66.4–69.1 ms |
| 1 text × 512 tokens | 215–226 ms | 216–226 ms |
| 256 texts × 32 tokens | 3,820–4,080 tokens/s | 3,580–4,180 tokens/s |
| 256 texts × 128 tokens | 3,120–4,150 tokens/s | 3,820–4,480 tokens/s |

Texts of 48 and 64 tokens run 2–6% faster on 12 threads than on 8 threads. 8 threads suit texts of up to 40 tokens,
even with only 6 P-cores. With 4 threads, a 32-token text takes twice as long as a 16-token text.

**`EG2_BF16=bfmmla`** (Arm `BFMMLA`) applies only with `EG2_SME=off` here, since SME takes the matmuls when present.
The M5 Pro reports `bf16`, but a P-core issues 2.1 G `BFMMLA`/s (68 GFLOP/s), half its fp32 `FMLA` rate, so the bf16
path is slower than NEON fp32. The kernel runs at that instruction rate:

| | NEON fp32 (`EG2_SME=off`) | `EG2_SME=off EG2_BF16=bfmmla` |
|---|---|---|
| matmul, one P-core | 90–93 GFLOP/s | 65–68 GFLOP/s |
| matmul, 6 threads | 410–460 GFLOP/s | 300–357 GFLOP/s |
| matmul, 18 threads | 550–1,030 GFLOP/s | 300–683 GFLOP/s |
| 1 text × 16 tokens | 16.8–17.0 ms | 22.5–23.0 ms |
| 1 text × 32 tokens | 26.0–26.2 ms | 35.3–37.2 ms |
| 1 text × 128 tokens | 68.8–69.0 ms | 96.5–98.0 ms |
| 1 text × 512 tokens | 220–222 ms | 312–323 ms |
| 256 texts × 32 tokens | 3,810–3,920 tokens/s | 2,520–2,790 tokens/s |
| 256 texts × 128 tokens | 3,760–4,150 tokens/s | 2,710–2,930 tokens/s |
| layer-8 exit, 256 × 32 tokens | 3.2 ms/text (10,000 tokens/s) | 4.4–4.5 ms/text (7,100–7,300 tokens/s) |
| layer-8 exit, 256 × 128 tokens | 11.5–12.1 ms/text (10,600–11,100 tokens/s) | 16.0–16.9 ms/text (7,600–8,000 tokens/s) |
| lowest embedding cosine vs PyTorch | 0.9999999 | 0.9999955 |
| worst layer-mean relative error | 1.6e-6 | 4.2e-3 |

**Early exit**, batched, at about 300-token tool results, in ms per item:

| exit after layer 4 | layer 8 | layer 12 | full pass | resume 8 → end |
|---|---|---|---|---|
| 12 | 24 | 38 | 70 | 44 |

**Concurrent callers**, each sending single texts (tokens/s; p95 latency):

| callers × tokens | throughput | p95 |
|---|---|---|
| 1 × 32 | 1,000 tokens/s | 33 ms |
| 16 × 32 | 3,413 tokens/s | 174 ms |
| 32 × 32 | 4,160 tokens/s | 266 ms |
| 1 × 128 | 1,878 tokens/s | 70 ms |
| 16 × 128 | 3,704 tokens/s | 580 ms |
| 32 × 128 | 4,102 tokens/s | 1,093 ms |
| 16 × 512 | 3,273 tokens/s | 2,784 ms |

Concurrent requests merge into batches and reach batch throughput (a single `forward_batch` of 512 × 128 tokens runs
at 4,176 tokens/s). Under mixed load, 4 callers sending 32-token requests while another thread runs a 512 × 128-token
batch, the single requests take p50 155 ms and p95 471 ms, and the batch finishes as fast as it does alone.

**Memory** (physical footprint):

| | footprint |
|---|---|
| loaded, idle | 315 MB |
| one short text, load included | 317 MB peak |
| one 1,024-token text | 371 MB peak |
| 256 texts × 128 tokens (`forward_batch`) | 466 MB |
| 256 texts × 512 tokens | 870 MB |
| `apply`, 256 spans × ~120 tokens | 470 MB |
| 32 callers × 128 tokens | 452 MB |

`EG2_MAX_IN_FLIGHT` trades speed for memory:

| `EG2_MAX_IN_FLIGHT` (256 × 512 tokens) | 4 | 8 | 12 | 18 (default here) |
|---|---|---|---|---|
| peak footprint | 578 MB | 639 MB | 720 MB | 863 MB |
| time | 41.3 s | 40.6 s | 40.1 s | 36.9 s |

## Intel Sapphire Rapids with AMX (AWS c7i.12xlarge, 24 cores)

Xeon Platinum 8488C, 24 cores and 48 threads, 48 MB of L2. The default is fp32 math on bf16 weights, exact against
the reference. `EG2_BF16=amx` multiplies in bf16 on AMX tiles instead, for the matmuls and both attention products.

| | fp32 (default) | `EG2_BF16=amx` | speedup |
|---|---|---|---|
| 1 text, 16 tokens | 17.9 ms | 15.4 ms | 1.16× |
| 1 text, 32 tokens | 31.0 ms | 21.0 ms | 1.47× |
| 1 text, 128 tokens | 95.6 ms | 57.6 ms | 1.66× |
| 1 text, 512 tokens | 267 ms | 228 ms | 1.17× |
| 1 text, 1,024 tokens | 532 ms | 376 ms | 1.42× |
| 256 texts × 32 tokens | 8,857 tokens/s | 17,385 tokens/s | 1.96× |
| 256 texts × 128 tokens | 8,776 tokens/s | 20,663 tokens/s | 2.35× |
| matmul, one core | 162 GFLOP/s | 460–575 GFLOP/s | |

On a faster host of the same type, AMX batches reached 27,500 tokens/s.

**Early exit**, batches of 256, ms per text:

| | fp32, ~32 tokens | AMX, ~32 tokens | fp32, ~127 tokens | AMX, ~127 tokens |
|---|---|---|---|---|
| layer 4 | 0.73 | 0.33 | 3.00 | 1.05 |
| layer 8 | 1.34 | 0.62 | 5.54 | 1.88 |
| layer 12 | 1.96 | 0.90 | 8.09 | 2.72 |
| full pass | 3.62 | 1.69 | 14.94 | 5.19 |

**Accuracy against PyTorch fp32** (`tests/reference.rs`, 5 texts of 4–1,024 tokens):

| mode | lowest embedding cosine | worst layer-mean relative error |
|---|---|---|
| fp32 (default) | 0.9999999 | 1.6e-6 |
| `EG2_BF16=dot` | 0.9999947 | 3.5e-3 |
| `EG2_BF16=amx` | 0.9999928 | 6.0e-3 |

Every example in `examples/` was also run in both modes. Of 766 rates (AUC, accuracy, QWK, agreement, local share),
the median changed by 0.0001 and the 99th percentile by 0.036; every layer choice, method choice and graduation mode
came out the same.

**Where single-text time goes.** A 128-token text is about 8 ms of matmul at AMX speed but takes 58 ms: the rest is
24 layers of fork-join synchronization, norms and attention bookkeeping on small tensors. The AMX kernel itself is
bound by loading weight tiles from L2, at about 500 GFLOP/s per core. `EG2_BF16=dot` (AVX-512 BF16 dot products) is
slower than fp32 here, because it issues on one port where fp32 FMAs issue on two.

**Decisions per second**, with Rust callers that each send one unit at a time, full pass, AMX:

| callers | 32-token units | 128-token units |
|---|---|---|
| 1 | 48/s, p50 21 ms | 19/s, p50 53 ms |
| 8 | 172/s, p50 25 ms | 60/s, p50 75 ms |
| 16 | 271/s, p50 37 ms | 80/s, p50 148 ms |
| 32 | 396/s, p50 59 ms | 96/s, p50 274 ms |

fp32 gives about 60–70% of those rates at 1.5–2× the latency. Python threads calling `Model.embed` (the bindings
release the GIL) reach 14,000–18,000 tokens/s with AMX, close to explicit batches; through `eg2cpu.Embedder`,
16,400 tokens/s at 256 callers × 128 tokens.

**Cost per million tokens**, with the machine fully busy:

| | Spot (~$1.10/h) | on-demand (~$2.14/h) |
|---|---|---|
| full pass (20,000–27,000 tokens/s) | $0.011–0.015 | $0.022–0.030 |
| layer-8 exit (~68,000 tokens/s, 128-token units) | $0.0045 | $0.0087 |

## Arm Neoverse V2 (AWS c8g.4xlarge, Graviton 4, 16 cores, NEON and BFMMLA)

Two modes: `EG2_BF16=bfmmla` (the default on Arm Linux), which runs the matmuls in bf16 with `BFMMLA` and rounds
activations to bf16, and fp32 math on bf16 weights (`EG2_BF16=off`). The numbers in this section have attention in
fp32 (`EG2_BF16_ATTN=off`); attention in bf16 as well gains up to 1.5x more on long inputs (see Neoverse N2 below).
Against PyTorch fp32 (`tests/reference.rs`):

| mode | lowest embedding cosine | worst layer-mean relative error |
|---|---|---|
| fp32 (default) | 0.9999999 | 1.3e-6 |
| `EG2_BF16=bfmmla` | 0.9999965 | 4.2e-3 |

| default pools | fp32 | `EG2_BF16=bfmmla` | speedup |
|---|---|---|---|
| 1 text, 16 tokens | 21.7 ms | 12.4 ms | 1.75× |
| 1 text, 32 tokens | 34.5 ms | 18.0 ms | 1.92× |
| 1 text, 128 tokens | 77.9 ms | 44.4 ms | 1.76× |
| 1 text, 512 tokens | 298 ms | 162 ms | 1.84× |
| 1 text, 1,024 tokens | 672 ms | 373 ms | 1.80× |
| 256 texts × 32 tokens | 2,438 tokens/s | 7,369 tokens/s | 3.02× |
| 256 texts × 128 tokens | 2,430 tokens/s | 7,075 tokens/s | 2.91× |
| matmul, one core | 53–56 GFLOP/s | 199–224 GFLOP/s | ~4× |
| matmul, 16 cores | 610–730 GFLOP/s | 1,240–2,310 GFLOP/s | |

The fp32 kernel runs at about 62% of the fp32 peak of the core. `BFMMLA` gives four times that rate per core. Short
texts use 8 threads up to 40 tokens and all 16 threads above 40 tokens. The thread count comes from the core count.

**Early exit**, ms per text:

| | fp32, 1 text, 32 tokens | fp32, batch, 32 / 128 tokens | bfmmla, 1 text, 32 tokens | bfmmla, batch, 32 / 128 tokens |
|---|---|---|---|---|
| layer 4 | 6.9 | 2.64 / 10.5 | 4.4 | 0.86 / 3.61 |
| layer 8 | 12.2 | 4.84 / 19.4 | 7.7 | 1.61 / 6.67 |
| layer 12 | 17.7 | 7.02 / 28.3 | 10.9 | 2.33 / 9.77 |
| full pass | | 13.0 / 52.4 | | 4.33 / 18.1 |

**Decisions per second**, Rust callers each sending one unit at a time, full pass:

| callers | fp32, 32-token units | fp32, 128-token units | bfmmla, 32-token units | bfmmla, 128-token units |
|---|---|---|---|---|
| 1 | 29/s, p50 34 ms | 13/s, p50 78 ms | 56/s, p50 18 ms | 23/s, p50 44 ms |
| 4 | 45/s, p50 54 ms | 15/s, p50 184 ms | 100/s, p50 22 ms | 34/s, p50 74 ms |
| 8 | 57/s, p50 104 ms | 17/s, p50 404 ms | 144/s, p50 37 ms | 41/s, p50 151 ms |
| 16 | 66/s, p50 198 ms | 17/s, p50 745 ms | 177/s, p50 68 ms | 46/s, p50 274 ms |
| 32 | 67/s, p50 369 ms | 19/s, p50 1,674 ms | 183/s, p50 133 ms | 47/s, p50 586 ms |

**Cost per million tokens**, with the machine fully busy, at Spot ($0.265/h):

| | fp32 | `EG2_BF16=bfmmla` |
|---|---|---|
| full pass | $0.031 (2,400 tokens/s) | $0.010 (7,100–7,400 tokens/s) |
| layer-8 exit, 128-token units | $0.011 (6,600 tokens/s) | $0.0038 (19,200 tokens/s) |

## Arm Neoverse N2 (GitHub Actions `ubuntu-24.04-arm`, Azure Cobalt 100, 4 cores, NEON and BFMMLA)

A shared CI runner, one run each, so the numbers are rough. fp32 (`EG2_BF16=off`) against `BFMMLA` with attention in
fp32 (`EG2_BF16_ATTN=off`) and in bf16 (the default):

| | fp32 | bfmmla, attention fp32 | bfmmla (default) | speedup |
|---|---|---|---|---|
| 1 text, 32 tokens | 91.7 ms | 31.9 ms | 32.7 ms | 2.8x |
| 1 text, 128 tokens | 306 ms | 92.1 ms | 87.5 ms | 3.5x |
| 1 text, 512 tokens | 1,279 ms | 436 ms | 327 ms | 3.9x |
| 256 texts x 32 tokens | 461 tokens/s | 1,794 tokens/s | 1,843 tokens/s | 4.0x |
| 256 texts x 128 tokens | 455 tokens/s | 1,947 tokens/s | 2,112 tokens/s | 4.6x |
| image, 70 soft tokens | 1,754 ms | 464 ms | 420 ms | 4.2x |
| image, 280 soft tokens | 8,473 ms | 3,530 ms | 2,387 ms | 3.5x |
| audio, 5 s | 960 ms | 258 ms | 250 ms | 3.8x |
| audio, 30 s | 5,618 ms | 1,606 ms | 1,359 ms | 4.1x |

Attention in bf16 matters where it is a large share of the work: 512-token texts, 280-soft-token images (2,340
patches in the vision tower) and 30 s clips (the text tower over 754 tokens). The audio tower's own attention spans 12
frames and stays in fp32. All of `cargo test --release` passes in both modes on this runner, in CI.

