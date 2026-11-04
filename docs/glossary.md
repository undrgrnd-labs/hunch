# Glossary: methods and maths

This is a quick reference for the terms used in eg2cpu and hunch: compiled decisions, the proxy, and the examples.
Each entry says what the term is. Where a formula helps, the entry gives it. Each entry also points to the code.

Sections: [encoder and features](#encoder-and-features), [decision kinds](#decision-kinds),
[directions](#directions-readout-methods), [calibration](#calibration), [abstain and escalate](#abstain-and-escalate),
[automatic choices](#automatic-choices-fixed-rules), [metrics and intervals](#metrics-and-intervals),
[the proxy (hunch)](#the-proxy-hunch), [pre-flight check](#pre-flight-check). Readers of the proxy docs need mostly
the proxy section and the metrics.
- `decisions.py` means `eg2cpu/python/eg2cpu/decisions.py`.
- `proxy.py` means `hunch/src/hunch/proxy.py`.
- `decision.rs` means `eg2cpu/src/decision.rs`.

Notation used throughout:
- `x` is an embedding vector (512 dimensions at an intermediate layer, 768 for the final embedding).
- `w` is a unit direction, and `s = w·x` is a raw score.
- `σ(t) = 1 / (1 + e^(−t))` is the logistic sigmoid.
- `n` is the sample size and `k` the number of classes or levels.

## Encoder and features

**Layer mean.** The mean over tokens of the hidden state after decoder layer *l*, a 512-dimensional vector.
`Partial.layer_means[l]`, `embed_batch_partial(texts, upto)`.

**Final embedding.** All 24 layers, then the final norm, a mean pool, a 512→768 projection and L2 normalization.
This is what `embed` returns.

**Early exit.** Stopping the forward pass after layer *l*, and resuming later if needed (`resume_batch`). The
result is bit-identical to a full pass. Layer 8 takes about a third of the full pass (see
`performance.md`).

**Unit.** The text a decision reads, rendered from a span by the spec's template, e.g. `"{call}\n{output}"`.

**Span.** The record a decision is applied to: a dict of fields such as `kind`, `tool`, `call`, `output` and
`is_error`.

**Gate (`applies_when`).** A boolean expression over span fields. If it is false, the verdict is
`not_applicable`. `gate_eval`.

**AMX.** Intel's matrix extension (Sapphire Rapids and later): eight 1 KB tile registers and `tdpbf16ps`, which
multiplies a 16 × 32 bf16 tile by a 32 × 16 bf16 tile into 16 × 16 fp32 sums, 16,384 multiply-adds per
instruction. Used when `EG2_BF16=amx`. `simd::amx`.

**bf16 mode.** `EG2_BF16=amx|dot|bfmmla`: matmuls multiply in bf16, activations included, with fp32 accumulation.
Embeddings stay at cosine ≥ 0.99999 to the reference, with layer means within about 0.6%. Off by default, except
`bfmmla` on Arm Linux CPUs with BF16 and no SME. `bf16_mode`.

**Pair packing.** Weights packed in panels of 32 output columns, with k in pairs, `(w[c][2q], w[c][2q+1])`: the input
layout of `vdpbf16ps`, and exactly AMX's B-tile layout.

## Decision kinds

**Decision / handle.** A decision is a spec (YAML): a typed question about a span, its matched contrasts and how to
calibrate, abstain and escalate. Compiling it gives a handle, a small safetensors file that eg2cpu applies in Rust.
`decisions.compile`, `Model.apply`.

**boolean.** Yes or no. The calibrated output is `p = P(yes)`.

**choice.** One of *k* unordered options. Each option has one direction and one probability.

**score.** In eg2cpu decisions, a continuous value, e.g. 0–1: one direction, then a linear map. In hunch, a `score`
question is one of ordered levels, like eg2cpu's `levels`.

**levels.** *k* ordered levels. There is one direction. Each level has a proportional-odds probability. The score is
the expected level.

## Directions (readout methods)

**Contrast.** Texts that differ only in the property being decided: a yes/no pair, one text per option, or one per
level. The difference between them isolates the property from the topic.

**Matched set.** The texts of one contrast. They all share the same template and context. For levels and score
contrasts, each set is centred. Only the grading of its texts against each other remains.

**Hard negative.** A text that looks like "yes" but is "no", e.g. "Payment declined by the card issuer" for "did the
tool call fail?". It anchors the readout against topic leakage.

**Diff of means.** `w ∝ μ_yes − μ_no`. For choice, `w_c ∝ μ_c − grand mean`. It has no parameters and is the
default. `_fit_generic`.

**LDA (shrinkage).** `w ∝ Σ⁻¹(μ_yes − μ_no)`, where Σ is the pooled within-class covariance (Ledoit-Wolf). It
down-weights directions that vary a lot within a class.

**Ridge.** For score and levels: `w ∝ (XᵀX + λI)⁻¹ Xᵀy` on per-set-centred rows, with `λ = 0.1·tr(XᵀX)/p`.

**Ledoit-Wolf shrinkage.** A covariance estimate usable when n is not much larger than p:
`Σ = δ·(tr S / p)·I + (1 − δ)·S`, with the shrinkage δ estimated from the data. `ledoit_wolf`.

**Whitening.** A matrix L with `‖L·v‖² = vᵀΣ⁻¹v` (the Cholesky factor of Σ⁻¹). `whitener`.

## Calibration

**Calibration sample.** About 60 labelled rows from the target domain. It is required. Rankings transfer from
contrasts, but cut-offs do not.

**Standardized score.** `z = (s − mean_cal) / std_cal`. It is taken over the calibration sample. Because of this,
every calibrator works on a unit-free scale.

**Platt scaling** (boolean). `p = σ(a·z + b)`, fitted by Newton's method on the log-loss with a tiny L2 penalty on
`a`. `fit_logistic`.

**Multinomial calibration** (choice). `logits = A·z + b`, softmax over the options. It starts at `A = I` (the
readout score is the logit), with L2 1e-3 and damped Newton. It has k² + k parameters. `fit_softmax`.

**Linear calibration** (score). `value = a·z + b`, fitted by least squares.

**Proportional odds** (levels). `P(Y ≤ j) = σ(θ_j − a·z)`, with one slope and increasing cut-points `θ_1 < … <
θ_(k−1)`. Level probabilities are the differences of adjacent cumulative probabilities. `fit_ordinal`,
`ordinal_probs`.

**Cut-points.** The `θ_j` above. They are written as a first value plus `exp` of each gap, so they can't cross. A
weak prior pulls the gaps toward their starting values, so a level missing from the sample doesn't collapse.

**Expected level.** `Σ_j j·P_j`: the levels score, and what `|value − c| < w` escalation rules test.

**LLM calibration.** Platt scaling of an escalation LLM's logit, `log(p / (1 − p))`, standardized and fitted on the
same calibration rows, so its probability is on the same footing as the readout's.

## Abstain and escalate

**Mahalanobis distance.** `d(x) = ‖L(x − μ)‖ = √((x − μ)ᵀΣ⁻¹(x − μ))`, the distance from the compile-time reference
set (contrasts plus calibration rows). It is computed in Rust at apply time.

**Held-out distances.** The distance of each reference row, under a mean and covariance fitted without its group
(grouped 5-fold). In-sample distances are far too small in high dimensions. They make every new span look out of
distribution. `heldout_distances`.

**Abstain threshold.** The 99th percentile (`mahalanobis_quantile`) of held-out distances. When calibration rows
exist, the percentile is taken over the calibration rows. When a distance is beyond the threshold, the verdict is
`abstain`.

**Escalate band.** `|p − c| < w` (boolean) or `|value − c| < w` (score and levels): an uncertain answer is sent to
an LLM with a rendered prompt.

**Confidence / p_max.** The top class probability. `confidence < t` escalates choice and levels verdicts.

**Margin.** The top probability minus the second. `margin < t` escalates.

## Automatic choices (fixed rules)

**Grouped k-fold.** Cross-validation where whole groups are held out together. A group is a contrast set, a document
type, a tool, or a session. A fold never sees a near-copy of its test rows.

**Contrast CV.** Grouped 5-fold over the contrasts: refit the direction without each fold and score the held-out
rows. The score is AUC (boolean), accuracy (choice), Spearman (levels) or correlation (score). `contrast_cv`.

**Paired accuracy.** The share of held-out matched pairs where the "yes" text outscores its "no" twin.

**Order accuracy** (levels). The share of held-out adjacent level pairs in the right order.

**Monotonicity check** (levels). Held-out mean scores per level must rise with the level. If not, the compiler
warns that the levels may not lie on one axis and suggests `choice`.

**`method: auto`.** Each method is fitted, and each hard negative is checked held out (refit without it, then read
it). The default (diff of means) is kept if it reads them all. Otherwise the first method that does is taken, or
else the one that misreads the fewest. It never chooses by calibration score.

**`layer: auto`.** Every candidate layer is compiled (final, and 4, 8, 12, 16, 20, 23). Then:
1. Drop any layer whose calibration interval lies wholly below the best layer's (the *calibration guard*).
2. Rank the rest by contrast CV, or by a transfer set if one is given.
3. Keep the final embedding if it is within one standard error of the best; otherwise take the best.

`_compile_layer_auto`.

**One-standard-error rule.** Prefer the simpler (or default) option unless the best one beats it by more than one
standard error. Here `SE ≈ (CI width) / 3.92`.

**Transfer set.** Labelled rows from a second domain (neither the contrasts nor the calibration sample). They are
only scored, never fitted, and rank layers instead of contrast CV.

## Metrics and intervals

**AUC.** The probability that a random positive outscores a random negative. It doesn't change under any monotone
calibration. `auc`.

**Hanley-McNeil interval.** A 95% interval for an AUC from its value and the class counts. `auc_ci`.

**Wilson interval.** A 95% interval for a proportion k/n:
`(p̂ + z²/2n ± z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)`, with z = 1.96. Unlike the normal approximation, it behaves
at small n and at p̂ near 0 or 1. `proportion_ci`, `proxy.wilson`.

**Spearman's ρ.** The Pearson correlation of ranks: does the score rise with the level? `spearman`.

**Fisher z interval.** An interval for a correlation r: `tanh(atanh(r) ± 1.96/√(n − 3))`. `corr_ci`.

**QWK (quadratic-weighted kappa).** Agreement on ordered levels, beyond chance, where a miss by two levels costs
four times a miss by one. 1 is perfect and 0 is chance. `qwk`.

**Within-one accuracy.** The share of predictions at most one level off.

**Cohen's κ.** Agreement beyond chance between two raters (e.g. student vs teacher):
`(p_o − p_e) / (1 − p_e)`, where p_e is the agreement expected from the two raters' class rates. It is 0 for a
student that always gives the majority answer. `examples/hunch_examples.kappa`.

**Majority baseline.** The agreement from always answering the most common class. Agreement must be read against
it: 95% agreement is worthless when 95% of answers are "no".

**Minority recall.** Recall on the answers other than the most common class: the guard against majority-baseline
agreement.

**Log-loss, Brier score, ECE.** Measures of whether stated probabilities match observed frequencies. The examples
comparisons with the teacher report the Brier score (the mean squared distance between the probabilities and the
true class, summed over classes) and ECE (top-class confidence against accuracy in 10 bins) for every model that
returns probabilities.

## The proxy (hunch)

**Teacher (backend, upstream).** The expensive model the proxy learns from (an LLM judge or a dedicated decision
model). The CLI calls it `--backend`; answers it gives say `"answered_by": "backend"`. The examples' teacher is Clef
(Clef-Flash, a 9B model run locally in 4-bit MLX).

**State.** What a question is asked about: a tool call and its output, a message, or any JSON. Its text
(`state_text`) is what the encoder and the teacher read.

**`noul`.** A yes/no question, answered as P(yes): the question protocol's name for `boolean`.

**Student.** One per question: logistic regression on standardized layer-8 means (see Layer mean, above).
`proxy.Student`.

**Answer object.** One question's answer, from the teacher or from hunch: `{"type": "noul", "noul": p_yes}`, or for
`choice` and `score` `{"type": ..., "probabilities": {option: p}}` (score options are level indices). hunch adds
`choice`/`score`, `confidence` and `answered_by`. `answer_object`, `parse_answer`.

**Graduation.** Graduation is the point where the student of a question starts to answer. It happens when
cross-validation on the examples of the question clears one of two bars. The bars are symmetric or one-sided (see
below). The system checks again at every refit.

**Question key (version).** A hash of a question's type, instructions and criteria. Rewording a question makes a
new version, with its own sampled answers and graduation. `question_key`.

**Question family.** A name (`"family": "injection"`) shared by every version of one question. The family keeps the
labelled examples, so a new wording starts with all the ground truth. All versions in a family must have the same
type and options. `family_key`.

**State hash.** A hash of the text the encoder and the teacher read (`state_text`), used to key cached answers.
`state_hash`.

**Labelled example.** Ground truth given through `/v1/train`. The newest 5,000 of each class are kept per family
(`--labelled-cap`); each is weighted 2.0, overrides a sampled answer for the same state, and is shared by every
version in its family.

**Sampled example.** The teacher's answer: every forwarded request, plus anything `/v1/train` gives as sampled. It
sits in a rotating window per question version (default 2,000) and is weighted by the teacher's confidence. A
question keeps one example per state: a new answer for a state replaces the old one.

**Stored answer.** If the store has an answer for every question in the state of a request (a label, or the teacher's
last answer), the store answers the request: `"answered_by": "cache"`.

**Refit.** A refit retrains a student and decides graduation again. A refit runs on a worker thread when a question
has 100 new examples. The teacher checks this count when it answers. A refit also runs on each `train` call with
`"refit": true` (the default). The refits of one question run one at a time. The store keeps the new fit, so a
restarted proxy starts from it.

**Out-of-fold predictions.** 5-fold cross-validated predictions on a student's own examples. Graduation thresholds
are chosen on them.

**Symmetric graduation.** Take the lowest confidence threshold t such that, among out-of-fold predictions with
confidence ≥ t:
- there are at least 30 of them;
- the Wilson lower bound on agreement is ≥ 0.90;
- there are at least 15 minority answers, with the Wilson lower bound on minority recall ≥ 0.80.

**One-sided graduation** (yes/no). Only "no" answers with p(yes) < τ are answered locally. τ must satisfy:
- at least 30 examples are answered locally;
- the Wilson lower bound on recall of "yes" among forwarded answers is ≥ 0.85;
- the Wilson upper bound on the local miss rate is ≤ 2%.

It's for rare-"yes" and safety questions.

**Coverage.** The share of requests a graduated student may answer locally.

**Novelty.** A state is novel if its nearest example (cosine, among the question's 1,000 most recent) is less similar
than the 5th percentile of those examples' own nearest-neighbour similarities. Novel states are forwarded.

**Audit.** A comparison of a student's answer with the teacher's on a state the student would have answered. 10% of
locally answered requests are audited in the background (the client gets the student's answer at once), and every
request forwarded for another reason audits each student that would have answered it.

**Demotion.** Consider a question's last 100 audits (at least 20). A symmetric question is demoted while the Wilson
upper bound on agreement is below 0.90. A one-sided question is demoted while the Wilson lower bound on its miss rate
is above 2%. The miss rate is the share of local "no" answers that the teacher calls "yes". A demoted question
forwards its requests. These requests keep auditing the question. The question answers again when its audits recover.

**Request-level routing.** The proxy answers a request locally only if it may answer every question in the request
locally. Otherwise, the whole request goes upstream. The reason is that the teacher reads the state once.

**Warm start.** Starting a student from a compiled handle's direction instead of from scratch.

**Cascade.** Try the layer-8 student first, and resume the forward pass to the full model only when it's unsure.

**Answer cache / offline backend.** Teacher answers stored as `{question hash, state hash, answer}` with no text
(`AnswerCache`), and replayed as a backend (`OfflineBackend`). The examples ship the teacher's answers this way,
one file per example.

**Teacher blind spot (rule 9).** A student distilled from a teacher copies the misses of the teacher. Before you
distill a teacher, check it against ground truth. Give labelled cases where the teacher is wrong.

## Pre-flight check

**Pre-flight.** Before deploying a question, cross-validate students at layers 4, 8, 12 and final on 150–300
teacher answers, using the proxy's graduation rules. The verdict is one of:
- **GRADUATES**, symmetrically or one-sided;
- **NOT ENOUGH DATA:** too few rare answers to prove recall;
- **BACKEND ONLY:** no readout passes.

`examples/preflight/`, `question-design.md`.
