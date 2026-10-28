//! EmbeddingGemma 2 text encoder: weights, tokenizer, and the forward pass.
//!
//! Mirrors `transformers.models.embedding_gemma2.EmbeddingGemma2TextModel` (fp32 math over bf16 weights).
//! The text tower is loaded up front; the vision tower (`vision.rs`) on the first image. The audio tower in the
//! checkpoint is never read.

use crate::simd::{Packed, bf16_to_f32, gelu_mul, gemm_packed, gemm_serial, softmax};
use anyhow::{Context, Result, anyhow, bail, ensure};
use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::Value;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::sched::Scheduler;
use crate::tensors::Tensors;
use crate::vision::{DEFAULT_SOFT_TOKENS, Rgb, Vision, VisionConfig};
use tokenizers::{Tokenizer, TruncationParams};

pub const MAX_TOKENS: usize = 1024;

// ---------------------------------------------------------------- public types

#[derive(Clone, Debug)]
pub struct LayerSpec {
    pub head_dim: usize,
    pub kv_heads: usize,
    pub full: bool,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub ple_dim: usize,
    pub embedding_dim: usize,
    pub vocab: usize,
    pub bos_token: u32,
    pub eos_token: u32,
    pub eps: f32,
    pub sliding_window: usize,
    pub rope_theta_sliding: f64,
    pub rope_theta_full: f64,
    pub layers: Vec<LayerSpec>,
}

impl Config {
    fn from_json(v: &Value) -> Result<Self> {
        let t = &v["text_config"];
        let u = |k: &str| t[k].as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("text_config.{k} missing"));
        let types = t["layer_types"].as_array().context("layer_types")?;
        let (head_dim, kv_heads) = (u("head_dim")?, u("num_key_value_heads")?);
        let mut layers: Vec<LayerSpec> = types
            .iter()
            .map(|ty| LayerSpec { head_dim, kv_heads, full: ty.as_str() == Some("full_attention") })
            .collect();
        if let Some(over) = t["per_layer_config"].as_object() {
            for (idx, o) in over {
                let l = &mut layers[idx.parse::<usize>()?];
                if let Some(h) = o["head_dim"].as_u64() {
                    l.head_dim = h as usize;
                }
                if let Some(h) = o["num_key_value_heads"].as_u64() {
                    l.kv_heads = h as usize;
                }
            }
        }
        if layers.len() != u("num_hidden_layers")? {
            bail!("layer_types does not match num_hidden_layers");
        }
        let rope = |ty: &str| t["rope_parameters"][ty]["rope_theta"].as_f64().context("rope_theta");
        Ok(Self {
            hidden: u("hidden_size")?,
            intermediate: u("intermediate_size")?,
            heads: u("num_attention_heads")?,
            ple_dim: u("hidden_size_per_layer_input")?,
            embedding_dim: u("embedding_dim")?,
            vocab: u("vocab_size")?,
            bos_token: u("bos_token_id")? as u32,
            eos_token: u("eos_token_id")? as u32,
            eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            sliding_window: u("sliding_window")?,
            rope_theta_sliding: rope("sliding_attention")?,
            rope_theta_full: rope("full_attention")?,
            layers,
        })
    }
}

/// What one text encodes to.
#[derive(Clone, Debug)]
pub struct Output {
    /// Mean-pooled, L2-normalized embedding (`embedding_dim`, 768).
    pub embedding: Vec<f32>,
    /// Mean-pooled residual stream after each layer (`num_layers` x `hidden`), for readouts at intermediate layers.
    pub layer_means: Vec<Vec<f32>>,
}

/// An early-exited forward pass: the layer means computed so far, and the state to finish it.
pub struct Partial {
    t: usize,
    h: Vec<f32>,
    inputs: Vec<f32>,
    next: usize,
    /// Mean-pooled residual stream after each layer run so far.
    pub layer_means: Vec<Vec<f32>>,
}

impl Partial {
    /// Layers run so far.
    pub fn layers_done(&self) -> usize {
        self.next
    }
}

pub struct Model {
    pub cfg: Config,
    mmap: Mmap,
    embed_offset: usize,
    embed_scale: f32,
    /// per_layer_model_projection, one [ple_dim, hidden] slice per layer; each layer's per-layer embedding is
    /// computed when that layer runs.
    ple_model_proj: Vec<Packed>,
    ple_proj_norm: Vec<f32>,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    out_proj: Vec<u16>,
    tokenizer: Tokenizer,
    revision: String,
    sched: Scheduler,
    vision_cfg: Option<VisionConfig>,
    /// The vision tower, loaded by the first image (about 300 MB of packed weights).
    vision: OnceLock<Vision>,
    vision_loading: Mutex<()>,
}

/// Concurrent forward passes in a batch: `$EG2_MAX_IN_FLIGHT`, else the pool's thread count.
pub fn max_in_flight() -> usize {
    std::env::var("EG2_MAX_IN_FLIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or_else(rayon::current_num_threads)
}

/// The traced (op, seconds, calls) since the last call, and reset.
pub fn take_trace() -> Vec<(&'static str, f64, u64)> {
    std::mem::take(&mut *TRACE.lock().unwrap())
}

// ---------------------------------------------------------------- loading

struct Layer {
    spec: LayerSpec,
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    pre_ff_ln: Vec<f32>,
    post_ff_ln: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wq: Packed,
    wk: Packed,
    wv: Packed,
    wo: Packed,
    w_gate: Packed,
    w_up: Packed,
    w_down: Packed,
    ple_gate: Packed,
    ple_proj: Packed,
    ple_norm: Vec<f32>,
    scalar: f32,
}

const GET_IT: &str = "download it with `uv run download.py embeddinggemma-2` (from examples/), or set EG2_MODEL_DIR \
                      to a folder holding config.json, tokenizer.json and model.safetensors";

/// `$EG2_MODEL_DIR`, else the Hugging Face cache snapshot of google/embeddinggemma-2.
pub fn default_model_dir() -> Result<PathBuf> {
    if let Ok(d) = std::env::var("EG2_MODEL_DIR") {
        return Ok(PathBuf::from(d));
    }
    let hub = std::env::var("HF_HUB_CACHE").map(PathBuf::from).unwrap_or_else(|_| {
        let home = std::env::var("HF_HOME").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface")
        });
        home.join("hub")
    });
    let snaps = hub.join("models--google--embeddinggemma-2/snapshots");
    std::fs::read_dir(&snaps)
        .with_context(|| format!("no EmbeddingGemma 2 in the Hugging Face cache ({}): {GET_IT}", snaps.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.join("model.safetensors").exists())
        .ok_or_else(|| anyhow!("no snapshot with model.safetensors under {}: {GET_IT}", snaps.display()))
}

impl Model {
    pub fn load_default() -> Result<Self> {
        Self::load(&default_model_dir()?)
    }

    pub fn load(dir: &Path) -> Result<Self> {
        for f in ["config.json", "tokenizer.json", "model.safetensors"] {
            if !dir.join(f).is_file() {
                let from = if std::env::var_os("EG2_MODEL_DIR").is_some() { " (EG2_MODEL_DIR)" } else { "" };
                return Err(anyhow!("no EmbeddingGemma 2 checkpoint in {}{from}: {f} is missing; {GET_IT}", dir.display()));
            }
        }
        let json: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let cfg = Config::from_json(&json)?;
        let vision_cfg = VisionConfig::from_json(&json)?;
        // The tokenizer first: parsing its 31 MB JSON briefly takes ~315 MB, which would otherwise stack on top of
        // the weights and set the process's peak.
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams { max_length: MAX_TOKENS, ..Default::default() }))
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        let file = File::open(dir.join("model.safetensors"))?;
        // SAFETY: the checkpoint is read-only and not modified while mapped.
        let mmap = unsafe { Mmap::map(&file)? };
        let t = Tensors::parse(&mmap)?;
        let (d, nl, pd) = (cfg.hidden, cfg.layers.len(), cfg.ple_dim);

        let layers = cfg
            .layers
            .iter()
            .enumerate()
            .map(|(i, spec)| -> Result<Layer> {
                let p = |n: &str| format!("language_model.layers.{i}.{n}.weight");
                let (hd, qd, kvd) = (spec.head_dim, cfg.heads * spec.head_dim, spec.kv_heads * spec.head_dim);
                let scalar = t.bf16_f32(&format!("language_model.layers.{i}.layer_scalar"), &[1])?[0];
                Ok(Layer {
                    spec: spec.clone(),
                    input_ln: t.bf16_f32(&p("input_layernorm"), &[d])?,
                    post_attn_ln: t.bf16_f32(&p("post_attention_layernorm"), &[d])?,
                    pre_ff_ln: t.bf16_f32(&p("pre_feedforward_layernorm"), &[d])?,
                    post_ff_ln: t.bf16_f32(&p("post_feedforward_layernorm"), &[d])?,
                    q_norm: t.bf16_f32(&p("self_attn.q_norm"), &[hd])?,
                    k_norm: t.bf16_f32(&p("self_attn.k_norm"), &[hd])?,
                    wq: t.packed(&p("self_attn.q_proj"), qd, d)?,
                    wk: t.packed(&p("self_attn.k_proj"), kvd, d)?,
                    wv: t.packed(&p("self_attn.v_proj"), kvd, d)?,
                    wo: t.packed(&p("self_attn.o_proj"), d, qd)?,
                    w_gate: t.packed(&p("mlp.gate_proj"), cfg.intermediate, d)?,
                    w_up: t.packed(&p("mlp.up_proj"), cfg.intermediate, d)?,
                    w_down: t.packed(&p("mlp.down_proj"), d, cfg.intermediate)?,
                    ple_gate: t.packed(&p("ple_block.per_layer_input_gate"), pd, d)?,
                    ple_proj: t.packed(&p("ple_block.per_layer_projection"), d, pd)?,
                    ple_norm: t.bf16_f32(&p("ple_block.post_per_layer_input_norm"), &[d])?,
                    scalar,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let embed_offset = t.offset("language_model.embed_tokens.weight", "BF16", &[cfg.vocab, d])?;
        let ple_model_proj = t
            .bytes("language_model.ple.per_layer_model_projection.weight", "BF16", &[nl * pd, d])?
            .chunks_exact(pd * d * 2)
            .map(|b| Packed::from_le_bytes(b, pd, d))
            .collect();
        let ple_proj_norm = t.bf16_f32("language_model.ple.per_layer_projection_norm.weight", &[pd])?;
        let final_norm = t.bf16_f32("language_model.norm.weight", &[d])?;
        let out_proj = t.bf16("language_model.embedding_projection.weight", &[cfg.embedding_dim, d])?;
        let revision = revision_of(dir, t.header(), mmap.len());
        drop(t);

        Ok(Self {
            embed_scale: (d as f32).sqrt(),
            cfg,
            mmap,
            embed_offset,
            ple_model_proj,
            ple_proj_norm,
            layers,
            final_norm,
            out_proj,
            tokenizer,
            revision,
            sched: Scheduler::new(),
            vision_cfg,
            vision: OnceLock::new(),
            vision_loading: Mutex::new(()),
        })
    }

    /// The vision tower, loading it on first use.
    fn vision(&self) -> Result<&Vision> {
        if let Some(v) = self.vision.get() {
            return Ok(v);
        }
        let _guard = self.vision_loading.lock().unwrap();
        if let Some(v) = self.vision.get() {
            return Ok(v);
        }
        let cfg = self.vision_cfg.as_ref().context("this checkpoint has no vision tower")?;
        let v = Vision::load(cfg, self.cfg.hidden, &Tensors::parse(&self.mmap)?)?;
        Ok(self.vision.get_or_init(|| v))
    }
}

fn revision_of(dir: &Path, header: &[u8], file_len: usize) -> String {
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.len() == 40 && name.bytes().all(|b| b.is_ascii_hexdigit()) {
        return name.to_string();
    }
    // FNV-1a over the header (names, shapes, offsets) and the file length.
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in header.iter().chain(&(file_len as u64).to_le_bytes()) {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    format!("local:{h:016x}")
}

// ---------------------------------------------------------------- the forward pass

impl Model {
    /// Embed an image (PNG, JPEG or WebP bytes) at the processor's default budget of 280 soft tokens.
    pub fn embed_image(&self, bytes: &[u8]) -> Result<Output> {
        self.embed_image_with(bytes, DEFAULT_SOFT_TOKENS)
    }

    /// Embed an image with at most `max_soft_tokens` soft tokens (70, 140, 280, 560 or 1120): the image is
    /// resized to about 9x that many 16x16 patches, so the vision tower's cost grows with the budget.
    pub fn embed_image_with(&self, bytes: &[u8], max_soft_tokens: usize) -> Result<Output> {
        let img = self.preprocess_image(bytes, max_soft_tokens)?;
        let soft = self.image_soft_tokens(&img)?;
        Ok(self.embed_soft_tokens(&soft))
    }

    /// Decode and resize an image as the processor does for a soft-token budget.
    pub fn preprocess_image(&self, bytes: &[u8], max_soft_tokens: usize) -> Result<Rgb> {
        let cfg = self.vision_cfg.as_ref().context("this checkpoint has no vision tower")?;
        crate::vision::preprocess(cfg, bytes, max_soft_tokens)
    }

    /// The vision tower's soft tokens for a preprocessed image: [n, hidden], what the text tower receives in place
    /// of the image placeholder tokens. Runs on the vision pool (`vision_pool`).
    pub fn image_soft_tokens(&self, img: &Rgb) -> Result<Vec<f32>> {
        let vision = self.vision()?;
        Ok(match vision_pool() {
            Some(pool) => pool.install(|| vision.forward(&self.mmap, img)),
            None => vision.forward(&self.mmap, img),
        })
    }

    /// The text tower over `<bos> <boi> soft tokens <eoi> <eos>`: the image's embedding and layer means. Runs on a
    /// pool sized to the sequence length, directly (not through the request scheduler).
    pub fn embed_soft_tokens(&self, soft: &[f32]) -> Output {
        let (cfg, d) = (&self.cfg, self.cfg.hidden);
        let vc = self.vision_cfg.as_ref().expect("soft tokens come from the vision tower");
        assert!(!soft.is_empty() && soft.len().is_multiple_of(d), "soft tokens must be [n, {d}]");
        let n = soft.len() / d;
        let mut ids = vec![cfg.bos_token, vc.boi_token];
        ids.extend(std::iter::repeat_n(vc.image_token, n));
        ids.extend([vc.eoi_token, cfg.eos_token]);
        let mut h = self.embed_ids(&ids);
        h[2 * d..(2 + n) * d].copy_from_slice(soft);
        let run = || {
            let p = self.partial_from_embeds(h, cfg.layers.len() - 1);
            self.resume_in_pool(p)
        };
        match pool_for(ids.len()) {
            Some(pool) => pool.install(run),
            None => run(),
        }
    }

    /// The checkpoint's revision: the Hugging Face snapshot commit when loaded from the hub cache, otherwise
    /// `local:` plus a fingerprint of the safetensors header and file size. Decision handles record it.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Token ids with `<bos>`/`<eos>`, truncated to `MAX_TOKENS` (as `tok(text, truncation=True, max_length=1024)`).
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.tokenizer.encode(text, true).map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn embed(&self, text: &str) -> Result<Output> {
        Ok(self.forward(&self.tokenize(text)?))
    }

    /// Texts run concurrently, each unpadded, at most `max_in_flight()` at a time (see `forward_batch`).
    /// Concurrent calls from several threads share that bound and are interleaved.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Output>> {
        let ids: Vec<Vec<u32>> = texts.par_iter().map(|t| self.tokenize(t)).collect::<Result<_>>()?;
        Ok(self.forward_batch(&ids.iter().map(Vec::as_slice).collect::<Vec<_>>()))
    }

    /// Many texts, at most `max_in_flight()` forward passes at once across the whole process: calls from
    /// outside the rayon pool go through the scheduler (see `sched`), so concurrent batches and single
    /// requests from other threads share the bound and are interleaved round-robin. Called from inside the
    /// pool (nested use), the texts run directly on it with the same per-call bound.
    pub fn forward_batch(&self, ids: &[&[u32]]) -> Vec<Output> {
        if ids.is_empty() {
            return vec![];
        }
        if rayon::current_thread_index().is_some() {
            return self.forward_batch_with(ids, max_in_flight());
        }
        ids.iter().for_each(|t| self.check_ids(t));
        self.sched.run(self, ids)
    }

    /// Run `ids` directly on the current pool, at most `max_in_flight` passes at once, bypassing the scheduler.
    /// Each pass's matmuls still spread over the whole pool. Without the bound, a thread waiting inside one
    /// text's matmul steals the next text and starts it, so every text in the batch can end up in flight with
    /// its activations (tens of MB each at a few hundred tokens) allocated at the same time.
    pub fn forward_batch_with(&self, ids: &[&[u32]], max_in_flight: usize) -> Vec<Output> {
        run_bounded(ids.len(), max_in_flight, |i| ids[i].len(), |i| self.forward_in_pool(ids[i]))
    }

    /// Early exit: run each text through layers 0..=`upto` only (on the current pool, at most `max_in_flight()`
    /// at a time, bypassing the scheduler; a single text runs on the pool sized to its length). The result holds the layer means so far and the state needed to
    /// `resume_batch` it to the end without recomputing those layers.
    pub fn forward_batch_partial(&self, ids: &[&[u32]], upto: usize) -> Vec<Partial> {
        ids.iter().for_each(|t| self.check_ids(t));
        if let [one] = ids {
            // One text runs on the pool sized to its length, as `forward` does, rather than waking every thread.
            if rayon::current_thread_index().is_none() {
                let run = || self.forward_partial_in_pool(one, upto);
                return vec![match pool_for(one.len()) {
                    Some(pool) => pool.install(run),
                    None => run(),
                }];
            }
        }
        run_bounded(ids.len(), max_in_flight(), |i| ids[i].len(), |i| self.forward_partial_in_pool(ids[i], upto))
    }

    /// Finish early-exited passes: the remaining layers, then the final embedding.
    pub fn resume_batch(&self, partials: Vec<Partial>) -> Vec<Output> {
        let lens: Vec<usize> = partials.iter().map(|p| p.t).collect();
        let slots: Vec<Mutex<Option<Partial>>> = partials.into_iter().map(|p| Mutex::new(Some(p))).collect();
        run_bounded(slots.len(), max_in_flight(), |i| lens[i], |i| {
            self.resume_in_pool(slots[i].lock().unwrap().take().expect("each partial resumed once"))
        })
    }

    /// Tokenize and run each text up to layer `upto` (see `forward_batch_partial`).
    pub fn embed_batch_partial(&self, texts: &[&str], upto: usize) -> Result<Vec<Partial>> {
        ensure!(upto < self.cfg.layers.len(), "upto must be below {}", self.cfg.layers.len());
        let ids: Vec<Vec<u32>> = texts.par_iter().map(|t| self.tokenize(t)).collect::<Result<_>>()?;
        Ok(self.forward_batch_partial(&ids.iter().map(Vec::as_slice).collect::<Vec<_>>(), upto))
    }

    /// One text. Alone, it runs at once on a pool sized to its length; while other requests are running it
    /// joins the next batch instead (see `sched`), so concurrent callers share the cores and the in-flight bound.
    pub fn forward(&self, ids: &[u32]) -> Output {
        if rayon::current_thread_index().is_some() {
            return self.forward_in_pool(ids);
        }
        self.check_ids(ids);
        self.sched.run(self, &[ids]).pop().expect("one output")
    }

    fn check_ids(&self, ids: &[u32]) {
        assert!(!ids.is_empty(), "empty input");
        if let Some(id) = ids.iter().find(|&&id| id as usize >= self.cfg.vocab) {
            panic!("token id {id} out of range");
        }
    }

    /// One text on the pool sized to its length (`pool_for`).
    pub(crate) fn forward_solo(&self, ids: &[u32]) -> Output {
        match pool_for(ids.len()) {
            Some(pool) => pool.install(|| self.forward_in_pool(ids)),
            None => self.forward_in_pool(ids),
        }
    }

    /// The forward pass on whatever rayon pool the caller is in.
    pub fn forward_in_pool(&self, ids: &[u32]) -> Output {
        let p = self.forward_partial_in_pool(ids, self.cfg.layers.len() - 1);
        self.resume_in_pool(p)
    }

    /// Layers 0..=`upto` of one text; `resume_in_pool` finishes it.
    pub fn forward_partial_in_pool(&self, ids: &[u32], upto: usize) -> Partial {
        assert!(!ids.is_empty(), "empty input");
        self.partial_from_embeds(self.embed_ids(ids), upto)
    }

    /// Scaled token embeddings, [t, hidden].
    fn embed_ids(&self, ids: &[u32]) -> Vec<f32> {
        let (cfg, d) = (&self.cfg, self.cfg.hidden);
        let mut h = vec![0.0f32; ids.len() * d];
        for (row, &id) in h.chunks_exact_mut(d).zip(ids) {
            assert!((id as usize) < cfg.vocab, "token id {id} out of range");
            let base = self.embed_offset + id as usize * d * 2;
            let src = &self.mmap[base..base + d * 2];
            for (x, b) in row.iter_mut().zip(src.chunks_exact(2)) {
                *x = bf16_to_f32(u16::from_le_bytes([b[0], b[1]])) * self.embed_scale;
            }
        }
        h
    }

    /// Layers 0..=`upto` over input embeddings `h` ([t, hidden]).
    fn partial_from_embeds(&self, h: Vec<f32>, upto: usize) -> Partial {
        let cfg = &self.cfg;
        let t = h.len() / cfg.hidden;
        assert!(t > 0, "empty input");
        assert!(upto < cfg.layers.len(), "upto {upto} out of range");
        // Projection-only per-layer embeddings come from the input embeddings; keep those (t x d) and project
        // one layer's [t, pd] slice at a time (all layers' slices together would be 48 KB per token).
        let inputs = h.clone();
        let mut p = Partial { t, h, inputs, next: 0, layer_means: Vec::with_capacity(cfg.layers.len()) };
        self.run_layers(&mut p, upto + 1);
        p
    }

    /// Layers p.next..end of one text, from where an early exit stopped.
    fn run_layers(&self, p: &mut Partial, end: usize) {
        let cfg = &self.cfg;
        let (t, d, pd) = (p.t, cfg.hidden, cfg.ple_dim);
        let mut ple = vec![0.0f32; t * pd];
        let s = (d as f32).powf(-0.5);
        // RoPE tables for this text's positions only; building them is cheap next to the layers.
        let rope_s = &Rope::new(cfg.rope_theta_sliding, sliding_head_dim(cfg), t);
        let rope_f = &Rope::new(cfg.rope_theta_full, full_head_dim(cfg), t);
        for i in p.next..end {
            let layer = &self.layers[i];
            let rope = if layer.spec.full { rope_f } else { rope_s };
            let window = (!layer.spec.full).then_some(cfg.sliding_window);
            trace("ple input", || {
                gemm_packed(t, &p.inputs, d, &self.ple_model_proj[i], &mut ple, pd);
                ple.par_chunks_mut(pd).with_min_len(64).for_each(|x| {
                    x.iter_mut().for_each(|v| *v *= s);
                    rmsnorm(x, Some(&self.ple_proj_norm), cfg.eps);
                });
            });
            self.layer_forward(layer, &mut p.h, &ple, rope, window, t);
            p.layer_means.push(trace("layer mean", || mean_rows(&p.h, d)));
        }
        p.next = end;
    }

    /// Finish one early-exited pass: the remaining layers, then the final embedding.
    pub fn resume_in_pool(&self, mut p: Partial) -> Output {
        let cfg = &self.cfg;
        let d = cfg.hidden;
        self.run_layers(&mut p, cfg.layers.len());
        let (mut h, layer_means) = (p.h, p.layer_means);
        rmsnorm_rows(&mut h, d, Some(&self.final_norm), cfg.eps);
        let pooled = mean_rows(&h, d);
        let mut embedding = vec![0.0f32; cfg.embedding_dim];
        gemm_serial(1, cfg.embedding_dim, d, &pooled, d, &self.out_proj, d, &mut embedding, cfg.embedding_dim, false);
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        embedding.iter_mut().for_each(|x| *x /= norm);
        Output { embedding, layer_means }
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self, l: &Layer, h: &mut [f32], ple: &[f32], rope: &Rope, window: Option<usize>, t: usize,
    ) {
        let cfg = &self.cfg;
        let (d, eps, nh, hd, kvh) = (cfg.hidden, cfg.eps, cfg.heads, l.spec.head_dim, l.spec.kv_heads);
        let (qd, kvd, inter, pd) = (nh * hd, kvh * hd, cfg.intermediate, cfg.ple_dim);

        // Attention block
        let mut x = h.to_vec();
        trace("norm", || rmsnorm_rows(&mut x, d, Some(&l.input_ln), eps));
        // q, k and v each get their own compact buffer, which keeps rows narrow when many texts are in flight.
        let mut q = vec![0.0f32; t * qd];
        let mut k = vec![0.0f32; t * kvd];
        let mut v = vec![0.0f32; t * kvd];
        trace("gemm qkv", || {
            gemm_packed(t, &x, d, &l.wq, &mut q, qd);
            gemm_packed(t, &x, d, &l.wk, &mut k, kvd);
            gemm_packed(t, &x, d, &l.wv, &mut v, kvd);
        });
        // Per-head norms and RoPE for q and k, and v's plain norm, in one pass over positions.
        trace("qkv norm+rope", || {
            q.par_chunks_mut(qd).zip(k.par_chunks_mut(kvd)).zip(v.par_chunks_mut(kvd)).with_min_len(32).enumerate().for_each(|(pos, ((q, k), v))| {
                for head in q.chunks_exact_mut(hd) {
                    rmsnorm(head, Some(&l.q_norm), eps);
                    rope.apply(head, pos);
                }
                for head in k.chunks_exact_mut(hd) {
                    rmsnorm(head, Some(&l.k_norm), eps);
                    rope.apply(head, pos);
                }
                for head in v.chunks_exact_mut(hd) {
                    rmsnorm(head, None, eps);
                }
            })
        });
        let attn = trace("attention", || attention(&q, &k, &v, t, nh, kvh, hd, window));
        let mut o = vec![0.0f32; t * d];
        trace("gemm o", || gemm_packed(t, &attn, qd, &l.wo, &mut o, d));
        // h += norm(o); x = norm(h) for the MLP, in one pass.
        trace("norm+add", || add_norm_then_norm(h, &mut o, &l.post_attn_ln, &mut x, &l.pre_ff_ln, d, eps));

        // MLP block
        let mut g = vec![0.0f32; t * inter];
        let mut u = vec![0.0f32; t * inter];
        trace("gemm gate+up", || {
            gemm_packed(t, &x, d, &l.w_gate, &mut g, inter);
            gemm_packed(t, &x, d, &l.w_up, &mut u, inter);
        });
        trace("gelu", || g.par_chunks_mut(inter).zip(u.par_chunks(inter)).with_min_len(16).for_each(|(g, u)| gelu_mul(g, u)));
        trace("gemm down", || gemm_packed(t, &g, inter, &l.w_down, &mut o, d));
        trace("norm+add", || add_norm(h, &mut o, &l.post_ff_ln, d, eps));

        // Per-layer-embedding block, ending in h = (h + norm(o)) * scalar in one pass.
        let mut pg = vec![0.0f32; t * pd];
        trace("ple", || {
            gemm_packed(t, h, d, &l.ple_gate, &mut pg, pd);
            pg.par_chunks_mut(pd).zip(ple.par_chunks(pd)).with_min_len(64).for_each(|(row, p)| gelu_mul(row, p));
            gemm_packed(t, &pg, pd, &l.ple_proj, &mut o, d);
            let scalar = l.scalar;
            h.par_chunks_mut(d).zip(o.par_chunks_mut(d)).with_min_len(64).for_each(|(h, o)| {
                rmsnorm(o, Some(&l.ple_norm), eps);
                h.iter_mut().zip(o.iter()).for_each(|(a, b)| *a = (*a + b) * scalar);
            })
        });
    }
}

fn sliding_head_dim(cfg: &Config) -> usize {
    cfg.layers.iter().find(|l| !l.full).map_or(cfg.layers[0].head_dim, |l| l.head_dim)
}

fn full_head_dim(cfg: &Config) -> usize {
    cfg.layers.iter().find(|l| l.full).map_or(cfg.layers[0].head_dim, |l| l.head_dim)
}

// ---------------------------------------------------------------- scheduling and thread pools

/// Run `n` jobs on the current pool, at most `cap` at a time, longest first (so a batch doesn't end waiting on
/// one long text started last); results in job order. Each worker runs one job at a time, so no more than `cap`
/// are ever in flight even when threads waiting inside a job's matmuls steal work.
fn run_bounded<T: Send>(n: usize, cap: usize, len: impl Fn(usize) -> usize, job: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(len(i)));
    rayon::scope(|s| {
        for _ in 0..cap.clamp(1, n.max(1)) {
            s.spawn(|_| {
                while let Some(&i) = order.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let out = job(i);
                    *slots[i].lock().unwrap() = Some(out);
                }
            });
        }
    });
    slots.into_iter().map(|m| m.into_inner().unwrap().expect("every job ran")).collect()
}

/// Whether to stay on the current pool instead of a sized one: on a pool worker already (e.g. one text of a parallel
/// batch), the outer split supplies the parallelism; a pinned RAYON_NUM_THREADS means the caller chose the count.
fn use_current_pool() -> bool {
    rayon::current_thread_index().is_some() || std::env::var_os("RAYON_NUM_THREADS").is_some()
}

/// Smaller pools for short texts: below a few hundred tokens, the barriers between ops cost more than extra threads
/// gain. `None` means use the current (global) pool, which is also the answer when `use_current_pool`.
fn pool_for(tokens: usize) -> Option<&'static rayon::ThreadPool> {
    // (up to this many tokens, threads): `EG2_POOLS`, e.g. "40:8,160:12", else `default_pools()`.
    static SIZES: OnceLock<Vec<(usize, usize)>> = OnceLock::new();
    let sizes = SIZES.get_or_init(|| {
        std::env::var("EG2_POOLS")
            .ok()
            .and_then(|v| v.split(',').map(|p| p.split_once(':').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))).collect())
            .unwrap_or_else(default_pools)
    });
    static POOLS: OnceLock<Vec<rayon::ThreadPool>> = OnceLock::new();
    if use_current_pool() {
        return None;
    }
    let all = rayon::current_num_threads();
    let i = sizes.iter().position(|&(max_tokens, threads)| tokens <= max_tokens && threads < all)?;
    let pools = POOLS.get_or_init(|| {
        sizes
            .iter()
            .map(|&(_, n)| rayon::ThreadPoolBuilder::new().num_threads(n.min(all)).build().expect("thread pool"))
            .collect()
    });
    Some(&pools[i])
}

/// The pool the vision tower runs on: `$EG2_VISION_THREADS` threads; else, with SME, the performance cores plus two
/// (the matmuls share one matrix unit per cluster, and tasks that land on the efficiency cores' slower unit
/// straggle); else the current pool. `None` (the current pool) also when `use_current_pool`.
fn vision_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    if use_current_pool() {
        return None;
    }
    POOL.get_or_init(|| {
        let all = rayon::current_num_threads();
        let n = match std::env::var("EG2_VISION_THREADS").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(n) if n > 0 => n,
            #[cfg(target_arch = "aarch64")]
            _ if sme() => perf_cores() + 2,
            _ => all,
        };
        (n < all).then(|| rayon::ThreadPoolBuilder::new().num_threads(n).build().expect("thread pool"))
    })
    .as_ref()
}

/// Default pool sizes for single texts. With SME the matmuls of every thread share one matrix unit per cluster, so
/// fewer threads pay off up to longer texts. Otherwise they come from the number of performance cores: half of them up
/// to 40 tokens and all of them up to 160, but at least 8 and 12 threads. The floors keep short texts spread over some
/// efficiency cores on hybrid CPUs, where that still pays. A pool as large as the global one means the global pool.
fn default_pools() -> Vec<(usize, usize)> {
    if sme() {
        return vec![(128, 4), (320, 8), (640, 10)];
    }
    let big = perf_cores();
    vec![(40, (big / 2).max(8)), (160, big.max(12))]
}

/// Physical performance cores: on macOS the cores of the fastest performance level (`hw.perflevel0.physicalcpu`),
/// elsewhere the logical CPUs divided by the hardware threads per core (cpu0's thread siblings on Linux).
fn perf_cores() -> usize {
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    #[cfg(target_os = "macos")]
    {
        let (mut n, mut len) = (0i32, std::mem::size_of::<i32>());
        // SAFETY: the name is NUL-terminated and `n` is an i32 buffer of `len` bytes.
        let name = c"hw.perflevel0.physicalcpu";
        let ret = unsafe { libc::sysctlbyname(name.as_ptr(), (&raw mut n).cast(), &mut len, std::ptr::null_mut(), 0) };
        if ret == 0 && n > 0 {
            return n as usize;
        }
    }
    #[cfg(target_os = "linux")]
    if let Ok(list) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list") {
        // e.g. "0", "0,48" or "0-1"
        let span = |r: &str| {
            let (a, b) = r.split_once('-')?;
            Some(b.parse::<usize>().ok()? + 1 - a.parse::<usize>().ok()?)
        };
        let per_core: usize = list.trim().split(',').map(|r| span(r).unwrap_or(1)).sum();
        return (logical / per_core.max(1)).max(1);
    }
    logical
}

/// Whether the matmuls run on SME (`sme::available`).
pub(crate) fn sme() -> bool {
    #[cfg(target_arch = "aarch64")]
    return crate::sme::available();
    #[cfg(not(target_arch = "aarch64"))]
    false
}

// ---------------------------------------------------------------- tracing

/// `EG2_TRACE=1`: wall time per op of the forward pass, summed until `take_trace` (for latency work; off when unset,
/// empty or `0`).
fn tracing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("EG2_TRACE").is_some_and(|v| !v.is_empty() && v != "0"))
}

static TRACE: Mutex<Vec<(&'static str, f64, u64)>> = Mutex::new(Vec::new());

#[inline]
pub(crate) fn trace<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
    if !tracing() {
        return f();
    }
    let t0 = std::time::Instant::now();
    let r = f();
    let dt = t0.elapsed().as_secs_f64();
    let mut tr = TRACE.lock().unwrap();
    match tr.iter_mut().find(|e| e.0 == name) {
        Some(e) => {
            e.1 += dt;
            e.2 += 1;
        }
        None => tr.push((name, dt, 1)),
    }
    r
}

// ---------------------------------------------------------------- kernels

/// Which attention products run on AMX in `EG2_BF16=amx` mode, as (Q K^T, P V). Both by default: the embeddings stay
/// within cosine 0.99999 of the fp32 reference, and batches are faster than with P V alone. `EG2_BF16_ATTN=pv` keeps
/// Q K^T in fp32, `EG2_BF16_ATTN=off` keeps both.
#[cfg(target_arch = "x86_64")]
fn attention_on_amx() -> (bool, bool) {
    static M: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        if crate::simd::bf16_mode() != crate::simd::Bf16::Amx {
            return (false, false);
        }
        match std::env::var("EG2_BF16_ATTN").as_deref() {
            Ok("off") => (false, false),
            Ok("pv") => (false, true),
            _ => (true, true),
        }
    })
}

/// Bidirectional multi-head attention with grouped KV heads; `window` limits |q - k| (inclusive).
/// q: [t, nh*hd], k/v: [t, kvh*hd] -> [t, nh*hd]. Attention scaling is 1.0 (q and k are RMS-normed).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(q: &[f32], k: &[f32], v: &[f32], t: usize, nh: usize, kvh: usize, hd: usize, window: Option<usize>) -> Vec<f32> {
    const QB: usize = 32;
    let (qd, kvd, groups) = (nh * hd, kvh * hd, nh / kvh);
    #[cfg(target_arch = "x86_64")]
    if let (amx_qk, true) = attention_on_amx() {
        return attention_amx(q, k, v, t, nh, kvh, hd, window, amx_qk);
    }
    // V transposed per KV head to [hd, t] so P V is another A B^T product.
    let mut vt = vec![0.0f32; kvh * hd * t];
    vt.par_chunks_mut(t).with_min_len(32).enumerate().for_each(|(row, dst)| {
        let (g, c) = (row / hd, row % hd);
        for (p, x) in dst.iter_mut().enumerate() {
            *x = v[p * kvd + g * hd + c];
        }
    });
    let mut out = vec![0.0f32; t * qd];
    let outp = OutPtr(out.as_mut_ptr());
    let nqb = t.div_ceil(QB);
    (0..nh * nqb).into_par_iter().for_each(|task| {
        let (hh, qb) = (task / nqb, task % nqb);
        let g = hh / groups;
        let (r0, r1) = (qb * QB, ((qb + 1) * QB).min(t));
        let (k0, k1) = match window {
            Some(w) => (r0.saturating_sub(w), (r1 - 1 + w + 1).min(t)),
            None => (0, t),
        };
        let (rows, kn) = (r1 - r0, k1 - k0);
        let mut s = vec![0.0f32; rows * kn];
        gemm_serial(rows, kn, hd, &q[r0 * qd + hh * hd..], qd, &k[k0 * kvd + g * hd..], kvd, &mut s, kn, false);
        for (i, row) in s.chunks_exact_mut(kn).enumerate() {
            let (lo, hi) = key_range(r0 + i, k0, k1, window);
            softmax_range(row, lo - k0, hi - k0);
        }
        // Rebind so the closure captures the Send wrapper rather than its raw-pointer field.
        #[allow(clippy::redundant_locals)]
        let outp = outp;
        // SAFETY: each task writes rows r0..r1 of head hh's column band; tasks never overlap.
        let dst = unsafe { std::slice::from_raw_parts_mut(outp.0.add(r0 * qd + hh * hd), (rows - 1) * qd + hd) };
        gemm_serial(rows, hd, kn, &s, kn, &vt[g * hd * t + k0..], t, dst, qd, false);
    });
    out
}

/// `attention` with P V (and, if `amx_qk`, Q K^T) on AMX tiles. Keys are handled in 32-aligned ranges around each
/// query block's window; the extra columns are masked out with the window. V^T (and K) are packed per KV head once.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn attention_amx(
    q: &[f32], k: &[f32], v: &[f32], t: usize, nh: usize, kvh: usize, hd: usize, window: Option<usize>, amx_qk: bool,
) -> Vec<f32> {
    use crate::simd::{amx, bf16dot, pack_pairs};
    const QB: usize = 32;
    let (qd, kvd, groups) = (nh * hd, kvh * hd, nh / kvh);
    let (tp, hdp) = (t.next_multiple_of(32), hd.next_multiple_of(32));
    // V^T per KV head: hd output columns, keys along k. K per KV head (for Q K^T): keys as columns, hd along k.
    let vp: Vec<Vec<u16>> = (0..kvh).map(|g| pack_pairs(t, hd, |key, c| v[key * kvd + g * hd + c])).collect();
    let kp: Vec<Vec<u16>> =
        if amx_qk { (0..kvh).map(|g| pack_pairs(hd, t, |d, key| k[key * kvd + g * hd + d])).collect() } else { vec![] };
    let mut out = vec![0.0f32; t * qd];
    let outp = OutPtr(out.as_mut_ptr());
    let nqb = t.div_ceil(QB);
    (0..nh * nqb).into_par_iter().for_each(|task| {
        let (hh, qb) = (task / nqb, task % nqb);
        let g = hh / groups;
        let (r0, r1) = (qb * QB, ((qb + 1) * QB).min(t));
        let (k0, k1) = match window {
            Some(w) => (r0.saturating_sub(w), (r1 - 1 + w + 1).min(t)),
            None => (0, t),
        };
        let (ka0, ka1) = (k0 / 32 * 32, k1.next_multiple_of(32).min(tp));
        let rows = r1 - r0;
        // scores over keys sk0..sk0 + skn: the aligned range on AMX, the exact range in fp32
        let (sk0, skn) = if amx_qk { (ka0, ka1 - ka0) } else { (k0, k1 - k0) };
        let mut s = vec![0.0f32; rows * skn];
        if amx_qk {
            let mut q16 = vec![0u16; 32 * hdp];
            // SAFETY: rows r0..r1 of head hh are in bounds; AMX availability was checked by bf16_mode.
            unsafe {
                bf16dot::round_rows(rows, hd, q.as_ptr().add(r0 * qd + hh * hd), qd, q16.as_mut_ptr(), hdp);
                amx::gemm_block(rows, hdp, q16.as_ptr(), hdp, false, kp[g].as_ptr().add(ka0 / 32 * hdp * 32), hdp * 32,
                                skn / 32, skn, s.as_mut_ptr(), skn);
            }
        } else {
            gemm_serial(rows, skn, hd, &q[r0 * qd + hh * hd..], qd, &k[k0 * kvd + g * hd..], kvd, &mut s, skn, false);
        }
        for (i, row) in s.chunks_exact_mut(skn).enumerate() {
            let (lo, hi) = key_range(r0 + i, k0, k1, window);
            softmax_range(row, lo - sk0, hi - sk0);
        }
        let pkn = ka1 - ka0;
        let mut p16 = vec![0u16; 32 * pkn];
        #[allow(clippy::redundant_locals)]
        let outp = outp;
        // SAFETY: p16 holds 32 rows of pkn; each task writes rows r0..r1 of head hh's column band, never overlapping.
        unsafe {
            bf16dot::round_rows(rows, skn, s.as_ptr(), skn, p16.as_mut_ptr().add(sk0 - ka0), pkn);
            amx::gemm_block(rows, pkn, p16.as_ptr(), pkn, false, vp[g].as_ptr().add(ka0 * 32), tp * 32, hdp / 32, hd,
                            outp.0.add(r0 * qd + hh * hd), qd);
        }
    });
    out
}

/// The keys query `qi` attends to, within the block's key range [k0, k1): |key - qi| <= window.
fn key_range(qi: usize, k0: usize, k1: usize, window: Option<usize>) -> (usize, usize) {
    match window {
        Some(w) => (k0.max(qi.saturating_sub(w)), k1.min(qi + w + 1)),
        None => (k0, k1),
    }
}

/// Softmax over row[lo..hi], zero elsewhere: the masked keys, without a per-element mask.
fn softmax_range(row: &mut [f32], lo: usize, hi: usize) {
    softmax(&mut row[lo..hi]);
    row[..lo].fill(0.0);
    row[hi..].fill(0.0);
}

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// Rotary tables for positions 0..t: cos/sin of pos * inv_freq, inv_freq = theta^(-2i/dim).
pub(crate) struct Rope {
    half: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    pub(crate) fn new(theta: f64, dim: usize, t: usize) -> Self {
        let half = dim / 2;
        let inv: Vec<f32> = (0..half).map(|i| (1.0 / theta.powf(((2 * i) as f32 / dim as f32) as f64)) as f32).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(t * half), Vec::with_capacity(t * half));
        for p in 0..t {
            for &f in &inv {
                // the angle in fp32, as PyTorch computes it
                let (s_, c_) = (p as f32 * f).sin_cos();
                cos.push(c_);
                sin.push(s_);
            }
        }
        Self { half, cos, sin }
    }

    /// x * cos + rotate_half(x) * sin, with rotate_half(x) = cat(-x2, x1).
    pub(crate) fn apply(&self, x: &mut [f32], pos: usize) {
        let h = self.half;
        let (c, s) = (&self.cos[pos * h..(pos + 1) * h], &self.sin[pos * h..(pos + 1) * h]);
        let (x1, x2) = x.split_at_mut(h);
        for i in 0..h {
            let (a, b) = (x1[i], x2[i]);
            x1[i] = a * c[i] - b * s[i];
            x2[i] = b * c[i] + a * s[i];
        }
    }
}

/// x * (mean(x^2) + eps)^-0.5 * w
pub(crate) fn rmsnorm(x: &mut [f32], w: Option<&[f32]>, eps: f32) {
    let mut acc = [0.0f32; 8];
    let mut chunks = x.chunks_exact(8);
    for c in &mut chunks {
        for i in 0..8 {
            acc[i] += c[i] * c[i];
        }
    }
    let tail: f32 = chunks.remainder().iter().map(|v| v * v).sum();
    let ss = acc.iter().sum::<f32>() + tail;
    let r = (ss / x.len() as f32 + eps).powf(-0.5);
    match w {
        Some(w) => x.iter_mut().zip(w).for_each(|(v, w)| *v = *v * r * w),
        None => x.iter_mut().for_each(|v| *v *= r),
    }
}

fn rmsnorm_rows(x: &mut [f32], d: usize, w: Option<&[f32]>, eps: f32) {
    x.par_chunks_mut(d).with_min_len(64).for_each(|row| rmsnorm(row, w, eps));
}

/// The residual add after a block, row by row: h += norm(o) with weights `post` (o is normed in place).
pub(crate) fn add_norm(h: &mut [f32], o: &mut [f32], post: &[f32], d: usize, eps: f32) {
    h.par_chunks_mut(d).zip(o.par_chunks_mut(d)).with_min_len(64).for_each(|(h, o)| {
        rmsnorm(o, Some(post), eps);
        h.iter_mut().zip(o.iter()).for_each(|(a, b)| *a += b);
    })
}

/// `add_norm`, then the next block's input x = norm(h) with weights `pre`, in one pass over the rows.
pub(crate) fn add_norm_then_norm(h: &mut [f32], o: &mut [f32], post: &[f32], x: &mut [f32], pre: &[f32], d: usize, eps: f32) {
    h.par_chunks_mut(d).zip(o.par_chunks_mut(d)).zip(x.par_chunks_mut(d)).with_min_len(64).for_each(|((h, o), x)| {
        rmsnorm(o, Some(post), eps);
        h.iter_mut().zip(o.iter()).for_each(|(a, b)| *a += b);
        x.copy_from_slice(h);
        rmsnorm(x, Some(pre), eps);
    })
}

fn mean_rows(x: &[f32], d: usize) -> Vec<f32> {
    let t = x.len() / d;
    let mut m = vec![0.0f64; d];
    for row in x.chunks_exact(d) {
        m.iter_mut().zip(row).for_each(|(a, b)| *a += *b as f64);
    }
    m.into_iter().map(|v| (v / t as f64) as f32).collect()
}
