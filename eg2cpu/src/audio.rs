//! EmbeddingGemma 2 audio path: WAV decoding, log-mel features, the audio tower, and the projection into the text
//! tower's input space.
//!
//! Mirrors `Gemma4AudioFeatureExtractor` and `EmbeddingGemma2Model.get_audio_features`: 16 kHz mono samples (at most
//! 30 s) become 128 log-mel bins every 10 ms; two stride-2 3x3 convolutions subsample them to one frame per 40 ms; a
//! 12-layer conformer (Universal Speech Model style: half-step feed-forwards, chunked local attention with relative
//! positions, a causal depthwise convolution) encodes the frames, and each becomes a soft token, RMS-normalized and
//! projected to the text tower's 512-d input. The text tower then encodes `<bos> <boa> soft tokens <eoa> <eos>`.
//!
//! Padding the processor adds (to a multiple of 128 samples) only ever follows the audio, and nothing in the tower
//! looks ahead past a frame it keeps (the subsampling's look-ahead lands on zeroed frames, attention and the
//! convolution are causal), so the tower runs on the valid frames alone.

use crate::model::{rmsnorm, trace};
use crate::simd::{Packed, gemm_packed};
use crate::tensors::Tensors;
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use serde_json::Value;

/// The sampling rate the feature extractor expects.
pub const SAMPLE_RATE: u32 = 16_000;
/// The processor keeps the first 30 s.
pub const MAX_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
/// Analysis window and hop, in samples (20 ms and 10 ms).
const FRAME: usize = 320;
const HOP: usize = 160;
const FFT: usize = 512;
const MELS: usize = 128;
const MEL_FLOOR: f64 = 1e-3;
/// The highest mel filter's edge.
const MAX_HZ: f64 = 8000.0;

#[derive(Clone, Debug)]
pub struct AudioConfig {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    /// Keys a query sees: itself and the `context - 1` frames before it.
    pub context: usize,
    pub conv_kernel: usize,
    pub out_dims: usize,
    /// Output channels of the two subsampling convolutions.
    pub channels: [usize; 2],
    pub eps: f32,
    pub softcap: f32,
    pub residual_weight: f32,
    pub audio_token: u32,
    pub boa_token: u32,
    pub eoa_token: u32,
}

impl AudioConfig {
    pub(crate) fn from_json(v: &Value) -> Result<Option<Self>> {
        let c = &v["audio_config"];
        if c.is_null() {
            return Ok(None);
        }
        let u = |k: &str| c[k].as_u64().map(|x| x as usize).with_context(|| format!("audio_config.{k} missing"));
        let f = |k: &str| c[k].as_f64().with_context(|| format!("audio_config.{k} missing"));
        let id = |k: &str| v[k].as_u64().map(|x| x as u32).with_context(|| format!("{k} missing"));
        ensure!(u("attention_context_right")? == 0, "audio attention with right context is not supported");
        // with chunks as long as the left context, the blocked attention's relative positions are plain distances
        ensure!(
            u("attention_chunk_size")? + 1 == u("attention_context_left")?,
            "audio attention chunks must match the left context"
        );
        ensure!(c["hidden_act"].as_str() == Some("silu"), "audio hidden_act is not silu");
        ensure!(c["use_clipped_linears"].as_bool() == Some(true), "audio without clipped linears is not supported");
        let ch: Vec<usize> = c["subsampling_conv_channels"]
            .as_array()
            .context("audio_config.subsampling_conv_channels")?
            .iter()
            .filter_map(|x| x.as_u64().map(|x| x as usize))
            .collect();
        ensure!(ch.len() == 2, "audio_config.subsampling_conv_channels must have two entries");
        let cfg = Self {
            hidden: u("hidden_size")?,
            heads: u("num_attention_heads")?,
            layers: u("num_hidden_layers")?,
            // the sliding window keeps attention_context_left - 1 positions: the query and the ones before it
            context: u("attention_context_left")? - 1,
            conv_kernel: u("conv_kernel_size")?,
            out_dims: u("output_proj_dims")?,
            channels: [ch[0], ch[1]],
            eps: f("rms_norm_eps")? as f32,
            softcap: f("attention_logit_cap")? as f32,
            residual_weight: f("residual_weight")? as f32,
            audio_token: id("audio_token_id")?,
            boa_token: id("boa_token_id")?,
            // the checkpoint's config spells it eoa_token_index
            eoa_token: id("eoa_token_index").or_else(|_| id("eoa_token_id"))?,
        };
        ensure!(cfg.hidden.is_multiple_of(cfg.heads), "audio hidden size not divisible by heads");
        ensure!(MELS / 4 * cfg.channels[1] == cfg.hidden, "audio subsampling output doesn't match the hidden size");
        Ok(Some(cfg))
    }
}

// ---------------------------------------------------------------- decoding

/// 16 kHz mono samples from a WAV file: PCM (8, 16, 24 or 32-bit) or IEEE float (32 or 64-bit), channels averaged.
/// Values are scaled to [-1, 1) as soundfile reads them. Other sampling rates are refused rather than resampled.
pub fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE", "not a WAV file");
    let le16 = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
    let le32 = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let (mut fmt, mut data) = (None, None);
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let (id, len) = (&bytes[at..at + 4], le32(&bytes[at + 4..]) as usize);
        let body = &bytes[at + 8..];
        // a streamed file may leave the data size unset (0xffffffff): take the rest
        let body = &body[..len.min(body.len())];
        match id {
            b"fmt " => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        at += 8 + len + (len & 1);
    }
    let fmt = fmt.context("WAV file has no fmt chunk")?;
    let data = data.context("WAV file has no data chunk")?;
    ensure!(fmt.len() >= 16, "WAV fmt chunk too short");
    let mut tag = le16(fmt);
    let (channels, rate, bits) = (le16(&fmt[2..]) as usize, le32(&fmt[4..]), le16(&fmt[14..]) as usize);
    if tag == 0xfffe {
        // WAVE_FORMAT_EXTENSIBLE: the format is the subformat GUID's first two bytes
        ensure!(fmt.len() >= 26, "WAV extensible fmt chunk too short");
        tag = le16(&fmt[24..]);
    }
    ensure!(channels > 0, "WAV file has no channels");
    if rate != SAMPLE_RATE {
        bail!("WAV is sampled at {rate} Hz; resample it to {SAMPLE_RATE} Hz first");
    }
    let width = bits.div_ceil(8);
    let sample: fn(&[u8]) -> f32 = match (tag, bits) {
        (1, 8) => |b| (b[0] as f32 - 128.0) / 128.0,
        (1, 16) => |b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
        (1, 24) => |b| (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0,
        (1, 32) => |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0,
        (3, 32) => |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        (3, 64) => |b| f64::from_le_bytes(b[..8].try_into().unwrap()) as f32,
        _ => bail!("unsupported WAV encoding (format {tag:#x}, {bits} bits): use PCM or float"),
    };
    let inv = 1.0 / channels as f32;
    Ok(data
        .chunks_exact(width * channels)
        .map(|frame| frame.chunks_exact(width).map(sample).sum::<f32>() * inv)
        .collect())
}

// ---------------------------------------------------------------- features

/// Mel frames the processor keeps for `n` samples: those whose whole window holds audio (after the 30 s cap).
pub fn frames_for(n: usize) -> usize {
    let n = n.min(MAX_SAMPLES);
    // frame i covers samples i * HOP - HOP .. i * HOP + HOP of the input, the last one only as the mask's check
    if n <= HOP { 0 } else { (n - HOP - 1) / HOP + 1 }
}

/// Soft tokens for `n` samples: the mel frames halved twice, rounding up.
pub fn soft_tokens_for(n: usize) -> usize {
    frames_for(n).div_ceil(2).div_ceil(2)
}

/// HTK mel filters, [FFT / 2 + 1][MELS], as transformers' `mel_filter_bank(norm=None, mel_scale="htk")`.
fn mel_filters() -> Vec<[f64; MELS]> {
    let mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let hz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
    // np.linspace(0, mel(MAX_HZ), MELS + 2): i * step, and the end point exactly
    let top = mel(MAX_HZ);
    let step = top / (MELS + 1) as f64;
    let edges: Vec<f64> = (0..MELS + 2).map(|i| hz(if i == MELS + 1 { top } else { i as f64 * step })).collect();
    let bins = FFT / 2 + 1;
    (0..bins)
        .map(|k| {
            let f = (SAMPLE_RATE / 2) as f64 * k as f64 / (bins - 1) as f64;
            std::array::from_fn(|m| {
                let down = (f - edges[m]) / (edges[m + 1] - edges[m]);
                let up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
                down.min(up).max(0.0)
            })
        })
        .collect()
}

/// In-place iterative radix-2 FFT of `FFT` complex values.
fn fft(re: &mut [f64; FFT], im: &mut [f64; FFT], tw: &[(f64, f64)]) {
    let bits = FFT.trailing_zeros();
    for i in 0..FFT {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= FFT {
        let step = FFT / len;
        for start in (0..FFT).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = tw[k * step];
                let (a, b) = (start + k, start + k + len / 2);
                let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                (re[b], im[b]) = (re[a] - xr, im[a] - xi);
                (re[a], im[a]) = (re[a] + xr, im[a] + xi);
            }
        }
        len *= 2;
    }
}

/// Log-mel features for 16 kHz mono samples: [frames, 128], frames = `frames_for(samples.len())`. As the processor:
/// the first frame is centred on the first sample (half a window of zeros before it), each frame is a periodic-Hann
/// window of 320 samples, zero-padded to a 512-point FFT; the magnitudes go through the mel filters and log(x + 1e-3).
/// The math after the window is in f64, as numpy does it.
pub fn log_mel(samples: &[f32]) -> Vec<f32> {
    let n = samples.len().min(MAX_SAMPLES);
    let frames = frames_for(n);
    let window: Vec<f32> =
        (0..FRAME).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FRAME as f64).cos()) as f32).collect();
    let tw: Vec<(f64, f64)> = (0..FFT / 2)
        .map(|k| {
            let a = -2.0 * std::f64::consts::PI * k as f64 / FFT as f64;
            (a.cos(), a.sin())
        })
        .collect();
    let filters = mel_filters();
    let mut out = vec![0.0f32; frames * MELS];
    out.par_chunks_mut(MELS).enumerate().for_each(|(i, dst)| {
        let (mut re, mut im) = ([0.0f64; FFT], [0.0f64; FFT]);
        for (j, w) in window.iter().enumerate() {
            // input sample index, after the HOP zeros the semicausal padding puts in front
            let s = (i * HOP + j).checked_sub(HOP).map_or(0.0, |s| samples[s]);
            re[j] = (s * w) as f64;
        }
        fft(&mut re, &mut im, &tw);
        let mut mel = [0.0f64; MELS];
        for (k, f) in filters.iter().enumerate() {
            let mag = re[k].hypot(im[k]);
            mel.iter_mut().zip(f).for_each(|(m, w)| *m += mag * w);
        }
        dst.iter_mut().zip(mel).for_each(|(d, m)| *d = (m + MEL_FLOOR).ln() as f32);
    });
    out
}

// ---------------------------------------------------------------- the audio tower

/// A linear layer whose input and output are clamped to ranges stored in the checkpoint.
struct Clipped {
    w: Packed,
    input: (f32, f32),
    output: (f32, f32),
}

/// q, k and v stacked into one matmul: they share an input range; each keeps its output range.
struct Qkv {
    w: Packed,
    input: (f32, f32),
    output: [(f32, f32); 3],
}

struct FeedForward {
    pre_norm: Vec<f32>,
    post_norm: Vec<f32>,
    up: Clipped,
    down: Clipped,
}

struct Layer {
    ff1: FeedForward,
    ff2: FeedForward,
    norm_pre_attn: Vec<f32>,
    norm_post_attn: Vec<f32>,
    norm_out: Vec<f32>,
    qkv: Qkv,
    post: Clipped,
    /// Per-dimension query scale: head_dim^-0.5 / ln 2 * softplus(per_dim_scale).
    q_scale: Vec<f32>,
    /// relative_k_proj of the sinusoidal position embeddings: [context + 1, hidden], row r for distance
    /// `context - r`.
    rel_k: Vec<f32>,
    conv_pre_norm: Vec<f32>,
    conv_norm: Vec<f32>,
    /// Rows of [value | gate] for the GLU.
    conv_start: Clipped,
    conv_end: Clipped,
    /// Depthwise causal kernel, [hidden][kernel].
    conv_w: Vec<f32>,
}

/// Per-clip activation buffers, reused by every layer.
struct Scratch {
    x: Vec<f32>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    o: Vec<f32>,
    /// the feed-forward's 4x hidden, also the convolution's GLU input (2x hidden)
    big: Vec<f32>,
}

/// The audio tower's weights, loaded on first use.
pub(crate) struct Audio {
    cfg: AudioConfig,
    /// The two subsampling convolutions, [out][in * 3 * 3]: the first as f32, the second packed for the GEMM.
    conv0: Vec<f32>,
    conv1: Packed,
    norm0: Vec<f32>,
    norm1: Vec<f32>,
    input_proj: Packed,
    layers: Vec<Layer>,
    output_proj: Packed,
    output_bias: Vec<f32>,
    /// embed_audio.embedding_projection: [text hidden, output dims]
    proj: Packed,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn clamp(x: &mut [f32], (lo, hi): (f32, f32)) {
    x.par_chunks_mut(4096).for_each(|c| c.iter_mut().for_each(|v| *v = v.clamp(lo, hi)));
}

/// LayerNorm over the last axis of rows of `d`, with a scale and no bias, then ReLU.
fn layer_norm_relu(x: &mut [f32], d: usize, w: &[f32], eps: f32) {
    x.par_chunks_mut(d).with_min_len(64).for_each(|row| {
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let r = 1.0 / (var + eps).sqrt();
        row.iter_mut().zip(w).for_each(|(v, w)| *v = ((*v - mean) * r * w).max(0.0));
    });
}

impl Clipped {
    fn load(t: &Tensors, p: &str, n: usize, k: usize) -> Result<Self> {
        Ok(Self {
            w: t.packed(&format!("{p}.linear.weight"), n, k)?,
            input: range(t, p, "input")?,
            output: range(t, p, "output")?,
        })
    }

    /// out[m, n] = clamp(clamp(x) W^T); x is clamped in place.
    fn apply(&self, m: usize, x: &mut [f32], out: &mut [f32]) {
        clamp(&mut x[..m * self.w.k], self.input);
        gemm_packed(m, x, self.w.k, &self.w, out, self.w.n);
        clamp(&mut out[..m * self.w.n], self.output);
    }
}

fn range(t: &Tensors, p: &str, which: &str) -> Result<(f32, f32)> {
    let get = |end: &str| -> Result<f32> { Ok(t.bf16_f32(&format!("{p}.{which}_{end}"), &[])?[0]) };
    Ok((get("min")?, get("max")?))
}

impl FeedForward {
    fn load(t: &Tensors, p: &str, d: usize) -> Result<Self> {
        Ok(Self {
            pre_norm: t.bf16_f32(&format!("{p}.pre_layer_norm.weight"), &[d])?,
            post_norm: t.bf16_f32(&format!("{p}.post_layer_norm.weight"), &[d])?,
            up: Clipped::load(t, &format!("{p}.ffw_layer_1"), 4 * d, d)?,
            down: Clipped::load(t, &format!("{p}.ffw_layer_2"), d, 4 * d)?,
        })
    }
}

impl Audio {
    pub(crate) fn load(cfg: &AudioConfig, text_hidden: usize, t: &Tensors) -> Result<Self> {
        let (d, nh) = (cfg.hidden, cfg.heads);
        let hd = d / nh;
        let [c0, c1] = cfg.channels;
        let pos = rel_positions(cfg.context, d);
        let layers = (0..cfg.layers)
            .map(|i| -> Result<Layer> {
                let p = |n: &str| format!("audio_tower.layers.{i}.{n}");
                let names = ["q", "k", "v"].map(|x| p(&format!("self_attn.{x}_proj")));
                let input = range(t, &names[0], "input")?;
                for n in &names[1..] {
                    ensure!(range(t, n, "input")? == input, "{n}: q, k and v have different input ranges");
                }
                let output = [0, 1, 2].map(|j| range(t, &names[j], "output"));
                let [oq, ok, ov] = output;
                let rel_w = t.packed(&p("self_attn.relative_k_proj.weight"), d, d)?;
                let mut rel_k = vec![0.0f32; (cfg.context + 1) * d];
                gemm_packed(cfg.context + 1, &pos, d, &rel_w, &mut rel_k, d);
                let pds = t.bf16_f32(&p("self_attn.per_dim_scale"), &[hd])?;
                let q_scale = pds
                    .iter()
                    .map(|&s| (hd as f32).powf(-0.5) / std::f32::consts::LN_2 * (1.0 + s.exp()).ln())
                    .collect();
                Ok(Layer {
                    ff1: FeedForward::load(t, &p("feed_forward1"), d)?,
                    ff2: FeedForward::load(t, &p("feed_forward2"), d)?,
                    norm_pre_attn: t.bf16_f32(&p("norm_pre_attn.weight"), &[d])?,
                    norm_post_attn: t.bf16_f32(&p("norm_post_attn.weight"), &[d])?,
                    norm_out: t.bf16_f32(&p("norm_out.weight"), &[d])?,
                    qkv: Qkv {
                        w: t.stacked(&names.clone().map(|n| format!("{n}.linear.weight")), d, d)?,
                        input,
                        output: [oq?, ok?, ov?],
                    },
                    post: Clipped::load(t, &p("self_attn.post"), d, d)?,
                    q_scale,
                    rel_k,
                    conv_pre_norm: t.bf16_f32(&p("lconv1d.pre_layer_norm.weight"), &[d])?,
                    conv_norm: t.bf16_f32(&p("lconv1d.conv_norm.weight"), &[d])?,
                    conv_start: Clipped::load(t, &p("lconv1d.linear_start"), 2 * d, d)?,
                    conv_end: Clipped::load(t, &p("lconv1d.linear_end"), d, d)?,
                    conv_w: t.bf16_f32(&p("lconv1d.depthwise_conv1d.weight"), &[d, 1, cfg.conv_kernel])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let sub = "audio_tower.subsample_conv_projection";
        Ok(Self {
            conv0: t.bf16_f32(&format!("{sub}.layer0.conv.weight"), &[c0, 1, 3, 3])?,
            conv1: Packed::from_le_bytes(
                t.bytes(&format!("{sub}.layer1.conv.weight"), "BF16", &[c1, c0, 3, 3])?,
                c1,
                c0 * 9,
            ),
            norm0: t.bf16_f32(&format!("{sub}.layer0.norm.weight"), &[c0])?,
            norm1: t.bf16_f32(&format!("{sub}.layer1.norm.weight"), &[c1])?,
            input_proj: t.packed(&format!("{sub}.input_proj_linear.weight"), d, d)?,
            layers,
            output_proj: t.packed("audio_tower.output_proj.weight", cfg.out_dims, d)?,
            output_bias: t.bf16_f32("audio_tower.output_proj.bias", &[cfg.out_dims])?,
            proj: t.packed("embed_audio.embedding_projection.weight", text_hidden, cfg.out_dims)?,
            cfg: cfg.clone(),
        })
    }

    /// Soft tokens for log-mel features ([frames, 128], from `log_mel`): [ceil(frames / 4), text hidden].
    pub(crate) fn forward(&self, mel: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.hidden;
        let mut h = trace("audio subsample", || self.subsample(mel));
        let t = h.len() / d;
        let mut s = Scratch {
            x: vec![0.0; t * d],
            qkv: vec![0.0; t * 3 * d],
            attn: vec![0.0; t * d],
            o: vec![0.0; t * d],
            big: vec![0.0; t * 4 * d],
        };
        for l in &self.layers {
            self.layer_forward(l, &mut h, &mut s, t);
        }
        let od = cfg.out_dims;
        let mut out = vec![0.0f32; t * od];
        trace("audio output proj", || {
            gemm_packed(t, &h, d, &self.output_proj, &mut out, od);
            out.par_chunks_mut(od).for_each(|row| {
                row.iter_mut().zip(&self.output_bias).for_each(|(v, b)| *v += b);
                rmsnorm(row, None, cfg.eps);
            });
        });
        let n_text = self.proj.n;
        let mut soft = vec![0.0f32; t * n_text];
        trace("audio project", || gemm_packed(t, &out, od, &self.proj, &mut soft, n_text));
        soft
    }

    /// The two stride-2 3x3 convolutions (each with LayerNorm over channels and ReLU) over [time, mel] and the input
    /// projection: [ceil(frames / 4), hidden]. Activations are laid out [time][frequency][channel].
    fn subsample(&self, mel: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let [c0, c1] = cfg.channels;
        let t_in = mel.len() / MELS;
        let (t0, f0) = (t_in.div_ceil(2), MELS / 2);
        let (t1, f1) = (t0.div_ceil(2), f0 / 2);
        // the input around output (t, f), tap (ky, kx): row 2t - 1 + ky, column 2f - 1 + kx, zero outside
        let tap = |o: usize, k: usize, n: usize| (2 * o + k).checked_sub(1).filter(|&i| i < n);

        let mut a = vec![0.0f32; t0 * f0 * c0];
        a.par_chunks_mut(f0 * c0).enumerate().for_each(|(ot, row)| {
            for (of, out) in row.chunks_exact_mut(c0).enumerate() {
                let mut x = [0.0f32; 9];
                for ky in 0..3 {
                    for kx in 0..3 {
                        if let (Some(y), Some(xx)) = (tap(ot, ky, t_in), tap(of, kx, MELS)) {
                            x[ky * 3 + kx] = mel[y * MELS + xx];
                        }
                    }
                }
                for (c, o) in out.iter_mut().enumerate() {
                    *o = self.conv0[c * 9..c * 9 + 9].iter().zip(&x).map(|(w, x)| w * x).sum();
                }
            }
        });
        layer_norm_relu(&mut a, c0, &self.norm0, cfg.eps);

        // second convolution as a GEMM over patches ordered (in channel, ky, kx), like the weight
        let k = c0 * 9;
        let mut patches = vec![0.0f32; t1 * f1 * k];
        patches.par_chunks_mut(f1 * k).enumerate().for_each(|(ot, row)| {
            for (of, p) in row.chunks_exact_mut(k).enumerate() {
                for ky in 0..3 {
                    for kx in 0..3 {
                        if let (Some(y), Some(x)) = (tap(ot, ky, t0), tap(of, kx, f0)) {
                            let src = &a[(y * f0 + x) * c0..][..c0];
                            for (ci, v) in src.iter().enumerate() {
                                p[ci * 9 + ky * 3 + kx] = *v;
                            }
                        }
                    }
                }
            }
        });
        let mut b = vec![0.0f32; t1 * f1 * c1];
        gemm_packed(t1 * f1, &patches, k, &self.conv1, &mut b, c1);
        layer_norm_relu(&mut b, c1, &self.norm1, cfg.eps);
        // each frame's [frequency][channel] block is the projection's input
        let mut h = vec![0.0f32; t1 * cfg.hidden];
        gemm_packed(t1, &b, f1 * c1, &self.input_proj, &mut h, cfg.hidden);
        h
    }

    fn feed_forward(&self, ff: &FeedForward, h: &mut [f32], s: &mut Scratch, t: usize) {
        let (d, eps, w) = (self.cfg.hidden, self.cfg.eps, self.cfg.residual_weight);
        let Scratch { x, o, big, .. } = s;
        let big = &mut big[..t * 4 * d];
        x.par_chunks_mut(d).zip(h.par_chunks(d)).for_each(|(x, h)| {
            x.copy_from_slice(h);
            rmsnorm(x, Some(&ff.pre_norm), eps);
        });
        ff.up.apply(t, x, big);
        big.par_chunks_mut(4096).for_each(|c| c.iter_mut().for_each(|v| *v = silu(*v)));
        ff.down.apply(t, big, o);
        h.par_chunks_mut(d).zip(o.par_chunks_mut(d)).for_each(|(h, o)| {
            rmsnorm(o, Some(&ff.post_norm), eps);
            h.iter_mut().zip(o.iter()).for_each(|(h, o)| *h += w * o);
        });
    }

    fn layer_forward(&self, l: &Layer, h: &mut [f32], s: &mut Scratch, t: usize) {
        let cfg = &self.cfg;
        let (d, eps) = (cfg.hidden, cfg.eps);
        trace("audio feed-forward", || self.feed_forward(&l.ff1, h, s, t));
        trace("audio attention", || self.attention(l, h, s, t));
        trace("audio conv", || self.conv(l, h, s, t));
        trace("audio feed-forward", || self.feed_forward(&l.ff2, h, s, t));
        h.par_chunks_mut(d).for_each(|row| rmsnorm(row, Some(&l.norm_out), eps));
    }

    /// Chunked local attention, as a sliding window: each frame attends to itself and the `context - 1` frames
    /// before it, with a learned relative-position term and a tanh soft cap on the logits.
    fn attention(&self, l: &Layer, h: &mut [f32], s: &mut Scratch, t: usize) {
        let cfg = &self.cfg;
        let (d, eps, nh, ctx, cap) = (cfg.hidden, cfg.eps, cfg.heads, cfg.context, cfg.softcap);
        let hd = d / nh;
        let Scratch { x, qkv, attn, o, .. } = s;
        x.par_chunks_mut(d).zip(h.par_chunks(d)).for_each(|(x, h)| {
            x.copy_from_slice(h);
            rmsnorm(x, Some(&l.norm_pre_attn), eps);
        });
        clamp(x, l.qkv.input);
        gemm_packed(t, x, d, &l.qkv.w, qkv, 3 * d);
        // Clamp each projection to its range, then scale the queries (the keys' scale is applied to q.k below: the
        // relative keys don't get it).
        qkv.par_chunks_mut(3 * d).for_each(|row| {
            for (j, part) in row.chunks_exact_mut(d).enumerate() {
                let (lo, hi) = l.qkv.output[j];
                part.iter_mut().for_each(|v| *v = v.clamp(lo, hi));
            }
            for head in row[..d].chunks_exact_mut(hd) {
                head.iter_mut().zip(&l.q_scale).for_each(|(v, s)| *v *= s);
            }
        });
        let k_scale = (1.0 + std::f32::consts::E).ln() / std::f32::consts::LN_2;
        let qkv = &*qkv;
        attn.par_chunks_mut(d).enumerate().for_each(|(i, out)| {
            let q = &qkv[i * 3 * d..][..d];
            let first = (i + 1).saturating_sub(ctx);
            let mut logits = [0.0f32; 64];
            assert!(ctx <= logits.len());
            for hh in 0..nh {
                let qh = &q[hh * hd..][..hd];
                let n = i + 1 - first;
                for (w, j) in logits[..n].iter_mut().zip(first..=i) {
                    let k = &qkv[j * 3 * d + d + hh * hd..][..hd];
                    let r = &l.rel_k[(ctx - (i - j)) * d + hh * hd..][..hd];
                    let ac: f32 = qh.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * k_scale;
                    let bd: f32 = qh.iter().zip(r).map(|(a, b)| a * b).sum();
                    *w = ((ac + bd) / cap).tanh() * cap;
                }
                let m = logits[..n].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0;
                logits[..n].iter_mut().for_each(|w| {
                    *w = (*w - m).exp();
                    sum += *w;
                });
                let oh = &mut out[hh * hd..][..hd];
                oh.fill(0.0);
                for (w, j) in logits[..n].iter().zip(first..=i) {
                    let v = &qkv[j * 3 * d + 2 * d + hh * hd..][..hd];
                    oh.iter_mut().zip(v).for_each(|(o, v)| *o += w / sum * v);
                }
            }
        });
        l.post.apply(t, attn, o);
        h.par_chunks_mut(d).zip(o.par_chunks_mut(d)).for_each(|(h, o)| {
            rmsnorm(o, Some(&l.norm_post_attn), eps);
            h.iter_mut().zip(o.iter()).for_each(|(h, o)| *h += o);
        });
    }

    /// The light convolution block: GLU, a causal depthwise convolution over time, RMS norm, SiLU, and back.
    fn conv(&self, l: &Layer, h: &mut [f32], s: &mut Scratch, t: usize) {
        let (d, eps, kw) = (self.cfg.hidden, self.cfg.eps, self.cfg.conv_kernel);
        let Scratch { x, attn: g, o, big, .. } = s;
        let big = &mut big[..t * 2 * d];
        x.par_chunks_mut(d).zip(h.par_chunks(d)).for_each(|(x, h)| {
            x.copy_from_slice(h);
            rmsnorm(x, Some(&l.conv_pre_norm), eps);
        });
        l.conv_start.apply(t, x, big);
        g.par_chunks_mut(d).zip(big.par_chunks(2 * d)).for_each(|(g, row)| {
            let (a, b) = row.split_at(d);
            g.iter_mut().zip(a.iter().zip(b)).for_each(|(g, (a, b))| *g = a / (1.0 + (-b).exp()));
        });
        let g = &*g;
        x.par_chunks_mut(d).enumerate().for_each(|(i, out)| {
            for (c, o) in out.iter_mut().enumerate() {
                let w = &l.conv_w[c * kw..][..kw];
                // tap j reads frame i - (kw - 1) + j
                *o = (0..kw).filter_map(|j| (i + j).checked_sub(kw - 1).map(|f| w[j] * g[f * d + c])).sum();
            }
            rmsnorm(out, Some(&l.conv_norm), eps);
            out.iter_mut().for_each(|v| *v = silu(*v));
        });
        l.conv_end.apply(t, x, o);
        h.par_chunks_mut(d).zip(o.par_chunks(d)).for_each(|(h, o)| h.iter_mut().zip(o).for_each(|(h, o)| *h += o));
    }
}

/// The sinusoidal relative position embeddings for distances `ctx` down to 0: [ctx + 1, d], each row
/// [sin | cos] over d / 2 timescales from 1 to 10^4.
fn rel_positions(ctx: usize, d: usize) -> Vec<f32> {
    let n = d / 2;
    let inc = (10_000f64.ln() / (n - 1).max(1) as f64) as f32;
    let inv: Vec<f32> = (0..n).map(|i| (i as f32 * -inc).exp()).collect();
    let mut out = vec![0.0f32; (ctx + 1) * d];
    for (r, row) in out.chunks_exact_mut(d).enumerate() {
        let p = (ctx - r) as f32;
        let (s, c) = row.split_at_mut(n);
        for ((s, c), f) in s.iter_mut().zip(c).zip(&inv) {
            (*s, *c) = ((p * f).sin(), (p * f).cos());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_and_token_counts_match_the_processor() {
        // from transformers' processor: (samples, mel frames, soft tokens)
        for (n, frames, tokens) in [(8000, 49, 13), (16000, 99, 25), (4800, 29, 8), (560_000, 2999, 750)] {
            assert_eq!(frames_for(n), frames, "{n} samples");
            assert_eq!(soft_tokens_for(n), tokens, "{n} samples");
        }
    }

    #[test]
    fn fft_matches_a_direct_dft() {
        let tw: Vec<(f64, f64)> = (0..FFT / 2)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / FFT as f64;
                (a.cos(), a.sin())
            })
            .collect();
        let x: Vec<f64> = (0..FFT).map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0).collect();
        let (mut re, mut im) = ([0.0; FFT], [0.0; FFT]);
        re.copy_from_slice(&x);
        fft(&mut re, &mut im, &tw);
        for k in [0, 1, 7, 100, 256] {
            let (mut r, mut i) = (0.0, 0.0);
            for (n, v) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * n) as f64 / FFT as f64;
                r += v * a.cos();
                i += v * a.sin();
            }
            assert!((re[k] - r).abs() < 1e-9 && (im[k] - i).abs() < 1e-9, "bin {k}");
        }
    }
}
