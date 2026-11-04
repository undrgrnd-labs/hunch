//! EmbeddingGemma 2 image path: decoding and preprocessing, the vision tower, pooling, and the projection into the
//! text tower's input space.
//!
//! Mirrors `Gemma4ImageProcessor` (the torchvision backend `AutoProcessor` loads) and
//! `EmbeddingGemma2Model.get_image_features`: the image is resized, keeping its aspect ratio, to the largest size
//! with at most `max_soft_tokens * 9` patches of 16x16 and sides divisible by 48; a 16-layer ViT with 2-D RoPE
//! encodes the patches; 3x3 average pooling gives the soft tokens, which are RMS-normalized and projected to the
//! text tower's 512-d input. The text tower then encodes `<bos> <boi> soft tokens <eoi> <eos>`.

use crate::model::{Rope, add_norm, add_norm_then_norm, attention, rmsnorm, trace};
use crate::simd::{Packed, bf16_to_f32, gelu_mul, gemm_packed};
use crate::tensors::Tensors;
use anyhow::{Context, Result, anyhow, bail, ensure};
use image::ImageDecoder;
use image::metadata::Orientation;
use rayon::prelude::*;
use serde_json::Value;

/// Soft-token budgets the image processor accepts.
pub const SOFT_TOKEN_BUDGETS: [usize; 5] = [70, 140, 280, 560, 1120];
/// The processor's default budget.
pub const DEFAULT_SOFT_TOKENS: usize = 280;

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub patch: usize,
    pub pool: usize,
    pub pos_size: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub image_token: u32,
    pub boi_token: u32,
    pub eoi_token: u32,
}

impl VisionConfig {
    pub(crate) fn from_json(v: &Value) -> Result<Option<Self>> {
        let c = &v["vision_config"];
        if c.is_null() {
            return Ok(None);
        }
        let u = |k: &str| c[k].as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("vision_config.{k} missing"));
        let id = |k: &str| v[k].as_u64().map(|x| x as u32).ok_or_else(|| anyhow!("{k} missing"));
        ensure!(c["rope_parameters"]["rope_type"].as_str() == Some("axial"), "vision rope is not axial");
        ensure!(!c["standardize"].as_bool().unwrap_or(false), "vision standardize is not supported");
        ensure!(!c["use_clipped_linears"].as_bool().unwrap_or(false), "vision clipped linears are not supported");
        ensure!(u("num_key_value_heads")? == u("num_attention_heads")?, "vision GQA is not supported");
        Ok(Some(Self {
            hidden: u("hidden_size")?,
            intermediate: u("intermediate_size")?,
            heads: u("num_attention_heads")?,
            head_dim: u("head_dim")?,
            layers: u("num_hidden_layers")?,
            patch: u("patch_size")?,
            pool: u("pooling_kernel_size")?,
            pos_size: u("position_embedding_size")?,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta: c["rope_parameters"]["rope_theta"].as_f64().context("vision rope_theta")?,
            image_token: id("image_token_id")?,
            boi_token: id("boi_token_id")?,
            eoi_token: id("eoi_token_id")?,
        }))
    }
}

// ---------------------------------------------------------------- preprocessing

/// A decoded RGB image, row-major, 3 bytes per pixel.
#[derive(Clone, Debug)]
pub struct Rgb {
    pub height: usize,
    pub width: usize,
    pub data: Vec<u8>,
}

/// Decode PNG, JPEG or WebP bytes to RGB, applying the EXIF orientation (as `transformers.image_utils.load_image`
/// does) and dropping any alpha channel (as PIL's `convert("RGB")` does).
///
/// JPEGs go through `jpeg-decoder`. About 5% of its decoded values differ from the libjpeg-turbo PIL uses, by at
/// most a few levels, since the IDCT and chroma upsampling differ.
pub fn decode(bytes: &[u8]) -> Result<Rgb> {
    if bytes.starts_with(&[0xff, 0xd8, 0xff])
        && let Some(img) = decode_jpeg(bytes)?
    {
        return Ok(img);
    }
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut decoder = reader.into_decoder().context("unsupported image format")?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(decoder).context("decoding image")?;
    img.apply_orientation(orientation);
    into_rgb(img)
}

/// RGB or grayscale baseline/progressive JPEGs; `None` for CMYK, which the `image` crate converts.
fn decode_jpeg(bytes: &[u8]) -> Result<Option<Rgb>> {
    use jpeg_decoder::PixelFormat;
    let mut dec = jpeg_decoder::Decoder::new(bytes);
    let px = dec.decode().context("decoding JPEG")?;
    let info = dec.info().context("JPEG without a frame")?;
    let (width, height) = (info.width as u32, info.height as u32);
    let rgb = match info.pixel_format {
        PixelFormat::RGB24 => px,
        PixelFormat::L8 => px.iter().flat_map(|&v| [v, v, v]).collect(),
        PixelFormat::L16 => px.chunks_exact(2).flat_map(|v| [v[0], v[0], v[0]]).collect(),
        PixelFormat::CMYK32 => return Ok(None),
    };
    let buf = image::RgbImage::from_raw(width, height, rgb).context("JPEG size")?;
    let mut img = image::DynamicImage::ImageRgb8(buf);
    if let Some(o) = dec.exif_data().and_then(Orientation::from_exif_chunk) {
        img.apply_orientation(o);
    }
    into_rgb(img).map(Some)
}

/// The decoded image as `Rgb`, without any alpha channel.
fn into_rgb(img: image::DynamicImage) -> Result<Rgb> {
    let rgb = img.into_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    ensure!(width > 0 && height > 0, "empty image");
    Ok(Rgb { height, width, data: rgb.into_raw() })
}

/// The processor's `get_aspect_ratio_preserving_size`: the largest (height, width) with at most `max_patches`
/// patches and both sides multiples of `patch * pool`.
pub fn target_size(
    height: usize, width: usize, patch: usize, pool: usize, max_patches: usize,
) -> Result<(usize, usize)> {
    let target_px = (max_patches * patch * patch) as f64;
    let factor = (target_px / (height * width) as f64).sqrt();
    let side = pool * patch;
    let mut th = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * width as f64) / side as f64).floor() as usize * side;
    if th == 0 && tw == 0 {
        bail!("cannot resize a {height}x{width} image to fit {max_patches} patches");
    }
    let max_side = (max_patches / (pool * pool)) * side;
    if th == 0 {
        th = side;
        tw = ((width / height) * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((height / width) * side).min(max_side);
    }
    ensure!((th * tw) as f64 <= target_px, "resizing {height}x{width} to {th}x{tw} exceeds {max_patches} patches");
    Ok((th, tw))
}

/// Keys' bicubic kernel, a = -0.5 (what PIL and torchvision's antialiased resize use).
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per output index along one axis: the first input index and int16 fixed-point weights, plus the precision.
/// Mirrors ATen's `_compute_index_ranges_int16_weights` for antialiased bicubic on uint8.
fn resize_weights(insz: usize, outsz: usize) -> (Vec<(usize, Vec<i32>)>, u32) {
    let scale = insz as f64 / outsz as f64;
    let (support, invscale) = if scale >= 1.0 { (2.0 * scale, 1.0 / scale) } else { (2.0, 1.0) };
    let max_interp = support.ceil() as i64 * 2 + 1;
    let mut wt_max = 0.0f64;
    let rows: Vec<(usize, Vec<f64>)> = (0..outsz)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xsize = (((center + support + 0.5) as i64).min(insz as i64) - xmin).clamp(0, max_interp);
            let mut w: Vec<f64> =
                (0..xsize).map(|j| cubic((j as f64 + xmin as f64 - center + 0.5) * invscale)).collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                w.iter_mut().for_each(|v| *v /= total);
            }
            wt_max = w.iter().fold(wt_max, |m, &v| m.max(v));
            (xmin as usize, w)
        })
        .collect();
    let mut prec = 0u32;
    while prec < 22 && ((0.5 + wt_max * (1u64 << (prec + 1)) as f64) as i64) < (1 << 15) {
        prec += 1;
    }
    let rows = rows
        .into_iter()
        .map(|(xmin, w)| {
            let wi = w
                .iter()
                .map(|&v| {
                    let v = v * (1u64 << prec) as f64;
                    if v < 0.0 { (-0.5 + v) as i32 } else { (0.5 + v) as i32 }
                })
                .collect();
            (xmin, wi)
        })
        .collect();
    (rows, prec)
}

/// One antialiased bicubic pass along the middle axis of a [outer, n, inner] uint8 array, giving [outer, out, inner].
fn resize_axis(src: &[u8], outer: usize, n: usize, inner: usize, out: usize) -> Vec<u8> {
    let (ws, prec) = resize_weights(n, out);
    let round = 1i32 << (prec - 1);
    let mut dst = vec![0u8; outer * out * inner];
    if inner == 3 {
        // Along rows of interleaved RGB: one output row per task, three accumulators per output pixel.
        dst.par_chunks_mut(out * 3).zip(src.par_chunks(n * 3)).for_each(|(dst, row)| {
            for ((xmin, w), px) in ws.iter().zip(dst.chunks_exact_mut(3)) {
                let mut a = [round; 3];
                for (s, &wj) in row[xmin * 3..(xmin + w.len()) * 3].chunks_exact(3).zip(w) {
                    a[0] += s[0] as i32 * wj;
                    a[1] += s[1] as i32 * wj;
                    a[2] += s[2] as i32 * wj;
                }
                for c in 0..3 {
                    px[c] = (a[c] >> prec).clamp(0, 255) as u8;
                }
            }
        });
        return dst;
    }
    dst.par_chunks_mut(inner).with_min_len((1024 / inner).max(1)).enumerate().for_each_init(
        Vec::new,
        |acc, (oi, dst)| {
            let (o, (xmin, w)) = (oi / out, &ws[oi % out]);
            acc.clear();
            acc.resize(inner, round);
            for (j, &wj) in w.iter().enumerate() {
                let row = &src[(o * n + xmin + j) * inner..][..inner];
                acc.iter_mut().zip(row).for_each(|(a, &x)| *a += x as i32 * wj);
            }
            dst.iter_mut().zip(acc.iter()).for_each(|(d, &a)| *d = (a >> prec).clamp(0, 255) as u8);
        },
    );
    dst
}

/// Resize as `torchvision.transforms.v2.functional.resize(uint8 tensor, bicubic, antialias=True)` on CPU: width
/// first, then height, each rounded to uint8, a pass skipped when that side keeps its size. Bit-exact.
pub fn resize(img: &Rgb, height: usize, width: usize) -> Rgb {
    let mut data = img.data.clone();
    if width != img.width {
        data = resize_axis(&data, img.height, img.width, 3, width);
    }
    if height != img.height {
        data = resize_axis(&data, 1, img.height, width * 3, height);
    }
    Rgb { height, width, data }
}

/// Decode and resize for a soft-token budget (one of `SOFT_TOKEN_BUDGETS`).
pub(crate) fn preprocess(cfg: &VisionConfig, bytes: &[u8], max_soft_tokens: usize) -> Result<Rgb> {
    ensure!(SOFT_TOKEN_BUDGETS.contains(&max_soft_tokens), "max_soft_tokens must be one of {SOFT_TOKEN_BUDGETS:?}");
    let img = trace("image decode", || decode(bytes))?;
    let (h, w) = target_size(img.height, img.width, cfg.patch, cfg.pool, max_soft_tokens * cfg.pool * cfg.pool)?;
    Ok(if (h, w) == (img.height, img.width) { img } else { trace("image resize", || resize(&img, h, w)) })
}

// ---------------------------------------------------------------- the vision tower

struct Layer {
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    pre_ff_ln: Vec<f32>,
    post_ff_ln: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// q, k and v projections stacked: one matmul gives rows of [q | k | v].
    wqkv: Packed,
    wo: Packed,
    /// gate and up projections stacked: rows of [gate | up].
    w_gate_up: Packed,
    w_down: Packed,
}

/// Per-image activation buffers, reused by every layer.
struct Scratch {
    x: Vec<f32>,
    qkv: Vec<f32>,
    /// q and k transposed per head, for the SME attention
    qkt: Vec<f32>,
    attn: Vec<f32>,
    o: Vec<f32>,
    gu: Vec<f32>,
}

/// The vision tower's weights, loaded on first use. The 2 x 10240 x 768 position table stays memory-mapped.
pub(crate) struct Vision {
    cfg: VisionConfig,
    input_proj: Packed,
    pos_offset: usize,
    layers: Vec<Layer>,
    /// embed_vision.embedding_projection: [text hidden, vision hidden]
    proj: Packed,
}

impl Vision {
    /// The tower's weights from the checkpoint; the position table stays in it, at `pos_offset`.
    pub(crate) fn load(cfg: &VisionConfig, text_hidden: usize, t: &Tensors) -> Result<Self> {
        let (d, hd, nh, inter) = (cfg.hidden, cfg.head_dim, cfg.heads, cfg.intermediate);
        let layers = (0..cfg.layers)
            .map(|i| -> Result<Layer> {
                let p = |n: &str| format!("vision_tower.encoder.layers.{i}.{n}");
                Ok(Layer {
                    input_ln: t.bf16_f32(&p("input_layernorm.weight"), &[d])?,
                    post_attn_ln: t.bf16_f32(&p("post_attention_layernorm.weight"), &[d])?,
                    pre_ff_ln: t.bf16_f32(&p("pre_feedforward_layernorm.weight"), &[d])?,
                    post_ff_ln: t.bf16_f32(&p("post_feedforward_layernorm.weight"), &[d])?,
                    q_norm: t.bf16_f32(&p("self_attn.q_norm.weight"), &[hd])?,
                    k_norm: t.bf16_f32(&p("self_attn.k_norm.weight"), &[hd])?,
                    wqkv: t.stacked(
                        &["q", "k", "v"].map(|x| p(&format!("self_attn.{x}_proj.linear.weight"))),
                        nh * hd,
                        d,
                    )?,
                    wo: t.packed(&p("self_attn.o_proj.linear.weight"), d, nh * hd)?,
                    w_gate_up: t.stacked(
                        &[p("mlp.gate_proj.linear.weight"), p("mlp.up_proj.linear.weight")],
                        inter,
                        d,
                    )?,
                    w_down: t.packed(&p("mlp.down_proj.linear.weight"), d, inter)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let pp = 3 * cfg.patch * cfg.patch;
        Ok(Self {
            input_proj: t.packed("vision_tower.patch_embedder.input_proj.weight", d, pp)?,
            pos_offset: t.offset(
                "vision_tower.patch_embedder.position_embedding_table",
                "BF16",
                &[2, cfg.pos_size, d],
            )?,
            proj: t.packed("embed_vision.embedding_projection.weight", text_hidden, d)?,
            layers,
            cfg: cfg.clone(),
        })
    }

    /// Soft tokens for one preprocessed image: [n, text hidden], n = patches / pool^2, in row-major order of the
    /// pooled grid.
    pub(crate) fn forward(&self, mmap: &[u8], img: &Rgb) -> Vec<f32> {
        let cfg = &self.cfg;
        let (ps, d, k) = (cfg.patch, cfg.hidden, cfg.pool);
        let (gh, gw) = (img.height / ps, img.width / ps);
        assert!(gh * ps == img.height && gw * ps == img.width && gh % k == 0 && gw % k == 0, "image not preprocessed");
        assert!(gh.max(gw) <= cfg.pos_size, "image too large for the position table");
        let t = gh * gw;

        // Patches, each flattened as (row, column, channel), scaled to [-1, 1] as the patch embedder does.
        let pp = 3 * ps * ps;
        let mut patches = vec![0.0f32; t * pp];
        trace("vision patches", || {
            patches.par_chunks_mut(pp).enumerate().for_each(|(i, dst)| {
                let (py, px) = (i / gw, i % gw);
                for r in 0..ps {
                    let row = &img.data[((py * ps + r) * img.width + px * ps) * 3..][..ps * 3];
                    for (o, &b) in dst[r * ps * 3..(r + 1) * ps * 3].iter_mut().zip(row) {
                        *o = 2.0 * (b as f32 * (1.0f32 / 255.0) - 0.5);
                    }
                }
            })
        });
        let mut h = vec![0.0f32; t * d];
        trace("vision embed", || {
            gemm_packed(t, &patches, pp, &self.input_proj, &mut h, d);
            let table = &mmap[self.pos_offset..];
            let row = |axis: usize, i: usize| &table[(axis * cfg.pos_size + i) * d * 2..][..d * 2];
            h.par_chunks_mut(d).enumerate().for_each(|(i, h)| {
                let (xs, ys) = (row(0, i % gw), row(1, i / gw));
                for ((v, x), y) in h.iter_mut().zip(xs.chunks_exact(2)).zip(ys.chunks_exact(2)) {
                    let (x, y) =
                        (bf16_to_f32(u16::from_le_bytes([x[0], x[1]])), bf16_to_f32(u16::from_le_bytes([y[0], y[1]])));
                    *v += x + y;
                }
            });
        });
        drop(patches);

        let rope = Rope::new(cfg.rope_theta, cfg.head_dim / 2, gh.max(gw));
        // Activations reused across layers: fresh multi-megabyte buffers per layer cost a page fault per page.
        let qd = cfg.heads * cfg.head_dim;
        let mut s = Scratch {
            x: vec![0.0; t * d],
            qkv: vec![0.0; t * 3 * qd],
            qkt: vec![0.0; 2 * qd * t],
            attn: vec![0.0; t * qd],
            o: vec![0.0; t * d],
            gu: vec![0.0; t * 2 * cfg.intermediate],
        };
        for l in &self.layers {
            self.layer_forward(l, &mut h, &mut s, &rope, gw, t);
        }

        // 3x3 average pooling over the patch grid, then the sqrt(hidden) scale.
        let (oh, ow) = (gh / k, gw / k);
        let mut pooled = vec![0.0f32; oh * ow * d];
        let (inv, scale) = (1.0 / (k * k) as f32, (d as f64).sqrt() as f32);
        trace("vision pool", || {
            pooled.par_chunks_mut(d).enumerate().for_each(|(o, dst)| {
                let (oy, ox) = (o / ow, o % ow);
                for dy in 0..k {
                    for dx in 0..k {
                        let src = &h[((oy * k + dy) * gw + ox * k + dx) * d..][..d];
                        dst.iter_mut().zip(src).for_each(|(a, b)| *a += b * inv);
                    }
                }
                dst.iter_mut().for_each(|a| *a *= scale);
                rmsnorm(dst, None, cfg.eps);
            })
        });
        let n_text = self.proj.n;
        let mut out = vec![0.0f32; oh * ow * n_text];
        trace("vision project", || gemm_packed(oh * ow, &pooled, d, &self.proj, &mut out, n_text));
        out
    }

    fn layer_forward(&self, l: &Layer, h: &mut [f32], s: &mut Scratch, rope: &Rope, gw: usize, t: usize) {
        let cfg = &self.cfg;
        let (d, eps, nh, hd, inter) = (cfg.hidden, cfg.eps, cfg.heads, cfg.head_dim, cfg.intermediate);
        let qd = nh * hd;
        // qkt is only read by the SME attention.
        #[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]
        let Scratch { x, qkv, qkt, attn, o, gu } = s;
        trace("vision norm", || {
            x.par_chunks_mut(d).zip(h.par_chunks(d)).with_min_len(64).for_each(|(x, h)| {
                x.copy_from_slice(h);
                rmsnorm(x, Some(&l.input_ln), eps);
            })
        });
        trace("vision gemm qkv", || gemm_packed(t, x, d, &l.wqkv, qkv, 3 * qd));
        // Per-head norms, then 2-D RoPE: the first half of each head rotates by the patch column, the second by
        // the row.
        let half = hd / 2;
        trace("vision qkv norm+rope", || {
            qkv.par_chunks_mut(3 * qd).with_min_len(32).enumerate().for_each(|(pos, row)| {
                let (px, py) = (pos % gw, pos / gw);
                let (qk, v) = row.split_at_mut(2 * qd);
                for (i, head) in qk.chunks_exact_mut(hd).enumerate() {
                    rmsnorm(head, Some(if i < nh { &l.q_norm } else { &l.k_norm }), eps);
                    let (a, b) = head.split_at_mut(half);
                    rope.apply(a, px);
                    rope.apply(b, py);
                }
                for head in v.chunks_exact_mut(hd) {
                    rmsnorm(head, None, eps);
                }
            })
        });
        trace("vision attention", || {
            #[cfg(target_arch = "aarch64")]
            if crate::model::sme() {
                return crate::sme::attention(qkv, 3 * qd, t, nh, hd, qkt, attn);
            }
            let part = |i: usize| -> Vec<f32> {
                qkv.chunks_exact(3 * qd).flat_map(|r| &r[i * qd..(i + 1) * qd]).copied().collect()
            };
            attn.copy_from_slice(&attention(&part(0), &part(1), &part(2), t, nh, nh, hd, None));
        });
        trace("vision gemm o", || gemm_packed(t, attn, qd, &l.wo, o, d));
        trace("vision norm+add", || add_norm_then_norm(h, o, &l.post_attn_ln, x, &l.pre_ff_ln, d, eps));
        trace("vision gemm gate+up", || gemm_packed(t, x, d, &l.w_gate_up, gu, 2 * inter));
        trace("vision gelu", || {
            gu.par_chunks_mut(2 * inter).with_min_len(16).for_each(|row| {
                let (g, u) = row.split_at_mut(inter);
                gelu_mul(g, u)
            })
        });
        trace("vision gemm down", || gemm_packed(t, gu, 2 * inter, &l.w_down, o, d));
        trace("vision norm+add", || add_norm(h, o, &l.post_ff_ln, d, eps));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_sizes_match_the_processor() {
        // values from transformers' get_aspect_ratio_preserving_size
        assert_eq!(target_size(200, 300, 16, 3, 2520).unwrap(), (624, 960));
        assert_eq!(target_size(512, 512, 16, 3, 2520).unwrap(), (768, 768));
        assert_eq!(target_size(240, 160, 16, 3, 2520).unwrap(), (960, 624));
        assert_eq!(target_size(24, 40, 16, 3, 630).unwrap(), (288, 480));
        assert_eq!(target_size(10, 2000, 16, 3, 630).unwrap(), (48, 3360));
    }
}
