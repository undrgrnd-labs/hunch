//! Arm SME kernels: fp32 outer products accumulated in the ZA tiles, for the matmuls and attention products.
//!
//! Apple M4 and later have a matrix unit per CPU cluster that SME drives. Measured on an M5 Pro, fp32 FMOPA runs
//! at about 1.5 TFLOP/s on one core and 2.5-3 TFLOP/s across the chip, against about 95 GFLOP/s per core for NEON
//! FMAs. The math stays fp32 (bf16 weights are widened exactly), so results differ from the NEON path only in
//! summation order.
//!
//! The kernel is inline assembly. Each call enters streaming mode, multiplies one strip of up to 32 rows against a
//! run of 32-column blocks, and leaves it, so no compiled Rust runs in streaming mode. Per k step it loads 32 A
//! values and 32 B values and issues four 16x16 FMOPAs (ZA holds four fp32 tiles at a 512-bit vector length).
//! Loads and stores are predicated on the valid rows and columns, so no operand needs padding.
//!
//! Used when the CPU has SME with a 512-bit streaming vector length (Apple M4 and later); `EG2_SME=off` disables it.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::simd::{Packed, SendPtr, Split, Weights, softmax_cols, task_target};
use rayon::prelude::*;
use std::arch::asm;
use std::cell::RefCell;
use std::sync::OnceLock;

/// Panel width of SME-packed weights: two 16-lane tiles.
pub(crate) const NRW: usize = 32;

/// Whether the SME kernels are in use: SME present, 512-bit streaming vectors, and `EG2_SME` not `off`.
pub(crate) fn available() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // SAFETY: rdsvl only runs once SME is known to be present.
    *ON.get_or_init(|| std::env::var("EG2_SME").as_deref() != Ok("off") && has_sme() && unsafe { svl_bytes() } == 64)
}

#[cfg(target_os = "macos")]
fn has_sme() -> bool {
    let (mut v, mut len) = (0i32, std::mem::size_of::<i32>());
    // SAFETY: a NUL-terminated name and an int-sized output buffer.
    let r = unsafe {
        libc::sysctlbyname(c"hw.optional.arm.FEAT_SME".as_ptr(), (&raw mut v).cast(), &mut len, std::ptr::null_mut(), 0)
    };
    r == 0 && v == 1
}

#[cfg(target_os = "linux")]
fn has_sme() -> bool {
    const HWCAP2_SME: libc::c_ulong = 1 << 23;
    // SAFETY: getauxval has no preconditions.
    unsafe { libc::getauxval(libc::AT_HWCAP2) & HWCAP2_SME != 0 }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn has_sme() -> bool {
    false
}

/// The streaming vector length in bytes. Only valid when SME is present.
unsafe fn svl_bytes() -> u64 {
    let v: u64;
    asm!(".arch_extension sme", "rdsvl {v}, #1", v = out(reg) v, options(nomem, nostack, pure));
    v
}

/// The type of B's values.
#[derive(Clone, Copy)]
enum B {
    F32 = 0,
    Bf16 = 1,
    /// int8 codes, widened to their values.
    Int8 = 2,
    /// e4m3 codes, widened to their values times 2^-8: a code shifted into half-precision bit positions.
    E4m3 = 3,
}

/// How B is laid out and C written.
#[derive(Clone, Copy)]
struct Mode {
    b: B,
    /// Store C^T: column `cc` of the block goes to `C + cc * ldc` as a contiguous run of `rows` values.
    transposed: bool,
}

/// One strip: C[rows, ncols] = A B, `rows` <= 32. Per k step `p`, A supplies 32 fp32 values (one per row) at
/// `a + p * astr` bytes and B supplies 32 values (one per column) at `b + p * bstr` bytes; column block `j` (32
/// columns) of B starts at `b + j * bjump` bytes. C is fp32 with a leading dimension of `ldc` elements.
///
/// Predicates: p0 all lanes, p1/p2 rows 0..15/16..31, p3/p4 the block's columns 0..15/16..31, p5 the block's 32
/// columns as halfwords (for bf16 B). 8-bit B is sign-extended into 32-bit lanes as it loads, two loads of 16 codes.
///
/// # Safety
/// Every addressed A, B and C element of the valid rows and columns must be in bounds; the CPU must have SME
/// with a 512-bit streaming vector length (`available`).
#[allow(clippy::too_many_arguments)]
unsafe fn strip(
    k: usize, a: *const f32, astr: usize, rows: usize, b: *const u8, bstr: usize, bjump: usize, ncols: usize,
    c: *mut f32, ldc: usize, mode: Mode,
) {
    debug_assert!(k > 0 && (1..=32).contains(&rows) && ncols > 0);
    let (rows0, rows1) = (rows.min(16), rows.saturating_sub(16));
    asm!(
        ".arch_extension sme",
        "smstart",
        "ptrue p0.s",
        "mov z31.h, #0",
        "mov {j}, #0",
        "mov {t}, #16",
        "whilelt p1.s, {j}, {rows}",
        "whilelt p2.s, {t}, {rows}",
        // for each 32-column block j
        "20:",
        "whilelt p3.s, {j}, {ncols}",
        "add {t}, {j}, #16",
        "whilelt p4.s, {t}, {ncols}",
        "whilelt p5.h, {j}, {ncols}",
        "zero {{za}}",
        "mov {pa}, {a}",
        "mov {pb}, {b}",
        "mov {kk}, {k}",
        "cmp {bk}, #1",
        "b.lo 23f",
        "b.eq 21f",
        "cmp {bk}, #2",
        "b.eq 26f",
        "b 27f",
        // k loop, bf16 B: widen by interleaving zero low halves
        "21:",
        "ld1w {{z0.s}}, p1/z, [{pa}]",
        "add {t}, {pa}, #64",
        "ld1w {{z1.s}}, p2/z, [{t}]",
        "ld1h {{z4.h}}, p5/z, [{pb}]",
        "zip1 z2.h, z31.h, z4.h",
        "zip2 z3.h, z31.h, z4.h",
        "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",
        "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
        "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",
        "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",
        "add {pa}, {pa}, {astr}",
        "add {pb}, {pb}, {bstr}",
        "subs {kk}, {kk}, #1",
        "b.ne 21b",
        "b 25f",
        // k loop, fp32 B
        "23:",
        "ld1w {{z0.s}}, p1/z, [{pa}]",
        "add {t}, {pa}, #64",
        "ld1w {{z1.s}}, p2/z, [{t}]",
        "ld1w {{z2.s}}, p3/z, [{pb}]",
        "add {t}, {pb}, #64",
        "ld1w {{z3.s}}, p4/z, [{t}]",
        "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",
        "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
        "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",
        "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",
        "add {pa}, {pa}, {astr}",
        "add {pb}, {pb}, {bstr}",
        "subs {kk}, {kk}, #1",
        "b.ne 23b",
        "b 25f",
        // k loop, int8 B
        "26:",
        "ld1w {{z0.s}}, p1/z, [{pa}]",
        "add {t}, {pa}, #64",
        "ld1w {{z1.s}}, p2/z, [{t}]",
        "ld1sb {{z2.s}}, p3/z, [{pb}]",
        "ld1sb {{z3.s}}, p4/z, [{pb}, #1, mul vl]",
        "scvtf z2.s, p0/m, z2.s",
        "scvtf z3.s, p0/m, z3.s",
        "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",
        "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
        "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",
        "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",
        "add {pa}, {pa}, {astr}",
        "add {pb}, {pb}, {bstr}",
        "subs {kk}, {kk}, #1",
        "b.ne 26b",
        "b 25f",
        // k loop, e4m3 B: shifted left 7, the sign-extended code is sign | exponent | mantissa in half-precision
        // positions once bit 14 (a copy of the sign) is cleared
        "27:",
        "ld1w {{z0.s}}, p1/z, [{pa}]",
        "add {t}, {pa}, #64",
        "ld1w {{z1.s}}, p2/z, [{t}]",
        "ld1sb {{z2.s}}, p3/z, [{pb}]",
        "ld1sb {{z3.s}}, p4/z, [{pb}, #1, mul vl]",
        "lsl z2.s, z2.s, #7",
        "lsl z3.s, z3.s, #7",
        "and z2.s, z2.s, #0xffffbfff",
        "and z3.s, z3.s, #0xffffbfff",
        "fcvt z2.s, p0/m, z2.h",
        "fcvt z3.s, p0/m, z3.h",
        "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",
        "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
        "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",
        "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",
        "add {pa}, {pa}, {astr}",
        "add {pb}, {pb}, {bstr}",
        "subs {kk}, {kk}, #1",
        "b.ne 27b",
        "25:",
        "cbnz {tr}, 40f",
        // row-major store: horizontal tile slices, row r of the strip to c + r * ldc
        "mov {pc}, {c}",
        "mov w12, #0",
        "30:",
        "cmp x12, {rows0}",
        "b.hs 31f",
        "st1w {{za0h.s[w12, 0]}}, p3, [{pc}]",
        "add {t}, {pc}, #64",
        "st1w {{za1h.s[w12, 0]}}, p4, [{t}]",
        "add {pc}, {pc}, {ldc}",
        "add w12, w12, #1",
        "b 30b",
        "31:",
        "mov w12, #0",
        "32:",
        "cmp x12, {rows1}",
        "b.hs 33f",
        "st1w {{za2h.s[w12, 0]}}, p3, [{pc}]",
        "add {t}, {pc}, #64",
        "st1w {{za3h.s[w12, 0]}}, p4, [{t}]",
        "add {pc}, {pc}, {ldc}",
        "add w12, w12, #1",
        "b 32b",
        "33:",
        "add {c}, {c}, #128",
        "b 50f",
        // transposed store: vertical tile slices, column cc of the block to c + cc * ldc
        "40:",
        "sub {t}, {ncols}, {j}",
        "mov {kk}, #16",
        "cmp {t}, #16",
        "csel {pa}, {t}, {kk}, lo",
        "sub {pb}, {t}, {pa}",
        "cmp {pb}, #16",
        "csel {pb}, {pb}, {kk}, lo",
        "mov {pc}, {c}",
        "mov w12, #0",
        "41:",
        "cmp x12, {pa}",
        "b.hs 42f",
        "st1w {{za0v.s[w12, 0]}}, p1, [{pc}]",
        "add {t}, {pc}, #64",
        "st1w {{za2v.s[w12, 0]}}, p2, [{t}]",
        "add {pc}, {pc}, {ldc}",
        "add w12, w12, #1",
        "b 41b",
        "42:",
        "mov w12, #0",
        "43:",
        "cmp x12, {pb}",
        "b.hs 44f",
        "st1w {{za1v.s[w12, 0]}}, p1, [{pc}]",
        "add {t}, {pc}, #64",
        "st1w {{za3v.s[w12, 0]}}, p2, [{t}]",
        "add {pc}, {pc}, {ldc}",
        "add w12, w12, #1",
        "b 43b",
        "44:",
        "mov {c}, {pc}",
        "50:",
        "add {b}, {b}, {bjump}",
        "add {j}, {j}, #32",
        "cmp {j}, {ncols}",
        "b.lo 20b",
        "smstop",
        a = in(reg) a, astr = in(reg) astr, rows = in(reg) rows, b = inout(reg) b => _, bstr = in(reg) bstr,
        bjump = in(reg) bjump, ncols = in(reg) ncols, c = inout(reg) c => _, ldc = in(reg) ldc * 4, k = in(reg) k,
        rows0 = in(reg) rows0, rows1 = in(reg) rows1, bk = in(reg) mode.b as u64,
        tr = in(reg) mode.transposed as u64,
        j = out(reg) _, t = out(reg) _, pa = out(reg) _, pb = out(reg) _, kk = out(reg) _, pc = out(reg) _,
        out("x12") _,
        out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _,
        out("v23") _, out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
        out("v30") _, out("v31") _,
        options(nostack),
    );
}

/// dst[c * ldd + r] = src[r * lds + c] for r < rows, c < cols: 4x4 blocks with NEON, scalar tails.
unsafe fn transpose(src: *const f32, lds: usize, rows: usize, cols: usize, dst: *mut f32, ldd: usize) {
    use std::arch::aarch64::*;
    let (r4, c4) = (rows / 4 * 4, cols / 4 * 4);
    let f = |x: float32x4_t| vreinterpretq_f64_f32(x);
    for r in (0..r4).step_by(4) {
        let s = src.add(r * lds);
        for c in (0..c4).step_by(4) {
            let x0 = vld1q_f32(s.add(c));
            let x1 = vld1q_f32(s.add(lds + c));
            let x2 = vld1q_f32(s.add(2 * lds + c));
            let x3 = vld1q_f32(s.add(3 * lds + c));
            let (t0, t1) = (vtrn1q_f32(x0, x1), vtrn2q_f32(x0, x1));
            let (t2, t3) = (vtrn1q_f32(x2, x3), vtrn2q_f32(x2, x3));
            let d = dst.add(c * ldd + r);
            vst1q_f32(d, vreinterpretq_f32_f64(vtrn1q_f64(f(t0), f(t2))));
            vst1q_f32(d.add(ldd), vreinterpretq_f32_f64(vtrn1q_f64(f(t1), f(t3))));
            vst1q_f32(d.add(2 * ldd), vreinterpretq_f32_f64(vtrn2q_f64(f(t0), f(t2))));
            vst1q_f32(d.add(3 * ldd), vreinterpretq_f32_f64(vtrn2q_f64(f(t1), f(t3))));
        }
        for c in c4..cols {
            for i in 0..4 {
                *dst.add(c * ldd + r + i) = *s.add(i * lds + c);
            }
        }
    }
    for r in r4..rows {
        for c in 0..cols {
            *dst.add(c * ldd + r) = *src.add(r * lds + c);
        }
    }
}

thread_local!(static STRIP: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) });

/// The least work, in FLOPs, a matmul task is given (about 20 us on one core's share of the matrix unit): smaller
/// tasks cost more in scheduling and repacked A strips than they gain in balance, since all threads share the unit.
const TASK_FLOPS: usize = 1 << 25;

/// C[m, w.n] = A[m, w.k] W^T with SME-packed weights (32-column panels), as (strip of 32 rows, panel group) tasks.
/// Each task transposes its strip of A into a k-major per-thread buffer, then runs the kernel over its panels.
/// Strips are only split across tasks when there are too few of them to go round.
pub(crate) fn gemm_packed(m: usize, a: &[f32], lda: usize, w: &Packed, c: &mut [f32], ldc: usize) {
    let (n, k) = (w.n, w.k);
    let panels = n.div_ceil(NRW);
    let groups = task_target(m, n, k, TASK_FLOPS).div_ceil(m.div_ceil(32)).clamp(1, panels);
    let (ap, cp) = (SendPtr(a.as_ptr() as *mut f32), SendPtr(c.as_mut_ptr()));
    Split::new(m, n, NRW, 32, panels.div_ceil(groups)).run(|t| {
        let (ap, cp) = (ap, cp);
        let mut buf = STRIP.take();
        buf.resize(k * 32, 0.0);
        // SAFETY: gemm_packed (simd) checked the bounds of A and C; tasks write disjoint (strip, column range)
        // regions of C.
        unsafe {
            transpose(ap.0.add(t.i0 * lda), lda, t.rows, k, buf.as_mut_ptr(), 32);
            let cb = cp.0.add(t.i0 * ldc + t.j0);
            match w.format() {
                Weights::Bf16 => {
                    let wb = w.bf16().as_ptr().add(t.p0 * k * NRW) as *const u8;
                    let mode = Mode { b: B::Bf16, transposed: false };
                    strip(k, buf.as_ptr(), 128, t.rows, wb, 64, k * NRW * 2, t.ncols, cb, ldc, mode)
                }
                // 8-bit weights widen in the kernel, then the task's columns are scaled
                format => {
                    let (b, factor) = if format == Weights::Int8 { (B::Int8, 1.0) } else { (B::E4m3, 256.0) };
                    let wb = w.codes().as_ptr().add(t.p0 * k * NRW);
                    let mode = Mode { b, transposed: false };
                    strip(k, buf.as_ptr(), 128, t.rows, wb, 32, k * NRW, t.ncols, cb, ldc, mode);
                    w.scale_task(t, cb, ldc, factor)
                }
            }
        }
        STRIP.set(buf);
    });
}

thread_local!(static SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) });

/// Full bidirectional multi-head attention, scaling 1.0, on SME. `qkv` holds t rows of [q | k | v] with a leading
/// dimension of `ld`, each nh heads of hd (one KV head per query head); the result goes to `out`, [t, nh * hd].
/// `qkt` is scratch for q and k transposed, at least 2 * nh * hd * t.
///
/// Per (head, 32-query block): the scores are stored transposed, [t keys][32 queries], so the softmax runs down
/// columns with SIMD across queries and the probabilities are already the k-major A operand of P V; the 1 / sum
/// normalization is applied to the output rows.
pub(crate) fn attention(qkv: &[f32], ld: usize, t: usize, nh: usize, hd: usize, qkt: &mut [f32], out: &mut [f32]) {
    let d = nh * hd;
    assert!(ld >= 3 * d && qkv.len() >= (t - 1) * ld + 3 * d && d.is_multiple_of(32));
    assert!(qkt.len() >= 2 * d * t && out.len() >= t * d);
    // Q and K per head, transposed to [hd][t] (the k-major operands for Q K^T), in blocks of 32 positions by 32
    // columns (128 bytes per row), so each task reads whole cache lines of qkv.
    let (qp, nb) = (SendPtr(qkt.as_mut_ptr()), t.div_ceil(32));
    (0..nb * (2 * d / 32)).into_par_iter().for_each(|task| {
        let (p0, c0) = (task % nb * 32, task / nb * 32);
        // Rebind so the closure captures the Send wrapper rather than its raw-pointer field.
        #[allow(clippy::redundant_locals)]
        let qp = qp;
        // SAFETY: rows p0.. of qkv and columns c0..c0 + 32 <= 2d are in bounds; tasks write disjoint blocks.
        unsafe { transpose(qkv.as_ptr().add(p0 * ld + c0), ld, 32.min(t - p0), 32, qp.0.add(c0 * t + p0), t) }
    });
    let qkt = &qkt[..2 * d * t];
    let (qt, kt) = qkt.split_at(d * t);
    let op = SendPtr(out.as_mut_ptr());
    let nqb = t.div_ceil(32);
    (0..nh * nqb).into_par_iter().for_each(|task| {
        let (h, qb) = (task / nqb, task % nqb);
        let (q0, rows) = (qb * 32, 32.min(t - qb * 32));
        let mut s = SCORES.take();
        s.resize(t * 32, 0.0);
        #[allow(clippy::redundant_locals)]
        let op = op;
        // SAFETY: qt/kt hold hd rows of t per head; s holds t rows of 32; v rows are in bounds by the assert; the
        // task writes rows q0..q0 + rows of head h's column band, disjoint from every other task.
        unsafe {
            strip(
                hd,
                qt.as_ptr().add(h * hd * t + q0),
                t * 4,
                rows,
                kt.as_ptr().add(h * hd * t) as *const u8,
                t * 4,
                128,
                t,
                s.as_mut_ptr(),
                32,
                Mode { b: B::F32, transposed: true },
            );
            let sums = softmax_cols(&mut s[..t * 32]);
            let dst = op.0.add(q0 * d + h * hd);
            strip(
                t,
                s.as_ptr(),
                128,
                rows,
                qkv.as_ptr().add(2 * d + h * hd) as *const u8,
                ld * 4,
                128,
                hd,
                dst,
                d,
                Mode { b: B::F32, transposed: false },
            );
            for (r, sum) in sums.iter().enumerate().take(rows) {
                let inv = 1.0 / sum;
                std::slice::from_raw_parts_mut(dst.add(r * d), hd).iter_mut().for_each(|x| *x *= inv);
            }
        }
        SCORES.set(s);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against the NEON attention, at lengths with partial 32-row blocks and partial key blocks.
    #[test]
    fn attention_matches_neon() {
        if !available() {
            return;
        }
        let (nh, hd) = (3usize, 64usize);
        let d = nh * hd;
        for t in [1usize, 5, 45, 97] {
            let qkv: Vec<f32> = (0..t * 3 * d).map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * 2.0).collect();
            let part = |i: usize| -> Vec<f32> {
                qkv.chunks_exact(3 * d).flat_map(|r| &r[i * d..(i + 1) * d]).copied().collect()
            };
            let want = crate::model::attention(&part(0), &part(1), &part(2), t, nh, nh, hd, None);
            let (mut qkt, mut got) = (vec![0.0f32; 2 * d * t], vec![f32::NAN; t * d]);
            attention(&qkv, 3 * d, t, nh, hd, &mut qkt, &mut got);
            let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-5, "t={t}: {err}");
        }
    }
}

#[cfg(test)]
mod speed {
    use super::*;
    use std::time::Instant;

    /// cargo test --release --lib -- --ignored --nocapture attention_phases
    #[test]
    #[ignore]
    fn attention_phases() {
        if !available() {
            return;
        }
        let (t, nh, hd) = (540usize, 12usize, 64usize);
        let d = nh * hd;
        let qkv: Vec<f32> = (0..t * 3 * d).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.1).collect();
        let qt: Vec<f32> = (0..d * t).map(|i| (i % 13) as f32 * 0.01).collect();
        let mut s = vec![0.0f32; t * 32];
        let mut out = vec![0.0f32; t * d];
        let iters = 2000;
        let (mut tqk, mut tsm, mut tpv) = (0.0, 0.0, 0.0);
        for i in 0..iters {
            let (h, q0) = (i % nh, (i / nh) % (t / 32) * 32);
            unsafe {
                let a = Instant::now();
                strip(
                    hd,
                    qt.as_ptr().add(h * hd * t + q0),
                    t * 4,
                    32,
                    qt.as_ptr().add(h * hd * t) as *const u8,
                    t * 4,
                    128,
                    t,
                    s.as_mut_ptr(),
                    32,
                    Mode { b: B::F32, transposed: true },
                );
                let b = Instant::now();
                let sums = softmax_cols(&mut s);
                let c = Instant::now();
                strip(
                    t,
                    s.as_ptr(),
                    128,
                    32,
                    qkv.as_ptr().add(2 * d + h * hd) as *const u8,
                    3 * d * 4,
                    128,
                    hd,
                    out.as_mut_ptr().add(q0 * d + h * hd),
                    d,
                    Mode { b: B::F32, transposed: false },
                );
                let e = Instant::now();
                std::hint::black_box(sums);
                tqk += (b - a).as_secs_f64();
                tsm += (c - b).as_secs_f64();
                tpv += (e - c).as_secs_f64();
            }
        }
        let us = |x: f64| x / iters as f64 * 1e6;
        let fl = 2.0 * 32.0 * t as f64 * hd as f64;
        println!(
            "per task (t={t}): QK {:.2} us ({:.0} GFLOP/s), softmax {:.2} us, PV {:.2} us ({:.0} GFLOP/s); {} tasks/layer",
            us(tqk),
            fl / (tqk / iters as f64) / 1e9,
            us(tsm),
            us(tpv),
            fl / (tpv / iters as f64) / 1e9,
            nh * t.div_ceil(32)
        );
        let (mut qkt, mut out) = (vec![0.0f32; 2 * d * t], vec![0.0f32; t * d]);
        let a = Instant::now();
        for _ in 0..20 {
            attention(&qkv, 3 * d, t, nh, hd, &mut qkt, &mut out);
        }
        println!("whole attention: {:.2} ms", a.elapsed().as_secs_f64() / 20.0 * 1e3);
    }
}
