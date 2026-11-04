//! Runtime-dispatched SIMD kernels: AVX-512, AVX2+FMA, NEON, and a scalar fallback.
//!
//! Every kernel is written once, generic over the `Simd` trait, and instantiated per ISA inside a
//! `#[target_feature]` function so LLVM inlines the intrinsics with the right features enabled.
//! `EG2_ISA=scalar|avx2|avx512|neon` caps the detected ISA (for testing the fallbacks).
//!
//! The matmuls against packed weights (`Packed`, `gemm_packed`) live here too: the fp32 broadcast kernels, AVX-512 BF16
//! dot products (`bf16dot`), AMX tiles (`amx`) and Arm `BFMMLA` (`bfmmla`), chosen by `bf16_mode` (`EG2_BF16`); the SME
//! kernels are in `sme`.
#![allow(unsafe_op_in_unsafe_fn)]
// Kernels index their register arrays explicitly so the accumulators stay in registers.
#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

static ISA: OnceLock<Isa> = OnceLock::new();

pub fn isa() -> Isa {
    *ISA.get_or_init(detect)
}

fn detect() -> Isa {
    let forced = std::env::var("EG2_ISA").ok();
    let allow = |name: &str| forced.as_deref().is_none_or(|f| f == name);
    #[cfg(target_arch = "x86_64")]
    {
        if allow("avx512") && is_x86_feature_detected!("avx512f") {
            return Isa::Avx512;
        }
        if allow("avx2") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Isa::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if allow("neon") && std::arch::is_aarch64_feature_detected!("neon") {
        return Isa::Neon;
    }
    Isa::Scalar
}

pub(crate) trait Simd {
    type V: Copy;
    const LANES: usize;
    unsafe fn splat(x: f32) -> Self::V;
    unsafe fn load(p: *const f32) -> Self::V;
    unsafe fn load_bf16(p: *const u16) -> Self::V;
    unsafe fn store(p: *mut f32, v: Self::V);
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn div(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn max(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn min(a: Self::V, b: Self::V) -> Self::V;
    /// a * b + c
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// Round to nearest, ties to even.
    unsafe fn round(a: Self::V) -> Self::V;
    /// 2^n for integral n in [-126, 127].
    unsafe fn pow2i(n: Self::V) -> Self::V;
    unsafe fn hsum(a: Self::V) -> f32;
    unsafe fn hmax(a: Self::V) -> f32;
    /// Whether `fma_lane` is available (and cheaper than a broadcast per element).
    const LANE_FMA: bool = false;
    /// c + b * a[lane]
    #[inline(always)]
    unsafe fn fma_lane(_a: Self::V, _lane: usize, _b: Self::V, _c: Self::V) -> Self::V {
        unreachable!()
    }
}

/// A weight element: f32, or bf16 stored as its raw u16 bits.
pub(crate) trait Elem: Copy + Send + Sync + 'static {
    unsafe fn load<S: Simd>(p: *const Self) -> S::V;
    fn to_f32(self) -> f32;
}

impl Elem for f32 {
    #[inline(always)]
    unsafe fn load<S: Simd>(p: *const f32) -> S::V {
        S::load(p)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

impl Elem for u16 {
    #[inline(always)]
    unsafe fn load<S: Simd>(p: *const u16) -> S::V {
        S::load_bf16(p)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        bf16_to_f32(self)
    }
}

#[inline(always)]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// Round to the nearest bf16, ties to even; NaN stays NaN (quieted, sign and top payload bits kept, as the hardware
/// conversions `vcvtne2ps2bf16` and `BFCVTN` do).
#[inline(always)]
fn f32_to_bf16(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return (u >> 16) as u16 | 0x0040;
    }
    // At most 0xff80_0000 (-inf) + 0x8000 for a non-NaN value, so the sum cannot overflow.
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

// ---------------------------------------------------------------- ISA implementations

#[derive(Clone, Copy)]
pub(crate) struct Scalar;

impl Simd for Scalar {
    type V = f32;
    const LANES: usize = 1;
    #[inline(always)]
    unsafe fn splat(x: f32) -> f32 {
        x
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> f32 {
        *p
    }
    #[inline(always)]
    unsafe fn load_bf16(p: *const u16) -> f32 {
        bf16_to_f32(*p)
    }
    #[inline(always)]
    unsafe fn store(p: *mut f32, v: f32) {
        *p = v
    }
    #[inline(always)]
    unsafe fn add(a: f32, b: f32) -> f32 {
        a + b
    }
    #[inline(always)]
    unsafe fn mul(a: f32, b: f32) -> f32 {
        a * b
    }
    #[inline(always)]
    unsafe fn div(a: f32, b: f32) -> f32 {
        a / b
    }
    #[inline(always)]
    unsafe fn max(a: f32, b: f32) -> f32 {
        a.max(b)
    }
    #[inline(always)]
    unsafe fn min(a: f32, b: f32) -> f32 {
        a.min(b)
    }
    #[inline(always)]
    unsafe fn fma(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
    #[inline(always)]
    unsafe fn round(a: f32) -> f32 {
        a.round_ties_even()
    }
    #[inline(always)]
    unsafe fn pow2i(n: f32) -> f32 {
        f32::from_bits(((n as i32 + 127) << 23) as u32)
    }
    #[inline(always)]
    unsafe fn hsum(a: f32) -> f32 {
        a
    }
    #[inline(always)]
    unsafe fn hmax(a: f32) -> f32 {
        a
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use super::Simd;
    use std::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub(crate) struct Avx2;

    impl Simd for Avx2 {
        type V = __m256;
        const LANES: usize = 8;
        #[inline(always)]
        unsafe fn splat(x: f32) -> __m256 {
            _mm256_set1_ps(x)
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> __m256 {
            _mm256_loadu_ps(p)
        }
        #[inline(always)]
        unsafe fn load_bf16(p: *const u16) -> __m256 {
            let h = _mm_loadu_si128(p as *const __m128i);
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(h)))
        }
        #[inline(always)]
        unsafe fn store(p: *mut f32, v: __m256) {
            _mm256_storeu_ps(p, v)
        }
        #[inline(always)]
        unsafe fn add(a: __m256, b: __m256) -> __m256 {
            _mm256_add_ps(a, b)
        }
        #[inline(always)]
        unsafe fn mul(a: __m256, b: __m256) -> __m256 {
            _mm256_mul_ps(a, b)
        }
        #[inline(always)]
        unsafe fn div(a: __m256, b: __m256) -> __m256 {
            _mm256_div_ps(a, b)
        }
        #[inline(always)]
        unsafe fn max(a: __m256, b: __m256) -> __m256 {
            _mm256_max_ps(a, b)
        }
        #[inline(always)]
        unsafe fn min(a: __m256, b: __m256) -> __m256 {
            _mm256_min_ps(a, b)
        }
        #[inline(always)]
        unsafe fn fma(a: __m256, b: __m256, c: __m256) -> __m256 {
            _mm256_fmadd_ps(a, b, c)
        }
        #[inline(always)]
        unsafe fn round(a: __m256) -> __m256 {
            _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a)
        }
        #[inline(always)]
        unsafe fn pow2i(n: __m256) -> __m256 {
            let e = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127));
            _mm256_castsi256_ps(_mm256_slli_epi32::<23>(e))
        }
        #[inline(always)]
        unsafe fn hsum(a: __m256) -> f32 {
            let s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps::<1>(s, s)))
        }
        #[inline(always)]
        unsafe fn hmax(a: __m256) -> f32 {
            let s = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
            let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
            _mm_cvtss_f32(_mm_max_ss(s, _mm_shuffle_ps::<1>(s, s)))
        }
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Avx512;

    impl Simd for Avx512 {
        type V = __m512;
        const LANES: usize = 16;
        #[inline(always)]
        unsafe fn splat(x: f32) -> __m512 {
            _mm512_set1_ps(x)
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> __m512 {
            _mm512_loadu_ps(p)
        }
        #[inline(always)]
        unsafe fn load_bf16(p: *const u16) -> __m512 {
            let h = _mm256_loadu_si256(p as *const __m256i);
            _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(h)))
        }
        #[inline(always)]
        unsafe fn store(p: *mut f32, v: __m512) {
            _mm512_storeu_ps(p, v)
        }
        #[inline(always)]
        unsafe fn add(a: __m512, b: __m512) -> __m512 {
            _mm512_add_ps(a, b)
        }
        #[inline(always)]
        unsafe fn mul(a: __m512, b: __m512) -> __m512 {
            _mm512_mul_ps(a, b)
        }
        #[inline(always)]
        unsafe fn div(a: __m512, b: __m512) -> __m512 {
            _mm512_div_ps(a, b)
        }
        #[inline(always)]
        unsafe fn max(a: __m512, b: __m512) -> __m512 {
            _mm512_max_ps(a, b)
        }
        #[inline(always)]
        unsafe fn min(a: __m512, b: __m512) -> __m512 {
            _mm512_min_ps(a, b)
        }
        #[inline(always)]
        unsafe fn fma(a: __m512, b: __m512, c: __m512) -> __m512 {
            _mm512_fmadd_ps(a, b, c)
        }
        #[inline(always)]
        unsafe fn round(a: __m512) -> __m512 {
            _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a)
        }
        #[inline(always)]
        unsafe fn pow2i(n: __m512) -> __m512 {
            let e = _mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127));
            _mm512_castsi512_ps(_mm512_slli_epi32::<23>(e))
        }
        #[inline(always)]
        unsafe fn hsum(a: __m512) -> f32 {
            _mm512_reduce_add_ps(a)
        }
        #[inline(always)]
        unsafe fn hmax(a: __m512) -> f32 {
            _mm512_reduce_max_ps(a)
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod arm {
    use super::Simd;
    use std::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub(crate) struct Neon;

    impl Simd for Neon {
        type V = float32x4_t;
        const LANES: usize = 4;
        #[inline(always)]
        unsafe fn splat(x: f32) -> float32x4_t {
            vdupq_n_f32(x)
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> float32x4_t {
            vld1q_f32(p)
        }
        #[inline(always)]
        unsafe fn load_bf16(p: *const u16) -> float32x4_t {
            vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(p)))
        }
        #[inline(always)]
        unsafe fn store(p: *mut f32, v: float32x4_t) {
            vst1q_f32(p, v)
        }
        #[inline(always)]
        unsafe fn add(a: float32x4_t, b: float32x4_t) -> float32x4_t {
            vaddq_f32(a, b)
        }
        #[inline(always)]
        unsafe fn mul(a: float32x4_t, b: float32x4_t) -> float32x4_t {
            vmulq_f32(a, b)
        }
        #[inline(always)]
        unsafe fn div(a: float32x4_t, b: float32x4_t) -> float32x4_t {
            vdivq_f32(a, b)
        }
        #[inline(always)]
        unsafe fn max(a: float32x4_t, b: float32x4_t) -> float32x4_t {
            vmaxq_f32(a, b)
        }
        #[inline(always)]
        unsafe fn min(a: float32x4_t, b: float32x4_t) -> float32x4_t {
            vminq_f32(a, b)
        }
        #[inline(always)]
        unsafe fn fma(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
            vfmaq_f32(c, a, b)
        }
        #[inline(always)]
        unsafe fn round(a: float32x4_t) -> float32x4_t {
            vrndnq_f32(a)
        }
        #[inline(always)]
        unsafe fn pow2i(n: float32x4_t) -> float32x4_t {
            let e = vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127));
            vreinterpretq_f32_s32(vshlq_n_s32::<23>(e))
        }
        #[inline(always)]
        unsafe fn hsum(a: float32x4_t) -> f32 {
            vaddvq_f32(a)
        }
        #[inline(always)]
        unsafe fn hmax(a: float32x4_t) -> f32 {
            vmaxvq_f32(a)
        }
        const LANE_FMA: bool = true;
        #[inline(always)]
        unsafe fn fma_lane(a: float32x4_t, lane: usize, b: float32x4_t, c: float32x4_t) -> float32x4_t {
            match lane {
                0 => vfmaq_laneq_f32::<0>(c, b, a),
                1 => vfmaq_laneq_f32::<1>(c, b, a),
                2 => vfmaq_laneq_f32::<2>(c, b, a),
                _ => vfmaq_laneq_f32::<3>(c, b, a),
            }
        }
    }
}

// ---------------------------------------------------------------- bf16 matmuls

/// How matmuls multiply in bf16, if at all. Activations are rounded to bf16 too, so results differ from the fp32
/// reference by more than 1e-3: `EG2_BF16=amx` (AMX tiles, `tdpbf16ps`), `EG2_BF16=dot` (AVX-512 BF16 `vdpbf16ps`,
/// which issues on one port where fp32 FMAs issue on two, so it gains nothing on current CPUs), or on Arm
/// `EG2_BF16=bfmmla` (`BFMMLA` 2 x 4 by 4 x 2 matrix products, where the SME kernels are not in use). `amx` falls back
/// to `dot` where AMX is unavailable but AVX-512 BF16 is present; a request the CPU can't honour gives `Off`.
///
/// On x86 and macOS they are opt-in. On Arm Linux with BF16 and no SME (Neoverse V2 and N2 servers, DGX Spark's
/// Cortex-X925 and A725) `bfmmla` is the default, as on Graviton 4 `BFMMLA` runs at four times the fp32 rate per core;
/// `EG2_BF16=off` keeps fp32. macOS stays opt-in: Apple's cores run `BFMMLA` at half their fp32 rate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bf16 {
    Off,
    Dot,
    Amx,
    Bfmmla,
}

pub fn bf16_mode() -> Bf16 {
    static MODE: OnceLock<Bf16> = OnceLock::new();
    *MODE.get_or_init(|| {
        let want = std::env::var("EG2_BF16").unwrap_or_default();
        #[cfg(target_arch = "x86_64")]
        {
            if isa() == Isa::Avx512 && want == "amx" && amx::available() {
                return Bf16::Amx;
            }
            if isa() == Isa::Avx512 && (want == "dot" || want == "amx") && is_x86_feature_detected!("avx512bf16") {
                return Bf16::Dot;
            }
        }
        // SME matmuls take precedence: with `sme::available`, `bfmmla` is ignored.
        #[cfg(target_arch = "aarch64")]
        if isa() == Isa::Neon
            && (want == "bfmmla" || (want.is_empty() && cfg!(target_os = "linux")))
            && std::arch::is_aarch64_feature_detected!("bf16")
            && !crate::sme::available()
        {
            return Bf16::Bfmmla;
        }
        let _ = want;
        Bf16::Off
    })
}

/// bf16 x bf16 products with fp32 accumulation (AVX-512 BF16 `vdpbf16ps`), against weights packed in k pairs.
#[cfg(target_arch = "x86_64")]
pub(crate) mod bf16dot {
    use super::MAX_PANEL;
    use std::arch::x86_64::*;
    use std::mem::transmute;

    /// Output columns per packed panel (two 16-lane vectors).
    pub(crate) const NRW: usize = 32;

    /// Round `rows` rows of `a` (stride `lda`, `k` used) to bf16 into `dst` (stride `kp`), ties to even, 32 at a time.
    /// The caller checks `bf16_mode() != Off`.
    #[target_feature(enable = "avx512f,avx512bf16")]
    pub(crate) unsafe fn round_rows(rows: usize, k: usize, a: *const f32, lda: usize, dst: *mut u16, kp: usize) {
        let full = k / 32 * 32;
        for r in 0..rows {
            let (src, out) = (a.add(r * lda), dst.add(r * kp));
            let mut j = 0;
            while j < full {
                let lo = _mm512_loadu_ps(src.add(j));
                let hi = _mm512_loadu_ps(src.add(j + 16));
                _mm512_storeu_si512(out.add(j) as *mut __m512i, transmute(_mm512_cvtne2ps_pbh(hi, lo)));
                j += 32;
            }
            for jj in j..k {
                *out.add(jj) = super::f32_to_bf16(*src.add(jj));
            }
        }
    }

    /// One MR x 32 tile: per k pair, broadcast each row's (a[2q], a[2q+1]) and multiply-add it into both column
    /// vectors. `kp` is k rounded up to even; `a` rows are bf16 with stride `lda`.
    #[inline(always)]
    unsafe fn tile<const MR: usize>(
        kp: usize, a: *const u16, lda: usize, panel: *const u16, c: *mut f32, ldc: usize, ncols: usize,
    ) {
        let mut acc = [[_mm512_setzero_ps(); 2]; MR];
        for q in 0..kp / 2 {
            let b0: __m512bh = transmute(_mm512_loadu_si512(panel.add(q * 2 * NRW) as *const __m512i));
            let b1: __m512bh = transmute(_mm512_loadu_si512(panel.add(q * 2 * NRW + NRW) as *const __m512i));
            for i in 0..MR {
                // a broadcast from memory (vbroadcastss) uses a load port, not port 5, which vdpbf16ps needs
                let av: __m512bh = transmute(_mm512_set1_ps(*(a.add(i * lda + 2 * q) as *const f32)));
                acc[i][0] = _mm512_dpbf16_ps(acc[i][0], av, b0);
                acc[i][1] = _mm512_dpbf16_ps(acc[i][1], av, b1);
            }
        }
        if ncols == NRW {
            for i in 0..MR {
                _mm512_storeu_ps(c.add(i * ldc), acc[i][0]);
                _mm512_storeu_ps(c.add(i * ldc + 16), acc[i][1]);
            }
        } else {
            let mut tmp = [0.0f32; MAX_PANEL];
            for i in 0..MR {
                _mm512_storeu_ps(tmp.as_mut_ptr(), acc[i][0]);
                _mm512_storeu_ps(tmp.as_mut_ptr().add(16), acc[i][1]);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), c.add(i * ldc), ncols);
            }
        }
    }

    /// C[m, cols] = A[m, kp] W^T over `np` consecutive pair-packed panels covering `ncols` output columns.
    #[target_feature(enable = "avx512f,avx512bf16")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gemm_block(
        m: usize, kp: usize, a: *const u16, lda: usize, w: *const u16, np: usize, ncols: usize, c: *mut f32, ldc: usize,
    ) {
        for pi in 0..np {
            let (panel, nc, cc) = (w.add(pi * kp * NRW), (ncols - pi * NRW).min(NRW), c.add(pi * NRW));
            let mut i = 0;
            while i + 12 <= m {
                tile::<12>(kp, a.add(i * lda), lda, panel, cc.add(i * ldc), ldc, nc);
                i += 12;
            }
            while i + 4 <= m {
                tile::<4>(kp, a.add(i * lda), lda, panel, cc.add(i * ldc), ldc, nc);
                i += 4;
            }
            while i < m {
                tile::<1>(kp, a.add(i * lda), lda, panel, cc.add(i * ldc), ldc, nc);
                i += 1;
            }
        }
    }
}

/// AMX bf16 tiles: C (32 x 32 f32, four 16 x 16 tiles) += A (32 rows x 32 k) B (32 k x 32 columns) per step, with B
/// read straight from the pair-packed panels (each panel row is one k pair for 32 columns: AMX's B-tile layout).
#[cfg(target_arch = "x86_64")]
pub(crate) mod amx {
    use super::bf16dot::NRW;
    use std::arch::asm;
    use std::arch::x86_64::*;
    use std::mem::transmute;
    use std::sync::OnceLock;

    /// AMX-TILE and AMX-BF16 in CPUID, and Linux's permission to use tile data (arch_prctl ARCH_REQ_XCOMP_PERM).
    /// AMX needs the CPU feature and, on Linux, the kernel's permission to use tile data; other systems report it
    /// unavailable.
    pub(crate) fn available() -> bool {
        if !cfg!(target_os = "linux") {
            return false;
        }
        static OK: OnceLock<bool> = OnceLock::new();
        // SAFETY: cpuid exists on every x86_64 CPU; the syscall is Linux's arch_prctl(ARCH_REQ_XCOMP_PERM,
        // XFEATURE_XTILEDATA), which only requests a permission and touches no memory, and rcx/r11 (which `syscall`
        // overwrites) are declared clobbered.
        *OK.get_or_init(|| unsafe {
            let edx = std::arch::x86_64::__cpuid_count(7, 0).edx;
            if edx & (1 << 24) == 0 || edx & (1 << 22) == 0 {
                return false;
            }
            let ret: i64;
            asm!("syscall", inlateout("rax") 158i64 => ret, in("rdi") 0x1023i64, in("rsi") 18i64,
                 lateout("rcx") _, lateout("r11") _, options(nostack));
            ret == 0
        })
    }

    #[repr(C, align(64))]
    struct Config {
        palette: u8,
        start_row: u8,
        reserved: [u8; 14],
        colsb: [u16; 16],
        rows: [u8; 16],
    }

    /// Round `rows` rows of `a` to bf16 in AMX tile order: each 16-row x 32-k tile is 1 KB contiguous, so a tile
    /// load streams one block instead of touching 16 rows `kp` apart. Tile (r / 16, j / 32) starts at
    /// `(r / 16) * 16 * kp + (j / 32) * 512`, the same offset a row-major [rows, kp] buffer has for its first row.
    #[target_feature(enable = "avx512f,avx512bf16")]
    pub(crate) unsafe fn round_rows_tiled(rows: usize, k: usize, a: *const f32, lda: usize, dst: *mut u16, kp: usize) {
        let full = k / 32 * 32;
        for r in 0..rows {
            let src = a.add(r * lda);
            let out = dst.add(r / 16 * 16 * kp + r % 16 * 32);
            let mut j = 0;
            while j < full {
                let lo = _mm512_loadu_ps(src.add(j));
                let hi = _mm512_loadu_ps(src.add(j + 16));
                _mm512_storeu_si512(out.add(j * 16) as *mut __m512i, transmute(_mm512_cvtne2ps_pbh(hi, lo)));
                j += 32;
            }
            for jj in j..k {
                *out.add(jj / 32 * 512 + jj % 32) = super::f32_to_bf16(*src.add(jj));
            }
        }
    }

    /// C[m, cols] = A[m, kp] W^T over `np` pair-packed panels covering `ncols` columns, `pstride` u16 apart (a window
    /// of a longer packed operand starts mid-panel). `a` has `m` rounded up to 32 rows (the padding rows zero), `kp`
    /// is a multiple of 32, and `lda` is in bf16 elements.
    /// With `tiled`, `a` is in tile order (`round_rows_tiled`, `lda` = `kp`): each A tile loads as one
    /// contiguous 1 KB block.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gemm_block(
        m: usize, kp: usize, a: *const u16, lda: usize, tiled: bool, w: *const u16, pstride: usize, np: usize,
        ncols: usize, c: *mut f32, ldc: usize,
    ) {
        // Configure the tiles once per thread and keep them configured.
        thread_local!(static CONFIGURED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });
        if !CONFIGURED.get() {
            let mut cfg = Config { palette: 1, start_row: 0, reserved: [0; 14], colsb: [0; 16], rows: [0; 16] };
            for t in 0..8 {
                cfg.colsb[t] = 64;
                cfg.rows[t] = 16;
            }
            asm!("ldtilecfg [{}]", in(reg) &cfg, options(nostack));
            CONFIGURED.set(true);
        }
        #[repr(align(64))]
        struct Tmp([f32; 32 * 32]);
        let mut tmp = Tmp([0.0; 32 * 32]);
        let (sa, sb, st): (usize, usize, usize) = (if tiled { 64 } else { lda * 2 }, 2 * NRW * 2, 32 * 4);
        let mut i = 0;
        while i < m {
            for pi in 0..np {
                let panel = w.add(pi * pstride);
                asm!("tilezero tmm0", "tilezero tmm1", "tilezero tmm2", "tilezero tmm3", options(nostack));
                let mut q = 0;
                while q < kp {
                    let (a0, a1) = if tiled {
                        (a.add(i * lda + q * 16), a.add((i + 16) * lda + q * 16))
                    } else {
                        (a.add(i * lda + q), a.add((i + 16) * lda + q))
                    };
                    let b0 = panel.add(q / 2 * 2 * NRW);
                    let b1 = b0.add(NRW);
                    asm!(
                        "tileloadd tmm4, [{a0} + {sa}*1]",
                        "tileloadd tmm5, [{a1} + {sa}*1]",
                        "tileloadd tmm6, [{b0} + {sb}*1]",
                        "tileloadd tmm7, [{b1} + {sb}*1]",
                        "tdpbf16ps tmm0, tmm4, tmm6",
                        "tdpbf16ps tmm1, tmm4, tmm7",
                        "tdpbf16ps tmm2, tmm5, tmm6",
                        "tdpbf16ps tmm3, tmm5, tmm7",
                        a0 = in(reg) a0, a1 = in(reg) a1, b0 = in(reg) b0, b1 = in(reg) b1,
                        sa = in(reg) sa, sb = in(reg) sb, options(nostack),
                    );
                    q += 32;
                }
                let t = tmp.0.as_mut_ptr();
                asm!(
                    "tilestored [{t0} + {st}*1], tmm0",
                    "tilestored [{t1} + {st}*1], tmm1",
                    "tilestored [{t2} + {st}*1], tmm2",
                    "tilestored [{t3} + {st}*1], tmm3",
                    t0 = in(reg) t, t1 = in(reg) t.add(16), t2 = in(reg) t.add(16 * 32), t3 = in(reg) t.add(16 * 32 + 16),
                    st = in(reg) st, options(nostack),
                );
                let nc = (ncols - pi * NRW).min(NRW);
                for r in 0..32.min(m - i) {
                    std::ptr::copy_nonoverlapping(t.add(r * 32), c.add((i + r) * ldc + pi * NRW), nc);
                }
            }
            i += 32;
        }
    }
}

/// Pack `n` output columns over `k`, value `at(kk, c)` for k index `kk` and column `c`, into pair-packed panels
/// (`Layout::Pairs`): 32 columns per panel, each `k.next_multiple_of(32)` long, k in pairs, zero-padded. This is AMX's
/// B-tile layout.
#[cfg(target_arch = "x86_64")]
pub(crate) fn pack_pairs(k: usize, n: usize, at: impl Fn(usize, usize) -> f32 + Sync) -> Vec<u16> {
    pack_panels(n, k, bf16dot::NRW, Layout::Pairs, |c, kk| f32_to_bf16(at(kk, c)))
}

/// Pack `n` output columns over `k`, value `at(kk, c)` for k index `kk` and column `c`, into `BFMMLA` panels
/// (`Layout::Quads`): `bfmmla::NRW` columns per panel, each `k.next_multiple_of(4)` long, k in groups of four,
/// zero-padded.
#[cfg(target_arch = "aarch64")]
pub(crate) fn pack_quads(k: usize, n: usize, at: impl Fn(usize, usize) -> f32 + Sync) -> Vec<u16> {
    pack_panels(n, k, bfmmla::NRW, Layout::Quads, |c, kk| f32_to_bf16(at(kk, c)))
}

/// Arm `BFMMLA`: one instruction multiplies a 2 x 4 bf16 block of A (two rows, four k) by a 4 x 2 block of W^T (four
/// k, two columns) into a 2 x 2 fp32 block. Weights are packed in panels of `NRW` columns, k in groups of four, each
/// column pair's 2 x 4 block one 16-byte vector; A is rounded into the same shape per `MR`-row tile.
#[cfg(target_arch = "aarch64")]
pub(crate) mod bfmmla {
    use std::arch::aarch64::*;
    use std::arch::asm;

    /// Rows per tile (four row pairs) and columns per panel (four column pairs): 16 accumulators.
    pub(crate) const MR: usize = 8;
    pub(crate) const NRW: usize = 8;

    /// acc += a b^T for a = two rows x four k, b = two columns x four k: acc is [r0c0, r0c1, r1c0, r1c1].
    #[inline(always)]
    unsafe fn bfmmla(acc: float32x4_t, a: uint16x8_t, b: uint16x8_t) -> float32x4_t {
        let mut acc = acc;
        asm!("bfmmla {acc:v}.4s, {a:v}.8h, {b:v}.8h", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
             options(pure, nomem, nostack, preserves_flags));
        acc
    }

    /// Offset of A's (row, k) in tile order: `MR`-row tiles of [k / 4][row pair][row % 2][k % 4], so each k group of a
    /// tile is one contiguous `MR` x 4 block. Tile r / MR starts where a row-major [rows, kq] buffer has row r.
    #[inline(always)]
    fn offset(r: usize, kk: usize, kq: usize) -> usize {
        r / MR * MR * kq + kk / 4 * MR * 4 + r % MR * 4 + kk % 4
    }

    /// Round `rows` rows of `a` (stride `lda`, `k` used) to bf16 in tile order with `kq` (k rounded up to 4) per row,
    /// to nearest (`BFCVTN`). `dst` holds `rows` rounded up to `MR` rows, zeroed beforehand.
    #[target_feature(enable = "neon,bf16")]
    pub(crate) unsafe fn round_rows_tiled(rows: usize, k: usize, a: *const f32, lda: usize, dst: *mut u16, kq: usize) {
        let full = k / 4 * 4;
        // A tile at a time, k group by k group, so the stores are sequential.
        for r0 in (0..rows).step_by(MR) {
            let n = MR.min(rows - r0);
            let (src, out) = (a.add(r0 * lda), dst.add(r0 * kq));
            let mut j = 0;
            while j < full {
                for i in 0..n {
                    let x = vld1q_f32(src.add(i * lda + j));
                    let h: uint16x4_t;
                    asm!("bfcvtn {h:v}.4h, {x:v}.4s", h = out(vreg) h, x = in(vreg) x,
                         options(pure, nomem, nostack, preserves_flags));
                    vst1_u16(out.add(j * MR + i * 4), h);
                }
                j += 4;
            }
            for i in 0..n {
                for jj in full..k {
                    *dst.add(offset(r0 + i, jj, kq)) = super::f32_to_bf16(*src.add(i * lda + jj));
                }
            }
        }
    }

    /// One `MR` x `NRW` tile of C over `kq` / 4 steps, storing `rows` x `ncols` of it.
    #[inline(always)]
    unsafe fn tile(kq: usize, a: *const u16, panel: *const u16, c: *mut f32, ldc: usize, rows: usize, ncols: usize) {
        const RP: usize = MR / 2;
        const CP: usize = NRW / 2;
        let mut acc = [[vdupq_n_f32(0.0); CP]; RP];
        for q in 0..kq / 4 {
            let (ap, bp) = (a.add(q * MR * 4), panel.add(q * NRW * 4));
            let mut av = [vdupq_n_u16(0); RP];
            let mut bv = [vdupq_n_u16(0); CP];
            for i in 0..RP {
                av[i] = vld1q_u16(ap.add(i * 8));
            }
            for j in 0..CP {
                bv[j] = vld1q_u16(bp.add(j * 8));
            }
            for i in 0..RP {
                for j in 0..CP {
                    acc[i][j] = bfmmla(acc[i][j], av[i], bv[j]);
                }
            }
        }
        // acc[i][j] holds rows 2i, 2i+1 x columns 2j, 2j+1; pairing column blocks gives four columns of a row.
        let full = rows == MR && ncols == NRW;
        let mut tmp = [0.0f32; MR * NRW];
        let (dst, ld) = if full { (c, ldc) } else { (tmp.as_mut_ptr(), NRW) };
        for i in 0..RP {
            for j in (0..CP).step_by(2) {
                let (x, y) = (vreinterpretq_f64_f32(acc[i][j]), vreinterpretq_f64_f32(acc[i][j + 1]));
                vst1q_f32(dst.add(2 * i * ld + 2 * j), vreinterpretq_f32_f64(vtrn1q_f64(x, y)));
                vst1q_f32(dst.add((2 * i + 1) * ld + 2 * j), vreinterpretq_f32_f64(vtrn2q_f64(x, y)));
            }
        }
        if !full {
            for r in 0..rows {
                std::ptr::copy_nonoverlapping(tmp.as_ptr().add(r * NRW), c.add(r * ldc), ncols);
            }
        }
    }

    /// C[m, cols] = A[m, kq] W^T over `np` packed panels `ldw` apart covering `ncols` columns, with `a` in tile order
    /// (`round_rows_tiled`). `ldw` is `kq * NRW` for whole panels; it is larger when `w` starts partway into longer
    /// panels (a range of k, as attention's P V takes a range of keys).
    #[target_feature(enable = "neon,bf16")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn gemm_block(
        m: usize, kq: usize, a: *const u16, w: *const u16, ldw: usize, np: usize, ncols: usize, c: *mut f32, ldc: usize,
    ) {
        for pi in 0..np {
            let (panel, nc, cc) = (w.add(pi * ldw), (ncols - pi * NRW).min(NRW), c.add(pi * NRW));
            let mut i = 0;
            while i < m {
                tile(kq, a.add(i * kq), panel, cc.add(i * ldc), ldc, MR.min(m - i), nc);
                i += MR;
            }
        }
    }
}

// ---------------------------------------------------------------- generic kernels

/// One MR x NR tile of C = A B^T, with A rows of f32 and B rows of W, both contiguous along k.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tile<S: Simd, W: Elem, const MR: usize, const NR: usize>(
    k: usize, a: *const f32, lda: usize, b: *const W, ldb: usize, c: *mut f32, ldc: usize, acc_c: bool,
) {
    let mut acc = [[S::splat(0.0); NR]; MR];
    let kv = k / S::LANES * S::LANES;
    let mut p = 0;
    while p < kv {
        let mut bv = [S::splat(0.0); NR];
        for j in 0..NR {
            bv[j] = W::load::<S>(b.add(j * ldb + p));
        }
        for i in 0..MR {
            let av = S::load(a.add(i * lda + p));
            for j in 0..NR {
                acc[i][j] = S::fma(av, bv[j], acc[i][j]);
            }
        }
        p += S::LANES;
    }
    for i in 0..MR {
        for j in 0..NR {
            let mut s = S::hsum(acc[i][j]);
            for q in kv..k {
                s += *a.add(i * lda + q) * (*b.add(j * ldb + q)).to_f32();
            }
            let o = c.add(i * ldc + j);
            *o = if acc_c { *o + s } else { s };
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_block_g<S: Simd, W: Elem, const MR: usize, const NR: usize>(
    m: usize, n: usize, k: usize, a: *const f32, lda: usize, b: *const W, ldb: usize, c: *mut f32, ldc: usize,
    acc: bool,
) {
    let mut j = 0;
    while j + NR <= n {
        let mut i = 0;
        while i + MR <= m {
            tile::<S, W, MR, NR>(k, a.add(i * lda), lda, b.add(j * ldb), ldb, c.add(i * ldc + j), ldc, acc);
            i += MR;
        }
        while i < m {
            tile::<S, W, 1, NR>(k, a.add(i * lda), lda, b.add(j * ldb), ldb, c.add(i * ldc + j), ldc, acc);
            i += 1;
        }
        j += NR;
    }
    while j < n {
        let mut i = 0;
        while i + MR <= m {
            tile::<S, W, MR, 1>(k, a.add(i * lda), lda, b.add(j * ldb), ldb, c.add(i * ldc + j), ldc, acc);
            i += MR;
        }
        while i < m {
            tile::<S, W, 1, 1>(k, a.add(i * lda), lda, b.add(j * ldb), ldb, c.add(i * ldc + j), ldc, acc);
            i += 1;
        }
        j += 1;
    }
}

/// One MR x (NV * LANES) tile of C = A W^T against a packed bf16 panel: each k broadcasts one A
/// element per row against NV weight vectors, so the accumulators stay in registers for the whole k loop.
#[inline(always)]
unsafe fn ptile<S: Simd, const MR: usize, const NV: usize>(
    k: usize, a: *const f32, lda: usize, panel: *const u16, c: *mut f32, ldc: usize, ncols: usize,
) {
    let nrw = NV * S::LANES;
    let mut acc = [[S::splat(0.0); NV]; MR];
    let mut p0 = 0;
    if S::LANE_FMA {
        // Load LANES consecutive k of each A row once and FMA against lanes: no per-element broadcasts.
        let ku = k / S::LANES * S::LANES;
        while p0 < ku {
            let mut av = [S::splat(0.0); MR];
            for i in 0..MR {
                av[i] = S::load(a.add(i * lda + p0));
            }
            for l in 0..S::LANES {
                let bp = panel.add((p0 + l) * nrw);
                let mut b = [S::splat(0.0); NV];
                for j in 0..NV {
                    b[j] = S::load_bf16(bp.add(j * S::LANES));
                }
                for i in 0..MR {
                    for j in 0..NV {
                        acc[i][j] = S::fma_lane(av[i], l, b[j], acc[i][j]);
                    }
                }
            }
            p0 += S::LANES;
        }
    }
    for p in p0..k {
        let bp = panel.add(p * nrw);
        let mut b = [S::splat(0.0); NV];
        for j in 0..NV {
            b[j] = S::load_bf16(bp.add(j * S::LANES));
        }
        for i in 0..MR {
            let av = S::splat(*a.add(i * lda + p));
            for j in 0..NV {
                acc[i][j] = S::fma(av, b[j], acc[i][j]);
            }
        }
    }
    if ncols == nrw {
        for i in 0..MR {
            for j in 0..NV {
                S::store(c.add(i * ldc + j * S::LANES), acc[i][j]);
            }
        }
    } else {
        let mut tmp = [0.0f32; MAX_PANEL];
        for i in 0..MR {
            for j in 0..NV {
                S::store(tmp.as_mut_ptr().add(j * S::LANES), acc[i][j]);
            }
            std::ptr::copy_nonoverlapping(tmp.as_ptr(), c.add(i * ldc), ncols);
        }
    }
}

/// C[m, cols] = A[m, k] W^T over `np` consecutive packed panels covering `ncols` output columns.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_packed_block_g<S: Simd, const MR: usize, const NV: usize>(
    m: usize, k: usize, a: *const f32, lda: usize, w: *const u16, np: usize, ncols: usize, c: *mut f32, ldc: usize,
) {
    let nrw = NV * S::LANES;
    for pi in 0..np {
        let (panel, nc, cc) = (w.add(pi * k * nrw), (ncols - pi * nrw).min(nrw), c.add(pi * nrw));
        let mut i = 0;
        while i + MR <= m {
            ptile::<S, MR, NV>(k, a.add(i * lda), lda, panel, cc.add(i * ldc), ldc, nc);
            i += MR;
        }
        while i + 4 <= m {
            ptile::<S, 4, NV>(k, a.add(i * lda), lda, panel, cc.add(i * ldc), ldc, nc);
            i += 4;
        }
        while i + 2 <= m {
            ptile::<S, 2, NV>(k, a.add(i * lda), lda, panel, cc.add(i * ldc), ldc, nc);
            i += 2;
        }
        while i < m {
            ptile::<S, 1, NV>(k, a.add(i * lda), lda, panel, cc.add(i * ldc), ldc, nc);
            i += 1;
        }
    }
}

const LOG2E: f32 = std::f32::consts::LOG2_E;

/// Cephes-style expf, ~1 ulp on [-87, 88]; inputs are clamped to that range.
#[inline(always)]
unsafe fn exp_v<S: Simd>(x: S::V) -> S::V {
    let x = S::min(S::max(x, S::splat(-87.0)), S::splat(88.0));
    let n = S::round(S::mul(x, S::splat(LOG2E)));
    let r = S::fma(n, S::splat(-0.693_359_4), x);
    let r = S::fma(n, S::splat(2.121_944_4e-4), r);
    let mut p = S::splat(1.987_569_1e-4);
    p = S::fma(p, r, S::splat(1.398_199_9e-3));
    p = S::fma(p, r, S::splat(8.333_452e-3));
    p = S::fma(p, r, S::splat(4.166_579_6e-2));
    p = S::fma(p, r, S::splat(1.666_666_5e-1));
    p = S::fma(p, r, S::splat(0.5));
    p = S::fma(p, S::mul(r, r), S::add(r, S::splat(1.0)));
    S::mul(p, S::pow2i(n))
}

/// gelu_tanh(x) = 0.5 x (1 + tanh(u)) = x / (1 + exp(-2u)),  u = sqrt(2/pi) (x + 0.044715 x^3)
#[inline(always)]
unsafe fn gelu_v<S: Simd>(x: S::V) -> S::V {
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    let x3 = S::mul(S::mul(x, x), x);
    let u = S::mul(S::fma(x3, S::splat(0.044715), x), S::splat(-2.0 * K));
    S::div(x, S::add(S::splat(1.0), exp_v::<S>(u)))
}

/// x[i] = gelu(x[i]) * y[i]
#[inline(always)]
unsafe fn gelu_mul_g<S: Simd>(x: &mut [f32], y: &[f32]) {
    let n = x.len().min(y.len());
    let (xp, yp) = (x.as_mut_ptr(), y.as_ptr());
    let nv = n / S::LANES * S::LANES;
    let mut i = 0;
    while i < nv {
        S::store(xp.add(i), S::mul(gelu_v::<S>(S::load(xp.add(i))), S::load(yp.add(i))));
        i += S::LANES;
    }
    for i in nv..n {
        *xp.add(i) = gelu_v::<Scalar>(*xp.add(i)) * *yp.add(i);
    }
}

/// In-place softmax of one row.
#[inline(always)]
unsafe fn softmax_g<S: Simd>(x: &mut [f32]) {
    let n = x.len();
    let p = x.as_mut_ptr();
    let nv = n / S::LANES * S::LANES;
    let mut mv = S::splat(f32::NEG_INFINITY);
    let mut i = 0;
    while i < nv {
        mv = S::max(mv, S::load(p.add(i)));
        i += S::LANES;
    }
    let mut m = S::hmax(mv);
    for i in nv..n {
        m = m.max(*p.add(i));
    }
    let (mv, mut sv) = (S::splat(m), S::splat(0.0));
    i = 0;
    while i < nv {
        let e = exp_v::<S>(S::add(S::load(p.add(i)), S::mul(mv, S::splat(-1.0))));
        S::store(p.add(i), e);
        sv = S::add(sv, e);
        i += S::LANES;
    }
    let mut s = S::hsum(sv);
    for i in nv..n {
        let e = exp_v::<Scalar>(*p.add(i) - m);
        *p.add(i) = e;
        s += e;
    }
    let inv = S::splat(1.0 / s);
    i = 0;
    while i < nv {
        S::store(p.add(i), S::mul(S::load(p.add(i)), inv));
        i += S::LANES;
    }
    for i in nv..n {
        *p.add(i) *= 1.0 / s;
    }
}

/// Softmax down each of the 32 columns of a row-major [rows, 32] block, without the division: x becomes
/// exp(x - column max) and the column sums are returned. All 32 / LANES column vectors advance together, so the
/// max and sum chains are independent.
#[inline(always)]
unsafe fn softmax_cols_g<S: Simd>(x: &mut [f32]) -> [f32; 32] {
    const W: usize = 32;
    let (rows, p, nv) = (x.len() / W, x.as_mut_ptr(), W / S::LANES);
    let mut m = [S::splat(f32::NEG_INFINITY); W];
    for r in 0..rows {
        for j in 0..nv {
            m[j] = S::max(m[j], S::load(p.add(r * W + j * S::LANES)));
        }
    }
    for j in 0..nv {
        m[j] = S::mul(m[j], S::splat(-1.0));
    }
    let mut sv = [S::splat(0.0); W];
    for r in 0..rows {
        for j in 0..nv {
            let q = p.add(r * W + j * S::LANES);
            let e = exp_v::<S>(S::add(S::load(q), m[j]));
            S::store(q, e);
            sv[j] = S::add(sv[j], e);
        }
    }
    let mut sums = [0.0f32; W];
    for j in 0..nv {
        S::store(sums.as_mut_ptr().add(j * S::LANES), sv[j]);
    }
    sums
}

// ---------------------------------------------------------------- per-ISA instantiation

macro_rules! isa_kernels {
    ($modname:ident, $S:ty, $MR:literal, $NR:literal, $PMR:literal, $PNV:literal $(, $feat:literal)?) => {
        mod $modname {
            use super::*;
            pub(super) const PMR: usize = $PMR;
            pub(super) const NRW: usize = $PNV * <$S as Simd>::LANES;

            $(#[target_feature(enable = $feat)])?
            #[allow(clippy::too_many_arguments)]
            pub(super) unsafe fn gemm_packed_block(
                m: usize, k: usize, a: *const f32, lda: usize, w: *const u16, np: usize, ncols: usize,
                c: *mut f32, ldc: usize,
            ) {
                gemm_packed_block_g::<$S, $PMR, $PNV>(m, k, a, lda, w, np, ncols, c, ldc)
            }

            $(#[target_feature(enable = $feat)])?
            #[allow(clippy::too_many_arguments)]
            pub(super) unsafe fn gemm_block<W: Elem>(
                m: usize, n: usize, k: usize, a: *const f32, lda: usize, b: *const W, ldb: usize,
                c: *mut f32, ldc: usize, acc: bool,
            ) {
                gemm_block_g::<$S, W, $MR, $NR>(m, n, k, a, lda, b, ldb, c, ldc, acc)
            }

            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn gelu_mul(x: &mut [f32], y: &[f32]) {
                gelu_mul_g::<$S>(x, y)
            }

            $(#[target_feature(enable = $feat)])?
            pub(super) unsafe fn softmax(x: &mut [f32]) {
                softmax_g::<$S>(x)
            }

            $(#[target_feature(enable = $feat)])?
            #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
            pub(super) unsafe fn softmax_cols(x: &mut [f32]) -> [f32; 32] {
                softmax_cols_g::<$S>(x)
            }
        }
    };
}

// (dot-kernel MR, NR), (packed-kernel MR, vectors per panel row): sized to each ISA's register file.
isa_kernels!(k_scalar, Scalar, 4, 4, 4, 4);
#[cfg(target_arch = "x86_64")]
isa_kernels!(k_avx2, x86::Avx2, 4, 3, 6, 2, "avx2,fma");
#[cfg(target_arch = "x86_64")]
isa_kernels!(k_avx512, x86::Avx512, 4, 4, 12, 2, "avx512f,avx2,fma");
#[cfg(target_arch = "aarch64")]
isa_kernels!(k_neon, arm::Neon, 4, 4, 4, 3, "neon");

macro_rules! dispatch {
    ($f:ident $(::<$t:ty>)? ($($arg:expr),*)) => {
        // SAFETY: `isa()` only returns an ISA whose features were detected at runtime; each caller checks its
        // operands' bounds.
        unsafe {
            match isa() {
                Isa::Scalar => k_scalar::$f$(::<$t>)?($($arg),*),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2 => k_avx2::$f$(::<$t>)?($($arg),*),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx512 => k_avx512::$f$(::<$t>)?($($arg),*),
                #[cfg(target_arch = "aarch64")]
                Isa::Neon => k_neon::$f$(::<$t>)?($($arg),*),
            }
        }
    };
}

macro_rules! isa_const {
    ($c:ident) => {
        match isa() {
            Isa::Scalar => k_scalar::$c,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => k_avx2::$c,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => k_avx512::$c,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => k_neon::$c,
        }
    };
}

/// bf16 weights W[n, k] repacked into panels of `nrw` consecutive output columns, panel p starting at column p * nrw
/// (zero-padded past n). The `layout` within a panel, and `nrw`, follow `bf16_mode` and `sme::available`:
/// - `Rows`: for each k, the panel's `nrw` columns; `nrw` is the ISA's broadcast-kernel width (4 scalar, 16 AVX2,
///   32 AVX-512, 12 NEON).
/// - `Sme`: as `Rows`, `nrw` = 32 (`sme::NRW`).
/// - `Pairs`: k in pairs, padded to a multiple of 32; `nrw` = 32 (`bf16dot::NRW`).
/// - `Quads`: k in groups of four, padded to a multiple of 4; `nrw` = 8 (`bfmmla::NRW`).
pub(crate) struct Packed {
    pub n: usize,
    pub k: usize,
    nrw: usize,
    layout: Layout,
    data: Vec<u16>,
}

/// The widest panel of any layout: bounds the stack buffers that hold a partial panel row.
const MAX_PANEL: usize = 64;

/// How a `Packed` panel orders its k values, following `bf16_mode` and `sme::available`.
/// Used by: `Rows` the fp32 broadcast kernels, `Sme` `sme::gemm_packed`, `Pairs` `bf16dot` and `amx`, `Quads` `bfmmla`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// One row of `nrw` columns per k, for the fp32 kernels.
    Rows,
    /// `Rows` in 32-column panels, for `sme::gemm_packed`.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    Sme,
    /// In k pairs for `bf16dot` and AMX: each panel row holds (w[c][2q], w[c][2q+1]) for its `nrw` columns, k padded
    /// to a multiple of 32.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    Pairs,
    /// In k groups of four for `bfmmla`: per group, each column pair's four k of both columns, k padded to a
    /// multiple of 4.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    Quads,
}

impl Layout {
    /// Offset of column `c`'s k index `kk` within a panel of `nrw` columns.
    #[inline(always)]
    fn offset(self, nrw: usize, c: usize, kk: usize) -> usize {
        match self {
            Layout::Rows | Layout::Sme => kk * nrw + c,
            Layout::Pairs => (kk / 2) * 2 * nrw + 2 * c + kk % 2,
            Layout::Quads => kk / 4 * 4 * nrw + c * 4 + kk % 4,
        }
    }

    /// k as stored in a panel: padded to a multiple of 32 for `Pairs` and of 4 for `Quads`.
    fn kpad(self, k: usize) -> usize {
        match self {
            Layout::Rows | Layout::Sme => k,
            Layout::Pairs => k.next_multiple_of(32),
            Layout::Quads => k.next_multiple_of(4),
        }
    }
}

/// Pack `n` output columns over `k`, value `at(c, kk)` for column `c` and k index `kk`, into panels of `nrw` columns
/// in `layout`, zero-padded, one panel per task.
fn pack_panels(n: usize, k: usize, nrw: usize, layout: Layout, at: impl Fn(usize, usize) -> u16 + Sync) -> Vec<u16> {
    let plen = layout.kpad(k) * nrw;
    let mut data = vec![0u16; n.div_ceil(nrw) * plen];
    data.par_chunks_mut(plen).enumerate().for_each(|(p, dst)| {
        for c in 0..nrw.min(n - p * nrw) {
            for kk in 0..k {
                dst[layout.offset(nrw, c, kk)] = at(p * nrw + c, kk);
            }
        }
    });
    data
}

impl Packed {
    #[cfg(test)]
    pub fn new(w: &[u16], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k);
        Self::pack(n, k, |i| w[i])
    }

    /// Pack row-major little-endian bf16 bytes (e.g. a slice of a memory-mapped checkpoint) without an
    /// intermediate copy.
    pub fn from_le_bytes(b: &[u8], n: usize, k: usize) -> Self {
        assert_eq!(b.len(), 2 * n * k);
        Self::pack(n, k, |i| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]))
    }

    fn pack(n: usize, k: usize, at: impl Fn(usize) -> u16 + Sync) -> Self {
        #[cfg(target_arch = "x86_64")]
        let (layout, nrw) =
            if bf16_mode() != Bf16::Off { (Layout::Pairs, bf16dot::NRW) } else { (Layout::Rows, isa_const!(NRW)) };
        #[cfg(target_arch = "aarch64")]
        let (layout, nrw) = if bf16_mode() == Bf16::Bfmmla {
            (Layout::Quads, bfmmla::NRW)
        } else if crate::sme::available() {
            (Layout::Sme, crate::sme::NRW)
        } else {
            (Layout::Rows, isa_const!(NRW))
        };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let (layout, nrw) = (Layout::Rows, isa_const!(NRW));
        assert!(nrw <= MAX_PANEL);
        let data = pack_panels(n, k, nrw, layout, |c, kk| at(c * k + kk));
        Self { n, k, nrw, layout, data }
    }

    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn data(&self) -> &[u16] {
        &self.data
    }
}

/// The least work, in FLOPs, a matmul task is given: smaller products are not split so finely that scheduling
/// dominates.
const TASK_FLOPS: usize = 1 << 21;

/// A raw pointer handed to the tasks of one parallel call.
#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(pub(crate) *mut T);
// SAFETY: a SendPtr only crosses threads within one parallel call, whose tasks access disjoint regions through it (each
// use states which) and finish before the call returns.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// How many tasks to split an [m, k] x [k, n] matmul into: ~3 per thread gives work stealing (and slower efficiency
/// cores) some slack, and each task gets at least `task_flops`.
pub(crate) fn task_target(m: usize, n: usize, k: usize, task_flops: usize) -> usize {
    (3 * rayon::current_num_threads()).min((2 * m * n * k / task_flops).max(1))
}

/// The tasks of a matmul against packed weights: C's `m` rows in blocks of `mb`, its `n` columns in panels of `nrw`,
/// grouped `pc` panels per task. Task `t` covers row block `t % mblocks` and panel group `t / mblocks`.
#[derive(Clone, Copy)]
pub(crate) struct Split {
    m: usize,
    n: usize,
    nrw: usize,
    mb: usize,
    mblocks: usize,
    panels: usize,
    pc: usize,
    groups: usize,
}

/// One task of a `Split`: rows `i0..i0 + rows` of C against panels `p0..p0 + np`, which cover columns
/// `j0..j0 + ncols`.
#[derive(Clone, Copy)]
pub(crate) struct Task {
    pub i0: usize,
    pub rows: usize,
    pub p0: usize,
    pub np: usize,
    pub j0: usize,
    pub ncols: usize,
}

impl Split {
    pub(crate) fn new(m: usize, n: usize, nrw: usize, mb: usize, pc: usize) -> Self {
        let panels = n.div_ceil(nrw);
        Self { m, n, nrw, mb, mblocks: m.div_ceil(mb), panels, pc, groups: panels.div_ceil(pc) }
    }

    /// Row blocks of `mb`, with the panels grouped so there are about `task_target` tasks in all.
    fn balanced(m: usize, n: usize, k: usize, nrw: usize, mb: usize) -> Self {
        let (panels, mblocks) = (n.div_ceil(nrw), m.div_ceil(mb));
        let pc = (panels * mblocks).div_ceil(task_target(m, n, k, TASK_FLOPS)).clamp(1, panels);
        Self::new(m, n, nrw, mb, pc)
    }

    #[inline]
    fn task(&self, t: usize) -> Task {
        let (i0, p0) = (t % self.mblocks * self.mb, t / self.mblocks * self.pc);
        let (rows, np) = (self.mb.min(self.m - i0), self.pc.min(self.panels - p0));
        let j0 = p0 * self.nrw;
        Task { i0, rows, p0, np, j0, ncols: (np * self.nrw).min(self.n - j0) }
    }

    /// Run every task: on the calling thread when there is one, else across the current pool.
    #[inline]
    pub(crate) fn run(&self, f: impl Fn(Task) + Sync) {
        let tasks = self.mblocks * self.groups;
        if tasks == 1 {
            f(self.task(0))
        } else {
            (0..tasks).into_par_iter().for_each(|t| f(self.task(t)));
        }
    }
}

/// Run `f` on this thread's bf16 copy of A, `len` values zeroed. The buffer is taken out while in use: a worker
/// waiting inside this matmul may steal another text's matmul, which then starts from an empty one.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn with_a16(len: usize, f: impl FnOnce(&mut [u16])) {
    use std::cell::RefCell;
    thread_local!(static A16: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) });
    let mut a16 = A16.take();
    a16.clear();
    a16.resize(len, 0);
    f(&mut a16);
    A16.set(a16);
}

/// C[m, w.n] = A[m, w.k] W^T with packed weights, split into (row block, panel group) tasks.
pub(crate) fn gemm_packed(m: usize, a: &[f32], lda: usize, w: &Packed, c: &mut [f32], ldc: usize) {
    let (n, k, nrw) = (w.n, w.k, w.nrw);
    if m == 0 {
        return;
    }
    assert!(lda >= k && ldc >= n && a.len() >= (m - 1) * lda + k && c.len() >= (m - 1) * ldc + n);
    #[cfg(target_arch = "aarch64")]
    if w.layout == Layout::Sme {
        return crate::sme::gemm_packed(m, a, lda, w, c, ldc);
    }
    #[cfg(target_arch = "x86_64")]
    if w.layout == Layout::Pairs {
        return gemm_pairs(m, a, lda, w, c, ldc);
    }
    #[cfg(target_arch = "aarch64")]
    if w.layout == Layout::Quads {
        return gemm_quads(m, a, lda, w, c, ldc);
    }
    let pmr: usize = isa_const!(PMR);
    let (ap, wp, cp) = (SendPtr(a.as_ptr() as *mut f32), SendPtr(w.data.as_ptr() as *mut u16), SendPtr(c.as_mut_ptr()));
    Split::balanced(m, n, k, nrw, pmr * 8).run(|t| {
        let (ap, wp, cp) = (ap, wp, cp);
        // SAFETY: bounds checked above; tasks write disjoint (row block, column range) regions of C.
        let (a, w, c) = unsafe {
            (ap.0.add(t.i0 * lda) as *const f32, wp.0.add(t.p0 * k * nrw) as *const u16, cp.0.add(t.i0 * ldc + t.j0))
        };
        dispatch!(gemm_packed_block(t.rows, k, a, lda, w, t.np, t.ncols, c, ldc))
    });
}

/// `gemm_packed` for pair-packed weights, with the same (row block, panel group) split. A is rounded to bf16 once,
/// on the calling thread, into the per-thread `with_a16` buffer (rows padded to a multiple of 32 for AMX, padding
/// zero).
#[cfg(target_arch = "x86_64")]
fn gemm_pairs(m: usize, a: &[f32], lda: usize, w: &Packed, c: &mut [f32], ldc: usize) {
    let (n, k, nrw) = (w.n, w.k, w.nrw);
    let (kp, amx) = (k.next_multiple_of(32), bf16_mode() == Bf16::Amx);
    // Rows per task: a multiple of 32 for AMX tiles; 96 balances weight reuse against parallelism for short texts.
    let split = Split::balanced(m, n, k, nrw, 96);
    with_a16(if amx { m.next_multiple_of(32) } else { m } * kp, |a16| {
        // SAFETY: gemm_packed checked the bounds of A; `bf16_mode` checked the CPU supports AVX-512 BF16.
        unsafe {
            if amx {
                amx::round_rows_tiled(m, k, a.as_ptr(), lda, a16.as_mut_ptr(), kp)
            } else {
                bf16dot::round_rows(m, k, a.as_ptr(), lda, a16.as_mut_ptr(), kp)
            }
        };
        let (ap, wp, cp) =
            (SendPtr(a16.as_ptr() as *mut u16), SendPtr(w.data.as_ptr() as *mut u16), SendPtr(c.as_mut_ptr()));
        split.run(|t| {
            let (ap, wp, cp) = (ap, wp, cp);
            // SAFETY: tasks write disjoint (row block, column range) regions of C; the CPU (and, for AMX, the OS)
            // support was checked by `bf16_mode`.
            unsafe {
                let (ab, wb, cb) = (ap.0.add(t.i0 * kp), wp.0.add(t.p0 * kp * nrw), cp.0.add(t.i0 * ldc + t.j0));
                if amx {
                    amx::gemm_block(t.rows, kp, ab, kp, true, wb, kp * nrw, t.np, t.ncols, cb, ldc)
                } else {
                    bf16dot::gemm_block(t.rows, kp, ab, kp, wb, t.np, t.ncols, cb, ldc)
                }
            }
        });
    });
}

/// `gemm_packed` for `BFMMLA`-packed weights: A is rounded to bf16 once, in `bfmmla` tile order, into the per-thread
/// `with_a16` buffer, then split into (row block, panel group) tasks like `gemm_pairs`.
#[cfg(target_arch = "aarch64")]
fn gemm_quads(m: usize, a: &[f32], lda: usize, w: &Packed, c: &mut [f32], ldc: usize) {
    let (n, k, nrw) = (w.n, w.k, w.nrw);
    let kq = k.next_multiple_of(4);
    // Rows per task: a multiple of bfmmla::MR; 64 balances weight reuse against parallelism for short texts.
    let split = Split::balanced(m, n, k, nrw, 64);
    with_a16(m.next_multiple_of(bfmmla::MR) * kq, |a16| {
        // Rounding runs at about 8 values per ns on one core: below 1 << 16 values (about 8 us) it stays on the
        // calling thread; above, it is split by tile across the pool, each job at least 1 << 14 values (about 2 us)
        // so it outweighs its scheduling.
        let tile = bfmmla::MR * kq;
        let ap0 = SendPtr(a.as_ptr() as *mut f32);
        let round = |(t, dst): (usize, &mut [u16])| {
            let (r0, ap0) = (t * bfmmla::MR, ap0);
            // SAFETY: gemm_packed checked the bounds of A; each tile of `a16` is written by one call.
            unsafe {
                bfmmla::round_rows_tiled(bfmmla::MR.min(m - r0), k, ap0.0.add(r0 * lda), lda, dst.as_mut_ptr(), kq)
            }
        };
        if m * k < 1 << 16 {
            a16.chunks_mut(tile).enumerate().for_each(round);
        } else {
            a16.par_chunks_mut(tile).with_min_len((1 << 14) / tile + 1).enumerate().for_each(round);
        }
        let (ap, wp, cp) =
            (SendPtr(a16.as_ptr() as *mut u16), SendPtr(w.data.as_ptr() as *mut u16), SendPtr(c.as_mut_ptr()));
        split.run(|t| {
            let (ap, wp, cp) = (ap, wp, cp);
            // SAFETY: tasks write disjoint (row block, column range) regions of C; `bf16_mode` checked the CPU has BF16.
            unsafe {
                let (ab, wb, cb) = (ap.0.add(t.i0 * kq), wp.0.add(t.p0 * kq * nrw), cp.0.add(t.i0 * ldc + t.j0));
                bfmmla::gemm_block(t.rows, kq, ab, wb, kq * nrw, t.np, t.ncols, cb, ldc)
            }
        });
    });
}

// ---------------------------------------------------------------- public entry points

#[allow(clippy::too_many_arguments)]
fn check_gemm<W>(m: usize, n: usize, k: usize, a: &[f32], lda: usize, b: &[W], ldb: usize, c: &[f32], ldc: usize) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(lda >= k && ldb >= k && ldc >= n, "gemm: bad leading dimensions");
    assert!(a.len() >= (m - 1) * lda + k, "gemm: A too small");
    assert!(b.len() >= (n - 1) * ldb + k, "gemm: B too small");
    assert!(c.len() >= (m - 1) * ldc + n, "gemm: C too small");
}

/// C[m, n] (+)= A[m, k] B[n, k]^T on the calling thread. B rows are f32 or bf16 (u16 bits).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_serial<W: Elem>(
    m: usize, n: usize, k: usize, a: &[f32], lda: usize, b: &[W], ldb: usize, c: &mut [f32], ldc: usize, acc: bool,
) {
    check_gemm(m, n, k, a, lda, b, ldb, c, ldc);
    if m == 0 || n == 0 {
        return;
    }
    let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    dispatch!(gemm_block::<W>(m, n, k, ap, lda, bp, ldb, cp, ldc, acc))
}

/// x[i] = gelu_tanh(x[i]) * y[i]
pub(crate) fn gelu_mul(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    dispatch!(gelu_mul(x, y))
}

pub(crate) fn softmax(x: &mut [f32]) {
    dispatch!(softmax(x))
}

/// exp(x - column max) down each column of a row-major [rows, 32] block; returns the column sums, for the caller
/// to normalize with.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) fn softmax_cols(x: &mut [f32]) -> [f32; 32] {
    assert!(x.len().is_multiple_of(32));
    dispatch!(softmax_cols(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (0..k).map(|p| a[i * k + p] as f64 * b[j * k + p] as f64).sum::<f64>() as f32;
            }
        }
        c
    }

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// The bf16 paths (`EG2_BF16=amx|dot|bfmmla`, else this checks the fp32 packed path) against a product of the
    /// bf16-rounded inputs: exact up to fp32 accumulation order. Shapes cover row, column and k tails.
    #[test]
    fn packed_matches_naive_in_its_precision() {
        for &(m, n, k) in &[(1, 1, 1), (5, 7, 13), (33, 70, 100), (67, 131, 257), (96, 512, 512), (130, 64, 2048)] {
            let a = rnd(m * k, 3);
            let b = rnd(n * k, 4);
            let bb: Vec<u16> = b.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
            let bt: Vec<f32> = bb.iter().map(|&x| bf16_to_f32(x)).collect();
            let bf16 = bf16_mode() != Bf16::Off;
            let ar: Vec<f32> = if bf16 { a.iter().map(|&x| bf16_to_f32(f32_to_bf16(x))).collect() } else { a.clone() };
            let want = naive(m, n, k, &ar, &bt);
            let mut c = vec![0.0; m * n];
            gemm_packed(m, &a, k, &Packed::new(&bb, n, k), &mut c, n);
            let err = c.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            assert!(err < 1e-4, "{:?} {m}x{n}x{k}: {err}", bf16_mode());
        }
    }

    #[test]
    fn gemm_matches_naive() {
        for &(m, n, k) in &[(1, 1, 1), (5, 7, 13), (67, 131, 257), (4, 512, 512)] {
            let a = rnd(m * k, 1);
            let b = rnd(n * k, 2);
            let want = naive(m, n, k, &a, &b);
            let mut c = vec![0.0; m * n];
            gemm_serial(m, n, k, &a, k, &b, k, &mut c, n, false);
            let err = c.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            assert!(err < 1e-4, "{m}x{n}x{k}: {err}");
            // bf16 weights: truncate b, compare against the truncated naive product
            let bb: Vec<u16> = b.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
            let bt: Vec<f32> = bb.iter().map(|&x| bf16_to_f32(x)).collect();
            let want = naive(m, n, k, &a, &bt);
            gemm_serial(m, n, k, &a, k, &bb, k, &mut c, n, false);
            let err = c.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            assert!(err < 1e-4, "bf16 {m}x{n}x{k}: {err}");
            if bf16_mode() != Bf16::Off {
                continue; // the packed path rounds A too: `packed_matches_naive_in_its_precision` covers it
            }
            c.fill(f32::NAN);
            gemm_packed(m, &a, k, &Packed::new(&bb, n, k), &mut c, n);
            let err = c.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            assert!(err < 1e-4, "packed {m}x{n}x{k}: {err}");
        }
    }

    #[test]
    fn transcendental_accuracy() {
        let xs: Vec<f32> = (0..20001).map(|i| -10.0 + i as f32 * 1e-3).collect();
        let mut g = xs.clone();
        gelu_mul(&mut g, &vec![1.0; xs.len()]);
        for (x, y) in xs.iter().zip(&g) {
            let x = *x as f64;
            let want = 0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
            assert!((want - *y as f64).abs() < 2e-6 * (1.0 + want.abs()), "gelu({x}) = {y}, want {want}");
        }
        let mut s: Vec<f32> = (0..37).map(|i| (i as f32 * 0.37).sin() * 20.0).collect();
        let e: Vec<f64> = s.iter().map(|&x| (x as f64).exp()).collect();
        let z: f64 = e.iter().sum();
        softmax(&mut s);
        for (p, q) in s.iter().zip(&e) {
            assert!((*p as f64 - q / z).abs() < 1e-6);
        }
    }
}

#[cfg(test)]
mod speed {
    use super::*;
    use std::time::Instant;

    /// cargo test --release -- --ignored --nocapture gemm_speed   (RAYON_NUM_THREADS=1 for per-core numbers)
    /// EG2_BF16 picks the path, as in the model.
    #[test]
    #[ignore]
    fn gemm_speed() {
        // the model's shapes for a 128-token text: q|k|v, gate|up, down, o; then larger m for comparison
        for &(m, n, k) in &[
            (128, 2048, 512),
            (128, 4096, 512),
            (128, 512, 2048),
            (128, 512, 1024),
            (512, 2048, 512),
            (512, 512, 2048),
            (16, 2048, 512),
            (2340, 768, 768),
            (2340, 3072, 768),
            (2340, 768, 3072),
            (540, 3072, 768),
        ] {
            let a = vec![0.5f32; m * k];
            let w = Packed::new(&vec![0x3f80u16; n * k], n, k);
            let mut c = vec![0.0f32; m * n];
            gemm_packed(m, &a, k, &w, &mut c, n);
            let iters = 20;
            let t = Instant::now();
            for _ in 0..iters {
                gemm_packed(m, &a, k, &w, &mut c, n);
            }
            let dt = t.elapsed().as_secs_f64() / iters as f64;
            println!("{m:>4}x{n:>4}x{k:>4}: {:>7.1} GFLOP/s", 2.0 * (m * n * k) as f64 / dt / 1e9);
        }
    }
}
