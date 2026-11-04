//! CPU (SIMD) inference for the EmbeddingGemma 2 encoder: text, images through the vision tower, and audio through the
//! audio tower.
//!
//! bf16 weights, fp32 math, runtime-dispatched kernels for AVX-512, AVX2+FMA, NEON, or scalar, and SME matmuls on
//! Apple M4 and later. `EG2_BF16` opts into bf16 matmuls: AMX tiles or AVX-512 BF16 dot products on x86, BFMMLA on
//! Arm.

mod audio;
mod decision;
mod model;
mod sched;
mod simd;
#[cfg(target_arch = "aarch64")]
mod sme;
mod tensors;
mod vision;

pub use audio::{AudioConfig, MAX_SAMPLES, SAMPLE_RATE, decode_wav, frames_for, log_mel, soft_tokens_for};
pub use decision::{CmpOp, FORMAT, Gate, Handle, Kind, Operand, Readout, Span, Status, Template, Verdict};
pub use model::{Config, LayerSpec, MAX_TOKENS, Model, Output, Partial, default_model_dir, max_in_flight, take_trace};
pub use simd::{Bf16, Isa, bf16_mode, isa};
pub use vision::{DEFAULT_SOFT_TOKENS, Rgb, SOFT_TOKEN_BUDGETS, VisionConfig, decode as decode_image, resize as resize_image};

#[cfg(feature = "python")]
mod python;
