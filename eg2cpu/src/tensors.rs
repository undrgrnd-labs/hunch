//! safetensors buffers: the header index, and tensors by name with dtype and shape checks.
//!
//! The checkpoint (`model.rs`, `vision.rs`) is read as bf16, widened to f32 or packed for the GEMM kernels straight
//! from the mapped bytes; decision handles (`decision.rs`) are f32.

use crate::simd::{Packed, bf16_to_f32};
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use std::collections::HashMap;

struct Entry {
    dtype: String,
    shape: Vec<usize>,
    /// Byte range in the whole buffer (header included).
    start: usize,
    end: usize,
}

/// The tensors of a safetensors buffer, indexed by name; the data stays in the buffer.
pub(crate) struct Tensors<'a> {
    buf: &'a [u8],
    header_len: usize,
    index: HashMap<String, Entry>,
    meta: HashMap<String, String>,
}

/// Bytes per element, for the dtypes whose sizes are checked against the shape.
fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "BOOL" | "U8" | "I8" => Some(1),
        "BF16" | "F16" | "U16" | "I16" => Some(2),
        "F32" | "U32" | "I32" => Some(4),
        "F64" | "U64" | "I64" => Some(8),
        _ => None,
    }
}

impl<'a> Tensors<'a> {
    /// Parse the header, checking each tensor's data range lies in `buf` and matches its dtype and shape.
    pub(crate) fn parse(buf: &'a [u8]) -> Result<Self> {
        ensure!(buf.len() >= 8, "not a safetensors file");
        let n = u64::from_le_bytes(buf[..8].try_into()?) as usize;
        let header_len = 8usize.checked_add(n).filter(|&h| h <= buf.len()).context("truncated safetensors header")?;
        let header: Value = serde_json::from_slice(&buf[8..header_len])?;
        let mut index = HashMap::new();
        let mut meta = HashMap::new();
        for (name, v) in header.as_object().context("safetensors header")? {
            if name == "__metadata__" {
                for (k, s) in v.as_object().context("__metadata__")? {
                    meta.insert(k.clone(), s.as_str().unwrap_or_default().to_string());
                }
                continue;
            }
            let dtype = v["dtype"].as_str().with_context(|| format!("{name}: dtype"))?.to_string();
            let shape = v["shape"]
                .as_array()
                .with_context(|| format!("{name}: shape"))?
                .iter()
                .map(|x| x.as_u64().map(|x| x as usize).with_context(|| format!("{name}: shape")))
                .collect::<Result<Vec<_>>>()?;
            let off = v["data_offsets"]
                .as_array()
                .filter(|o| o.len() == 2)
                .with_context(|| format!("{name}: data_offsets"))?;
            let at = |i: usize| {
                off[i]
                    .as_u64()
                    .and_then(|x| usize::try_from(x).ok()?.checked_add(header_len))
                    .with_context(|| format!("{name}: data_offsets"))
            };
            let (start, end) = (at(0)?, at(1)?);
            let bytes = |s: usize| shape.iter().try_fold(s, |n, &d| n.checked_mul(d));
            let size_ok =
                dtype_size(&dtype).is_none_or(|s| bytes(s).is_some_and(|b| Some(b) == end.checked_sub(start)));
            ensure!(start <= end && end <= buf.len() && size_ok, "{name}: bad data offsets");
            index.insert(name.clone(), Entry { dtype, shape, start, end });
        }
        Ok(Self { buf, header_len, index, meta })
    }

    /// The length prefix and JSON header.
    pub(crate) fn header(&self) -> &'a [u8] {
        &self.buf[..self.header_len]
    }

    /// The `__metadata__` string map.
    pub(crate) fn metadata(&self) -> &HashMap<String, String> {
        &self.meta
    }

    /// Every tensor's name and dtype.
    pub(crate) fn dtypes(&self) -> impl Iterator<Item = (&str, &str)> {
        self.index.iter().map(|(name, e)| (name.as_str(), e.dtype.as_str()))
    }

    /// A tensor's shape, `None` if there is no such tensor.
    pub(crate) fn shape(&self, name: &str) -> Option<&[usize]> {
        self.index.get(name).map(|e| e.shape.as_slice())
    }

    fn entry(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<&Entry> {
        let e = self.index.get(name).with_context(|| format!("missing tensor {name}"))?;
        if e.dtype != dtype {
            bail!("{name}: expected {dtype}, got {}", e.dtype);
        }
        if e.shape != shape {
            bail!("{name}: expected shape {shape:?}, got {:?}", e.shape);
        }
        Ok(e)
    }

    /// Where a `dtype` tensor of `shape` starts in the buffer.
    pub(crate) fn offset(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<usize> {
        Ok(self.entry(name, dtype, shape)?.start)
    }

    /// The raw little-endian bytes of a `dtype` tensor of `shape`.
    pub(crate) fn bytes(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<&'a [u8]> {
        let e = self.entry(name, dtype, shape)?;
        Ok(&self.buf[e.start..e.end])
    }

    /// A bf16 tensor's raw values.
    pub(crate) fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        Ok(self.bytes(name, "BF16", shape)?.as_chunks::<2>().0.iter().map(|&b| u16::from_le_bytes(b)).collect())
    }

    /// A bf16 tensor widened to f32.
    pub(crate) fn bf16_f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self
            .bytes(name, "BF16", shape)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| bf16_to_f32(u16::from_le_bytes(b)))
            .collect())
    }

    /// An f32 tensor.
    pub(crate) fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self.bytes(name, "F32", shape)?.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect())
    }

    /// A [n, k] bf16 matrix packed for the GEMM kernels straight from the buffer.
    pub(crate) fn packed(&self, name: &str, n: usize, k: usize) -> Result<Packed> {
        Ok(Packed::from_le_bytes(self.bytes(name, "BF16", &[n, k])?, n, k))
    }

    /// Several [n, k] bf16 matrices stacked along n into one packed matrix.
    pub(crate) fn stacked(&self, names: &[String], n: usize, k: usize) -> Result<Packed> {
        let mut bytes = Vec::with_capacity(names.len() * n * k * 2);
        for name in names {
            bytes.extend_from_slice(self.bytes(name, "BF16", &[n, k])?);
        }
        Ok(Packed::from_le_bytes(&bytes, names.len() * n, k))
    }
}
