//! Python bindings (`--features python`, built with maturin): `eg2cpu.Model` and decision `Handle`s.

use std::collections::HashMap;
use std::path::PathBuf;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::{Gate, Handle, Kind, Model, Output, Partial, Span, Template, Verdict};

type Pair<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>);

fn err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

/// Per-layer means as one [layers, hidden] array.
fn means_array(layer_means: &[Vec<f32>]) -> Array2<f32> {
    let d = layer_means.first().map_or(0, Vec::len);
    Array2::from_shape_vec((layer_means.len(), d), layer_means.concat()).expect("layer means shape")
}

fn to_numpy(py: Python<'_>, out: Output) -> Pair<'_> {
    (out.embedding.into_pyarray(py), means_array(&out.layer_means).into_pyarray(py))
}

/// EmbeddingGemma 2 encoder on CPU, for text, images and audio. `path` is the model snapshot directory; by default
/// `$EG2_MODEL_DIR`, else the Hugging Face cache snapshot of `google/embeddinggemma-2`.
#[pyclass(name = "Model", module = "eg2cpu", frozen)]
struct PyModel {
    inner: Model,
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (path=None))]
    fn new(py: Python<'_>, path: Option<PathBuf>) -> PyResult<Self> {
        let inner = py.detach(|| match path {
            Some(p) => Model::load(&p),
            None => Model::load_default(),
        });
        Ok(Self { inner: inner.map_err(err)? })
    }

    /// `(embedding[768], layer_means[24, 512])` for one text: the L2-normalized embedding and the mean-pooled
    /// residual stream after each layer.
    fn embed<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Pair<'py>> {
        let out = py.detach(|| self.inner.embed(text)).map_err(err)?;
        Ok(to_numpy(py, out))
    }

    /// `embed` over many texts, run concurrently on all cores.
    fn embed_batch<'py>(&self, py: Python<'py>, texts: Vec<String>) -> PyResult<Vec<Pair<'py>>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let outs = py.detach(|| self.inner.embed_batch(&refs)).map_err(err)?;
        Ok(outs.into_iter().map(|o| to_numpy(py, o)).collect())
    }

    /// Early exit: run each text through layers 0..=`upto` only. Each result exposes the layer means so far
    /// and can be finished later with `resume_batch`, without recomputing those layers.
    fn embed_batch_partial(&self, py: Python<'_>, texts: Vec<String>, upto: usize) -> PyResult<Vec<PyPartial>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let parts = py.detach(|| self.inner.embed_batch_partial(&refs, upto)).map_err(err)?;
        Ok(parts.into_iter().map(|p| PyPartial { inner: std::sync::Mutex::new(Some(p)) }).collect())
    }

    /// Finish early-exited passes: `(embedding[768], layer_means[24, 512])` per text, as `embed_batch` returns.
    fn resume_batch<'py>(&self, py: Python<'py>, partials: Vec<Bound<'py, PyPartial>>) -> PyResult<Vec<Pair<'py>>> {
        let parts: Vec<Partial> = partials
            .iter()
            .map(|p| {
                p.get().inner.lock().unwrap().take().ok_or_else(|| PyRuntimeError::new_err("partial already resumed"))
            })
            .collect::<PyResult<_>>()?;
        let outs = py.detach(|| self.inner.resume_batch(parts));
        Ok(outs.into_iter().map(|o| to_numpy(py, o)).collect())
    }

    /// `(embedding[768], layer_means[24, 512])` for one image: PNG, JPEG or WebP, given as bytes or a path. The
    /// image is resized to about 9 x `max_soft_tokens` patches (70, 140, 280, 560 or 1120; 280 is the processor's
    /// default, 70 is about 5x faster) and its soft tokens are encoded by the text tower, so the embedding shares the
    /// text embeddings' space.
    #[pyo3(signature = (image, max_soft_tokens=crate::DEFAULT_SOFT_TOKENS))]
    fn embed_image<'py>(
        &self, py: Python<'py>, image: &Bound<'py, PyAny>, max_soft_tokens: usize,
    ) -> PyResult<Pair<'py>> {
        let bytes = match image.cast::<PyBytes>() {
            Ok(b) => b.as_bytes().to_vec(),
            Err(_) => {
                let path: PathBuf = image.extract()?;
                std::fs::read(&path).map_err(|e| PyRuntimeError::new_err(format!("{}: {e}", path.display())))?
            }
        };
        let out = py.detach(|| self.inner.embed_image_with(&bytes, max_soft_tokens)).map_err(err)?;
        Ok(to_numpy(py, out))
    }

    /// `(embedding[768], layer_means[24, 512])` for one audio clip: a WAV file (PCM or float, 16 kHz; channels are
    /// averaged) given as bytes or a path, or 16 kHz mono samples as a 1-D array: float32 in [-1, 1], or int16 (16-bit
    /// PCM, scaled by 1/32768 as a 16-bit WAV is). At most the first 30 s are used. Other formats and sampling rates need
    /// converting first (e.g. with soundfile or ffmpeg).
    fn embed_audio<'py>(&self, py: Python<'py>, audio: &Bound<'py, PyAny>) -> PyResult<Pair<'py>> {
        let slice_err = |e: numpy::AsSliceError| PyRuntimeError::new_err(e.to_string());
        let out = if let Ok(samples) = audio.extract::<PyReadonlyArray1<f32>>() {
            let samples = samples.as_slice().map_err(slice_err)?.to_vec();
            py.detach(|| self.inner.embed_audio_samples(&samples))
        } else if let Ok(pcm) = audio.extract::<PyReadonlyArray1<i16>>() {
            let samples: Vec<f32> = pcm.as_slice().map_err(slice_err)?.iter().map(|&v| v as f32 / 32768.0).collect();
            py.detach(|| self.inner.embed_audio_samples(&samples))
        } else {
            let bytes = match audio.cast::<PyBytes>() {
                Ok(b) => b.as_bytes().to_vec(),
                Err(_) => {
                    let path: PathBuf = audio.extract()?;
                    std::fs::read(&path).map_err(|e| PyRuntimeError::new_err(format!("{}: {e}", path.display())))?
                }
            };
            py.detach(|| self.inner.embed_audio(&bytes))
        };
        Ok(to_numpy(py, out.map_err(err)?))
    }

    /// Token ids with `<bos>`/`<eos>`, truncated to 1024.
    fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner.tokenize(text).map_err(err)
    }

    #[getter]
    fn num_layers(&self) -> usize {
        self.inner.cfg.layers.len()
    }

    /// Checkpoint revision (Hugging Face snapshot commit, or `local:<fingerprint>`); handles record it.
    #[getter]
    fn revision(&self) -> String {
        self.inner.revision().to_string()
    }

    /// Apply decision handles to spans (dicts of str -> str). Each distinct unit text is encoded once, in one
    /// concurrent batch. Returns, per span, one verdict dict per handle.
    fn apply<'py>(
        &self, py: Python<'py>, handles: Vec<Bound<'py, PyHandle>>, spans: Vec<Span>,
    ) -> PyResult<Vec<Vec<Bound<'py, PyDict>>>> {
        let hs: Vec<&Handle> = handles.iter().map(|h| &h.get().inner).collect();
        let out = py.detach(|| self.inner.apply(&hs, &spans)).map_err(err)?;
        out.into_iter().map(|row| row.into_iter().zip(&hs).map(|(v, h)| verdict_dict(py, h, v)).collect()).collect()
    }
}

fn verdict_dict<'py>(py: Python<'py>, h: &Handle, v: Verdict) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("decision", &h.name)?;
    d.set_item("status", v.status.as_str())?;
    d.set_item("answer", v.answer)?;
    match h.kind {
        Kind::Boolean => d.set_item("p", v.p.first().copied())?,
        Kind::Choice => {
            let p: HashMap<&str, f32> = h.labels.iter().map(String::as_str).zip(v.p.iter().copied()).collect();
            d.set_item("p", if v.p.is_empty() { None } else { Some(p) })?
        }
        Kind::Score => d.set_item("p", None::<f32>)?,
        Kind::Levels => {
            // Per-level probabilities keyed by level index, the expected level as the score,
            // the top probability as confidence, and the legend.
            let probs: HashMap<String, f32> = v.p.iter().enumerate().map(|(i, q)| (i.to_string(), *q)).collect();
            let legend: HashMap<String, &str> =
                h.labels.iter().enumerate().map(|(i, l)| (i.to_string(), l.as_str())).collect();
            d.set_item("p", if v.p.is_empty() { None } else { Some(probs) })?;
            // "value" below carries the same number; Python callers may read either key.
            d.set_item("score", v.value)?;
            d.set_item("confidence", v.p.iter().cloned().reduce(f32::max))?;
            d.set_item("legend", legend)?;
        }
    }
    d.set_item("value", v.value)?;
    d.set_item("z", v.z)?;
    d.set_item("distance", v.distance)?;
    d.set_item("prompt", v.prompt)?;
    Ok(d)
}

/// An early-exited forward pass (see `Model.embed_batch_partial`).
#[pyclass(name = "Partial", module = "eg2cpu", frozen)]
struct PyPartial {
    inner: std::sync::Mutex<Option<Partial>>,
}

#[pymethods]
impl PyPartial {
    /// Mean-pooled residual stream after each layer run so far, [layers_done, 512].
    #[getter]
    fn layer_means<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let g = self.inner.lock().unwrap();
        let p = g.as_ref().ok_or_else(|| PyRuntimeError::new_err("partial already resumed"))?;
        Ok(means_array(&p.layer_means).into_pyarray(py))
    }

    /// Layers run so far (the exit layer + 1).
    #[getter]
    fn layers_done(&self) -> usize {
        self.inner.lock().unwrap().as_ref().map_or(0, Partial::layers_done)
    }
}

/// A compiled contrastive decision (see `eg2cpu.decisions`). Load with `Handle(path)` or `Handle.from_bytes`.
#[pyclass(name = "Handle", module = "eg2cpu", frozen)]
struct PyHandle {
    inner: Handle,
}

#[pymethods]
impl PyHandle {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        Ok(Self { inner: Handle::load(&path).map_err(err)? })
    }

    #[staticmethod]
    fn from_bytes(data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        Ok(Self { inner: Handle::from_bytes(data.as_bytes()).map_err(err)? })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            Kind::Boolean => "boolean",
            Kind::Choice => "choice",
            Kind::Score => "score",
            Kind::Levels => "levels",
        }
    }

    #[getter]
    fn labels(&self) -> Vec<String> {
        self.inner.labels.clone()
    }

    #[getter]
    fn revision(&self) -> &str {
        &self.inner.revision
    }

    /// The handle's metadata as a JSON string (the Python wrapper parses it).
    #[getter]
    fn metadata_json(&self) -> String {
        self.inner.meta.to_string()
    }

    /// The text this handle embeds for `span`, or None when the gate rejects it or a field is missing.
    fn render(&self, span: Span) -> Option<String> {
        self.inner.render(&span)
    }

    fn __repr__(&self) -> String {
        format!("Handle({:?}, kind={})", self.inner.name, self.kind())
    }
}

/// Render a `{field}` template over a span, as handles do (None if a field is missing).
#[pyfunction]
#[pyo3(signature = (template, span, unit=None))]
fn render_template(template: &str, span: Span, unit: Option<&str>) -> PyResult<Option<String>> {
    let t = Template::parse(template).map_err(err)?;
    Ok(t.render(&span, unit.map(|u| ("unit", u))))
}

/// Evaluate an `applies_when` expression on a span (raises on a syntax error).
#[pyfunction]
fn gate_eval(expr: &str, span: Span) -> PyResult<bool> {
    Ok(Gate::parse(expr).map_err(err)?.eval(&span))
}

/// SIMD instruction set the kernels dispatched to (e.g. "neon", "avx512").
#[pyfunction]
fn isa() -> String {
    format!("{:?}", crate::isa()).to_lowercase()
}

#[pymodule]
fn _eg2cpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyHandle>()?;
    m.add_class::<PyPartial>()?;
    m.add_function(wrap_pyfunction!(isa, m)?)?;
    m.add_function(wrap_pyfunction!(render_template, m)?)?;
    m.add_function(wrap_pyfunction!(gate_eval, m)?)?;
    m.add("HANDLE_FORMAT", crate::FORMAT)?;
    m.add("MAX_TOKENS", crate::MAX_TOKENS)?;
    m.add("SOFT_TOKEN_BUDGETS", crate::SOFT_TOKEN_BUDGETS.to_vec())?;
    Ok(())
}
