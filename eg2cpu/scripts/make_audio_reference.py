# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "transformers @ git+https://github.com/huggingface/transformers",
#   "sentencepiece",
#   "safetensors",
#   "numpy",
#   "soundfile",
#   "pyarrow",
#   "huggingface_hub",
# ]
# ///
"""Dump PyTorch fp32 CPU reference outputs for audio, for the Rust port to test against.

    uv run scripts/make_audio_reference.py tests/fixtures

Writes small WAV files to <dir>/audio/ and <dir>/audio_reference.json:
- three LibriSpeech utterances (validation-clean, via hf-internal-testing/librispeech_asr_dummy; LibriSpeech is
  CC BY 4.0, Panayotov et al. 2015), as 16-bit mono;
- a synthetic stereo float32 file (a chirp and noise) and a short 24-bit tone, for the WAV decoder and short inputs.

The samples are read with soundfile (channels averaged) and go through the checkpoint's own processor
(`AutoProcessor`: log-mel features, at most 30 s) and the full model, as `model(**processor(audio=[x]))`. One case
repeats an utterance past 30 s to exercise the truncation. For each case: token ids, the number of mel frames and a
sample of the features, the soft tokens the audio tower hands the text tower, all 24 layer means, and the final
embedding. The utterances' transcripts are embedded too, for a speech-text similarity check.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor, AutoTokenizer

out = Path(sys.argv[1])
(out / "audio").mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
proc = AutoProcessor.from_pretrained("google/embeddinggemma-2")
tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
model = AutoModel.from_pretrained("google/embeddinggemma-2", dtype=torch.float32).eval()
SR = 16000

table = pq.read_table(hf_hub_download(
    "hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet", repo_type="dataset"))
rows = table.to_pylist()
transcripts = {}


def speech(i, name):
    x, sr = sf.read(io.BytesIO(rows[i]["audio"]["bytes"]), dtype="float32")
    assert sr == SR
    path = out / "audio" / f"{name}.wav"
    sf.write(path, x, SR, subtype="PCM_16")
    transcripts[name] = rows[i]["text"].lower()
    return path


def chirp_stereo():
    t = np.arange(int(1.3 * SR)) / SR
    left = 0.4 * np.sin(2 * np.pi * (200 * t + 1500 * t * t))
    right = 0.2 * np.random.default_rng(0).standard_normal(t.size)
    path = out / "audio" / "chirp_stereo_f32.wav"
    sf.write(path, np.stack([left, right], 1).astype(np.float32), SR, subtype="FLOAT")
    return path


def tone_24bit():
    t = np.arange(int(0.3 * SR)) / SR
    path = out / "audio" / "tone_24bit.wav"
    sf.write(path, 0.5 * np.sin(2 * np.pi * 440 * t), SR, subtype="PCM_24")
    return path


files = {
    "speech_a": speech(0, "speech_a"),
    "speech_b": speech(1, "speech_b"),
    "speech_c": speech(2, "speech_c"),
    "chirp": chirp_stereo(),
    "tone": tone_24bit(),
}
# (file, times repeated): the repeated utterance runs past the processor's 30 s cap
cases_spec = [("speech_a", 1), ("speech_b", 1), ("speech_c", 1), ("chirp", 1), ("tone", 1), ("speech_b", 7)]


def samples(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR
    return x.mean(1, dtype=np.float32)


layers = list(model.language_model.layers)


def run(**inputs):
    caught = []
    hooks = [m.register_forward_hook(lambda mod, i, o: caught.append(o[0] if isinstance(o, tuple) else o)) for m in layers]
    try:
        o = model(**inputs)
    finally:
        for h in hooks:
            h.remove()
    return o, caught


rng = np.random.default_rng(0)
cases = []
with torch.no_grad():
    for key, repeat in cases_spec:
        path = files[key]
        x = np.tile(samples(path), repeat)
        inputs = proc(audio=[x], return_tensors="pt")
        mask = inputs["input_features_mask"][0]
        feats_in = inputs["input_features"][0][mask]  # the valid mel frames
        idx = rng.integers(0, feats_in.numel(), 512)
        o, caught = run(**inputs)
        soft = o.audio_hidden_states
        cases.append({
            "file": f"audio/{path.name}",
            "repeat": repeat,
            "ids": inputs["input_ids"][0].tolist(),
            "frames": int(feats_in.shape[0]),
            "feature_index": idx.tolist(),
            "feature_values": feats_in.reshape(-1)[idx].tolist(),
            "feature_mean": float(feats_in.mean()),
            "soft_tokens_head": soft[:2].tolist(),
            "soft_tokens_mean": soft.mean(0).tolist(),
            "layer_means": [c[0].mean(0).tolist() for c in caught],
            "embedding": F.normalize(o.last_hidden_state[0].mean(0), dim=-1).tolist(),
        })
        print(path.name, "x", repeat, "frames:", feats_in.shape[0], "soft tokens:", soft.shape[0])

    captions = [transcripts[k] for k in ("speech_a", "speech_b", "speech_c")]
    text_embs = []
    for c in captions:
        o, _ = run(**tok(c, return_tensors="pt"))
        text_embs.append(F.normalize(o.last_hidden_state[0].mean(0), dim=-1))
    sims = [[float(torch.tensor(case["embedding"]) @ t) for t in text_embs] for case in cases]
    for case, row in zip(cases, sims):
        print(f"{case['file']:<28} x{case['repeat']}", " ".join(f"{s:.3f}" for s in row))

(out / "audio_reference.json").write_text(json.dumps({"cases": cases, "captions": captions, "similarity": sims}))
