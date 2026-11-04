# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "eg2cpu",
#   "torch",
#   "transformers @ git+https://github.com/huggingface/transformers",
#   "sentencepiece",
#   "safetensors",
#   "accelerate",
#   "numpy",
# ]
#
# [tool.uv.sources]
# eg2cpu = { path = ".." }
# ///
"""Check the Python binding: against tests/fixtures/reference.json, image_reference.json and audio_reference.json,
then against PyTorch fp32 on CPU (the reference implementation, with hooks for the per-layer means) on 20 generated
texts. Cosine must exceed 0.99999.

    uv run scripts/check_python.py
"""
import json
import random
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import eg2cpu

HERE = Path(__file__).resolve().parent.parent


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def compare(label, got, want):
    """got/want: list of 24 layer means + embedding. Returns the worst cosine."""
    c = [cos(g, w) for g, w in zip(got, want, strict=True)]
    rel = max(float(np.abs(g - w).max() / np.abs(w).max()) for g, w in zip(got, want))
    print(f"  {label:<44} emb cos {c[-1]:.7f}  worst layer cos {min(c[:-1]):.7f}  max rel err {rel:.1e}")
    return min(c)


def texts(rng, n):
    """Tool calls and outputs of varied shapes and lengths."""
    cmds = ["git status", "cargo build", "pytest -q", "ls src/", "npm install", "make test", "cat README.md"]
    outs = ["On branch main\nnothing to commit, working tree clean", "error[E0308]: mismatched types\n --> src/main.rs:12:5",
            "13 passed in 0.7s", "app.yaml  db.yaml  main.rs", "npm ERR! code ERESOLVE", "make: *** [test] Error 2",
            "# Project\nA small app that embeds text on the CPU."]
    return [f"bash: {rng.choice(cmds)}\n" + "\n".join(rng.choice(outs) for _ in range(rng.randint(1, 40))) for _ in range(n)]


print(f"isa: {eg2cpu.isa()}")
emb = eg2cpu.Embedder()
assert emb.model.num_layers == 24

print("vs tests/fixtures/reference.json (torch fp32 CPU)")
worst = 1.0
for case in json.loads((HERE / "tests/fixtures/reference.json").read_text()):
    assert emb.model.tokenize(case["text"]) == case["ids"]
    e, m = emb.model.embed(case["text"])
    assert e.shape == (768,) and m.shape == (24, 512) and e.dtype == m.dtype == np.float32
    want = [np.array(x, dtype=np.float32) for x in case["layer_means"]] + [np.array(case["embedding"], dtype=np.float32)]
    worst = min(worst, compare(f"{len(case['ids'])} tokens", [*m, e], want))

print("vs tests/fixtures/image_reference.json (torch fp32 CPU)")
for case in json.loads((HERE / "tests/fixtures/image_reference.json").read_text())["cases"]:
    path = HERE / "tests/fixtures" / case["file"]
    e, m = emb.model.embed_image(path.read_bytes() if path.suffix == ".png" else str(path), case["max_soft_tokens"])
    assert e.shape == (768,) and m.shape == (24, 512)
    want = [np.array(x, dtype=np.float32) for x in case["layer_means"]] + [np.array(case["embedding"], dtype=np.float32)]
    # JPEG decoders differ slightly from PIL's libjpeg-turbo; PNG goes through bit-exact preprocessing.
    c = compare(f"{case['file']} ({case['max_soft_tokens']} soft tokens)", [*m, e], want)
    if path.suffix == ".png":
        worst = min(worst, c)

print("vs tests/fixtures/audio_reference.json (torch fp32 CPU)")
for case in json.loads((HERE / "tests/fixtures/audio_reference.json").read_text())["cases"]:
    path = HERE / "tests/fixtures" / case["file"]
    if case["repeat"] == 1:
        e, m = emb.model.embed_audio(str(path))
    else:  # a clip repeated past 30 s goes in as int16 samples (the repeated clips are 16-bit)
        with wave.open(str(path)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        e, m = emb.model.embed_audio(np.tile(pcm, case["repeat"]))
    assert e.shape == (768,) and m.shape == (24, 512)
    want = [np.array(x, dtype=np.float32) for x in case["layer_means"]] + [np.array(case["embedding"], dtype=np.float32)]
    worst = min(worst, compare(f"{case['file']} x{case['repeat']}", [*m, e], want))

sample = texts(random.Random(0), 20)
t0 = time.perf_counter()
emb.prefetch(sample)
print(f"\neg2cpu batch of {len(sample)}: {time.perf_counter() - t0:.2f}s")
assert all(emb(t) is emb.cache[t] for t in sample)

tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
model = AutoModel.from_pretrained("google/embeddinggemma-2", dtype=torch.float32).eval()
layers = list(model.language_model.layers)


@torch.no_grad()
def reference(text):
    """The mean-pooled output of every decoder layer plus the final normalized embedding, from PyTorch. The forward
    hooks capture layer outputs the same way `make_reference.py` does."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    caught = []
    hooks = [m.register_forward_hook(lambda mod, i, o: caught.append(o[0] if isinstance(o, tuple) else o)) for m in layers]
    try:
        out = model(**enc)
    finally:
        for h in hooks:
            h.remove()
    means = [c[0].mean(0).numpy() for c in caught[: len(layers)]]
    return means + [F.normalize(out.last_hidden_state[0].mean(0), dim=-1).numpy()]


print(f"vs PyTorch fp32 CPU on {len(sample)} generated texts")
for t in sample:
    n = len(emb.model.tokenize(t))
    worst = min(worst, compare(f"{n:>4} tokens  {t[:28]!r}", emb(t), reference(t)))

print(f"\nworst cosine {worst:.7f}")
assert worst > 0.99999, worst
print("OK")
