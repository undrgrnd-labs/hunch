"""Extract the tensors EmbeddingGemma 2's image path needs into weights.bin + manifest.json.

Matrices stay bf16 (raw u16, exact); the shaders widen them to f32. Norm weights, layer scalars and the four
special-token embedding rows are stored as f32. Matrices that run on the same input are stacked along their
output dimension (q|k|v, gate|up) so one matmul computes them.

usage: python convert.py [out_dir]     # weights/ next to this script by default
"""

import glob
import json
import os
import struct
import sys

import numpy as np



def checkpoint():
    """model.safetensors where eg2cpu looks: $EG2_MODEL_DIR, else the Hugging Face cache ($HF_HUB_CACHE, or
    $HF_HOME/hub, default ~/.cache/huggingface/hub)."""
    if os.environ.get("EG2_MODEL_DIR"):
        found = glob.glob(os.path.join(os.environ["EG2_MODEL_DIR"], "model.safetensors"))
    else:
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
        found = glob.glob(os.path.join(hub, "models--google--embeddinggemma-2/snapshots/*/model.safetensors"))
    if not found:
        sys.exit("no EmbeddingGemma 2 checkpoint found: download it with `uv run download.py embeddinggemma-2` "
                 "(from examples/), or set EG2_MODEL_DIR to a folder holding model.safetensors and config.json")
    return found[0]


SRC = checkpoint()
CFG = json.load(open(os.path.join(os.path.dirname(SRC), "config.json")))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
os.makedirs(OUT, exist_ok=True)
POS_ROWS = 840  # largest patch-grid side at a 280 soft-token budget (2520 patches)

f = open(SRC, "rb")
hlen = struct.unpack("<Q", f.read(8))[0]
header = json.loads(f.read(hlen))
base = 8 + hlen
mm = np.memmap(SRC, dtype=np.uint8, mode="r")


def raw(name):
    e = header[name]
    assert e["dtype"] == "BF16", name
    a, b = e["data_offsets"]
    return np.frombuffer(mm[base + a: base + b], dtype="<u2").reshape(e["shape"])


def f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


tensors = {}
out = open(os.path.join(OUT, "weights.bin"), "wb")


def put(name, arr, dtype):
    pos = out.tell()
    pad = (-pos) % 256
    out.write(b"\0" * pad)
    pos += pad
    data = np.ascontiguousarray(arr)
    out.write(data.tobytes())
    tensors[name] = {"offset": pos, "nbytes": data.nbytes, "dtype": dtype, "shape": list(arr.shape)}


def bf(name, *srcs, rows=None):
    a = np.concatenate([raw(s) for s in srcs], axis=0) if len(srcs) > 1 else raw(srcs[0])
    if rows is not None:
        a = a[:, :rows]
    put(name, a, "bf16")


def fl(name, src):
    put(name, f32(raw(src)), "f32")


vc, tc = CFG["vision_config"], CFG["text_config"]
bf("v.input_proj", "vision_tower.patch_embedder.input_proj.weight")
bf("v.pos", "vision_tower.patch_embedder.position_embedding_table", rows=POS_ROWS)
for i in range(vc["num_hidden_layers"]):
    p = f"vision_tower.encoder.layers.{i}."
    bf(f"v{i}.wqkv", *[p + f"self_attn.{x}_proj.linear.weight" for x in "qkv"])
    bf(f"v{i}.wo", p + "self_attn.o_proj.linear.weight")
    bf(f"v{i}.wgu", p + "mlp.gate_proj.linear.weight", p + "mlp.up_proj.linear.weight")
    bf(f"v{i}.wd", p + "mlp.down_proj.linear.weight")
    for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
              "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm"]:
        fl(f"v{i}.{n}", p + n + ".weight")
bf("v.proj", "embed_vision.embedding_projection.weight")

for i in range(tc["num_hidden_layers"]):
    p = f"language_model.layers.{i}."
    bf(f"t{i}.wqkv", *[p + f"self_attn.{x}_proj.weight" for x in "qkv"])
    bf(f"t{i}.wo", p + "self_attn.o_proj.weight")
    bf(f"t{i}.wgu", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight")
    bf(f"t{i}.wd", p + "mlp.down_proj.weight")
    bf(f"t{i}.ple_gate", p + "ple_block.per_layer_input_gate.weight")
    bf(f"t{i}.ple_proj", p + "ple_block.per_layer_projection.weight")
    for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
              "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm",
              "ple_block.post_per_layer_input_norm"]:
        fl(f"t{i}.{n}", p + n + ".weight")
    fl(f"t{i}.layer_scalar", p + "layer_scalar")
bf("t.ple_model_proj", "language_model.ple.per_layer_model_projection.weight")
fl("t.ple_proj_norm", "language_model.ple.per_layer_projection_norm.weight")
fl("t.norm", "language_model.norm.weight")
bf("t.out_proj", "language_model.embedding_projection.weight")

# Special-token embedding rows (unscaled): <bos>, <boi>, <eoi>, <eos>.
ids = [tc["bos_token_id"], CFG["boi_token_id"], CFG["eoi_token_id"], tc["eos_token_id"]]
emb = raw("language_model.embed_tokens.weight")
put("t.special", f32(emb[ids]), "f32")
out.close()

json.dump({"config": CFG, "special_ids": ids, "pos_rows": POS_ROWS, "tensors": tensors},
          open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(f"{len(tensors)} tensors, {os.path.getsize(os.path.join(OUT, 'weights.bin')) / 1e6:.1f} MB")
