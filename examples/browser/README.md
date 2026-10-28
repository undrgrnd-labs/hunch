# browser: EG2 on WebGPU, in a tab

EmbeddingGemma 2's image path, hand-written in WGSL, runs a webcam object namer entirely in the browser: the camera,
the embedding, the student and the voice all stay in the tab, and nothing is installed.

From the repository's root, after the [setup](../README.md#setup) (which downloads the checkpoint):

```bash
uv run examples/browser/convert.py          # the image path's weights -> examples/browser/weights/ (580 MB, once)
python3 -m http.server 8803 --bind 127.0.0.1
# open http://localhost:8803/examples/browser/ in Chrome, press Start, allow the camera
```

**Needs:** the EmbeddingGemma 2 checkpoint (`convert.py` finds it where eg2cpu does: `EG2_MODEL_DIR`, else the Hugging
Face cache), a browser with WebGPU (tested in Chrome 156 on Apple silicon), and a webcam; browsers allow the camera on
`http://localhost` but not on other plain-HTTP addresses. **Ships:** `things.json`, the prompt vectors below, and
hunch.live's bundle (`hunch/web/dist/live.js`), so nothing needs building; the weights don't ship. The page fetches
the 580 MB of weights from the server each time it loads.

## What it does

- **The engine** ([`hunch/web/eg2/`](../../hunch/web/eg2/), `model.js` and `kernels.js`): EG2's image path as about 680 WebGPU compute dispatches, recorded once
  per image size and replayed in one command buffer per frame. The 16-layer vision tower (2-D RoPE, flash
  attention), 3x3 pooling, the 24-layer text tower over the soft tokens, mean pooling, `out_proj` and the L2 norm all
  run on the GPU; decoding and the antialiased bicubic resize are ports of `eg2cpu/src/vision.rs` in JavaScript.
- **The weights** (`convert.py`): only what the image path needs, from the checkpoint's safetensors: the vision tower,
  the text layers and their per-layer embeddings, `out_proj` and four special-token rows. Matrices stay bf16 (so
  they are exact) and are widened to f32 in the shaders; q/k/v and gate/up are stacked so one matmul does each. The
  audio tower and the 262k-row vocabulary table are left out: 580 MB of the checkpoint's 1.4 GB.
- **The things it knows** (`things.py`): the browser runs no text path, so the ~45 everyday things of the
  [objects](../objects/) demo arrive as one prompt vector each, embedded once with eg2cpu. The demo is that one's
  zero-shot prompts, empty-hands baseline and student, without the server: a `Choice` from hunch.live's browser
  version ([`hunch/web/live.ts`](../../hunch/web/live.ts)), whose student trains in a web worker, and the browser's
  own voices speak.
  There is no teacher here.
- **Teaching:** hold a thing up, type its name (an emoji in the name becomes its emoji: "seashell 🐚") and press
  enter; the next two seconds of frames are labelled with it.

## What it found

On an Apple M5 Pro in Chrome 156, at 70 soft tokens (the webcam demos' size):

| | ms per frame (median) | cosine with eg2cpu |
|---|---|---|
| WebGPU, stock Chrome | 55–57 | 1.000000 on every test image |
| WebGPU, `--enable-unsafe-webgpu` (subgroup-matrix matmuls) | 45–48 | 1.000000 |
| eg2cpu on the same machine's CPU (SME) | about 110 | – |

- **Parity:** the 5 eg2cpu image fixtures and a 640×480 webcam frame, fed to both engines as the same resized
  pixels; at 280 soft tokens the cosine is 0.999999–1.000000 (about 280 ms a frame). The JavaScript resize gives
  byte-identical pixels to eg2cpu's on the fixtures.
- **Live:** the page runs at about 17 fps, twice the CPU demo's rate. Loading the weights from localhost and setting
  up the GPU takes about 0.3 s; a first frame at a new size takes up to 0.2 s more, for shader compiles.
- **Where the time goes** (subgroup-matrix path): vision matmuls about 30 ms, vision attention 5 ms, the text tower
  about 8 ms.
- **An ONNX export** of the same image path on onnxruntime-web's WebGPU backend (1.30.0) reaches about the same speed
  in fp16 (56–58 ms, cosine 0.9995–0.9999), and 8-bit weights halve its download to about 300 MB at 68 ms and cosine
  0.9995 or better. Round-to-nearest 4-bit weights fall to a cosine of about 0.98.
