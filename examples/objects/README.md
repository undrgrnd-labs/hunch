# objects: a webcam that names things and learns new ones

A `hunch.live` demo. Hold something up and it says what it is out loud: about forty everyday things are known from
text prompts alone. When it's unsure it asks, and you tell it ("that's a seashell"); it learns the new thing from the
next two seconds of frames. With a teacher (Clef), the teacher labels the frames it's sure about too, and checks the
student's answers until each thing graduates.

![the objects demo naming an orange held up to a webcam, with Clef teaching](../../docs/demo/objects.png)

## How it works

It is the proxy's loop on a camera: a student answers every frame, a teacher checks it, and an answer is trusted
once it graduates. In `hunch.live`'s terms ([hunch's README](../../hunch/README.md#live-images-hunchlive)):
- **Every frame** is embedded once by EG2's image path (about 110 ms on an Apple M5 Pro's CPU, at 70 *soft tokens*:
  the number of image tokens the frame becomes; more see smaller things, more slowly).
- **The answers** are a `Choice`: each thing has two text prompts ("a person holding up a banana to a webcam"), so it
  can be named before anyone teaches it, by how close the frame's embedding is to the prompts'. "Nothing" and "hand"
  are answers too, but quiet: never said out loud.
- **Empty hands first:** the first two seconds are labelled "nothing". Those frames are the baseline: how far a frame
  is from them says whether something is being shown, and what differs from them says how much it looks like a
  taught thing.
- **Teaching:** naming a thing labels the next two seconds of frames (about 15) with it; a new name becomes a new
  answer with its own prompts and emoji. A logistic regression on the labelled frames (the student) refits in the
  background, and answers as far as a frame resembles the frames it was taught.
- **The teacher** is a `Watcher`: Clef looks at a frame when that's worth it and labels the frames it's sure about.
- **Graduation:** each of Clef's looks checks the student's answer, if that answer was steady (the same for three
  frames in a row). A thing graduates once the 95% lower bound of its agreement with Clef reaches 0.90 over at least
  20 checks, which takes about 35 checks with no disagreement; it is demoted while the upper bound is below 0.90.
  Until then it says "I think that's a lemon"; after, "That's a lemon!". The bar under its name is the share of the
  checks it needs that it has.

From `examples/`, after the [setup](../README.md#setup):

```bash
uv run --with faster-whisper --with kokoro-onnx objects/objects.py      # then open http://localhost:8766 in Chrome
```

Press Start and allow the camera and microphone. Hold <kbd>space</kbd> to talk ("that's a lemon"), or press
<kbd>Enter</kbd> to type a name; hold <kbd>0</kbd> to show it your empty hands again.

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), a webcam and a microphone, and a browser on the same
machine: browsers allow the camera on `http://localhost` but not on other plain-HTTP addresses. **Time:** the server
is ready in about 30 s; a frame takes about 110 ms on an Apple M5 Pro's CPU.

**Downloads on first use:** Kokoro's voice (90 MB, to `~/.cache/hunch/kokoro`) when the server starts, and Whisper's
`small.en` the first time you speak. Without `kokoro-onnx`, with `--voice browser`, or when Kokoro can't speak on the
machine, the browser's own voices speak instead.

**Options:** `--port` (default 8766), `--tokens 140` (smaller things, about half the frame rate), `--voice` (a Kokoro
voice, or `browser`) and `--teacher` (Clef, below).

## Without a camera

To check the server works, send it any image; it answers with the thing it sees:

```bash
curl -s --data-binary @../docs/demo/objects.png localhost:8766/frame
```

The README's screenshot holds an orange, so the answer starts `{"ms": 124.1, "tokens": 70, "state": "sure",
"name": "orange", "emoji": "🍊", "p": 0.991, ...}`, followed by the top five answers and what it would say.

## A teacher

```bash
OPENROUTER_API_KEY=... uv run --with faster-whisper --with kokoro-onnx objects/objects.py --teacher openrouter:cloudflare/clef-flash
uv run --with faster-whisper --with kokoro-onnx --with "mlx-vlm>=0.7.4,<0.8" --with pillow \
    objects/objects.py --teacher path/to/clef-flash-4bit          # a local Clef on Apple silicon
```

Clef looks at a frame about once a second while something hasn't graduated, and at random moments (every 10 s on
average) once it has. A thing graduates after about 35 steady answers that Clef agrees with; the bar under its name
shows how far it is. Through OpenRouter, a look costs about $0.00001.

## What it keeps

Labelled frames go in `objects/frames/` (git-ignored): each JPEG, its embedding and `labels.jsonl`. They're taught
again when the server starts. "Forget what I taught" sets the labels aside as `labels.<time>.jsonl`. The frames are
pictures of you and your room, so they stay on your machine.

`objects.py`'s docstring has how it answers, step by step.
