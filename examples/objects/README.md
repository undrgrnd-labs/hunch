# objects: a webcam that names things and learns new ones

A `hunch.live` demo. Hold something up and it says what it is out loud: about forty everyday things are known from
text prompts alone. When it's unsure it asks, and you tell it ("that's a seashell"); it learns the new thing from the
next two seconds of frames. With a teacher (Clef), the teacher labels the frames it's sure about too, and checks the
student's answers until each thing graduates.

![the objects demo naming an orange held up to a webcam, with Clef teaching](../../docs/demo/objects.png)

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
