# gestures: hand signs, faces and one-shot detectors on a webcam

A `hunch.live` demo. EG2 embeds each webcam frame once, and five questions answer from that one embedding: what the
hand is doing, what the face is showing, and three yes/no detectors (fire? a cat? your thing?). Text prompts answer
from the start; holding a key while you make a sign labels the frames, and the question learns from them as it goes.

![the gestures demo reading a thumbs up and a smile from a webcam](../../docs/demo/gestures.png)

From `examples/`, after the [setup](../README.md#setup):

```bash
uv run gestures/gestures.py                 # then open http://localhost:8765 in Chrome and allow the camera
uv run gestures/gestures.py --tokens 140    # sharper (smaller hands, farther away) at about half the frame rate
```

Make a sign, then hold its key (shown next to each answer) to teach it; a detector takes its key for yes and
<kbd>shift</kbd>+key for no, and a second of each, in the same scene, is enough.

**Needs:** EmbeddingGemma 2 (see [Setup](../README.md#setup)), a webcam, and a browser on the same machine: browsers
allow the camera on `http://localhost` but not on other plain-HTTP addresses. **Time:** the server is ready in about
30 s; a frame takes about 110 ms on an Apple M5 Pro's CPU, for all five questions. **Options:** `--port` (default 8765)
and `--tokens`.

## Without a camera

To check the server works, send it any image; it answers every question:

```bash
curl -s --data-binary @../docs/demo/gestures.png localhost:8765/frame
```

## What it keeps

Labelled frames go in `gestures/frames/` (git-ignored): each JPEG, its embedding and `labels.jsonl`. They're taught
again when the server starts. "Forget what I taught" sets the labels aside as `labels.<time>.jsonl`. The frames are
pictures of you and your room, so they stay on your machine.

`gestures.py`'s docstring has how each question answers.
