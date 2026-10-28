"""A webcam that names what you hold up, out loud, and learns what it doesn't know: a `hunch.live.Choice` over about
forty everyday things, known from text prompts alone. When it's unsure it asks "what's that?", listens, and learns
the thing from the next two seconds of frames. With --teacher, Clef watches too (a `hunch.live.Watcher`): a vision
model, local on the GPU or from OpenRouter, that answers in under a second, labels the frames it's sure about for the
student, and says when it sees something that isn't any of the known things.

    uv run --with faster-whisper --with kokoro-onnx objects/objects.py   # from examples/; then open http://localhost:8766
    uv run --with faster-whisper --with kokoro-onnx --with "mlx-vlm>=0.7.4,<0.8" --with pillow \\
        objects/objects.py --teacher path/to/clef-flash-4bit        # Clef as the teacher (Apple silicon)
    OPENROUTER_API_KEY=... uv run --with faster-whisper --with kokoro-onnx \\
        objects/objects.py --teacher openrouter:cloudflare/clef-flash   # Clef from OpenRouter
    ... --tokens 140                                              # smaller things, about half the frame rate

How it answers (hunch/src/hunch/live.py has the details):
  empty hands  the first two seconds after start (or holding 0) are labelled "nothing": your face and room, empty
               handed. They're the Choice's baseline: a frame's distance from them says whether anything is being
               shown, and what differs from them says how much a frame resembles a taught thing.
  zero-shot    every thing has two text prompts ("a person holding up a banana to a webcam"). "Nothing" and "hand"
               (a bare hand or a gesture) have prompts too, and are quiet: never named out loud.
  taught       saying a thing's name labels the next two seconds of frames with it (a new thing gets prompts and an
               emoji from its name), and the teacher labels the frames it's sure about.
  the teacher  Clef picks one of the things or "something else". Sure of something else, it makes the page ask. It
               also says how each thing is named out loud: "that's an egg", "those are scissors", "that's rice".
  graduation   each look of Clef's checks the student's answer for that frame, if it was steady (the same sure answer
               three frames in a row, as the page waits for before it speaks). A thing graduates once its checks
               agree often enough (hunch.live's rules, as the proxy's), and only then is it named for sure; before,
               it's "I think that's a lemon". Clef audits graduated things at random moments, about every 10 s.

Speech: Kokoro speaks on the CPU (82M parameters, about 0.2 s a line; its 90 MB of files are downloaded once to
~/.cache/hunch/kokoro). --voice picks a voice; "--voice browser", no kokoro-onnx, or a Kokoro that fails its first line
at start uses the browser's own voices. What you say is sent here as 16 kHz samples and
transcribed by faster-whisper on the CPU (small.en, downloaded once, told the names of the things it knows). Hold
space to talk at any time ("that's a lemon"); Enter types a name instead.

Labelled frames are saved in frames/ (JPEG, embedding and labels.jsonl, with who labelled each) and taught again at
start; "forget" sets the labels aside as labels.<time>.jsonl and goes back to the things known from prompts alone.
"""

import argparse
import base64
import io
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from hunch.live import ELSE, Album, Choice, Eyes, Watcher, grey_png

NOTHING, HAND = "nothing", "hand"  # quiet: never named out loud, never asked about
KNOWN = {  # thing: emoji, known from text prompts alone
    "banana": "🍌", "apple": "🍎", "orange": "🍊", "lemon": "🍋", "avocado": "🥑", "strawberry": "🍓", "grapes": "🍇",
    "carrot": "🥕", "broccoli": "🥦", "egg": "🥚", "donut": "🍩", "cookie": "🍪", "coffee mug": "☕",
    "wine glass": "🍷", "water bottle": "💧", "soda can": "🥫", "spoon": "🥄", "fork": "🍴", "book": "📕",
    "phone": "📱", "headphones": "🎧", "sunglasses": "🕶️", "watch": "⌚", "keys": "🔑", "scissors": "✂️", "pen": "🖊️",
    "pencil": "✏️", "cat": "🐱", "dog": "🐶", "teddy bear": "🧸", "tennis ball": "🎾", "game controller": "🎮",
    "remote control": "📺", "light bulb": "💡", "potted plant": "🪴", "flower": "🌸", "toothbrush": "🪥",
    "baseball cap": "🧢", "shoe": "👟", "sock": "🧦", "camera": "📷", "rubber duck": "🦆", "toy car": "🚗",
    "candle": "🕯️", "laptop": "💻",
}
NOTHING_PROMPTS = ["a person at a desk looking at a webcam with empty hands", "a webcam photo of a face, nothing held up",
                   "a person sitting in a room, holding nothing"]
HAND_PROMPTS = ["an empty hand held up to a webcam", "a person making a hand gesture",
                "a person waving their hand at the camera"]
TEACHER_SAYS = {NOTHING: "Nothing is held up; just a person or a room",
                HAND: "A bare hand or a hand gesture, with nothing in it",
                ELSE: "Something is held up, but none of the other options"}
TOP = 5  # things shown on the page, most likely first

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8766)
ap.add_argument("--tokens", type=int, default=70, help="soft tokens per frame (70, 140, 280)")
ap.add_argument("--voice", default="af_heart", help="a Kokoro voice (af_heart, af_bella, am_michael, bf_emma...) or browser")
ap.add_argument("--teacher", help="Clef, to teach the student as it watches: a local MLX model directory "
                                   "(clef-flash-4bit), or openrouter:cloudflare/clef-flash (OPENROUTER_API_KEY)")
args = ap.parse_args()
try:  # the port first, so a taken one fails before the model loads
    server = ThreadingHTTPServer(("127.0.0.1", args.port), None)
except OSError as e:
    sys.exit(f"can't listen on port {args.port} ({e.strerror}): pass --port to use another")

eyes = Eyes(tokens=args.tokens)


def a_or_an(name):
    return ("an " if name[0] in "aeiou" else "a ") + name


PHRASES = {"a": ("That's", "a {}"), "an": ("That's", "an {}"), "plural": ("Those are", "{}"), "mass": ("That's", "{}")}


def naming(name):
    """How to say a thing out loud: Clef's phrasing once it has one, "a" or "an" before that."""
    kind = clef.phrases.get(name) if clef is not None else None
    verb, noun = PHRASES[kind] if kind else ("That's", a_or_an(name))
    return {"sentence": f"{verb} {noun.format(name)}!", "guess": f"I think {verb.lower()} {noun.format(name)}.",
            "noun": noun.format(name)}


def prompts_for(name):
    return [f"a person holding up {a_or_an(name)} to a webcam", f"a close-up photo of {a_or_an(name)}"]


class Emojis:
    """Every single-character emoji, by its Unicode name, so a spoken name finds the nearest one."""

    def __init__(self, cache):
        chars = [chr(c) for r in ((0x1F300, 0x1F650), (0x1F680, 0x1F700), (0x1F900, 0x1FB00), (0x2600, 0x2800))
                 for c in range(*r)]
        self.chars = [c for c in chars if unicodedata.name(c, "") and unicodedata.category(c) == "So"]
        self.names = [unicodedata.name(c).lower() for c in self.chars]
        if cache.exists() and len(np.load(cache)) == len(self.chars):
            self.vectors = np.load(cache)
        else:
            self.vectors = eyes.read([f"an emoji of {n}" for n in self.names])
            cache.parent.mkdir(exist_ok=True)
            np.save(cache, self.vectors)
        self.given = {}  # thing: its emoji

    def find(self, name):
        if name not in self.given:
            for words in (name, name.rstrip("s")):
                if words in self.names:
                    self.given[name] = self.chars[self.names.index(words)]
                    break
            else:
                self.given[name] = self.chars[int(np.argmax(self.vectors @ eyes.read([f"an emoji of {name}"])[0]))]
        return self.given[name]


def new_thing(name):
    emojis.find(name)
    return prompts_for(name)


class LocalClef:
    """Clef from a local MLX model directory, on the GPU."""

    def __init__(self, path):
        sys.path.insert(0, str(Path(path).resolve()))
        import clef_mlx
        from PIL import Image
        self.Image = Image
        self.model = clef_mlx.load(path)
        self.spent = 0.0

    def predict(self, text, questions, image=None):
        record = {"state": {"task": text}, "questions": questions}
        if image is not None:
            record["images"] = [self.Image.open(io.BytesIO(image)).convert("RGB")]
        return self.model.predict(record)


class OpenRouterClef:
    """Clef from OpenRouter's decisions endpoint (OPENROUTER_API_KEY), adding up what it costs."""

    URL = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, model):
        self.model, self.key = model, os.environ["OPENROUTER_API_KEY"]
        self.spent = 0.0

    def predict(self, text, questions, image=None):
        state = [{"type": "text", "text": text}]
        if image is not None:
            kind = "png" if image.startswith(b"\x89PNG") else "jpeg"
            url = f"data:image/{kind};base64," + base64.b64encode(image).decode()
            state.append({"type": "image_url", "image_url": {"url": url}})
        request = urllib.request.Request(self.URL, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                                         data=json.dumps({"model": self.model, "state": state, "questions": questions}).encode())
        out = json.load(urllib.request.urlopen(request, timeout=30))
        self.spent += out.get("usage", {}).get("cost", 0.0)
        return {q: a["probabilities"] for q, a in out["answers"].items()}


class Clef:
    """Clef, asked one question at a time: what a frame shows (the Watcher's teacher), and how to name a thing."""

    def __init__(self, where):
        self.clef = OpenRouterClef(where.removeprefix("openrouter:")) if where.startswith("openrouter:") else LocalClef(where)
        self.busy = threading.Lock()
        self.phrases = json.loads(PHRASED.read_text()) if PHRASED.exists() else {}
        self(grey_png(), things.names)  # loads the vision tower before the first frame
        threading.Thread(target=lambda: [self.phrase(n) for n in things.names[2:]], daemon=True).start()

    def predict(self, text, questions, image=None):
        with self.busy:
            return self.clef.predict(text, questions, image)

    def __call__(self, image, names):
        criteria = {n: TEACHER_SAYS.get(n, naming(n)["noun"].capitalize()) for n in names} | {ELSE: TEACHER_SAYS[ELSE]}
        question = {"type": "choice", "instructions": "What is the person holding up to the camera?", "criteria": criteria}
        return self.predict("A webcam frame.", {"thing": question}, image)["thing"]

    def phrase(self, name):
        if name not in self.phrases:
            question = {"type": "choice", "criteria": {k: f"{v} {n.format(name)}." for k, (v, n) in PHRASES.items()},
                        "instructions": f"Someone points at a {name!r} (or some) and names it out loud. "
                                        "Which sentence is correct English?"}
            out = self.predict(json.dumps({"thing": name}), {"phrase": question})["phrase"]
            self.phrases[name] = max(out, key=out.get)
            FRAMES.mkdir(exist_ok=True)
            PHRASED.write_text(json.dumps(self.phrases, indent=0))
        return self.phrases[name]


FILLER = re.compile(r"^(?:(?:no|nope|wrong|actually|um+|uh+|so|well|okay|ok|oh|hmm+)\b[,.!]?\s*)*"
                    r"(?:(?:it'?s|it is|that'?s|that is|this is|this'?s|these are|those are|i'?ve got|i have)\s+)?"
                    r"(?:(?:a|an|the|my|some|one)\s+)?")


def name_from(text):
    words = re.sub(r"[^a-z' ]", " ", text.lower()).strip()
    words = FILLER.sub("", words).strip()
    words = " ".join(words.split()[:3])
    return words if len(words) > 1 else ""


class Ears:
    """faster-whisper on the CPU (small.en), loaded on first use, told the names of the things it may hear."""

    def __init__(self):
        self.model = None

    def hear(self, samples):
        if self.model is None:
            from faster_whisper import WhisperModel
            self.model = WhisperModel("small.en", device="cpu", compute_type="int8")
        hint = "Naming things held up to a camera: " + ", ".join(things.names[2:]) + "."
        segments, _ = self.model.transcribe(samples, language="en", beam_size=1, vad_filter=True, initial_prompt=hint)
        return " ".join(s.text for s in segments).strip()


class Voice:
    """Kokoro on the CPU, its model downloaded on first start; each line is made once and kept. Its first line is made
    in a separate process: espeak-ng, under Kokoro, ends the process when it can't find its data (as when its install
    path is long), and a server should fall back to the browser's voices rather than go down with it."""

    FILES = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    TRY = ("import sys; from kokoro_onnx import Kokoro; "
           "Kokoro(sys.argv[1], sys.argv[2]).create('Hello.', voice=sys.argv[3], lang='en-us')")

    def __init__(self, name, home):
        from kokoro_onnx import Kokoro
        home.mkdir(parents=True, exist_ok=True)
        for f in ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin"):
            if not (home / f).exists():
                print(f"downloading {f}", flush=True)
                urllib.request.urlretrieve(self.FILES + f, home / f".{f}")
                (home / f".{f}").rename(home / f)
        model, voices = str(home / "kokoro-v1.0.int8.onnx"), str(home / "voices-v1.0.bin")
        tried = subprocess.run([sys.executable, "-c", self.TRY, model, voices, name], capture_output=True, text=True)
        if tried.returncode:
            raise RuntimeError((tried.stderr.strip().splitlines() or [f"exit {tried.returncode}"])[-1])
        self.kokoro = Kokoro(model, voices)
        self.name = name
        self.lines = {}
        self.busy = threading.Lock()

    def say(self, text):
        if text not in self.lines:
            with self.busy:
                samples, rate = self.kokoro.create(text, voice=self.name, speed=1.05, lang="en-us")
            pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
            self.lines[text] = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
                                + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
                                + b"data" + struct.pack("<I", len(pcm)) + pcm)
        return self.lines[text]


def voice_for(name):
    if name == "browser":
        return None
    try:
        return Voice(name, Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "hunch" / "kokoro")
    except ImportError:
        print("kokoro-onnx isn't installed: the browser's voices will speak", flush=True)
    except RuntimeError as e:
        print(f"Kokoro can't speak here ({e}): the browser's voices will speak", flush=True)
    return None


def emoji_of(name):
    return EMOJI.get(name) or emojis.find(name)


def for_page(a):
    """The Choice's answer for a frame, as the page shows it."""
    top = sorted(a["probs"], key=a["probs"].get, reverse=True)[:TOP]
    counts = {n: {"you": c.get("you", 0), "clef": c.get("clef", 0)} for n, c in things.counts().items()}
    return {"state": a["state"], "name": a["name"], "emoji": emoji_of(a["name"]), "p": round(a["p"], 3),
            "top": [{"name": n, "emoji": emoji_of(n), "p": round(a["probs"][n], 3), "zero": round(a["zero"][n], 3)}
                    for n in top],
            "shown": None if a["shown"] is None else round(a["shown"], 2), "resemble": round(a["resemble"], 3),
            "labelled": counts, "ready": a["ready"],
            "graduation": things.progress(a["name"]) if watcher is not None and a["state"] == "sure" else None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.reply((Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/frame":  # JPEG bytes in, the answer out; X-Label: <thing> labels the frame
            x, ms = eyes.see(body)
            name = (self.headers.get("X-Label") or "").strip().lower()
            if name:
                album.keep(body, x, thing=things.teach(x, name), by="you")
            a = things.ask(x)
            out = {"ms": round(ms, 1), "tokens": args.tokens} | for_page(a)
            # with Clef watching, a thing is named for sure once it has graduated, and guessed at before
            out["say"] = naming(out["name"])["sentence" if watcher is None or a["graduated"] else "guess"]
            if watcher is not None:
                watcher.look(body, x, a)
                said = watcher.describe()
                if said is not None:
                    progress = [{"name": n, "emoji": emoji_of(n), "checks": r["checks"], "agree": round(r["agree"], 3),
                                 "low": round(r["low"], 3), "graduated": r["graduated"]}
                                for n, r in things.report().items() if n not in things.quiet]
                    out["teacher"] = said | {"emoji": emoji_of(said["name"]) if said["known"] else "❓",
                                             "spent": round(clef.clef.spent, 6), "target": things.target,
                                             "progress": sorted(progress, key=lambda r: -r["checks"])[:6]}
                if watcher.saw_else() and out["state"] != "quiet":
                    out["state"] = "unsure"  # the teacher sees a thing that isn't any of the known ones
            self.reply(out)
        elif self.path == "/hear":  # speech in (16 kHz mono float32 samples), the words and the thing they name out
            t0 = time.perf_counter()
            text = ears.hear(np.frombuffer(body, dtype=np.float32))
            name = name_from(text)
            found = things.find(name) if name else None
            name = found or name
            if name and clef is not None:
                clef.phrase(name)
            self.reply({"text": text, "name": name, "emoji": emoji_of(name) if name else "", "known": found is not None,
                        "noun": naming(name)["noun"] if name else "", "ms": round((time.perf_counter() - t0) * 1000)})
        elif self.path == "/say":  # a line of text in, the voice saying it out (WAV), or 404 for the browser's voices
            if voice is None:
                self.send_error(404)
            else:
                self.reply(voice.say(body.decode()), "audio/wav")
        elif self.path == "/forget":
            things.forget()
            if watcher is not None:
                watcher.forget()
            album.set_aside()
            self.reply({})
        else:
            self.send_error(404)


FRAMES = Path(__file__).parent / "frames"
PHRASED = FRAMES / "phrases.json"  # how Clef said to name each thing, kept across restarts

EMOJI = {NOTHING: "🫥", HAND: "✋"} | KNOWN
emojis = Emojis(FRAMES / "emoji-names.npy")
things = Choice(eyes, {NOTHING: NOTHING_PROMPTS, HAND: HAND_PROMPTS} | {n: prompts_for(n) for n in KNOWN},
                quiet={NOTHING, HAND}, empty=NOTHING, new=new_thing)
album = Album(FRAMES, eyes)
ears = Ears()
voice = voice_for(args.voice)
kept = album.frames()
for x, row in kept:  # teach again what was taught before the restart (not checks: the student has seen them)
    things.teach(x, row["thing"], row.get("by", "you"), check=False)
if kept:
    print(f"taught again from {len(kept)} saved frames", flush=True)
eyes.warm()
clef = None  # naming() reads it while Clef loads
clef = Clef(args.teacher) if args.teacher else None
watcher = Watcher(things, clef, by="clef", on_label=lambda image, x, name: album.keep(image, x, thing=name, by="clef")) \
    if clef is not None else None
print(f"objects: http://localhost:{args.port}  ({args.tokens} soft tokens per frame, {len(things.names)} things"
      + (", Clef teaching)" if clef else ")"), flush=True)
server.RequestHandlerClass = Handler
server.serve_forever()
