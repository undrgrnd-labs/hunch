"""A live webcam recognizer: EG2 embeds each frame once, and every question answers from that one embedding. Two
questions pick one answer (what's the hand doing? what's the face showing?), each a `hunch.live.Choice`, and three
`hunch.live.Detector`s say yes or no (fire? a cat? your thing?). Text prompts answer zero-shot, and frames you label
teach each question as it goes.

    uv run gestures/gestures.py            # then open http://localhost:8765 and allow the camera
    uv run gestures/gestures.py --tokens 140   # sharper (smaller hands, farther away) at about half the frame rate

How a question that picks one answer answers (hunch/src/hunch/live.py has the details):
  zero-shot  every answer is a few text prompts ("a hand giving a thumbs up"); a frame scores each answer by its
             similarity to that answer's prompts, standardised across the question's answers, so the nearest one wins
  taught     holding an answer's key labels frames with it; a logistic regression on the question's labelled frames
             (standardised, so it learns what differs between your frames rather than the room and face they share;
             classes weighted equally, however long each key was held) answers once two of its answers have frames.
             Its weight grows with the frames you label (half and half at 8 frames) until the question's quiet answer
             (no hands, no face) has frames of its own; from then on it answers as far as a frame resembles the
             frames you taught, so a sign you never taught keeps its zero-shot answer.

How a detector answers:
  zero-shot  a frame's similarity to the "yes" prompts minus its similarity to the "no" prompts, against a threshold
             set between webcam frames and frames with a fire or cat pasted in
  taught     one frame with the thing and one without, in the same scene, is enough: the direction from the "no"
             frames to the "yes" frames is added to the prompts' direction, and the threshold moves halfway between
             them. A detector without prompts ("your thing") is that direction alone, so it only answers once taught.

Labelled frames are saved in frames/ (JPEG, embedding and labels.jsonl) and taught again at start; "forget" sets the
labels aside as labels.<time>.jsonl.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hunch.live import Album, Choice, Detector, Eyes

QUESTIONS = {  # question: (title, {answer: (emoji, key, prompts)}, the answer that means nothing to show)
    "gesture": ("What's the hand doing?", {
        "none": ("🙈", "0", ["a photo of a person with no hands visible", "a webcam photo of a room with nobody in it",
                             "a webcam photo of a face, hands out of view"]),
        "thumbs_up": ("👍", "1", ["a hand giving a thumbs up", "a photo of a thumbs up gesture"]),
        "thumbs_down": ("👎", "2", ["a hand giving a thumbs down", "a photo of a thumbs down gesture"]),
        "open_palm": ("✋", "3", ["an open hand with the palm facing the camera", "a hand raised with all five fingers spread"]),
        "fist": ("✊", "4", ["a closed fist", "a hand clenched into a fist"]),
        "peace": ("✌️", "5", ["a hand making a peace sign with two fingers", "a hand showing the V sign"]),
        "point": ("☝️", "6", ["a hand pointing up with the index finger", "one finger raised"]),
        "ok": ("👌", "7", ["a hand making the OK sign", "thumb and index finger touching in a circle"]),
        "rock": ("🤘", "8", ["a hand making the rock and roll horns sign", "index and little finger raised, devil horns"]),
    }, "none"),
    "emotion": ("What's the face showing?", {
        "no_face": ("🫥", "x", ["a webcam photo of an empty room", "a photo with no face in it"]),
        "neutral": ("😐", "n", ["a person with a neutral, expressionless face", "a calm face looking at the camera"]),
        "happy": ("😄", "h", ["a person smiling happily", "a big grin, a happy laughing face"]),
        "sad": ("😢", "s", ["a person with a sad face", "a frowning, unhappy face about to cry"]),
        "surprised": ("😮", "u", ["a surprised face with mouth wide open", "a shocked person with raised eyebrows"]),
        "angry": ("😠", "a", ["an angry face with furrowed brows", "a person scowling in anger"]),
        "silly": ("😜", "t", ["a person sticking their tongue out", "a silly goofy face pulling a funny expression"]),
    }, "no_face"),
}
DETECTORS = {  # detector: (title, emoji, key for yes (shift for no), no prompts, yes prompts, zero-shot threshold)
    "fire": ("Fire?", "🔥", "f", ["a normal indoor webcam photo", "a room with no fire or flames"],
             ["a photo of fire with bright flames", "a burning flame from a lighter or candle", "a fire burning"], -0.04),
    "cat": ("A cat?", "🐱", "c", ["a webcam photo with no animals", "a person in a room, no cat"],
            ["a photo of a cat", "a cat in front of the camera", "a kitten"], -0.03),
    "thing": ("Your thing? (no prompts: teach it once)", "⭐", "m", [], [], None),
}
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--tokens", type=int, default=70, help="soft tokens per frame (70, 140, 280)")
args = ap.parse_args()

eyes = Eyes(tokens=args.tokens)


def rounded(p):
    return {n: round(float(v), 4) for n, v in p.items()}


class Question:
    """Picks one answer: a Choice, and how the page shows it."""

    def __init__(self, title, answers, quiet):
        self.title, self.answers = title, answers
        self.choice = Choice(eyes, {a: prompts for a, (_, _, prompts) in answers.items()}, quiet={quiet}, empty=quiet)
        self.names, self.quiet = list(answers), quiet

    def describe(self):
        return {"title": self.title, "answers": [{"name": n, "emoji": e, "key": k, "quiet": n == self.quiet}
                                                 for n, (e, k, _) in self.answers.items()]}

    def teach(self, x, name):
        self.choice.teach(x, name)

    def forget(self):
        self.choice.forget()

    def answer(self, x):
        a = self.choice.ask(x)
        counts = self.choice.counts()
        return {"p": rounded(a["probs"]), "zero": rounded(a["zero"]),
                "mode": a["mode"], "labelled": {n: sum(counts.get(n, {}).values()) for n in self.answers}}


class Yes:
    """Says yes or no: a Detector, and how the page shows it."""

    def __init__(self, title, emoji, key, no, yes, threshold):
        self.title, self.emoji, self.key = title, emoji, key
        self.detector = Detector(eyes, yes=yes, no=no, threshold=threshold or 0.0)
        self.names = Detector.names

    def describe(self):
        return {"title": self.title, "answers": [
            {"name": "no", "emoji": "✖️", "key": "⇧" + self.key, "press": self.key.upper(), "quiet": True},
            {"name": "yes", "emoji": self.emoji, "key": self.key, "press": self.key, "alert": True}]}

    def teach(self, x, name):
        self.detector.teach(x, name)

    def forget(self):
        self.detector.forget()

    def answer(self, x):
        a, n = self.detector.ask(x), self.detector.counts()
        if a["taught"]:
            mode = f"taught: {n['yes']} frames with, {n['no']} without"
        elif a["zero"] is not None:
            mode = "zero-shot: text prompts only" + (" (teach a frame with and one without)" if n["yes"] + n["no"] else "")
        else:
            mode = f"show it once: hold {self.key} with your thing in view, shift+{self.key} without"
        p = a["p"] or 0.0
        return {"p": rounded({"no": 1 - p, "yes": p}),
                "zero": None if a["zero"] is None else rounded({"no": 1 - a["zero"], "yes": a["zero"]}),
                "mode": mode, "labelled": n}


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
        elif self.path == "/questions":
            self.reply({"tokens": args.tokens, "questions": {q: question.describe() for q, question in questions.items()}})
        else:
            self.send_error(404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/frame":  # JPEG bytes in, every question's answer out; X-Label: question/answer labels the frame
            x, ms = eyes.see(body)
            q, _, name = (self.headers.get("X-Label") or "").partition("/")
            if q in questions and name in questions[q].names:
                questions[q].teach(x, name)
                album.keep(body, x, question=q, answer=name)
            self.reply({"ms": round(ms, 1), "questions": {q: question.answer(x) for q, question in questions.items()}})
        elif self.path == "/forget":
            for question in questions.values():
                question.forget()
            album.set_aside()
            self.reply({})
        else:
            self.send_error(404)


questions = {q: Question(*spec) for q, spec in QUESTIONS.items()} | {d: Yes(*spec) for d, spec in DETECTORS.items()}
album = Album(Path(__file__).parent / "frames", eyes)
kept = album.frames()
for x, row in kept:  # teach again what was taught before the restart
    if row["question"] in questions:
        questions[row["question"]].teach(x, row["answer"])
if kept:
    print(f"taught again from {len(kept)} saved frames", flush=True)
eyes.warm()
print(f"gestures: http://localhost:{args.port}  ({args.tokens} soft tokens per frame)", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
