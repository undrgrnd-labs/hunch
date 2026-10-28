"""hunch.live: questions about a live stream (a webcam, or texts as they arrive), answered from one EG2 embedding per
item.

The proxy (`hunch.proxy`) learns typed questions about text states behind a teacher. This is the same idea for a
stream that a person watches and teaches as it goes: every item wants an answer as it arrives, and the answers it
can give grow while it runs. An item is an image (JPEG or PNG bytes) or a text; the classes below call either a
frame.

  Eyes      EG2 on frames (one forward pass at a time) and on text prompts. A text frame is embedded like a prompt,
            by the full pass (the proxy reads layer 8 instead), so it shares the prompts' space.
  Choice    picks one answer. Every answer has text prompts, so it answers zero-shot from the start; labelled
            frames train a student, which answers as far as a frame resembles what it was taught. Answers can be
            added while it runs, and `forget` goes back to the ones it started with.
  Detector  says yes or no: text prompts, plus a direction taught from one frame with the thing and one without.
  Watcher   a teacher model watching the stream: it looks at the newest frame when that's worth it, one look at a
            time, and labels the frames it's sure about for a Choice's student.
  Album     labelled frames on disk (the image or text, its embedding, labels.jsonl), to teach again after a restart.

How a Choice answers:
  zero-shot  a frame scores each answer by its mean similarity to the answer's prompts, standardised across the
             answers; a softmax at `temperature` makes those probabilities.
  baseline   frames labelled with the `empty` answer (empty hands, nothing to show) are the baseline: a frame's
             distance from them, in their own spreads, says whether anything is being shown (`shown`), and what
             differs from them, compared with the same for every taught frame, says how much the frame resembles
             what was taught (`resemble`, the best cosine).
  taught     a logistic regression on the labelled frames (standardised, classes weighted equally), refitted in the
             background. With a baseline, its weight is a sigmoid of the resemblance (half at `resemble`), and full
             when the frame is as close to the baseline as the baseline frames are to each other; without one, it
             grows with the labelled frames (half at `half_at`). Answers never taught keep their zero-shot share of
             what the student leaves.
  state      "quiet" when the best answer is a quiet one (never named), "sure" when it's at least `sure` likely,
             else "unsure" if something is being shown (or there's no baseline to tell) and "quiet" if not.
"""

from __future__ import annotations

import json
import struct
import threading
import time
import zlib
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .proxy import encoder_text

ELSE = "something else"  # a teacher's answer for a frame that shows none of the answers


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))


def grey_png(side: int = 64) -> bytes:
    """A plain grey PNG: a first frame that loads a vision tower before the real ones arrive."""
    rows = b"".join(b"\0" + b"\x80" * 3 * side for _ in range(side))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


Frame = bytes | str  # an image's bytes, or a text


class Eyes:
    """EG2 on frames, one forward pass at a time (frames queue with the caller): an image at `tokens` soft tokens, a
    text as the encoder reads it (the head and tail of a long one, as the proxy clips it). And on text prompts."""

    def __init__(self, model=None, tokens: int = 70):
        if model is None:
            from eg2cpu import Model
            model = Model()
        self.model, self.tokens = model, tokens
        self.lock = threading.Lock()

    def see(self, frame: Frame) -> tuple[np.ndarray, float]:
        """(unit embedding, milliseconds of EG2), timed inside the lock so the wait for another frame isn't counted."""
        with self.lock:
            t0 = time.perf_counter()
            if isinstance(frame, str):
                x = unit(self.model.embed(encoder_text(frame))[0])
            else:
                x = unit(self.model.embed_image(frame, max_soft_tokens=self.tokens)[0])
            return x, (time.perf_counter() - t0) * 1000

    def read(self, texts: list[str]) -> np.ndarray:
        """Unit embeddings of `texts`, one row each."""
        return unit(np.stack([self.model.embed(t)[0] for t in texts]))

    def warm(self) -> None:
        self.see(grey_png())


class _Fit:
    def __init__(self, student, baseline, residuals):
        self.student, self.baseline, self.residuals = student, baseline, residuals


class Choice:
    """Picks one answer from text prompts and labelled frames; see the module's docstring for how.

    `answers` maps each answer to its prompts. `quiet` answers are never named and never asked about; `empty` is the
    one whose frames are the baseline. `new(name)` gives prompts for a name taught that isn't an answer yet (without
    it, teaching an unknown name is an error)."""

    def __init__(self, eyes: Eyes, answers: dict[str, list[str]], *, quiet=(), empty: str | None = None,
                 new: Callable[[str], list[str]] | None = None, temperature: float = 2.5, sure: float = 0.5,
                 shown: float = 3.0, resemble: float = 0.6, resemble_scale: float = 0.05, half_at: int = 8):
        self.eyes, self.quiet, self.empty, self.new = eyes, set(quiet), empty, new
        self.temperature, self.sure, self.shown, self.half_at = temperature, sure, shown, half_at
        self.resemble, self.resemble_scale = resemble, resemble_scale
        self.names: list[str] = []
        self.prompts: list[np.ndarray] = []
        self.labelled: list[tuple[np.ndarray, int, str]] = []
        self.fit: _Fit | None = None
        self.generation = 0  # forget() moves it on, so a refit begun before it isn't installed after it
        self.adding = threading.Lock()
        self.changed = threading.Event()
        for name, prompts in answers.items():
            self._add(name, prompts)
        self.start = len(self.names)  # the answers it started with; names taught later come after
        threading.Thread(target=self._refitting, daemon=True).start()

    def _add(self, name, prompts):
        self.prompts = self.prompts + [self.eyes.read(prompts)]
        self.names = self.names + [name]  # last: a name is only answered once its prompts are in place

    def find(self, name: str) -> str | None:
        """The answer a name means, allowing for a plural."""
        names = self.names
        for n in (name, name + "s", name.rstrip("s"), name.removesuffix("es")):
            if n in names:
                return n
        return None

    def add(self, name: str, prompts: list[str]) -> str:
        """An answer, added with its prompts unless there's one by that name already."""
        with self.adding:
            found = self.find(name)
            if found is None:
                self._add(name, prompts)
            return found or name

    def teach(self, x: np.ndarray, name: str, by: str = "you") -> str:
        """Labels frame `x` with `name` (added as an answer, with `new(name)`'s prompts, if it isn't one) and refits in
        the background; returns the answer it was labelled with."""
        with self.adding:
            found = self.find(name)
            if found is None:
                if self.new is None:
                    raise KeyError(f"{name!r} isn't an answer")
                self._add(name, self.new(name))
                found = name
            self.labelled.append((x, self.names.index(found), by))
        self.changed.set()
        return found

    def forget(self) -> None:
        """Back to the answers it started with: the labelled frames go, and so do the answers added since."""
        with self.adding:
            self.generation += 1
            self.labelled, self.fit = [], None
            self.names, self.prompts = self.names[:self.start], self.prompts[:self.start]
        self.changed.set()

    def counts(self) -> dict[str, dict[str, int]]:
        """Labelled frames per answer, by who labelled them."""
        out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        names = self.names
        for _, i, by in list(self.labelled):
            if i < len(names):
                out[names[i]][by] += 1
        return {n: dict(c) for n, c in out.items()}

    def _refitting(self):
        while True:
            self.changed.wait()
            self.changed.clear()
            self.refit()
            time.sleep(0.2)  # labels arriving meanwhile are taken together

    def refit(self) -> None:
        with self.adding:
            generation, labelled, names = self.generation, list(self.labelled), self.names
        fit = None
        if labelled:
            ys = np.array([y for _, y, _ in labelled])
            X = np.stack([x for x, _, _ in labelled])
            student = baseline = residuals = None
            e = names.index(self.empty) if self.empty in names else -1
            if (ys == e).sum() >= 2:  # the baseline frames: where they sit, and how far apart they spread
                E = X[ys == e]
                centre = E.mean(0)
                d = np.linalg.norm(E - centre, axis=1)
                baseline = {"centre": centre, "d": float(np.median(d)), "spread": float(d.std() + 1e-4)}
                loud = np.array([names[y] not in self.quiet for y in ys])
                residuals = unit(X[loud] - centre) if loud.any() else None
            if len(set(ys.tolist())) >= 2:
                student = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
                student.fit(X, ys)
            fit = _Fit(student, baseline, residuals)
        with self.adding:
            if generation == self.generation:
                self.fit = fit

    def ask(self, x: np.ndarray) -> dict:
        """The answer for frame `x`: {"name", "p", "state", "probs", "zero", "shown", "resemble", "weight", "ready",
        "mode"}. `probs` and `zero` give every answer's probability, with and without the student."""
        names, prompts, fit, k = self.names, self.prompts, self.fit, len(self.labelled)
        n = min(len(names), len(prompts))
        student = fit.student if fit is not None else None
        if student is not None and student.classes_.max() >= n:  # fitted before a forget
            student = None
        s = np.array([float((P @ x).mean()) for P in prompts[:n]])
        p = zero = softmax(self.temperature * (s - s.mean()) / (s.std() + 1e-6))
        baseline = fit.baseline if fit is not None else None
        shown, resemble, w = None, 0.0, 0.0
        if baseline is not None:
            shown = (float(np.linalg.norm(x - baseline["centre"])) - baseline["d"]) / baseline["spread"]
            if fit.residuals is not None:
                resemble = float(np.max(fit.residuals @ unit(x - baseline["centre"])))
        if student is not None:
            taught = np.zeros(n)
            taught[student.classes_] = student.predict_proba(x[None])[0]
            if baseline is None:
                w = k / (k + self.half_at)
            else:  # as far as the frame looks like the things it was taught, or like the baseline
                w = sigmoid((resemble - self.resemble) / self.resemble_scale) if fit.residuals is not None else 0.0
                w = max(w, sigmoid(2 * (self.shown - shown)))
            p = (1 - w) * zero + w * taught
        best = int(np.argmax(p))
        if names[best] in self.quiet:
            state = "quiet"
        elif p[best] >= self.sure:
            state = "sure"
        else:  # no answer stands out: worth asking about only if something is being shown
            state = "unsure" if shown is None or shown >= self.shown else "quiet"
        if student is not None:
            mode = f"taught: {k} labelled frames, student weight {w:.0%}"
        elif k:
            mode = f"zero-shot until two answers have frames ({k} labelled)"
        else:
            mode = "zero-shot: text prompts only"
        return {"name": names[best], "p": float(p[best]), "state": state,
                "probs": dict(zip(names, p.tolist())), "zero": dict(zip(names, zero.tolist())),
                "shown": shown, "resemble": resemble, "weight": w, "ready": baseline is not None, "mode": mode}


class Detector:
    """Says yes or no. Zero-shot, a frame's similarity to the `yes` prompts minus its similarity to the `no` prompts,
    against `threshold`, with `scale` the distance past it that makes p = 0.73. Taught (a frame labelled "yes" and
    one labelled "no" is enough), the direction from the "no" frames to the "yes" frames turns the prompts'
    direction by `nudge`, and the threshold moves halfway between them. Without prompts it's that direction alone,
    so it only answers once taught."""

    names = ["no", "yes"]

    def __init__(self, eyes: Eyes, yes: list[str] = (), no: list[str] = (), threshold: float = 0.0,
                 scale: float = 0.01, nudge: float = 0.5):
        self.threshold, self.scale, self.nudge = threshold, scale, nudge
        self.v0 = (eyes.read(list(yes)).mean(0) - eyes.read(list(no)).mean(0)) if yes else None
        self.labelled: list[tuple[np.ndarray, int]] = []
        self.taught = None  # (direction, cut, scale)

    def teach(self, x: np.ndarray, name: str) -> str:
        self.labelled.append((x, self.names.index(name)))
        self.refit()
        return name

    def forget(self) -> None:
        self.labelled = []
        self.refit()

    def counts(self) -> dict[str, int]:
        return {n: sum(1 for _, y in self.labelled if y == i) for i, n in enumerate(self.names)}

    def refit(self) -> None:
        no = [x for x, y in self.labelled if y == 0]
        yes = [x for x, y in self.labelled if y == 1]
        taught = None
        if no and yes:
            mn, my = np.mean(no, 0), np.mean(yes, 0)
            d = unit(my - mn)
            v = d if self.v0 is None else unit(self.v0) + self.nudge * d
            a, b = float(mn @ v), float(my @ v)
            taught = (v, (a + b) / 2, max(abs(b - a) / 8, 1e-4))  # the taught frames sit at p ~ 0.02 and 0.98
        self.taught = taught

    def ask(self, x: np.ndarray) -> dict:
        """{"p": p(yes) or None before it can answer, "zero": the prompts' p(yes) or None, "taught": bool}."""
        zero = None if self.v0 is None else sigmoid((float(x @ self.v0) - self.threshold) / self.scale)
        taught = self.taught
        p = sigmoid((float(x @ taught[0]) - taught[1]) / taught[2]) if taught is not None else zero
        return {"p": p, "zero": zero, "taught": taught is not None}


class Watcher:
    """A teacher model watching the stream for a Choice. `teacher(frame, names)` answers which of `names` (or `ELSE`)
    the frame (an image's bytes or a text) shows, as {answer: probability}; it's called one look at a time, from the Watcher's own thread.

    It looks at the newest frame when the student's answer is a thing it hasn't agreed on in the last `confirmed`
    seconds, or, when the answer is quiet, every `idle` seconds while something is being shown (a held thing can read
    as a bare hand), resting `rest` seconds between looks. Sure of an answer (`sure`), it labels the frame for the
    student, up to `cap` frames per answer, and calls `on_label(frame, x, name)`."""

    def __init__(self, choice: Choice, teacher: Callable[[Frame, list[str]], dict[str, float]], *, by: str = "teacher",
                 sure: float = 0.5, cap: int = 60, rest: float = 0.3, confirmed: float = 3.0, idle: float = 3.0,
                 fresh: float = 1.5, on_label: Callable[[Frame, np.ndarray, str], None] | None = None):
        self.choice, self.teacher, self.by, self.on_label = choice, teacher, by, on_label
        self.sure, self.cap, self.rest, self.confirmed, self.idle, self.fresh = sure, cap, rest, confirmed, idle, fresh
        self.newest = None
        self.wake = threading.Event()
        self.last = None
        self.agree: deque[bool] = deque(maxlen=50)
        self.agreed: dict[str, float] = {}  # answer: when the teacher last agreed with the student on it
        self.looked: deque[float] = deque(maxlen=200)
        threading.Thread(target=self._watching, daemon=True).start()

    def look(self, frame: Frame, x: np.ndarray, answer: dict) -> None:
        """Offers the newest frame and the student's answer for it; the teacher looks if that's worth it."""
        now = time.time()
        if answer["state"] == "quiet":
            since = now - self.looked[-1] if self.looked else self.idle
            worth = (answer["shown"] or 0) >= self.choice.shown and since >= self.idle
        else:
            worth = now - self.agreed.get(answer["name"], 0) > self.confirmed
        if worth:
            self.newest = (frame, x)
            self.wake.set()

    def forget(self) -> None:
        self.agree.clear()
        self.agreed.clear()
        self.last = None

    def _watching(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            frame, x = self.newest
            self.looked.append(time.time())
            t0 = time.perf_counter()
            out = self.teacher(frame, list(self.choice.names))
            ms = (time.perf_counter() - t0) * 1000
            best = max(out, key=out.get)
            p = out[best]
            name = self.choice.find(best) if best != ELSE else None
            self.last = {"name": name or best, "known": name is not None, "p": p, "ms": ms, "at": time.time()}
            if p >= self.sure and name is not None:
                agreed = self.choice.ask(x)["name"] == name
                self.agree.append(agreed)
                if agreed:
                    self.agreed[name] = time.time()
                if self.choice.counts().get(name, {}).get(self.by, 0) < self.cap:
                    self.choice.teach(x, name, by=self.by)
                    if self.on_label is not None:
                        self.on_label(frame, x, name)
            time.sleep(self.rest)

    def saw_else(self) -> bool:
        """Whether the teacher's latest look, still fresh, was sure it saw none of the answers."""
        last = self.last
        return (last is not None and not last["known"] and last["p"] >= self.sure
                and time.time() - last["at"] < self.fresh)

    def describe(self) -> dict | None:
        last = self.last
        if last is None:
            return None
        now = time.time()
        return {"name": last["name"], "known": last["known"], "p": round(last["p"], 3), "ms": round(last["ms"]),
                "age": round(now - last["at"], 2), "fresh": now - last["at"] < self.fresh,
                "per_minute": sum(1 for t in self.looked if now - t < 60),
                "agree": round(float(np.mean(self.agree)), 3) if self.agree else None, "checked": len(self.agree)}


class Album:
    """Labelled frames in `folder`: each image (.jpg) or text (.txt), its embedding (an image's cached per soft-token
    budget) and a line of labels.jsonl with whatever fields describe the label."""

    def __init__(self, folder: str | Path, eyes: Eyes):
        self.folder, self.eyes = Path(folder), eyes
        self.labels = self.folder / "labels.jsonl"
        self.lock = threading.Lock()

    def keep(self, frame: Frame, x: np.ndarray, **fields) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{time.time_ns()}.{'txt' if isinstance(frame, str) else 'jpg'}"
        if isinstance(frame, str):
            path.write_text(frame)
        else:
            path.write_bytes(frame)
        np.save(self._cache(path), x)
        with self.lock, self.labels.open("a") as f:
            f.write(json.dumps({"file": path.name} | fields) + "\n")

    def frames(self) -> list[tuple[np.ndarray, dict]]:
        """Every kept frame's embedding (embedded again at this budget if it hasn't been) and its label line."""
        if not self.labels.exists():
            return []
        out = []
        for line in self.labels.read_text().splitlines():
            row = json.loads(line)
            path = self.folder / row["file"]
            cached = self._cache(path)
            if cached.exists():
                x = np.load(cached)
            else:
                x, _ = self.eyes.see(path.read_text() if path.suffix == ".txt" else path.read_bytes())
                np.save(cached, x)
            out.append((x, row))
        return out

    def _cache(self, path: Path) -> Path:
        return path.with_suffix(".npy" if path.suffix == ".txt" else f".{self.eyes.tokens}.npy")

    def set_aside(self) -> None:
        """Moves the labels to labels.<time>.jsonl, so they aren't taught again; the frames stay."""
        with self.lock:
            if self.labels.exists():
                self.labels.rename(self.folder / f"labels.{int(time.time())}.jsonl")
