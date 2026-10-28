import time

import numpy as np
import pytest

from hunch.live import ELSE, Album, Choice, Detector, Eyes, Watcher

D = 32


def direction(word: str) -> np.ndarray:
    v = np.random.default_rng(abs(hash(word)) % 2**32).standard_normal(D).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeModel:
    """Texts embed as the direction of their last word; an image is bytes naming a word (b"mug") or b"mug+room"."""

    def embed(self, text):
        return direction(text.split()[-1]), None

    def embed_image(self, image, max_soft_tokens=70):
        if image.startswith(b"\x89PNG"):
            return direction("grey"), None
        words = image.decode().split("+")
        return sum(direction(w) for w in words) + 0.01 * np.random.default_rng(len(image)).standard_normal(D), None


@pytest.fixture
def eyes():
    return Eyes(FakeModel())


def frame(eyes, image):
    return eyes.see(image)[0]


def settled(choice, timeout=5.0):
    """Waits for the background refit to take the labels in."""
    end = time.time() + timeout
    while time.time() < end:
        if choice.fit is not None and not choice.changed.is_set():
            time.sleep(0.05)
            return
        time.sleep(0.02)
    raise AssertionError("no refit")


def make_choice(eyes, **kw):
    return Choice(eyes, {"nothing": ["a room"], "mug": ["a mug"], "pen": ["a pen"], "cat": ["a cat"]},
                  quiet={"nothing"}, empty="nothing", **kw)


def test_zero_shot_picks_the_nearest_prompts(eyes):
    c = make_choice(eyes)
    a = c.ask(frame(eyes, b"pen"))
    assert a["name"] == "pen" and a["state"] == "sure" and a["mode"].startswith("zero-shot")
    assert c.ask(frame(eyes, b"room"))["state"] == "quiet"
    assert abs(sum(a["probs"].values()) - 1) < 1e-6


def test_teaching_a_new_name_adds_it_and_forget_takes_it_away(eyes):
    c = make_choice(eyes, new=lambda name: [f"a {name}"])
    for _ in range(4):
        c.teach(frame(eyes, b"room"), "nothing")
        c.teach(frame(eyes, b"room+shell"), "seashell")
    assert "seashell" in c.names
    settled(c)
    a = c.ask(frame(eyes, b"room+shell"))
    assert a["name"] == "seashell" and a["ready"] and a["weight"] > 0.9
    assert c.counts()["seashell"] == {"you": 4}
    c.forget()
    assert c.names == ["nothing", "mug", "pen", "cat"] and c.counts() == {} and c.fit is None
    assert c.ask(frame(eyes, b"room+shell"))["name"] != "seashell"


def test_unknown_names_need_new(eyes):
    with pytest.raises(KeyError):
        make_choice(eyes).teach(frame(eyes, b"shell"), "seashell")


def test_plurals_find_the_answer(eyes):
    c = make_choice(eyes)
    assert c.find("mugs") == "mug" and c.find("pen") == "pen" and c.find("dog") is None


def test_a_refit_begun_before_forget_is_not_installed(eyes, monkeypatch):
    import hunch.live
    c = make_choice(eyes)
    c.teach(frame(eyes, b"room"), "nothing")
    c.teach(frame(eyes, b"room+mug"), "mug")
    settled(c)
    real = hunch.live.make_pipeline

    def forgetting(*steps):  # a forget arrives while the student fits
        c.forget()
        return real(*steps)

    monkeypatch.setattr(hunch.live, "make_pipeline", forgetting)
    c.labelled = [(frame(eyes, b"room"), 0, "you"), (frame(eyes, b"mug"), 1, "you")]
    c.refit()
    assert c.fit is None


def test_without_a_baseline_the_student_grows_with_the_frames(eyes):
    c = Choice(eyes, {"a": ["one"], "b": ["two"]}, half_at=2)
    for _ in range(4):
        c.teach(frame(eyes, b"two"), "a")
        c.teach(frame(eyes, b"one"), "b")
    settled(c)
    a = c.ask(frame(eyes, b"two"))
    assert not a["ready"] and a["weight"] == pytest.approx(8 / 10) and a["shown"] is None
    assert a["name"] == "a"


def test_detector_answers_once_taught(eyes):
    d = Detector(eyes)
    assert d.ask(frame(eyes, b"star"))["p"] is None
    d.teach(frame(eyes, b"room+star"), "yes")
    d.teach(frame(eyes, b"room"), "no")
    assert d.ask(frame(eyes, b"room+star"))["p"] > 0.9 and d.ask(frame(eyes, b"room"))["p"] < 0.1
    d.forget()
    assert d.ask(frame(eyes, b"star"))["p"] is None and d.counts() == {"no": 0, "yes": 0}


def test_detector_zero_shot(eyes):
    d = Detector(eyes, yes=["a fire"], no=["a room"], threshold=0.0)
    assert d.ask(frame(eyes, b"fire"))["p"] > 0.9 and d.ask(frame(eyes, b"room"))["p"] < 0.1


def test_watcher_labels_what_the_teacher_is_sure_of(eyes):
    c = make_choice(eyes)
    seen, kept = [], []

    def teacher(image, names):
        seen.append(names)
        return {"cat": 0.9, ELSE: 0.1} if image == b"cat" else {ELSE: 0.95}

    w = Watcher(c, teacher, by="clef", rest=0, cap=2, on_label=lambda image, x, name: kept.append(name))
    x = frame(eyes, b"cat")
    for _ in range(3):
        w.look(b"cat", x, {"state": "unsure", "name": "pen", "shown": 5})
        time.sleep(0.1)
    assert c.counts()["cat"] == {"clef": 2} and kept == ["cat", "cat"]
    assert "cat" in seen[0] and w.describe()["name"] == "cat"
    w.look(b"shell", frame(eyes, b"shell"), {"state": "unsure", "name": "pen", "shown": 5})
    time.sleep(0.1)
    assert w.saw_else()
    assert w.describe()["name"] == ELSE and not w.describe()["known"]


def test_album_keeps_frames_and_sets_them_aside(tmp_path, eyes):
    album = Album(tmp_path, eyes)
    x = frame(eyes, b"mug")
    album.keep(b"mug", x, thing="mug", by="you")
    [(y, row)] = album.frames()
    assert np.allclose(x, y) and row["thing"] == "mug" and row["by"] == "you"
    (tmp_path / row["file"]).with_suffix(".70.npy").unlink()
    [(z, _)] = album.frames()  # embedded again
    assert z.shape == x.shape
    album.set_aside()
    assert album.frames() == [] and list(tmp_path.glob("labels.*.jsonl"))


def test_texts_are_frames_too(tmp_path, eyes):
    c = Choice(eyes, {"fine": ["the call worked fine"], "failed": ["the call failed"]}, new=lambda name: [name])
    x, _ = eyes.see("read_file returned: the call failed")
    assert c.ask(x)["name"] == "failed"
    for _ in range(4):
        c.teach(eyes.see("pytest: 3 passed, then timeout")[0], "timeout")
        c.teach(eyes.see("pytest: 3 passed, all fine")[0], "fine")
    settled(c)
    assert c.ask(eyes.see("npm test: timeout")[0])["name"] == "timeout"

    album = Album(tmp_path, eyes)
    album.keep("pytest: 3 passed, then timeout", x, thing="timeout")
    [(y, row)] = album.frames()
    assert row["file"].endswith(".txt") and np.allclose(x, y)
    (tmp_path / row["file"]).with_suffix(".npy").unlink()
    [(z, _)] = album.frames()  # embedded again from the text
    assert np.allclose(z, eyes.see("pytest: 3 passed, then timeout")[0])


def test_long_texts_are_clipped_like_the_proxys(eyes):
    long = "start " + "x " * 4000 + "failed"
    assert np.allclose(eyes.see(long)[0], eyes.see("failed")[0])  # the tail survives the clip


def test_watcher_passes_texts_to_the_teacher(eyes):
    c = make_choice(eyes)
    seen = []

    def teacher(text, names):
        seen.append(text)
        return {"pen": 0.9}

    Watcher(c, teacher, rest=0).look("a pen", eyes.see("a pen")[0], {"state": "unsure", "name": "cat", "shown": None})
    time.sleep(0.1)
    assert seen == ["a pen"] and c.counts()["pen"] == {"teacher": 1}
