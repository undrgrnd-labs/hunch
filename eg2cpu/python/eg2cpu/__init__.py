"""CPU (SIMD) inference for the EmbeddingGemma 2 encoder, text, images and audio; no torch or transformers needed."""

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from ._eg2cpu import MAX_TOKENS, SOFT_TOKEN_BUDGETS, Handle, Model, Partial, isa
from . import decisions

__all__ = ["MAX_TOKENS", "SOFT_TOKEN_BUDGETS", "Embedder", "Handle", "Model", "Partial", "decisions", "isa"]


class Embedder:
    """`embed(text)` returns the mean-pooled state after every layer (24 x 512-dim) plus the final normalized embedding
    (768-dim), as a list of 25 arrays, cached: `cache` maps each embedded text to that list.

    Calls are thread-safe and batched in the background: a worker thread takes every text queued while the previous
    batch ran and embeds them together, and concurrent calls for the same text share one forward pass. A plain
    sequential loop still embeds one text at a time, since each call waits for its result; `map` runs the loop body
    concurrently so the batcher has work to merge.
    """

    def __init__(self, *, path=None, model: Model | None = None, max_batch: int = 256):
        self.model = model or Model(path)
        self.max_batch = max_batch
        self.cache: dict[str, list[np.ndarray]] = {}
        self._pending: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue[list[str]] = queue.SimpleQueue()
        self._worker: threading.Thread | None = None

    @staticmethod
    def _unpack(out) -> list[np.ndarray]:
        emb, means = out
        return [*means, emb]

    def _submit(self, texts) -> list[Future]:
        """A future per text: already resolved if cached, else shared with any in-flight request for it."""
        futs, new = [], []
        with self._lock:
            for t in texts:
                if t in self.cache:
                    f = Future()
                    f.set_result(self.cache[t])
                elif (f := self._pending.get(t)) is None:
                    f = self._pending[t] = Future()
                    new.append(t)
                futs.append(f)
            # Queued as chunks, so the worker never wakes on the first text of a prefetch and runs it alone.
            for i in range(0, len(new), self.max_batch):
                self._queue.put(new[i : i + self.max_batch])
            if new and self._worker is None:
                self._worker = threading.Thread(target=self._run, name="eg2cpu-batcher", daemon=True)
                self._worker.start()
        return futs

    def _run(self) -> None:
        while True:
            batch = self._queue.get()
            while len(batch) < self.max_batch:
                try:
                    batch += self._queue.get_nowait()
                except queue.Empty:
                    break
            try:
                outs, err = [self._unpack(o) for o in self.model.embed_batch(batch)], None
            except BaseException as e:  # noqa: BLE001 - handed to every waiting caller (Rust panics are BaseException)
                outs, err = None, e
            with self._lock:
                futs = [self._pending.pop(t) for t in batch]
                if err is None:
                    self.cache.update(zip(batch, outs))
            for i, f in enumerate(futs):
                f.set_exception(err) if err is not None else f.set_result(outs[i])

    def __call__(self, text: str) -> list[np.ndarray]:
        if (r := self.cache.get(text)) is not None:
            return r
        return self._submit([text])[0].result()

    def prefetch(self, texts) -> None:
        """Embed every uncached text in batches, so later calls are cache hits."""
        for f in self._submit(texts):
            f.result()

    def map(self, fn, items, workers: int = 64) -> list:
        """`[fn(x) for x in items]`, run on a thread pool so the embed calls inside `fn` batch together. `fn` must
        be safe to run concurrently (it should not mutate shared state)."""
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(fn, items))
