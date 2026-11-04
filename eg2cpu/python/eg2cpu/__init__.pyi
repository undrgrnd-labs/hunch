import os
from collections.abc import Callable, Iterable
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from . import decisions as decisions

_T = TypeVar("_T")
_R = TypeVar("_R")

MAX_TOKENS: int
SOFT_TOKEN_BUDGETS: list[int]

Span = dict[str, str]
Verdict = dict[str, object]
Embedding = tuple[NDArray[np.float32], NDArray[np.float32]]

def isa() -> str: ...

class Handle:
    name: str
    kind: str
    labels: list[str]
    revision: str
    metadata_json: str
    def __init__(self, path: str) -> None: ...
    @staticmethod
    def from_bytes(data: bytes) -> Handle: ...
    def render(self, span: Span) -> str | None: ...

class Partial:
    """A forward pass stopped after some layers; finish it with `Model.resume_batch`."""

    layers_done: int
    layer_means: NDArray[np.float32]

class Model:
    num_layers: int
    revision: str
    def __init__(self, path: str | None = None) -> None: ...
    def embed(self, text: str) -> Embedding: ...
    def embed_batch(self, texts: list[str]) -> list[Embedding]: ...
    def embed_image(self, image: bytes | str | os.PathLike[str], max_soft_tokens: int = 280) -> Embedding: ...
    def embed_audio(self, audio: bytes | str | os.PathLike[str] | NDArray[np.float32] | NDArray[np.int16]) -> Embedding: ...
    def embed_batch_partial(self, texts: list[str], upto: int) -> list[Partial]: ...
    def resume_batch(self, partials: list[Partial]) -> list[Embedding]: ...
    def tokenize(self, text: str) -> list[int]: ...
    def apply(self, handles: list[Handle], spans: list[Span]) -> list[list[Verdict]]: ...

class Embedder:
    model: Model
    cache: dict[str, list[NDArray[np.float32]]]
    max_batch: int
    def __init__(self, *, path: str | None = None, model: Model | None = None, max_batch: int = 256) -> None: ...
    def __call__(self, text: str) -> list[NDArray[np.float32]]: ...
    def prefetch(self, texts: Iterable[str]) -> None: ...
    def map(self, fn: Callable[[_T], _R], items: Iterable[_T], workers: int = 64) -> list[_R]: ...
