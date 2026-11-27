"""Shared test helpers: golden files, condition lists, row lookup, and offline stand-ins for chat()."""
import copy
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from common.config import ROOT, model_spec, models

GOLDEN = ROOT / "tests" / "golden"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_golden(name: str):
    return json.loads((GOLDEN / name).read_text())


def require(path: Path, reason: str | None = None) -> Path:
    """Skip the test when ``path`` does not exist (e.g. a condition that was never run)."""
    if not path.exists():
        pytest.skip(reason or f"missing: {path.relative_to(ROOT)}")
    return path


def runs(experiment: str, *factors) -> list[tuple]:
    """(model, *factor levels, answer format) for every condition run in an experiment."""
    return [(m, *levels) for m in models(experiment)
            for levels in product(*factors, model_spec(experiment, m)["instructions"])]


def select(table, **match):
    """The single row of ``table`` whose columns equal ``match``."""
    rows = table[np.logical_and.reduce([table[k] == v for k, v in match.items()])]
    assert len(rows) == 1, match
    return rows.iloc[0]


def _sent(kwargs: dict) -> dict:
    """The options chat() actually sends: it omits every None."""
    return {k: v for k, v in kwargs.items() if v is not None}


class FakeLLM:
    """Records a deep copy of every request and returns a fixed reply."""

    def __init__(self, reply: str = "50"):
        self.reply = reply
        self.requests: list[dict] = []

    def __call__(self, provider, model_id, messages, **kwargs):
        self.requests.append({"provider": provider, "model": model_id, "messages": copy.deepcopy(messages),
                              **_sent(kwargs)})
        return self.reply


class RequestHasher:
    """Returns a fixed reply and hashes the whole request sequence.

    Each request is hashed as its difference from the previous request plus reply, which pins
    the sequence exactly while keeping long multi-turn image dialogues (Exp 1) cheap.
    """

    def __init__(self, reply: str = "50"):
        self.reply, self.n = reply, 0
        self._hash, self._prev = hashlib.sha256(), []

    def __call__(self, provider, model_id, messages, **kwargs):
        k = len(self._prev)
        start = k if len(messages) > k and messages[:k] == self._prev else 0
        self._hash.update(json.dumps([provider, model_id, start, messages[start:], _sent(kwargs)],
                                     sort_keys=True).encode())
        self._prev = [*messages, {"role": "assistant", "content": self.reply}]
        self.n += 1
        return self.reply

    def hexdigest(self) -> str:
        return self._hash.hexdigest()
