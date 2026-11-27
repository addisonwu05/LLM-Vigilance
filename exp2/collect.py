"""Experiment 2 data collection: elicit trust, incentive and offer-quality ratings.

Protocol (unchanged from the paper's runs). For each simulated subject, all speakers
(4 characters for trust; 4 characters x 4 incentives otherwise) are presented in one
multi-turn dialogue, in a freshly shuffled order. Every answer is appended to the
context before the next question. Model-specific details (token caps, scale flipping,
how the opening context is sent, answer formats) come from models.yaml, because
the original runs differed per model.

Randomness is consumed exactly like the original scripts: seven in-place shuffles of
the same speaker list per subject, then (for per-subject flipping) one coin flip. So
a seeded run reproduces the original request sequence (tests/test_exp2.py).
"""
from __future__ import annotations

import random
from itertools import product
from pathlib import Path

from common import files, llm
from common.config import model_spec
from common.design import PERSPECTIVES, PURPOSES, response_file
from . import DATA, prompts


def response_path(model: str, scenario: str, purpose: str, perspective: str, instructions: str,
                  root: Path = DATA) -> Path:
    return root / "responses" / model / scenario / response_file(purpose, perspective, instructions)


def flip_for(spec: dict, subject_index: int) -> bool | None:
    """Fixed flip for this subject, or None when it is drawn at random per subject."""
    schedule = spec["flip"]
    if schedule == "per_subject":
        return None
    if isinstance(schedule, bool):
        return schedule
    start = 0
    for batch in schedule:                      # e.g. [{flip: false, n: 80}, {flip: true, n: 80}]
        if subject_index < start + batch["n"]:
            return batch["flip"]
        start += batch["n"]
    return schedule[-1]["flip"]


def run_subject(rng: random.Random, speakers: list, scenario: str, perspective: str, purpose: str,
                instructions: str, spec: dict, chat: llm.ChatFn, fixed_flip: bool | None = None) -> list[dict]:
    """One simulated subject = one dialogue over all speakers. Mutates ``speakers`` order."""
    for _ in range(7):
        rng.shuffle(speakers)
    flip = rng.choice([True, False]) if fixed_flip is None else fixed_flip
    opening = prompts.context(scenario, perspective, purpose)
    merged = spec["context"] == "merged"
    history = [] if merged else [{"role": "user", "content": opening}]
    max_tokens = (spec.get("max_tokens") or {}).get(instructions)
    trials = []
    for i, speaker in enumerate(speakers):
        person, reward = (speaker, None) if purpose == "trust" else speaker
        text = prompts.question(scenario, perspective, purpose, person, reward, flip, instructions)
        history = llm.ask(chat, spec, history, opening + " " + text if merged and i == 0 else text,
                          max_tokens=max_tokens)
        trials.append({"person": person, **({} if purpose == "trust" else {"reward": reward}),
                       "flip": flip, "messages": history})
    return trials


def collect(model: str, scenario: str, perspective: str, purpose: str, instructions: str, *,
            n: int | None = None, seed: int | None = None, out: Path | None = None,
            chat: llm.ChatFn = llm.chat, spec: dict | None = None) -> Path:
    """Append subjects to the response file until it holds ``n`` (resumable).

    ``spec`` overrides the model's settings from models.yaml (used by the tests).
    """
    spec = spec or model_spec("exp2", model)
    chars, rewards = prompts.characters(scenario, perspective), prompts.incentives(scenario)
    speakers = list(chars) if purpose == "trust" else list(product(chars, rewards))
    rng = random.Random(seed)
    return files.extend(
        out or response_path(model, scenario, purpose, perspective, instructions), n or spec["samples"],
        lambda i: run_subject(rng, speakers, scenario, perspective, purpose, instructions, spec, chat,
                              flip_for(spec, i)))


def collect_all(model: str, *, n: int | None = None, seed: int | None = None) -> None:
    """Every scenario x perspective x purpose x answer format for one model."""
    for scenario, perspective, purpose, instructions in product(
            prompts.scenarios(), PERSPECTIVES, PURPOSES, model_spec("exp2", model)["instructions"]):
        collect(model, scenario, perspective, purpose, instructions, n=n, seed=seed)
