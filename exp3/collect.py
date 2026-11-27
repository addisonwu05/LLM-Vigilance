"""Experiment 3 data collection: one independent, single-turn query per video and rating.

Matches the paper's runs: temperature 0, token caps per answer format (models.yaml), one
sample per video, and a fresh random flip of the rating scale per video (stored as `flipped`).
"""
from __future__ import annotations

import random
from pathlib import Path

from common import files, llm
from common.config import model_spec
from common.design import response_file
from common.parsing import parse_exp3_score, unflip
from . import DATA, prompts


def videos_path(root: Path = DATA) -> Path:
    return root / "videos_censored.json"


def load_videos(root: Path = DATA) -> list[dict]:
    return files.read_json(videos_path(root))


def response_path(steer: str, model: str, purpose: str, perspective: str, instructions: str,
                  root: Path = DATA) -> Path:
    return root / "responses" / steer / model / response_file(purpose, perspective, instructions)


def collect(model: str, steer: str, perspective: str, purpose: str, instructions: str, *,
            videos: list[dict] | None = None, seed: int | None = None, out: Path | None = None,
            chat: llm.ChatFn = llm.chat) -> Path:
    spec = model_spec("exp3", model)
    videos = videos if videos is not None else load_videos()
    rng = random.Random(seed)

    def query(i: int) -> dict:
        flip = rng.choice([True, False])
        text = prompts.question(perspective, purpose, videos[i], flip, instructions, steer)
        reply = llm.ask(chat, spec, [], text, max_tokens=spec["max_tokens"][instructions],
                        temperature=spec["temperature"])[-1]["content"]
        return {**videos[i], "model_response": reply, "flipped": flip}

    return files.extend(out or response_path(steer, model, purpose, perspective, instructions), len(videos), query)


def score(path: Path) -> None:
    """Add `score` / `flipped_score` to a response file (the original parser.py step)."""
    data = files.read_json(path)
    for entry in data:
        if "model_response" in entry:
            s = parse_exp3_score(entry["model_response"])
            entry["score"] = s
            entry["flipped_score"] = unflip(s, entry.get("flipped") is True)
    files.write_json(path, data, indent=2)
