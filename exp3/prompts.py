"""Render Experiment 3 (YouTube sponsorship) prompts from prompts.yaml.

A steering sentence (none / incentives / bias / gricean) is placed directly before the
answer-format instruction, exactly where the original scripts put it.
"""
from __future__ import annotations

from common.config import load_prompts
from common.prompting import answer_format, rating_question


def _cfg() -> dict:
    return load_prompts("exp3")


def steers() -> list[str]:
    return list(_cfg()["steers"])


def question(perspective: str, purpose: str, video: dict, flip: bool, instructions: str,
             steer: str = "none") -> str:
    cfg = _cfg()
    text = rating_question(
        (cfg, cfg["questions"][purpose]), flip, perspective,
        video_title=video["video_title"],
        channel_name=video["video_channel_name"],
        channel_description=video["video_channel_description"],
        transcript=video["video_sponsorship_transcript"],
    )
    return text + cfg["steers"][steer] + answer_format(instructions)
