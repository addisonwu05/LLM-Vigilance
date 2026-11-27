"""The factors that define an experimental condition, named once for all experiments."""
from __future__ import annotations

PERSPECTIVES = ("first", "user")                          # the model is the listener ("you") / answers for a user ("I")
PURPOSES = ("trust", "reward_quality", "offer_quality")   # benevolence, incentive and recommendation-quality ratings
INSTRUCTIONS = ("cot", "direct", "reasoning")             # answer formats, in the paper's row order


def conditions(spec: dict) -> list[tuple[str, str]]:
    """(instructions, perspective) pairs a model was run in (``spec`` from config.model_spec),
    in the paper's row order."""
    return [(i, p) for i in INSTRUCTIONS if i in spec["instructions"] for p in PERSPECTIVES]


def response_file(purpose: str, perspective: str, instructions: str) -> str:
    """File name of one rating condition's responses (Experiments 2 and 3)."""
    return f"{purpose}_{perspective}_{instructions}.json"
