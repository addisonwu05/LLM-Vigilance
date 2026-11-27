"""Render Experiment 2 prompts from prompts.yaml (its header explains the layout).

tests/test_exp2.py checks that every prompt is byte-identical to the original scripts.
"""
from __future__ import annotations

from common.config import load_prompts
from common.prompting import answer_format, pronouns, rating_question


def _cfg() -> dict:
    return load_prompts("exp2")


def scenarios() -> list[str]:
    return list(_cfg()["scenarios"])


def definite(person: str) -> str:
    """'a stranger' -> 'the stranger', 'an agent ...' -> 'the agent ...' (original replace_first_word)."""
    if person.startswith("a "):
        return person.replace("a ", "the ", 1)
    if person.startswith("an "):
        return person.replace("an ", "the ", 1)
    return person


def characters(scenario: str, perspective: str) -> list[str]:
    return [c.format(**pronouns(perspective)) for c in _cfg()["scenarios"][scenario]["characters"]]


def incentives(scenario: str) -> list[int]:
    return list(_cfg()["scenarios"][scenario]["incentives"])


def context(scenario: str, perspective: str, purpose: str) -> str:
    """Opening message of every dialogue: scenario setup plus trust or incentive framing."""
    cfg = _cfg()
    p = cfg["scenarios"][scenario][perspective]
    return p["context"] + (cfg["trust_context"] if purpose == "trust" else p["incentive_context"])


def question(scenario: str, perspective: str, purpose: str, person: str, reward, flip: bool,
             instructions: str) -> str:
    """One rating question. ``flip`` reverses the scale anchors; ``instructions`` is
    'direct', 'cot' or 'reasoning' (the answer-format suffix)."""
    cfg = _cfg()
    s = cfg["scenarios"][scenario]
    p = s[perspective]
    layers = (cfg["questions"][purpose], s, s.get(purpose, {}), p, p.get(purpose, {}))
    text = rating_question(layers, flip, perspective, person=person, person_the=definite(person), reward=reward)
    return text + answer_format(instructions)
