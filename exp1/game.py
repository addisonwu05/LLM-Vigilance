"""Experiment 1: the Watson & Morgan vigilance game played by a vision LLM.

Player 1 sees easier images, answers, and leaves "advice" (which need not be honest).
Player 2 sees harder images with the same answers, guesses, then is shown either Player
1's actual guess ("spied") or their advice, and may revise. One run = 20 image pairs in a
fresh random order, with spied/advice assigned 10/10 at random. The same model plays
both roles.

Request details reproduce the paper's runs: token caps per turn type come from
models.yaml (only Claude's runs set any); images go to OpenAI as base64 'image/jpeg'
data URLs (the files are PNG, kept as in the original) and to Claude as PNG blocks.
"""
from __future__ import annotations

import base64
import random
import re
from pathlib import Path

from common import files, llm
from common.config import load_prompts, merge, model_spec
from common.parsing import parse_exp1_number

from . import DATA

RECORD_KEYS = ("correct_answer", "player_1_guess", "player_1_advice", "player_1_advice_message",
               "player_2_guess_before", "info_provided", "player_2_guess_after", "player_2_update_message")


def stimuli_dirs(root: Path = DATA) -> tuple[Path, Path]:
    base = root / "stimuli"
    return base / "player_1", base / "player_2"


def results_path(model: str, instructions: str, payoff: str, root: Path = DATA) -> Path:
    return root / "responses" / f"{model}_{instructions}_{payoff}.json"


def correct_answer(filename: str) -> int | None:
    """'{yellow}_yellow_{blue}_blue.png' -> blue - yellow."""
    m = re.match(r"(\d+)_yellow_(\d+)_blue\.png", filename)
    return int(m.group(2)) - int(m.group(1)) if m else None


def game_prompts(variant: str = "standard") -> dict:
    """prompts.yaml; a variant other than 'standard' only lists what it changes."""
    cfg = load_prompts("exp1")
    return cfg["standard"] if variant == "standard" else merge(cfg["standard"], cfg[variant])


def system_prompt(player: int, payoff: str, variant: str = "standard", **fmt) -> str:
    p = game_prompts(variant)
    text = "\n".join([p["intro"], p[f"player_{player}"], p["payoff"][payoff], p["start"]])
    return text.format(**fmt) if fmt else text


def _image_content(provider: str, path: Path, question: str) -> list[dict]:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    if provider == "anthropic":
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}
    else:
        image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
    return [image, {"type": "text", "text": question}]


def _questions(rng: random.Random, root: Path) -> tuple[tuple, tuple, tuple]:
    """Player 1 images, player 2 images and answers, in one shuffled order. Files are sorted
    naturally by the counts in their names; player 2's counts are both +2, so the i-th files
    of the two folders are the same question."""
    d1, d2 = stimuli_dirs(root)
    natural = lambda p: [int(n) for n in re.findall(r"\d+", p.name)]  # noqa: E731
    pairs = [(a, b, correct_answer(a.name))
             for a, b in zip(sorted(d1.glob("*.png"), key=natural), sorted(d2.glob("*.png"), key=natural))]
    rng.shuffle(pairs)
    return tuple(zip(*pairs))


def play_run(model: str, instructions: str, payoff: str, rng: random.Random, *, chat: llm.ChatFn = llm.chat,
             root: Path = DATA) -> list[dict]:
    """One game (20 rounds) in the standard blue/yellow-circle condition."""
    spec = model_spec("exp1", model)
    caps = spec["max_tokens"]
    p = game_prompts()
    imgs1, imgs2, answers = _questions(rng, root)

    def play(player: int, images, follow_ups) -> tuple[list, list]:
        """One player's dialogue: per image, a guess, then one follow-up turn. Returns
        (parsed guesses, follow-up replies)."""
        history = [{"role": "user", "content": system_prompt(player, payoff)}]
        guesses, replies = [], []
        for image, follow_up in zip(images, follow_ups):
            history = llm.ask(chat, spec, history, _image_content(spec["provider"], image, p["question"]),
                              max_tokens=caps.get("guess"))
            guesses.append(parse_exp1_number(history[-1]["content"]))
            history = llm.ask(chat, spec, history, follow_up, max_tokens=caps.get("talk"))
            replies.append(history[-1]["content"])
        return guesses, replies

    p1_guess, p1_advice_msg = play(1, imgs1, [p["advice"][instructions]] * len(imgs1))
    p1_advice = [parse_exp1_number(m) for m in p1_advice_msg]

    info = 10 * ["spied"] + 10 * ["advice"]
    for _ in range(7):
        rng.shuffle(info)
    shown = [p1_guess[i] if kind == "spied" else p1_advice[i] for i, kind in enumerate(info)]
    before, update_msg = play(2, imgs2, [p["info"][kind].format(value=value) + p["update"][instructions]
                                         for kind, value in zip(info, shown)])
    after = [parse_exp1_number(m) for m in update_msg]

    return [dict(zip(RECORD_KEYS, row))
            for row in zip(answers, p1_guess, p1_advice, p1_advice_msg, before, info, after, update_msg)]


def collect(model: str, instructions: str, payoff: str, *, n: int = 30, seed: int | None = None,
            chat: llm.ChatFn = llm.chat, root: Path = DATA) -> Path:
    """Append runs until the results file holds ``n`` games (resumable)."""
    rng = random.Random(seed)
    return files.extend(results_path(model, instructions, payoff, root), n,
                        lambda _: play_run(model, instructions, payoff, rng, chat=chat, root=root))
