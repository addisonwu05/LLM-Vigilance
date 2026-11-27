"""Experiment 1: seeded game replays, Player-2 exports vs. the original pipeline, the paper's numbers.

exp1_replays.json is a characterization golden captured from this package before its deduplication
pass: it pins the collector's complete request sequences.
"""
import json
import random

import pytest
from conftest import RequestHasher, load_golden, require, runs, sha256

from exp1 import DATA, analysis, export, game


@pytest.mark.parametrize("case", list(load_golden("exp1_replays.json")))
def test_seeded_game_replays(case):
    g = load_golden("exp1_replays.json")[case]
    fake = RequestHasher()
    run = game.play_run(*case.split("|"), random.Random(g["seed"]), chat=fake)
    assert (fake.n, fake.hexdigest()) == (g["n_requests"], g["requests_sha256"])
    assert sha256(json.dumps(run)) == g["run_sha256"]


@pytest.mark.parametrize("model,instr", runs("exp1"))
def test_player_two_export_matches_original(model, instr, tmp_path):
    ref = require(DATA / "reference" / f"Player_two_{model}_{instr}.csv", "the original pipeline wrote no file")
    assert export.write_player_two(model, instr, tmp_path).read_bytes() == ref.read_bytes()


def test_influence_numbers():
    """Section 3: humans shift 0.34 (advice) / 0.37 (spied); CoT shifts exceed 60% in 3 of 4 cases."""
    t = analysis.mean_shifts()
    human = t[t.model == "human"].set_index("Type")["mean"]
    assert (round(human["Advice"], 2), round(human["Spied"], 2)) == (0.34, 0.37)
    cot = t[t.instructions == "cot"]
    assert len(cot) == 4 and (cot["mean"] > 0.6).sum() == 3
