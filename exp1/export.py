"""Exp 1: game results -> CSVs in the Watson & Morgan human-data format (for analysis.Rmd).

Mirrors json_to_player2_csv.ipynb (LLM data) and clean_player_2.ipynb (human data).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from common.files import read_json
from . import DATA, game

NUM_ROUNDS = 20
DEFAULT_TIME = 5
DEFAULT_COMP_ATTEMPTS = 1
PAYOFF_ORDER = ("competitive", "cooperative")   # the original export's order


def player_two_rows(model: str, instructions: str, root: Path = DATA) -> list[dict]:
    """One row per Player-2 round, competitive games first, ids restarting per payoff
    condition (as in the original notebook)."""
    rows = []
    for payoff in PAYOFF_ORDER:
        runs = read_json(game.results_path(model, instructions, payoff, root))
        for pid, run in enumerate(runs):
            for j in range(NUM_ROUNDS):
                r = run[j]
                rows.append({
                    "Group": pid,
                    "participant_id": pid,
                    "Condition": payoff.capitalize(),
                    "Player": "B",
                    "Question": j,
                    "Comp_attemps": DEFAULT_COMP_ATTEMPTS,   # sic: column name as in the original export
                    "Time_spent_first": DEFAULT_TIME,
                    "First_answer": r["player_2_guess_before"],
                    "Time_spent_second": DEFAULT_TIME,
                    "Second_answer": r["player_2_guess_after"],
                    "Type": r["info_provided"].capitalize(),
                    "Social_info": r["player_1_guess"] if r["info_provided"] == "spied" else r["player_1_advice"],
                    "Result": "Correct" if r["player_2_guess_before"] == r["correct_answer"] else "Incorrect",
                })
    return rows


def write_player_two(model: str, instructions: str, out_dir: Path, root: Path = DATA) -> Path:
    rows = player_two_rows(model, instructions, root)
    out = out_dir / f"Player_two_{model}_{instructions}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out


def clean_human_player_two(raw_csv: Path, out_csv: Path) -> Path:
    """Watson & Morgan Player 2 data restricted to the Cooperative/Competitive conditions and
    the first 10 questions (the original notebook did this in place; here it writes a copy)."""
    df = pd.read_csv(raw_csv)
    df = df[(df["Condition"] != "Neutral") & (df["Question"] <= 10)]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv
