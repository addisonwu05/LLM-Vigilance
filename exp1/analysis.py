"""Experiment 1 headline statistic: how far Player 2 moves toward Player 1's information.

shift = (second answer - first answer) / (Player 1's number - first answer), restricted to
(-0.5, 1.5), averaged per model, answer format and information type (advice vs. spied).
``shifts()`` is the single definition: `vigilance exp1 export` writes it to
exp1/outputs/shifts.csv, which analysis.Rmd reads for its figures and tests.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.config import label, model_spec, models
from . import DATA, export

HUMAN_PLAYER_TWO = Path("human") / "watson_morgan_player_two.csv"
INFO_TYPES = ["Advice", "Spied"]
SHIFT_COLUMNS = ["model", "label", "instructions", "Condition", "Type", "Result", "shift"]   # outputs/shifts.csv


def player_two(root: Path = DATA) -> pd.DataFrame:
    """Every Player-2 round: each LLM x answer format, then the human participants."""
    frames = [pd.DataFrame(export.player_two_rows(m, i, root)).assign(model=m, label=label(m), instructions=i)
              for m in models("exp1") for i in model_spec("exp1", m)["instructions"]]
    frames.append(pd.read_csv(root / HUMAN_PLAYER_TWO).assign(model="human", label="Humans", instructions="-"))
    return pd.concat(frames, ignore_index=True)


def shifts(root: Path = DATA) -> pd.DataFrame:
    df = player_two(root)
    df["shift"] = (df.Second_answer - df.First_answer) / (df.Social_info - df.First_answer)
    # comparisons also drop 0/0 (NaN) and x/0 (inf), like dplyr::filter in the Rmd
    return df[(df["shift"] < 1.5) & (df["shift"] > -0.5) & df.Type.isin(INFO_TYPES)]


def mean_shifts(root: Path = DATA) -> pd.DataFrame:
    return (shifts(root).groupby(["model", "instructions", "Type"], sort=False)["shift"]
            .agg(mean="mean", n="count").reset_index())
