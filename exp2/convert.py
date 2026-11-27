"""Experiment 2 responses -> one row per (subject, speaker) with the model's three ratings.

Matches the original webppl_json_converter.py: the i-th subject of the trust file, the
reward file and the offer file are merged into one 'subject' (they were independent
dialogues; that pairing is how the paper's analysis was done). One row per reward-file
trial, in stored order, with the three ratings as model inputs (bayes.model_input; the trust
rounding formula is per model, ``trust_rounding`` in models.yaml).
Scores are un-flipped from (score, flip). Older files only stored `score`, newer ones
also `flipped_score`; both give the same number (checked in the tests).
(The earlier converter listed rows in offer-file order; row order affects no statistic.)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.bayes import model_input
from common.config import model_spec
from common.design import PURPOSES
from common.files import read_json
from common.parsing import unflip
from . import DATA, collect


def unflipped(trial: dict) -> int:
    if "flipped_score" in trial:
        return trial["flipped_score"]
    return unflip(trial["score"], trial.get("flip", False))


def trials(model: str, scenario: str, perspective: str, instructions: str, root: Path = DATA) -> pd.DataFrame:
    """Model inputs for one condition: Subject, rec (character), val (incentive), rec_vr, val_vr, value."""
    trust, reward, offer = (read_json(collect.response_path(model, scenario, purpose, perspective, instructions, root))
                            for purpose in PURPOSES)
    rule = model_spec("exp2", model)["trust_rounding"]
    rows = []
    for subj in range(len(trust)):
        offer_by = {(t["person"], t.get("reward")): t for t in offer[subj]}
        trust_by = {t["person"]: t for t in trust[subj]}
        for r in reward[subj]:
            o, t = offer_by.get((r["person"], r.get("reward"))), trust_by.get(r["person"])
            if not o or not t:
                continue
            row = {"Subject": subj, "rec": r["person"], "val": r.get("reward")}
            for purpose, trial in zip(PURPOSES, (t, r, o)):
                field, value = model_input(purpose, unflipped(trial), rule=rule)
                row[field] = value
            rows.append(row)
    return pd.DataFrame(rows)
