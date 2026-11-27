"""Experiment 3 responses -> one row per video with the model's three ratings as model inputs.

Mirrors webppl_compatible_json_converter.py followed by remove_nulls.py: offer, reward and trust
files are merged by video_id (offer first, so row order follows the offer file); unparseable
scores are skipped; videos missing any of the three ratings are dropped. Trust is put on the
decile grid WITHOUT clamping, as in the original.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.bayes import model_input
from common.files import read_json
from . import DATA, collect

COLUMNS = ["video_id", "val_vr", "rec_vr", "value"]


def trials(model: str, steer: str, perspective: str, instructions: str, root: Path = DATA) -> pd.DataFrame:
    combined: dict[str, dict] = {}
    for purpose in ("offer_quality", "reward_quality", "trust"):        # the original's order
        for item in read_json(collect.response_path(steer, model, purpose, perspective, instructions, root)):
            vid, raw = item.get("video_id"), item.get("flipped_score")
            if vid and raw is not None:
                field, value = model_input(purpose, raw, clamp=False)
                combined.setdefault(vid, {"video_id": vid})[field] = value
    return pd.DataFrame([r for r in combined.values() if len(r) == len(COLUMNS)], columns=COLUMNS)
