"""Audit of the stored Exp 2 scores against the rating each reply states explicitly. Changes no data.

Counts replies containing a bolded or line-start "Rating: N" / "Score: N" whose last such statement
differs from the stored (raw, pre-flip) score; see docs/PROVENANCE.md, section 5.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pandas as pd

from common.files import read_json

from . import DATA

STATED = re.compile(
    r"\*\*\s*(?:final\s+)?(?:rating|score)\s*:?\s*\**\s*(\d{1,3})\s*\**"   # **Rating: 70** / **Final Rating: 65**
    r"|^\s*(?:final\s+)?(?:rating|score)\s*:\s*(\d{1,3})\s*$",             # Rating: 70   (on its own line)
    re.IGNORECASE | re.MULTILINE)


def responses_dir(root: Path = DATA) -> Path:
    return root / "responses"


def scored_trials(root: Path = DATA) -> Iterator[tuple[Path, dict]]:
    """(response file, trial) for every stored Exp 2 trial that has a score."""
    for path in sorted(responses_dir(root).glob("*/*/*.json")):
        for subject in read_json(path):
            for trial in subject:
                if "score" in trial:
                    yield path, trial


def stated_rating(reply: str) -> int | None:
    """The last explicitly stated rating in a reply, if any."""
    found = [a or b for a, b in STATED.findall(reply)]
    return int(found[-1]) if found else None


def audit(root: Path = DATA) -> pd.DataFrame:
    """Per model: scored replies, replies stating a rating, and those whose stored score differs."""
    rows: dict[str, dict] = {}
    for path, trial in scored_trials(root):
        model = path.relative_to(responses_dir(root)).parts[0]
        row = rows.setdefault(model, {"model": model, "scores": 0, "stated": 0, "differ": 0, "first_example": ""})
        row["scores"] += 1
        stated = stated_rating(trial["messages"][-1]["content"])
        if stated is None:
            continue
        row["stated"] += 1
        if stated != trial["score"]:
            row["differ"] += 1
            row["first_example"] = row["first_example"] or (
                f"{path.relative_to(responses_dir(root))}: stored {trial['score']}, stated {stated}")
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r["model"]))
