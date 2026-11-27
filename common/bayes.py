"""Bayesian vigilance model: rating -> model inputs, posterior lookup, best-fitting parameters.

Pipeline (shared by Experiments 2 and 3):
  1. ``posterior_table()`` runs common/motivated_model.wppl once. It gives
     P(option is good | incentive, benevolence, alphaL0, alphaS, w) for every grid point.
  2. ``model_input()`` turns each of an LLM's three 0-100 ratings of a speaker into the model's
     inputs: trust -> benevolence (rec_vr), incentive -> utility (val_vr), quality -> 1-7 (value).
  3. ``best_fit()`` picks the grid point maximising Pearson r(value, posterior) across trials,
     exactly as the original best_bayesian_fit.R did (first maximum in grid order; incomplete
     pairs dropped).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import POSTERIOR_TABLE, WEBPPL_MODEL
from .stats import pearson_complete

PARAMS = ["alphaL0", "alphaS", "w"]


# ---------------------------------------------------------------- rating -> model inputs
def rescale_offer(x: float) -> float:
    """0-100 quality rating -> the 1-7 scale used by human participants."""
    return round((x / 100) * 6 + 1, 2)


def round_to_quartile(x: float) -> float:
    """0-100 incentive rating -> speaker utility grid {0, .25, .5, .75, 1}."""
    return round(round(x / 100 / 0.25) * 0.25, 2)


def round_to_decile(x: float, clamp: bool = True, rule: str = "grid") -> float:
    """0-100 trust rating -> benevolence grid {0, .1, ..., 1}.

    The original converters used two formulas, which differ only for ratings 5, 45 and 65
    (binary floating point decides the ties):
      "grid"           round(round(x/100/0.1)*0.1, 2)  ->  0.0, 0.4, 0.6   (converter from July 2025)
      "nearest_tenth"  round(x/100, 1)                 ->  0.1, 0.5, 0.7   (earlier converter)
    Exp 2 clamped to [0, 1]; the Exp 3 converter did not (``clamp=False``).
    """
    if rule == "grid":
        val = round(round(x / 100 / 0.1) * 0.1, 2)
    elif rule == "nearest_tenth":
        val = round(x / 100, 1)
    else:
        raise ValueError(f"unknown decile rule {rule!r}")
    return min(1.0, max(0.0, val)) if clamp else val


def model_input(purpose: str, rating: float, *, clamp: bool = True, rule: str = "grid") -> tuple[str, float]:
    """One un-flipped 0-100 rating -> (column, value) of the model's input for that rating purpose.
    ``clamp`` and ``rule`` apply to trust ratings (see round_to_decile)."""
    if purpose == "trust":
        return "rec_vr", round_to_decile(rating, clamp=clamp, rule=rule)
    if purpose == "reward_quality":
        return "val_vr", round_to_quartile(rating)
    if purpose == "offer_quality":
        return "value", rescale_offer(rating)
    raise ValueError(f"unknown rating purpose {purpose!r}")


# ---------------------------------------------------------------- posterior table
def posterior_table_path() -> Path:
    return POSTERIOR_TABLE


def build_posterior_table(out: Path | None = None) -> Path:
    """Run the WebPPL model (a few seconds) and write the posterior lookup table."""
    out = out or posterior_table_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["webppl", str(WEBPPL_MODEL), "--require", "webppl-json", "--", "--output", str(out)],
                   check=True, capture_output=True)
    return out


@lru_cache(maxsize=1)
def posterior_table() -> pd.DataFrame:
    path = posterior_table_path()
    if not path.exists():
        build_posterior_table(path)
    return pd.DataFrame(json.loads(path.read_text()))


# ---------------------------------------------------------------- fitting
@dataclass
class Fit:
    r: float                 # Pearson r(value, prob) at the best grid point
    params: dict             # {"alphaL0":…, "alphaS":…, "w":…}
    rows: pd.DataFrame       # the input trials with their posterior at the best grid point
    all_r: pd.Series         # r for every grid point (index = grid_index)


def attach_posteriors(trials: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross every trial with every grid point and look up its posterior ``prob``."""
    table = posterior_table() if table is None else table
    return trials.merge(table, on=["val_vr", "rec_vr"], how="left")


def best_fit(trials: pd.DataFrame, table: pd.DataFrame | None = None) -> Fit:
    """Grid point maximising r(value, prob); ties keep the earliest grid point."""
    joined = attach_posteriors(trials, table)
    all_r = pd.Series({gi: pearson_complete(g["value"], g["prob"])
                       for gi, g in joined.groupby("grid_index", sort=True)}).sort_index()
    best_gi, best_r = None, -np.inf
    for gi, r in all_r.items():          # strict '>' => first maximum wins, like the R script
        if not np.isnan(r) and r > best_r:
            best_gi, best_r = gi, r
    rows = joined[joined.grid_index == best_gi].drop(columns="grid_index").reset_index(drop=True)
    params = {p: rows[p].iloc[0] for p in PARAMS}
    return Fit(r=float(best_r), params=params, rows=rows, all_r=all_r)


def cell_means(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean LLM rating ('llm') and mean model posterior ('bayes') per design cell."""
    return rows.groupby(keys, sort=True).agg(llm=("value", "mean"), bayes=("prob", "mean")).reset_index()
