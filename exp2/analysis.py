"""Experiment 2 results (paper Section 4, Table 'exp_2_corrs1', Appendix B tables).

Per condition (model x scenario x perspective x answer format):
  * fit the Bayesian model to the LLM's own trust / incentive ratings (bayes.best_fit);
  * average the LLM's offer ratings and the model's posteriors within each of the 16
    character x incentive cells (posteriors rounded to 4 decimals, as the original
    aggregator read them from jsonlite JSON);
  * correlate those 16-cell profiles with each other and with the human means
    (data/human/oktar_cell_means.csv).
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from common import bayes
from common.config import model_spec, models
from common.design import conditions
from common.stats import fisher_mean, pearson
from . import DATA, convert, prompts

HUMAN_MEANS = DATA / "human" / "oktar_cell_means.csv"


@lru_cache(maxsize=1)
def human_means() -> pd.DataFrame:
    return pd.read_csv(HUMAN_MEANS)


@lru_cache(maxsize=None)
def fit(model: str, scenario: str, perspective: str, instructions: str) -> bayes.Fit:
    return bayes.best_fit(convert.trials(model, scenario, perspective, instructions))


def cell_profile(model: str, scenario: str, perspective: str, instructions: str) -> pd.DataFrame:
    """16 rows (character_index, incentive) with LLM mean, Bayes mean, human mean."""
    rows = fit(model, scenario, perspective, instructions).rows.copy()
    order = {c: i for i, c in enumerate(prompts.characters(scenario, perspective))}
    rows["character_index"] = rows["rec"].map(order)
    rows["prob"] = rows["prob"].round(4)              # jsonlite::write_json(digits = 4) in the original
    cells = bayes.cell_means(rows, ["character_index", "val"]).rename(columns={"val": "incentive"})
    cells["bayes"] = cells["bayes"].round(4)          # aggregator printed "{:.4f}" -> pasted into the sheet
    h = human_means()
    h = h[h.scenario == scenario][["character_index", "incentive", "human_mean"]]
    return cells.merge(h, on=["character_index", "incentive"], how="left")


def condition_metrics(model: str, scenario: str, perspective: str, instructions: str) -> dict:
    f = fit(model, scenario, perspective, instructions)
    cells = cell_profile(model, scenario, perspective, instructions)
    per_char = [pearson(g.llm, g.human_mean) for _, g in cells.groupby("character_index")]
    return {
        "model": model, "scenario": scenario, "perspective": perspective, "instructions": instructions,
        "n_subjects": int(f.rows["Subject"].nunique()),
        **f.params,
        "bayes_llm_trial": f.r,                                     # best-fit r across trials
        "bayes_llm_cells": pearson(cells.bayes, cells.llm),        # r across 16 cell means
        "bayes_human": pearson(cells.bayes, cells.human_mean),
        "llm_human": pearson(cells.llm, cells.human_mean),         # = cross-character calibration
        "intra_character": fisher_mean(per_char),                  # Fisher-z mean of 4 within-character r
    }


def all_conditions(model_list: list[str] | None = None) -> pd.DataFrame:
    """One row per model x answer format x perspective x scenario, in the paper's row order."""
    return pd.DataFrame([condition_metrics(m, scenario, perspective, instructions)
                         for m in model_list or models("exp2")
                         for instructions, perspective in conditions(model_spec("exp2", m))
                         for scenario in prompts.scenarios()])
