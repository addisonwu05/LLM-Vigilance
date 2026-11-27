"""Paper tables for Experiment 2, built from analysis.all_conditions().

How the paper's numbers are formed (reverse-engineered from the analysis spreadsheet and
verified in tests/test_exp2.py):
  1. Per scenario, correlate 16-cell profiles (character x incentive means):
     Bayesian-LLM (posterior means vs LLM offer ratings), Bayesian-Human, LLM-Human.
     The posterior cell means are rounded to 4 decimals first, because they were pasted
     into the sheet from the aggregator's "{:.4f}" printout.
  2. Combine the three scenarios with a Fisher-z mean: tanh(mean(atanh(r))).
  3. Table 1 (per model): plain mean of that model's rows from step 2 (o1 Bayesian-LLM:
     (0.763 + 0.646) / 2 = 0.705 as printed; a Fisher-z mean would give 0.709).
A correlation is undefined (NaN) when a profile is constant, e.g. a best fit whose posterior
is the same in every cell. Such a NaN propagates through step 2, since the printed numbers average
all scenarios; step 3 skips it.
"""
from __future__ import annotations

import pandas as pd

from common.config import label
from common.stats import fisher_mean

METRICS = {"bayes_llm_cells": "Bayesian-LLM", "bayes_human": "Bayesian-Human", "llm_human": "LLM-Human"}
ALIGNMENT = {"intra_character": "intra_character", "llm_human": "cross_character"}


def _across_scenarios(per_condition: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """One row per model x answer format x perspective: the Fisher-z mean over the scenarios of
    each column in ``columns`` (renamed to its value)."""
    return pd.DataFrame([
        {"model": label(model), "instructions": instructions, "perspective": perspective,
         **{name: fisher_mean(g[col]) for col, name in columns.items()}}
        for (model, instructions, perspective), g in per_condition.groupby(
            ["model", "instructions", "perspective"], sort=False)])


def table_full(per_condition: pd.DataFrame) -> pd.DataFrame:
    """Appendix table 'exp_2_corrs_full': one row per model x answer format x perspective."""
    return _across_scenarios(per_condition, METRICS)


def table_main(per_condition: pd.DataFrame) -> pd.DataFrame:
    """Table 'exp_2_corrs1': one row per model, plain mean over its prompt conditions.
    A NaN row is skipped here (pandas' Series.mean, which the original np.mean call dispatched to)."""
    return pd.DataFrame([{"model": model, **{name: float(g[name].mean()) for name in METRICS.values()}}
                         for model, g in table_full(per_condition).groupby("model", sort=False)])


def table_alignment(per_condition: pd.DataFrame) -> pd.DataFrame:
    """Appendix table 'exp2_prompting_alignment': intra-character alignment (Fisher-z mean of
    the 4 characters x 3 scenarios within-character LLM-human correlations, each over the 4
    incentive levels) and cross-character calibration (= the LLM-Human column)."""
    return _across_scenarios(per_condition, ALIGNMENT)
