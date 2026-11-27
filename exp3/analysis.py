"""Experiment 3 results (paper Section 5, Tables 'steer' and 'length', Appendix C).

The reported correlation for a model x prompt condition is the best Pearson r, over the
WebPPL parameter grid, between the model's product-quality rating and the Bayesian
posterior computed from its own trust and incentive ratings (one row per video).
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from common import bayes
from common.config import model_spec, models
from common.design import conditions
from common.stats import fisher_z_test, pearson
from . import collect, convert

ADDITIONAL_STEERS = ("none", "incentives", "gricean", "bias")   # the appendix table's column order
QUARTILES = ("Q1", "Q2", "Q3", "Q4")


@lru_cache(maxsize=None)
def fit(model: str, steer: str, perspective: str, instructions: str) -> bayes.Fit:
    return bayes.best_fit(convert.trials(model, steer, perspective, instructions))


def _runs(model_list=None):
    """(model, instructions, perspective) for every condition, in the paper's row order."""
    return [(m, i, p) for m in model_list or models("exp3") for i, p in conditions(model_spec("exp3", m))]


def steering_table(model_list=None, steer: str = "incentives") -> pd.DataFrame:
    """Table 'youtube_steering': default vs steering prompt, with a one-sided Fisher z-test."""
    rows = []
    for model, instructions, perspective in _runs(model_list):
        d, s = fit(model, "none", perspective, instructions), fit(model, steer, perspective, instructions)
        z, p = fisher_z_test(d.r, len(d.rows), s.r, len(s.rows), alternative="greater")
        rows.append({"model": model, "instructions": instructions, "perspective": perspective,
                     "default": d.r, steer: s.r, "n_default": len(d.rows), f"n_{steer}": len(s.rows),
                     "z": z, "p_one_sided": p})
    return pd.DataFrame(rows)


def additional_steers_table(model: str = "gpt-4o") -> pd.DataFrame:
    """Appendix table 'youtube_steering_additional' (GPT-4o only)."""
    return pd.DataFrame([{"instructions": instructions, "perspective": perspective,
                          **{steer: fit(model, steer, perspective, instructions).r for steer in ADDITIONAL_STEERS}}
                         for _, instructions, perspective in _runs([model])])


def word_counts(videos: list[dict] | None = None) -> dict[str, int]:
    videos = videos if videos is not None else collect.load_videos()
    return {v["video_id"]: (len(v["video_sponsorship_transcript"].split()) if v["video_sponsorship_transcript"] else 0)
            for v in videos if "video_id" in v and "video_sponsorship_transcript" in v}


def length_table(model_list=None, steer: str = "incentives") -> pd.DataFrame:
    """Table 'model_correlation_quartiles': r within transcript-length quartiles.

    Replicates add_word_counts.ipynb: posteriors rounded to 4 decimals (they were read
    back from jsonlite JSON), pandas.qcut quartiles, numpy.corrcoef per quartile. Where the
    model gave the same rating to every video in a quartile, r is mathematically
    undefined and numpy returns floating-point noise (~1e-16), as printed in the paper.
    """
    wc = word_counts()
    out = []
    for model, instructions, perspective in _runs(model_list):
        rows = fit(model, steer, perspective, instructions).rows.copy()
        rows["prob"] = rows["prob"].round(4)
        rows["word_count"] = rows["video_id"].map(wc).fillna(0)
        rows = rows.dropna(subset=["value", "prob", "word_count"])
        rows["quartile"] = pd.qcut(rows["word_count"], 4, labels=list(QUARTILES))
        for q in QUARTILES:
            sub = rows[rows["quartile"] == q]
            out.append({"model": model, "instructions": instructions, "perspective": perspective,
                        "quartile": q, "n": len(sub), "r": pearson(sub["value"], sub["prob"]),
                        "constant_rating": bool(sub["value"].nunique() <= 1),
                        "constant_prediction": bool(sub["prob"].nunique() <= 1)})
    return pd.DataFrame(out)
