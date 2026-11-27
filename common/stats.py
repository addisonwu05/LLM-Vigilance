"""Statistics used by the analyses."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def pearson(x, y) -> float:
    """Pearson r via numpy.corrcoef, as the original notebooks computed it. NaN (without a
    warning) when r is undefined: fewer than two points, or a constant input."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(x, y)[0, 1])


def pearson_complete(x, y) -> float:
    """R's cor(x, y, use = 'complete.obs'): pairs containing a NaN are dropped first.

    The best fit uses this exact arithmetic, because near-ties between grid points are decided
    by it (as in the original best_bayesian_fit.R).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    den = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    return float((xm * ym).sum() / den) if den > 0 else float("nan")


def fisher_mean(rs) -> float:
    """Average correlations on the Fisher-z scale: tanh(mean(atanh(r))). NaN propagates."""
    return float(np.tanh(np.mean(np.arctanh(np.asarray(list(rs), float)))))


def fisher_z_test(r1: float, n1: int, r2: float, n2: int, alternative: str = "greater") -> tuple[float, float]:
    """Compare two independent correlations via Fisher's r-to-z.

    Returns (z, p) for H1: r2 > r1 (``alternative='greater'``) or r2 != r1 ('two-sided').
    """
    z1, z2 = math.atanh(r1), math.atanh(r2)
    se = math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z2 - z1) / se
    if alternative == "greater":
        p = 1 - stats.norm.cdf(z)
    elif alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        raise ValueError(alternative)
    return z, float(p)
