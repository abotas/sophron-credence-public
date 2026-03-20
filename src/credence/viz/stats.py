"""Statistical helpers for the visualization app."""

from collections.abc import Callable

import numpy as np
from scipy import stats as sp_stats

from credence.viz.constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    BUCKET_RANGES,
)


def bootstrap_ci(
    values: list[float],
    stat_fn: Callable[[np.ndarray], float],
) -> tuple[float, float, float]:
    """Compute (point_estimate, ci_low, ci_high) via bootstrap percentile method."""
    arr = np.array(values)
    point = float(stat_fn(arr))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.array([
        float(stat_fn(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(BOOTSTRAP_N)
    ])
    alpha = (100 - BOOTSTRAP_CI_LEVEL) / 2
    lo = float(np.percentile(boot, alpha))
    hi = float(np.percentile(boot, 100 - alpha))
    return point, lo, hi


def bootstrap_mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Bootstrap CI for the mean."""
    return bootstrap_ci(values, np.mean)


def bootstrap_pass_rate_ci(passes: list[bool]) -> tuple[float, float, float]:
    """Bootstrap CI for a pass rate (list of booleans)."""
    return bootstrap_ci([float(p) for p in passes], np.mean)


def in_bucket(value: float, bucket_name: str) -> bool:
    """Check if a credence value falls within the named calibration bucket."""
    lo, hi = BUCKET_RANGES[bucket_name]
    return lo <= value <= hi


def fisher_z_ci(r: float, n: int) -> tuple[float, float]:
    """95% confidence interval for Pearson r via Fisher z-transform."""
    if n < 4:
        return (r, r)
    z = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
    se_z = 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se_z)), float(np.tanh(z + 1.96 * se_z))


def sig_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_shift(shift: float) -> str:
    """Format a shift value with color (green positive, red negative)."""
    if abs(shift) < 0.005:
        return ""
    color = "#2ca02c" if shift > 0 else "#d62728"
    return f' <span style="color:{color}">({shift:+.2f})</span>'


def wilcoxon_one_sided(values: list[float]) -> tuple[float, float]:
    """One-tailed Wilcoxon signed-rank test (greater than 0).

    Returns (statistic, p_value).
    """
    arr = np.array(values)
    result = sp_stats.wilcoxon(arr, alternative="greater")
    return float(result.statistic), float(result.pvalue)
