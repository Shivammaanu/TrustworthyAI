"""Data-driven definition of a time gap (where the onboard logger stopped)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def otsu(values, bins: int = 256) -> float:
    """Otsu (1979): the threshold that best splits a 1-D distribution into two groups
    (maximises between-group variance)."""
    hist, edges = np.histogram(values, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(edges[1:][np.argmax(between[:-1])])


def gap_threshold(ts: pd.Series, configured="auto") -> float:
    """Seconds above which the interval between readings counts as a gap.
    'auto' = Otsu break on the log-intervals longer than the median interval."""
    if configured != "auto":
        return float(configured)
    dt = ts.diff().dt.total_seconds().dropna()
    irregular = dt[dt > dt.median()]
    return float(10 ** otsu(np.log10(irregular))) if len(irregular) else float("inf")


def block_ids(ts: pd.Series, gap_s: float) -> np.ndarray:
    """Consecutive gap-free stretches get the same id."""
    return (ts.diff().dt.total_seconds().fillna(0).to_numpy() > gap_s).cumsum()
