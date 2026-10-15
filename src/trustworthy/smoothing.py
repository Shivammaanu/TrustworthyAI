"""
Score smoothing before the conformal step (Phase 5, step 4; Hundman et al. 2018, sec. 3.2: exponentially weighted
average of the errors "to dampen spikes ... that frequently occur ... even when this behavior is normal").

Used here for a second effect: a leak's build-up is weak but persistent, so averaging accumulates its evidence while
isolated noise spikes are damped. The EWMA is causal (adjust=False: s_t = a*x_t + (1-a)*s_(t-1), a = 2/(span+1)) and
restarts at every data gap (segment). Calibration scores are smoothed the same way, so the conformal false-alarm
control is unchanged in form. The span is given in hours and converted to decisions with the detector's own rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma(df: pd.DataFrame, span_decisions: float, col: str = "score") -> np.ndarray:
    """Causal EWMA of df[col] within each segment, in time order (df must be sorted by decision_time)."""
    if span_decisions <= 1:
        return df[col].to_numpy(float)
    key = df["segment"].astype(str) + "_" + df["part"].astype(str) if "part" in df else df["segment"]
    return df.groupby(key, sort=False)[col].transform(lambda s: s.ewm(span=span_decisions, adjust=False).mean()).to_numpy()
