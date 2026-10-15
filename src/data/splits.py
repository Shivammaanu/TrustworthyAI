"""Chronological splits. Training data never contains anything from PH hours before a failure
until its maintenance (problem statement: pre-failure behaviour must not leak into 'normal')."""
from __future__ import annotations

import numpy as np
import pandas as pd


def exclusion_intervals(failures: pd.DataFrame, ph_h: float):
    return [(f.start - pd.Timedelta(hours=ph_h), f.maintenance) for f in failures.itertuples()]


def in_intervals(times, intervals) -> np.ndarray:
    t = pd.to_datetime(pd.Series(times)).to_numpy()
    m = np.zeros(len(t), bool)
    for a, b in intervals:
        m |= (t >= np.datetime64(a)) & (t <= np.datetime64(b))
    return m


def make_parts(scheme: str, failures: pd.DataFrame, ph_h: float, data_end) -> list[dict]:
    """fixed: one model trained before F1 tests everything after.
    rolling (walk-forward): model k is trained on everything before cut k and tests (cut k, cut k+1]."""
    first_cut = failures.iloc[0].start - pd.Timedelta(hours=ph_h)
    if scheme == "fixed":
        cuts = [first_cut]
    elif scheme == "rolling":
        cuts = [first_cut] + list(failures.maintenance.iloc[:-1])
    else:
        raise ValueError(scheme)
    ends = cuts[1:] + [pd.Timestamp(data_end)]
    return [{"scheme": scheme, "part": i, "cut": c, "test_end": e} for i, (c, e) in enumerate(zip(cuts, ends))]


def masks(times, part: dict, exclusions, val_fraction: float, invalid=None):
    """Boolean train / val / test masks over decision times. `invalid` marks samples that must not be
    used for training (e.g. windows that reach back into an excluded period)."""
    t = pd.to_datetime(pd.Series(times)).to_numpy()
    cut, end = np.datetime64(part["cut"]), np.datetime64(part["test_end"])
    usable = (t < cut) & ~in_intervals(times, exclusions)
    if invalid is not None:
        usable &= ~invalid
    idx = np.flatnonzero(usable)
    n_val = int(round(val_fraction * len(idx)))
    train = np.zeros(len(t), bool); val = np.zeros(len(t), bool)
    train[idx[: len(idx) - n_val]] = True
    val[idx[len(idx) - n_val:]] = True
    test = (t > cut) & (t <= end)
    return train, val, test
