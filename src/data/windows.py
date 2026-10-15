"""Sliding windows that never cross a segment boundary (data gap or cycle-block break).
Positions before the segment start are filled by repeating its first sample
(replication padding, as in the TranAD code)."""
from __future__ import annotations

import numpy as np


def window_index(segment: np.ndarray, k: int) -> np.ndarray:
    n = len(segment)
    i = np.arange(n)
    seg_start = np.r_[0, np.flatnonzero(segment[1:] != segment[:-1]) + 1]
    first = seg_start[np.searchsorted(seg_start, i, side="right") - 1]
    idx = i[:, None] - np.arange(k)[::-1][None, :]
    return np.maximum(idx, first[:, None])


def make_windows(x: np.ndarray, segment: np.ndarray, k: int) -> np.ndarray:
    """(n, f) -> (n, k, f): window i ends at sample i."""
    return x[window_index(segment, k)]


def window_touches(flag: np.ndarray, segment: np.ndarray, k: int) -> np.ndarray:
    """True if any sample inside window i has flag set."""
    return flag[window_index(segment, k)].any(axis=1)
