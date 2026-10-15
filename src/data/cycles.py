"""
Compressor-cycle segmentation and the 44 cycle features of Davari et al. (SAC 2024, Table 2).

A cycle = one run period (compressor pumping) followed by one idle period, inside one gap-free
block. Running is defined from the data (Phase 0): running <=> COMP == 0.
A cycle is only known to be complete when the next run starts, so that moment is its decision time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BINNED = ["TP2", "TP3", "H1", "Motor_current"]          # 7 bins each
DIGITAL = ["COMP", "DV_electric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]


def segment_cycles(df: pd.DataFrame, running: np.ndarray, block: np.ndarray):
    """Returns (cycle id per row, -1 if not in a complete cycle; table of complete cycles)."""
    n = len(df)
    prev_run = np.r_[False, running[:-1]]
    new_block = np.r_[True, block[1:] != block[:-1]]
    run_start = running & (~prev_run | new_block)
    starts = np.flatnonzero(run_start)
    if len(starts) < 2:
        return np.full(n, -1), pd.DataFrame()
    # cycle k spans rows [starts[k], starts[k+1]) and is complete only if both starts share a block
    same_block = block[starts[:-1]] == block[starts[1:]]
    cid = np.full(n, -1)
    ts = df["timestamp"].to_numpy()
    rows = []
    k_valid = 0
    for k in np.flatnonzero(same_block):
        a, b = starts[k], starts[k + 1]
        seg_running = running[a:b]
        idle_rel = np.flatnonzero(~seg_running)
        if len(idle_rel) == 0:
            continue
        cid[a:b] = k_valid
        rows.append({"cycle": k_valid, "start": ts[a], "idle_start": ts[a + idle_rel[0]],
                     "decision_time": ts[b], "block": block[a], "n_rows": b - a})
        k_valid += 1
    return cid, pd.DataFrame(rows)


def cycle_features(df: pd.DataFrame, cid: np.ndarray, cycles: pd.DataFrame, running: np.ndarray, n_bins: int = 7):
    """44 features per cycle: 7 time-bins (mean) for TP2, TP3, H1, Motor current; oil temperature
    min / max / mean-in-run (MA1) / mean-in-idle (MA2); median DV pressure; median Reservoirs;
    run and idle durations (s); count of ones for the 8 digital signals."""
    m = cid >= 0
    d = df.loc[m].copy()
    d["cycle"] = cid[m]
    d["running"] = running[m]
    t = d["timestamp"].astype("int64").to_numpy() / 1e9
    g_first = d.groupby("cycle")["timestamp"].transform("first").astype("int64").to_numpy() / 1e9
    g_last = d.groupby("cycle")["timestamp"].transform("last").astype("int64").to_numpy() / 1e9
    d["bin"] = np.minimum(((t - g_first) / np.maximum(g_last - g_first, 1e-9) * n_bins).astype(int), n_bins - 1)

    parts = []
    binned = d.groupby(["cycle", "bin"])[BINNED].mean().unstack("bin")
    binned = binned.reindex(columns=pd.MultiIndex.from_product([BINNED, range(n_bins)]))
    for s in BINNED:                       # short cycles can leave a bin empty: fill from the neighbouring bins
        blk = binned[s].ffill(axis=1).bfill(axis=1)
        blk.columns = [f"{s}_bin{i + 1}" for i in range(n_bins)]
        parts.append(blk)
    oil = d.groupby("cycle")["Oil_temperature"].agg(["min", "max"]).add_prefix("Oil_")
    oil["Oil_MA1"] = d[d["running"]].groupby("cycle")["Oil_temperature"].mean()
    oil["Oil_MA2"] = d[~d["running"]].groupby("cycle")["Oil_temperature"].mean()
    oil["Oil_MA1"] = oil["Oil_MA1"].fillna(oil["Oil_MA2"]); oil["Oil_MA2"] = oil["Oil_MA2"].fillna(oil["Oil_MA1"])
    parts.append(oil)
    parts.append(d.groupby("cycle")[["DV_pressure", "Reservoirs"]].median().add_suffix("_median"))
    c = cycles.set_index("cycle")
    dur = pd.DataFrame({"T_run": (c["idle_start"] - c["start"]).dt.total_seconds(),
                        "T_idle": (c["decision_time"] - c["idle_start"]).dt.total_seconds()})
    parts.append(dur)
    parts.append(d.groupby("cycle")[DIGITAL].sum().add_prefix("N_"))
    feats = pd.concat(parts, axis=1).sort_index()
    assert feats.shape[1] == 4 * n_bins + 4 + 2 + 2 + 8, feats.shape
    return feats
