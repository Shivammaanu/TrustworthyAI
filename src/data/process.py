"""Turn raw rows into both model inputs (cycles under both running rules, 10-minute bins) with frozen flags.
Used for the synthetic benchmark copies, with exactly the rules derived on the real data (step 01)."""
from __future__ import annotations

import pandas as pd

from . import quality as q
from .cycles import cycle_features, segment_cycles
from .resample import resample

RULES = ["comp", "comp_and_motor"]


def representations(df, block, motor_thr, dc) -> dict:
    fz_bins = q.frozen_bins(df, dc["frozen_bin"])
    frozen = q.frozen_rows(df, fz_bins, dc["frozen_bin"])
    out = {}
    for rule in RULES:
        running = q.running_mask(df, rule, dc["running_signal"], dc["running_value"], motor_thr)
        cid, cycles = segment_cycles(df, running, block)
        feats = cycle_features(df, cid, cycles, running, dc["cycle_bins"])
        cyc = cycles.set_index("cycle").join(feats).reset_index()
        m = cid >= 0
        cyc["frozen"] = pd.Series(frozen[m]).groupby(cid[m]).max().reindex(cyc["cycle"]).to_numpy(bool)
        out[f"cycles_{rule}"] = cyc
    res = resample(df, dc["resample"])
    res["frozen"] = fz_bins.reindex(res["bin_start"]).fillna(False).to_numpy(bool)
    out["resampled"] = res
    return out