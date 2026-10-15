"""
Combining detectors with a controlled false-alarm budget (Phase 5, step 3).

Bonferroni on the rate scale: the operator's budget F (false alarms per day) is split evenly over the m detectors.
Detector k alarms when p_k < eps_k(F) / m, where eps_k(F) = F / (decisions per day of k) is its own level for F.
Each detector then contributes at most F/m expected false alarms per day, so the union contributes at most F,
whatever the dependence between detectors (Bonferroni is valid under any dependence; Fisher's combination is not
valid for conformal p-values, Bates et al. 2023). Each detector decides on its own grid (cycles or 10-min bins);
the union is formed on the common 10-min bin grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def combine(frames: dict, bins: pd.DataFrame, rule: str = "10min") -> pd.DataFrame:
    """frames: {model: conformal frame with decision_time, p, eps_<t>, part, frozen [, split]}.
    bins: the data's non-empty 10-min bins (bin_start, segment, frozen).
    The union is formed on this common grid: a bin is in alarm if ANY member flagged a decision falling in it.
    (Interleaving the members' decisions on one timeline would let one detector's unflagged decisions split another
    detector's alarm into many intervals - e.g. the LSTM-AE's flicker cycles inside frozen data.)"""
    m = len(frames)
    parts = []
    for model, f in frames.items():
        g = pd.DataFrame({"bin_start": pd.to_datetime(f["decision_time"] - pd.Timedelta(seconds=1)).dt.floor(rule),
                          "part": f["part"].to_numpy(), "p_adj": np.minimum(f["p"].to_numpy() * m, 1.0)})
        if "split" in f:
            g["split"] = f["split"].to_numpy()
        for c in [c for c in f.columns if c.startswith("eps_")]:      # ungated member flags; gate applied below
            g[f"flag_{c[4:]}"] = (f["p"] < f[c] / m).to_numpy()
        parts.append(g)
    dec = pd.concat(parts, ignore_index=True)
    flag_cols = [c for c in dec.columns if c.startswith("flag_")]
    agg = {c: "max" for c in flag_cols}
    agg.update(p_adj="min", part="max")
    if "split" in dec:
        agg["split"] = "first"
    grid = dec.groupby("bin_start").agg(agg)
    b = bins.set_index("bin_start")[["segment", "frozen"]]
    out = grid.join(b, how="inner").reset_index()           # decisions outside recorded bins are dropped
    for c in flag_cols:                                    # R1 gate on the common grid
        out[f"{c}_gated"] = out[c].astype(bool) & ~out["frozen"].astype(bool)
    out["decision_time"] = out["bin_start"] + pd.Timedelta(rule)
    out["score"] = -np.log10(out["p_adj"])          # for the threshold-free AD1 metric only
    return out.sort_values("decision_time").reset_index(drop=True)