"""
Data-quality rules derived from the data (Phase 0 diagnostics), see DOCUMENTATION.md section 2a.

R1 frozen data   : a time bin in which every analog signal shows zero variation (the logger repeats stale
                   values; 9 periods, ~170 h, 7 of them far from any failure).
R2 running       : the compressor runs when COMP == 0 AND the motor draws load current. The load threshold is
                   the Otsu break of Motor current over the rows where COMP == 0 (COMP alone also flickers to 0
                   inside frozen periods with the motor off).
R3 episodes      : unreported fault episodes = real runs that are long, have high DV pressure and belong to the
                   hours-long class of the reported failures (three Otsu stages, see derive_episodes); the short
                   high-DV runs are kept as 'minor' evidence of a degraded state (Mar -> early Jun), not as windows. Backed by the operator's report of 21 faulty
                   windows, only 4 of them failures (Davari et al. 2024, citing Davari et al. 2021).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gaps import otsu
from .load import ANALOG


def frozen_bins(df: pd.DataFrame, rule: str = "10min") -> pd.Series:
    """Boolean per non-empty bin (index = bin start): all analog signals constant inside the bin."""
    x = df.set_index("timestamp")[ANALOG]
    r = x.resample(rule)
    cnt = x[ANALOG[0]].resample(rule).count()
    fz = ((r.max() - r.min()).abs().max(axis=1) == 0) & (cnt > 1)
    return fz[cnt > 0]


def frozen_rows(df: pd.DataFrame, fz: pd.Series, rule: str = "10min") -> np.ndarray:
    return fz.reindex(df["timestamp"].dt.floor(rule)).fillna(False).to_numpy(bool)


def frozen_periods(fz: pd.Series, rule: str = "10min") -> pd.DataFrame:
    """Merge consecutive frozen bins into periods [start, end)."""
    step = pd.Timedelta(rule)
    t = fz.index[fz.to_numpy(bool)]
    if len(t) == 0:
        return pd.DataFrame(columns=["start", "end", "hours"])
    new = np.r_[True, np.diff(t.values) != np.timedelta64(step)]
    gid = np.cumsum(new)
    g = pd.DataFrame({"t": t, "g": gid}).groupby("g")["t"].agg(["min", "max"])
    out = pd.DataFrame({"start": g["min"].values, "end": (g["max"] + step).values})
    out["hours"] = (out["end"] - out["start"]).dt.total_seconds() / 3600
    return out


def loaded_threshold(df: pd.DataFrame, signal: str = "COMP", value=0) -> float:
    return otsu(df.loc[df[signal] == value, "Motor_current"].to_numpy())


def running_mask(df: pd.DataFrame, rule: str, signal: str = "COMP", value=0, motor_thr: float | None = None):
    comp = (df[signal] == value).to_numpy()
    if rule == "comp":
        return comp
    if rule == "comp_and_motor":
        return comp & (df["Motor_current"].to_numpy() > motor_thr)
    raise ValueError(rule)


def runs_table(df: pd.DataFrame, running: np.ndarray, block: np.ndarray, frozen: np.ndarray) -> pd.DataFrame:
    """One row per contiguous running segment inside a gap-free block."""
    brk = np.r_[True, (running[1:] != running[:-1]) | (block[1:] != block[:-1])]
    seg = np.cumsum(brk)
    d = pd.DataFrame({"seg": seg, "run": running, "t": df["timestamp"].to_numpy(),
                      "dv": df["DV_pressure"].to_numpy(), "fz": frozen})
    d = d[d["run"]]
    r = d.groupby("seg").agg(start=("t", "first"), end=("t", "last"), dv_mean=("dv", "mean"), frozen_share=("fz", "mean"))
    r["minutes"] = (r["end"] - r["start"]).dt.total_seconds() / 60
    return r.reset_index(drop=True)


def derive_episodes(runs: pd.DataFrame, failures: pd.DataFrame, ph_h: float):
    """R3, in the same two stages as the Phase 0 analysis:
    stage 1  Otsu on log run length over all non-frozen runs separates short fragments (motor dips, flicker) from runs;
    stage 2  over the runs above that break: Otsu on log run length gives the 'long run' break and Otsu on mean
             DV pressure gives the 'high DV' break. Episodes are runs above both breaks, outside reported failures."""
    real = runs[(runs["frozen_share"] == 0) & (runs["minutes"] > 0)]
    frag_break = float(10 ** otsu(np.log10(real["minutes"])))
    proper = real[real["minutes"] > frag_break]
    len_break = float(10 ** otsu(np.log10(proper["minutes"])))
    dv_break = float(otsu(proper["dv_mean"]))
    cand = proper[(proper["minutes"] > len_break) & (proper["dv_mean"] > dv_break)].copy()
    keep = np.ones(len(cand), bool)
    for f in failures.itertuples():
        a, b = f.start - pd.Timedelta(hours=ph_h), f.maintenance
        keep &= ~((cand["start"] <= b) & (cand["end"] >= a)).to_numpy()
    cand = cand[keep].sort_values("start").reset_index(drop=True)
    # stage 3: the high-DV long runs split into hours-long 'major' episodes (the class of the reported failures)
    # and short 'minor' runs (an elevated-DV degraded state); Otsu on log duration separates them
    major_break = float(10 ** otsu(np.log10(cand["minutes"]))) if len(cand) > 2 else 0.0
    cand["major"] = cand["minutes"] > major_break
    cand["name"] = ""
    cand.loc[cand["major"], "name"] = [f"E{i + 1:02d}" for i in range(int(cand["major"].sum()))]
    cand.loc[~cand["major"], "name"] = [f"m{i + 1:02d}" for i in range(int((~cand["major"]).sum()))]
    return cand, {"fragment_break_min": frag_break, "n_runs_above_fragment_break": int(len(proper)),
                  "run_length_break_min": len_break, "dv_pressure_break_bar": dv_break,
                  "major_episode_break_min": major_break,
                  "n_major": int(cand["major"].sum()), "n_minor": int((~cand["major"]).sum())}


def event_windows(failures: pd.DataFrame, episodes: pd.DataFrame, company_csv=None) -> pd.DataFrame:
    """One table of all event windows: reported failures, derived episodes and (optionally) the operator's list."""
    rows = [{"name": f.name, "start": f.start, "end": f.eval_end, "source": "failure"} for f in failures.itertuples()]
    rows += [{"name": e.name, "start": e.start, "end": e.end, "source": "derived"}
             for e in episodes.itertuples() if e.major]        # minor high-DV runs are listed separately, not windows
    if company_csv is not None:
        c = pd.read_csv(company_csv, parse_dates=["start", "end"])
        rows += [{"name": r.name, "start": r.start, "end": r.end, "source": "company"} for r in c.itertuples()]
    return pd.DataFrame(rows)