"""
Alarm triage with a guaranteed risk (Phase 5, human-oversight pillar; Geifman & El-Yaniv 2017, Algorithm 1 'SGR').

Selective classifier (f, g): f = 'this alarm is a real leak', g = confirm automatically if kappa >= theta, otherwise
send to engineer review. Coverage = share of alarms confirmed; selective risk = share of confirmed alarms that are
false. SGR searches theta by bisection over the sorted kappas (ceil(log2 m) steps) and returns the most permissive theta
whose binomial tail bound b* on the selective risk is below r*:
    b* = B*(r_hat, delta / ceil(log2 m), S_theta) = largest b with P(Bin(n, b) <= errors) >= delta'
(the exact Clopper-Pearson upper bound, beta.ppf(1 - delta', errors + 1, n - errors)). Then
P(selective risk > r*) < delta for alarms drawn like the labelled ones.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import beta


def alarm_table(df: pd.DataFrame, flag_col: str, p_col: str = "p_adj") -> pd.DataFrame:
    """One row per alarm interval (consecutive flagged decisions within a segment): start, end, evidence =
    -log10 min p, duration_h. Confidence functions kappa are in KAPPAS."""
    f = df[flag_col].astype(bool).to_numpy()
    if not f.any():
        return pd.DataFrame(columns=["start", "end", "evidence", "duration_h", "kappa", "split"])
    seg = df["segment"].to_numpy()
    new = f & ~np.r_[False, f[:-1] & (seg[1:] == seg[:-1])]
    gid = np.cumsum(new)
    x = df.loc[f].assign(_g=gid[f])
    agg = {"start": ("decision_time", "min"), "end": ("decision_time", "max"), "p_min": (p_col, "min")}
    if "split" in x:
        agg["split"] = ("split", "first")
    agg["n_decisions"] = ("decision_time", "size")
    t = x.groupby("_g").agg(**agg).reset_index(drop=True)
    t["evidence"] = -np.log10(t["p_min"].clip(lower=1e-12))
    t["duration_h"] = (t["end"] - t["start"]).dt.total_seconds() / 3600 + 1 / 6
    t["kappa"] = t["evidence"]
    return t


KAPPAS = {
    # strongest p-value of the alarm; ties at the floor 1/(n+1) for every strong alarm
    "evidence": lambda t: t["evidence"].to_numpy(float),
    # how long the alarm lasts (a leak keeps the score up for hours), evidence breaks ties
    "duration": lambda t: t["duration_h"].to_numpy(float) + 1e-3 * t["evidence"].to_numpy(float),
}


def risk_bound(errors: int, n: int, delta: float) -> float:
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, n - errors))


def sgr(kappa, is_error, r_star: float, delta: float) -> dict:
    """Geifman & El-Yaniv Algorithm 1. Returns theta (np.inf = confirm nothing), the bound, coverage and risk."""
    k = np.asarray(kappa, float)
    e = np.asarray(is_error, bool)
    m = len(k)
    if m == 0:
        return {"theta": np.inf, "bound": 1.0, "coverage": 0.0, "risk": np.nan, "m": 0}
    order = np.argsort(k)
    ks, es = k[order], e[order]
    steps = max(1, math.ceil(math.log2(m)))
    d = delta / steps
    lo, hi = 0, m - 1                     # candidate thresholds = ks[i]; accept alarms with kappa >= ks[i]
    best = None
    for _ in range(steps):
        mid = (lo + hi) // 2
        acc = es[mid:]
        b = risk_bound(int(acc.sum()), len(acc), d)
        if b < r_star:
            best = (mid, b)
            hi = mid                      # bound satisfied -> try more coverage (lower threshold)
        else:
            lo = mid + 1                  # not satisfied -> stricter threshold
        if lo >= hi:
            break
    if best is None:                      # last candidate check
        acc = es[lo:]
        b = risk_bound(int(acc.sum()), len(acc), d)
        if b < r_star:
            best = (lo, b)
    if best is None:
        return {"theta": np.inf, "bound": 1.0, "coverage": 0.0, "risk": np.nan, "m": m}
    i, b = best
    acc = es[i:]
    return {"theta": float(ks[i]), "bound": b, "coverage": len(acc) / m, "risk": float(acc.mean()), "m": m}
