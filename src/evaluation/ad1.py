"""
Benchmark metric of Papadopoulos et al. (SIGMOD 2026), so our baselines can be compared with theirs.
For each failure (start time f): predictive horizon PH = [f - L - PH, f - L), lead window L = [f - L, f),
with PH = 48 h and L = 2 h. Alarms inside L are disregarded (too late to act).
  precision_AD1 = flagged decisions inside a PH / all flagged decisions considered
  recall_AD1    = failures with >= 1 flagged decision inside their PH / failures
AD1 AUC-PR = area under the (recall, precision) curve over a sweep of thresholds.
Deviation from the benchmark (documented): decisions from failure start until maintenance are also
left out, because the failure period is not normal operation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ad1_regions(times, failures: pd.DataFrame, ph_h: float, lead_h: float, reference: str = "start"):
    """reference='start': PH ends L hours before the failure starts.
    reference='end': PH ends L hours before the failure ends (the window then covers the failure itself)."""
    t = pd.to_datetime(pd.Series(times)).to_numpy()
    ph, L = pd.Timedelta(hours=ph_h), pd.Timedelta(hours=lead_h)
    in_ph = np.zeros(len(t), bool); ignore = np.zeros(len(t), bool); which = np.full(len(t), -1)
    for i, fl in enumerate(failures.itertuples()):
        f = fl.start if reference == "start" else fl.eval_end
        m = (t >= np.datetime64(f - L - ph)) & (t < np.datetime64(f - L))
        in_ph |= m; which[m] = i
        ignore |= (t >= np.datetime64(f - L)) & (t <= np.datetime64(fl.maintenance))
    return in_ph, ignore, which


def ad1_point(flags, in_ph, ignore, which, n_failures):
    f = np.asarray(flags, bool) & ~ignore
    prec = (f & in_ph).sum() / f.sum() if f.sum() else np.nan
    rec = len(np.unique(which[f & in_ph])) / n_failures if n_failures else np.nan
    return prec, rec


def ad1_auc_pr(scores, times, failures, ph_h, lead_h, n_thresholds=200, reference="start"):
    s = np.asarray(scores, float)
    in_ph, ignore, which = ad1_regions(times, failures, ph_h, lead_h, reference)
    present = sorted(set(which[which >= 0]))
    n_f = len(present)
    if n_f == 0:
        return np.nan, pd.DataFrame()
    ths = np.unique(np.quantile(s[~ignore], np.linspace(0, 1, n_thresholds)))[::-1]
    pts = [(*ad1_point(s >= th, in_ph, ignore, which, n_f), th) for th in ths]
    curve = pd.DataFrame(pts, columns=["precision", "recall", "threshold"]).dropna()
    curve = curve.sort_values(["recall", "precision"], ascending=[True, False]).drop_duplicates("recall")
    r = np.r_[0.0, curve["recall"].to_numpy()]
    p = curve["precision"].to_numpy()
    return float(np.sum((r[1:] - r[:-1]) * p)), curve
