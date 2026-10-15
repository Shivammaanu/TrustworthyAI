"""
Conformal anomaly detection (Phase 5, uncertainty pillar).

p-value (Laxhammar & Falkman 2014, ICAD; Bates et al. 2023, eq. 3, written for 'higher score = more anomalous'):
    p(s) = (1 + #{calibration scores >= s}) / (n + 1)
Alarm if p < eps. Under exchangeability P(p < eps) <= eps for a normal decision (marginal validity).

Calibration-conditional validity (Bates et al. 2023, Thm 4 with the DKWM bound):
    p_ccv = h(p) = b_ceil((n+1) p),   b_i = min(i/n + sqrt(log(2/delta) / (2n)), 1)
holds for this calibration set with probability >= 1 - delta. Its smallest value is b_1; if b_1 >= eps, no alarm at
level eps can ever be raised with the conditional guarantee -> reported as a feasibility check.

Recalibration (Gama et al. 2014: blind sliding window; Laxhammar & Falkman: semi-offline ICAD), with our guard:
a decision joins the calibration window only once it is CONFIRMED NORMAL - `lag` hours have passed, it does not lie in
[failure start - PH, maintenance] of a failure already logged by then and (guard) no strict alarm followed within the
lag. The guard stops a slowly developing leak from being absorbed into 'normal' before it is detected, at the price of
slower adaptation when drift itself causes strict alarms; static / rolling / rolling+guard are compared.
"""
from __future__ import annotations

import bisect
from collections import deque

import numpy as np
import pandas as pd


def p_values(cal_scores, scores) -> np.ndarray:
    cal = np.sort(np.asarray(cal_scores, float))
    n = len(cal)
    ge = n - np.searchsorted(cal, np.asarray(scores, float), side="left")      # #{cal >= s}
    return (1 + ge) / (n + 1)


def dkwm_floor(n: int, delta: float) -> float:
    """Smallest attainable calibration-conditional p-value b_1 (DKWM)."""
    return min(1 / n + np.sqrt(np.log(2 / delta) / (2 * n)), 1.0)


def p_ccv(p, n: int, delta: float) -> np.ndarray:
    i = np.ceil((n + 1) * np.asarray(p, float))
    return np.minimum(i / n + np.sqrt(np.log(2 / delta) / (2 * n)), 1.0)


def rolling_p_values(cal_scores, times, scores, eps_confirm: float, lag_h: float, failures: pd.DataFrame,
                     ph_h: float, guard: bool = True) -> np.ndarray:
    """p-values with a sliding calibration window (FIFO, constant size). A decision is admitted `lag_h` hours after
    it was made, unless it lies in a logged failure's window [start - PH, maintenance]; with guard=True also unless a
    strict alarm (p < eps_confirm, the smallest eps) followed within lag_h. times must be sorted."""
    n0 = len(cal_scores)
    window = deque(np.asarray(cal_scores, float))                      # FIFO order (oldest first)
    sorted_cal = sorted(window)
    t = pd.to_datetime(pd.Series(times)).to_numpy()
    s = np.asarray(scores, float)
    lag = np.timedelta64(int(lag_h * 3600), "s")
    ph = np.timedelta64(int(ph_h * 3600), "s")
    fs = [(np.datetime64(f.start), np.datetime64(f.start) - ph, np.datetime64(f.maintenance)) for f in failures.itertuples()]
    alarm_times: list = []
    pending: deque = deque()
    out = np.empty(len(s))
    for i in range(len(s)):
        now = t[i]
        while pending and pending[0][0] + lag <= now:                   # confirm decisions older than the lag
            ti, si = pending.popleft()
            k = bisect.bisect_left(alarm_times, ti)
            alarmed = guard and k < len(alarm_times) and alarm_times[k] <= ti + lag
            in_failure = any(start <= ti + lag and a <= ti <= b for start, a, b in fs)
            if not alarmed and not in_failure:
                old = window.popleft()
                del sorted_cal[bisect.bisect_left(sorted_cal, old)]
                window.append(si)
                bisect.insort(sorted_cal, si)
        n = len(sorted_cal)
        p = (1 + n - bisect.bisect_left(sorted_cal, s[i])) / (n + 1)
        out[i] = p
        if p < eps_confirm:
            alarm_times.append(now)
        pending.append((now, s[i]))
    assert len(window) == n0
    return out


def decisions_per_day(cal_times, bin_starts, rule: str = "10min") -> float:
    """Decisions per recorded day in the calibration window (recorded time from the non-empty 10-min bins)."""
    t = pd.to_datetime(pd.Series(cal_times))
    b = pd.to_datetime(pd.Series(bin_starts))
    days = ((b >= t.min()) & (b <= t.max())).sum() * pd.Timedelta(rule).total_seconds() / 86400
    return len(t) / days if days > 0 else np.nan
