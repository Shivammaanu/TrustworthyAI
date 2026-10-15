"""
Event-level evaluation against the four failures (company protocol, Veloso et al. 2022, as in the
problem statement):

* alarm interval   = a run of consecutive flagged decisions within one segment
* detected failure = some flagged decision lies in [start - PH, eval_end]  (warning window or failure)
* lead time        = failure start - first flagged decision in that range (negative = detected late)
* false alarm      = an alarm interval that starts in normal test time, i.e. outside every
                     [start - PH, maintenance]. Alarms between failure end and maintenance are ignored
                     (the fault is known and being handled).
* false alarms per day uses recorded normal time (gap time is not counted).
* chance level: expected detections of a random alarm process with the same false-alarm rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def alarm_intervals(times, flags, segment) -> pd.DataFrame:
    t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    f = np.asarray(flags, bool); seg = np.asarray(segment)
    new = f & ~np.r_[False, f[:-1] & (seg[1:] == seg[:-1])]
    if not f.any():
        return pd.DataFrame({"start": pd.Series(dtype="datetime64[ns]"), "end": pd.Series(dtype="datetime64[ns]"),
                             "n": pd.Series(dtype=int)})
    gid = np.cumsum(new)[f]
    d = pd.DataFrame({"t": t[f].to_numpy(), "g": gid})
    return d.groupby("g")["t"].agg(start="min", end="max", n="count").reset_index(drop=True)


def evaluate_events(times, flags, segment, failures: pd.DataFrame, ph_h: float, warning_h: float,
                    normal_days: float, test_start, test_end) -> tuple[pd.DataFrame, dict]:
    t = pd.to_datetime(pd.Series(times)).to_numpy()
    f = np.asarray(flags, bool)
    ph = pd.Timedelta(hours=ph_h)
    rows = []
    for fl in failures.itertuples():
        if not (pd.Timestamp(test_start) < fl.start <= pd.Timestamp(test_end)):
            continue
        win = (t >= np.datetime64(fl.start - ph)) & (t <= np.datetime64(fl.eval_end))
        hit = np.flatnonzero(win & f)
        lead = (fl.start - pd.Timestamp(t[hit[0]])).total_seconds() / 3600 if len(hit) else np.nan
        rows.append({"failure": fl.name, "severity": fl.severity, "detected": bool(len(hit)),
                     "first_alarm": pd.Timestamp(t[hit[0]]) if len(hit) else pd.NaT,
                     "lead_time_h": lead, "meets_warning": bool(len(hit)) and lead >= warning_h,
                     "decisions_in_window": int(win.sum()), "flagged_in_window": int((win & f).sum())})
    per_failure = pd.DataFrame(rows)

    iv = alarm_intervals(times, flags, segment)
    abnormal = [(fl.start - ph, fl.maintenance) for fl in failures.itertuples()]
    in_abn = np.zeros(len(iv), bool)
    tp_int = np.zeros(len(iv), bool)
    if len(iv):
        for a, b in abnormal:
            in_abn |= ((iv["start"] >= a) & (iv["start"] <= b)).to_numpy()
        for fl in failures.itertuples():
            tp_int |= ((iv["start"] <= fl.eval_end) & (iv["end"] >= fl.start - ph)).to_numpy()
    n_fa = int((~in_abn & ~tp_int).sum()) if len(iv) else 0
    n_tp_int = int(tp_int.sum()) if len(iv) else 0
    normal_mask = np.ones(len(t), bool)
    for a, b in abnormal:
        normal_mask &= ~((t >= np.datetime64(a)) & (t <= np.datetime64(b)))
    summary = {
        "failures_in_test": len(per_failure),
        "detected": int(per_failure["detected"].sum()) if len(per_failure) else 0,
        "meets_warning": int(per_failure["meets_warning"].sum()) if len(per_failure) else 0,
        "false_alarms": n_fa,
        "normal_days": round(normal_days, 2),
        "false_alarms_per_day": n_fa / normal_days if normal_days > 0 else np.nan,
        "event_precision": n_tp_int / (n_tp_int + n_fa) if (n_tp_int + n_fa) else np.nan,
        "share_flagged_normal": float(f[normal_mask].mean()) if normal_mask.any() else np.nan,
    }
    # Chance level: a random alarm process with the same false-alarm rate fires inside a warning window
    # of PH hours with probability 1 - exp(-rate * PH_days). Detections near this number are not evidence.
    # For a >= warning_h warning the alarm must fall in [start - PH, start - warning_h]: (PH - warning_h) hours.
    rate = summary["false_alarms_per_day"]
    p_chance = 1 - np.exp(-rate * ph_h / 24) if np.isfinite(rate) else np.nan
    p_warn = 1 - np.exp(-rate * (ph_h - warning_h) / 24) if np.isfinite(rate) else np.nan
    summary["p_detect_by_chance"] = p_chance
    summary["detected_by_chance"] = p_chance * len(per_failure)
    summary["meets_warning_by_chance"] = p_warn * len(per_failure)
    return per_failure, summary


def context_metrics(times, flags, segment, failures: pd.DataFrame, episodes: pd.DataFrame, frozen: pd.DataFrame,
                    ph_h: float, normal_days_clean: float, test_start, test_end) -> dict:
    """Where do the 'false' alarms actually fall?  (R1 / R3 reporting, for every model.)
    * alarms starting inside frozen data (a logger fault, not an air leak)
    * alarms inside an unreported fault episode window [episode start - PH, episode end]
    * the remaining false alarms per day of clean normal time (no failure, episode or frozen time)
    * how many unreported episodes in the test period got >= 1 alarm (and how many by chance)"""
    t = pd.to_datetime(pd.Series(times)).to_numpy()
    f = np.asarray(flags, bool)
    ph = pd.Timedelta(hours=ph_h)
    iv = alarm_intervals(times, flags, segment)
    n = len(iv)
    abn = np.zeros(n, bool); ep_hit = np.zeros(n, bool); fz_hit = np.zeros(n, bool)
    if n:
        for fl in failures.itertuples():
            abn |= ((iv["start"] <= fl.maintenance) & (iv["end"] >= fl.start - ph)).to_numpy()
        for e in episodes.itertuples():
            ep_hit |= ((iv["start"] <= e.end) & (iv["end"] >= e.start - ph)).to_numpy()
        for z in frozen.itertuples():
            fz_hit |= ((iv["start"] >= z.start) & (iv["start"] < z.end)).to_numpy()
    fa = ~abn
    clean = fa & ~ep_hit & ~fz_hit
    in_test = episodes[(episodes["start"] > pd.Timestamp(test_start)) & (episodes["start"] <= pd.Timestamp(test_end))]
    detected = 0
    for e in in_test.itertuples():
        win = (t >= np.datetime64(e.start - ph)) & (t <= np.datetime64(e.end))
        detected += int((win & f).any())
    rate = int(clean.sum()) / normal_days_clean if normal_days_clean > 0 else np.nan
    return {"fa_in_frozen": int((fa & fz_hit).sum()), "fa_in_episodes": int((fa & ep_hit & ~fz_hit).sum()),
            "false_alarms_per_day_clean": rate, "normal_days_clean": round(normal_days_clean, 2),
            "episodes_in_test": len(in_test), "episodes_detected": detected,
            "episodes_detected_by_chance": (1 - np.exp(-rate * ph_h / 24)) * len(in_test) if np.isfinite(rate) else np.nan}