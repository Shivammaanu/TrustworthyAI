"""
Synthetic benchmark: air leaks and sensor faults injected into copies of the real data (Phase 5, step 1).

Why: Hundman's pruning, the EWMA span and Geifman's triage need labelled events, and the four real failures may
never be used to choose anything. Events are placed only in normal time (outside failures, unreported episodes and
frozen data, see R1-R3) and are labelled by construction.

Leak model (signature from the Phase 0 data, see DOCUMENTATION.md 5a):
  build-up  [onset, fail_start)  the compressor cycle is rebuilt from the host's own cycles: runs lengthen by a
            factor (1 + a*sev), idle periods shorten by (1 - b*sev); oil temperature rises, TP3/Reservoirs sag and
            DV pressure rises during runs, all proportional to sev(t) = severity * (t - onset) / (fail_start - onset)
  failure   [fail_start, fail_end)  continuous loaded running (template = the host's own loaded rows) with high DV
            pressure, lower TP3/Reservoirs and rising oil temperature
Sensor faults: 'frozen' (all analog signals held, COMP flickering ~20% zeros, as in the 9 real frozen periods),
'stuck' (one analog signal held) and 'noise' (one analog signal + N(0, (3 sd)^2)).
All magnitudes scale with a per-event severity in [0.2, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.load import ANALOG

# magnitudes at severity 1 (ranges span the training-period episodes E09-E23 and the reported failures)
LEAK = {"run_factor": 3.0, "idle_factor": 0.5, "oil_buildup_C": 8.0, "tp3_buildup_bar": 0.8, "dv_buildup_bar": 0.6,
        "dv_fail_min": 0.6, "dv_fail_span": 1.7, "tp3_fail_bar": 0.7, "oil_fail_C": 8.0, "tp3_fail_min": 0.3, "oil_fail_min": 4.0}
DURATION_H = {"buildup": (6, 24), "failure": (1, 12), "sensor": (1, 12)}
KINDS = {"leak": 0.6, "frozen": 0.15, "stuck": 0.15, "noise": 0.10}


def _pieces(running: np.ndarray):
    """Contiguous (start, end, is_running) pieces of a boolean array."""
    brk = np.flatnonzero(np.r_[True, running[1:] != running[:-1], True])
    return [(brk[i], brk[i + 1], bool(running[brk[i]])) for i in range(len(brk) - 1)]


def plan_events(times: pd.Series, forbidden, rng, spacing_h=48.0, min_coverage=0.7, median_dt=10.0):
    """Non-overlapping event slots in normal time. forbidden = list of (start, end) never touched."""
    t = pd.to_datetime(times).to_numpy()
    t0, t1 = t[0], t[-1]
    events, cur = [], pd.Timestamp(t0) + pd.Timedelta(hours=rng.uniform(0, spacing_h))
    while cur < pd.Timestamp(t1):
        kind = str(rng.choice(list(KINDS), p=list(KINDS.values())))
        sev = float(rng.uniform(0.2, 1.0))
        if kind == "leak":
            b = pd.Timedelta(hours=rng.uniform(*DURATION_H["buildup"]))
            f = pd.Timedelta(hours=rng.uniform(*DURATION_H["failure"]))
            onset, fs, fe = cur, cur + b, cur + b + f
        else:
            d = pd.Timedelta(hours=rng.uniform(*DURATION_H["sensor"]))
            onset, fs, fe = cur, cur, cur + d
        ok = all(not (onset <= b_ and fe >= a_) for a_, b_ in forbidden)
        n = int(((t >= np.datetime64(onset)) & (t < np.datetime64(fe))).sum())
        cover = n * median_dt / max((fe - onset).total_seconds(), 1)
        if ok and cover >= min_coverage:
            events.append({"kind": kind, "severity": sev, "onset": onset, "fail_start": fs, "fail_end": fe,
                           "sensor": str(rng.choice(ANALOG)) if kind in ("stuck", "noise") else ""})
            cur = fe + pd.Timedelta(hours=spacing_h)
        else:
            cur = cur + pd.Timedelta(hours=6)
    return pd.DataFrame(events)


def _apply_leak(df, ev, running, motor_thr, rng, block):
    t = df["timestamp"].to_numpy()
    cols = [c for c in df.columns if c != "timestamp"]
    onset, fs, fe = map(np.datetime64, (ev.onset, ev.fail_start, ev.fail_end))
    span = (fs - onset) / np.timedelta64(1, "s")
    s = ev.severity

    def sev(ti):
        return s * float(np.clip((ti - onset) / np.timedelta64(1, "s") / span, 0, 1))

    # --- build-up: rebuild each gap-free block's cycle sequence inside the window
    idx = np.flatnonzero((t >= onset) & (t < fs))
    for b in np.unique(block[idx]):
        rows = idx[block[idx] == b]
        vals = df.loc[rows, cols].to_numpy()
        run_b = running[rows]
        seq = []
        for a, e, is_run in _pieces(run_b):
            n = e - a
            k = sev(t[rows[a]])
            if is_run:
                target = int(round(n * (1 + LEAK["run_factor"] * k)))
                core = np.arange(a + n // 3, max(a + n // 3 + 1, e - 1))        # steady part of the run
                extra = np.resize(core, max(target - n, 0))
                seq.append(np.r_[np.arange(a, e - 1), extra, [e - 1]] if n > 1 else np.r_[np.arange(a, e), extra])
            else:
                target = max(1, int(round(n * (1 - LEAK["idle_factor"] * k))))
                seq.append(np.arange(a, a + target))
        order = np.concatenate(seq) if seq else np.arange(len(rows))
        order = np.resize(order, len(rows))                                     # same number of rows, same timestamps
        new = vals[order]
        sv = np.array([sev(x) for x in t[rows]])
        c = {name: i for i, name in enumerate(cols)}
        new[:, c["Oil_temperature"]] += LEAK["oil_buildup_C"] * sv
        new[:, c["TP3"]] -= LEAK["tp3_buildup_bar"] * sv
        new[:, c["Reservoirs"]] -= LEAK["tp3_buildup_bar"] * sv
        run_new = new[:, c["Motor_current"]] > motor_thr
        new[run_new, c["DV_pressure"]] += LEAK["dv_buildup_bar"] * sv[run_new]
        df.loc[rows, cols] = new

    # --- failure phase: continuous loaded running from the host's own loaded rows
    rows = np.flatnonzero((t >= fs) & (t < fe))
    if len(rows):
        # template: loaded rows within a day of the event but outside it (the build-up rows were just rewritten)
        loaded = (df["COMP"].to_numpy() == 0) & (df["Motor_current"].to_numpy() > motor_thr)
        around = (t >= onset - np.timedelta64(1, "D")) & (t < fe + np.timedelta64(1, "D")) & ~((t >= onset) & (t < fe))
        near = np.flatnonzero(around & loaded)
        if len(near) == 0:
            near = np.flatnonzero(loaded)
        tmpl = df.loc[near, cols].to_numpy()
        new = tmpl[rng.integers(0, len(tmpl), len(rows))]
        c = {name: i for i, name in enumerate(cols)}
        frac = np.linspace(0, 1, len(rows))
        dv = LEAK["dv_fail_min"] + LEAK["dv_fail_span"] * s
        new[:, c["DV_pressure"]] = dv + rng.normal(0, 0.05, len(rows))
        drop = LEAK["tp3_fail_min"] + LEAK["tp3_fail_bar"] * s
        new[:, c["TP3"]] -= drop
        new[:, c["Reservoirs"]] = new[:, c["TP3"]]
        oil0 = df.loc[rows[0], "Oil_temperature"]
        new[:, c["Oil_temperature"]] = oil0 + (LEAK["oil_fail_min"] + LEAK["oil_fail_C"] * s) * frac + rng.normal(0, 0.3, len(rows))
        df.loc[rows, cols] = new


def _apply_sensor(df, ev, rng, sds):
    t = df["timestamp"].to_numpy()
    rows = np.flatnonzero((t >= np.datetime64(ev.onset)) & (t < np.datetime64(ev.fail_end)))
    if not len(rows):
        return
    if ev.kind == "frozen":
        held = df.loc[rows[0], ANALOG].to_numpy()
        df.loc[rows, ANALOG] = np.tile(held, (len(rows), 1))
        comp = (rng.random(len(rows)) > 0.2).astype(float)                     # ~20% zeros, as in the real data
        df.loc[rows, "COMP"] = comp
        df.loc[rows, "Towers"] = comp
        df.loc[rows, "DV_electric"] = 0.0
    elif ev.kind == "stuck":
        df.loc[rows, ev.sensor] = df.loc[rows[0], ev.sensor]
    elif ev.kind == "noise":
        df.loc[rows, ev.sensor] = df.loc[rows, ev.sensor].to_numpy() + rng.normal(0, 3 * sds[ev.sensor], len(rows))


def make_copy(df, events, running, motor_thr, block, seed):
    """Return a modified copy of the raw data with all events applied."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    sds = df[ANALOG].std()
    for ev in events.itertuples():
        if ev.kind == "leak":
            _apply_leak(out, ev, running, motor_thr, rng, block)
        else:
            _apply_sensor(out, ev, rng, sds)
    return out