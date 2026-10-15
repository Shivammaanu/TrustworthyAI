"""Evaluation on the synthetic benchmark (Phase 5).

Leak events:   detected = some flag in [onset, fail_end]; early = some flag in [onset, fail_start);
               lead = fail_start - first flag (h); meets_warning = lead >= warning_h;
               chance = 1 - exp(-rate * event length), rate = alarm intervals per day outside all events.
Sensor faults: false_leak_alarm = some flag during the fault (the system calls a sensor fault an air leak);
               r1_frozen = share of the fault's decisions that the frozen-data rule (R1) marks as frozen.
Split: 'val' events (in each model's validation window) are for choosing parameters; 'test' events for reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .events import alarm_intervals

SEV_BINS = [0.2, 0.47, 0.73, 1.0001]
SEV_LABELS = ["low", "mid", "high"]


def evaluate_copy(scores: pd.DataFrame, flags: np.ndarray, events: pd.DataFrame, ph_h: float, warning_h: float,
                  real_windows, bin_starts=None, bin_rule: str = "10min") -> pd.DataFrame:
    """bin_starts: the copy's non-empty 10-min bins; background time is counted on them (recorded time), as on the
    real data. (Counting decisions x median spacing fails for cycles: flicker cycles in frozen data are seconds apart.)"""
    t = pd.to_datetime(scores["decision_time"]).to_numpy()
    f = np.asarray(flags, bool)
    split = scores["split"].to_numpy()
    ph = pd.Timedelta(hours=ph_h)
    # background alarm rate: decisions outside every event (incl. its horizon) and outside real failure windows
    bg = np.ones(len(t), bool)
    for e in events.itertuples():
        bg &= ~((t >= np.datetime64(e.onset - ph)) & (t <= np.datetime64(e.fail_end)))
    for a, b in real_windows:
        bg &= ~((t >= np.datetime64(a)) & (t <= np.datetime64(b)))
    rows = []
    for sp in ("val", "test"):
        m = bg & (split == sp)
        if m.sum() < 2:
            continue
        if bin_starts is not None:
            b = pd.to_datetime(pd.Series(bin_starts)).to_numpy()
            keep = (b >= t[split == sp].min()) & (b <= t[split == sp].max())
            for e in events.itertuples():
                keep &= ~((b >= np.datetime64(e.onset - ph)) & (b <= np.datetime64(e.fail_end)))
            for a, b_ in real_windows:
                keep &= ~((b >= np.datetime64(a)) & (b <= np.datetime64(b_)))
            days = keep.sum() * pd.Timedelta(bin_rule).total_seconds() / 86400
        else:
            dt = np.median(np.diff(t[m]).astype("timedelta64[s]").astype(float))
            days = m.sum() * dt / 86400
        iv = alarm_intervals(t[m], f[m], scores["segment"].to_numpy()[m])
        rate = len(iv) / days if days > 0 else np.nan
        sel = (split == sp)
        for e in events.itertuples():
            w = sel & (t >= np.datetime64(e.onset)) & (t <= np.datetime64(e.fail_end))
            if w.sum() == 0:
                continue
            hit = np.flatnonzero(w & f)
            r = {"event": e.event, "kind": e.kind, "severity": e.severity, "split": sp, "decisions": int(w.sum()),
                 "background_fa_per_day": rate}
            length_days = (e.fail_end - e.onset).total_seconds() / 86400
            r["chance"] = 1 - np.exp(-rate * length_days)
            if e.kind == "leak":
                early = w & f & (t < np.datetime64(e.fail_start))
                lead = (e.fail_start - pd.Timestamp(t[hit[0]])).total_seconds() / 3600 if len(hit) else np.nan
                r.update(detected=bool(len(hit)), early=bool(early.any()), lead_h=lead,
                         meets_warning=bool(len(hit)) and lead >= warning_h)
            else:
                r.update(false_leak_alarm=bool(len(hit)), flagged_share=float(f[w].mean()),
                         r1_frozen=float(scores["frozen"].to_numpy()[w].mean()))
            rows.append(r)
    return pd.DataFrame(rows)


def summarise(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["threshold", "scheme", "model", "split"]
    leak = ev[ev["kind"] == "leak"].copy()
    leak["sev"] = pd.cut(leak["severity"], SEV_BINS, labels=SEV_LABELS, right=False)
    agg = dict(n=("event", "size"), detected=("detected", "mean"), chance=("chance", "mean"), early=("early", "mean"),
               meets_warning=("meets_warning", "mean"), lead_median_h=("lead_h", "median"),
               background_fa_per_day=("background_fa_per_day", "mean"))
    by_all = leak.groupby(keys).agg(**agg).assign(sev="all").set_index("sev", append=True)
    by_sev = leak.groupby(keys + ["sev"], observed=True).agg(**agg)
    leaks = pd.concat([by_all, by_sev]).sort_index().round(3)
    sens = ev[ev["kind"] != "leak"].groupby(keys + ["kind"]).agg(
        n=("event", "size"), false_leak_alarm=("false_leak_alarm", "mean"), flagged_share=("flagged_share", "mean"),
        r1_frozen=("r1_frozen", "mean"), chance=("chance", "mean")).round(3)
    return leaks, sens


def benchmark_events_table(cfg, d, sub: str = "benchmark", models=None) -> pd.DataFrame:
    """Evaluate every score file under results/scores/<sub>/<scheme>/<model>/seed*_copy*.csv."""
    from ..utils.config import failure_table
    from .report import flag_variants
    events = pd.read_csv(d["processed"] / "benchmark" / "events.csv", parse_dates=["onset", "fail_start", "fail_end"])
    ph_h, warn = cfg["evaluation"]["predictive_horizon_h"], cfg["evaluation"]["warning_h"]
    real = [(f.start - pd.Timedelta(hours=ph_h), f.maintenance) for f in failure_table(cfg).itertuples()]
    bins = {c: pd.read_pickle(d["processed"] / "benchmark" / f"copy{c}" / "resampled.pkl")["bin_start"]
            for c in range(cfg["benchmark"]["n_copies"])}
    rows = []
    for path in sorted((d["scores"] / sub).glob("*/*/seed*_copy*.csv")):
        scheme, model = path.parts[-3], path.parts[-2]
        seed, copy = path.stem.split("_copy")
        seed, copy = int(seed[4:]), int(copy)
        if model not in (models or cfg["models"]) or scheme not in cfg["splits"]["schemes"]:
            continue
        s = pd.read_csv(path, parse_dates=["decision_time"])
        for v, flags in flag_variants(s, cfg).items():
            r = evaluate_copy(s, flags, events[events["copy"] == copy], ph_h, warn, real, bins.get(copy),
                              cfg["data"]["resample"])
            rows.append(r.assign(threshold=v, scheme=scheme, model=model, seed=seed, copy=copy))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()