"""Collect baseline scores into the metric tables, figures and RESULTS_SUMMARY.md."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .ad1 import ad1_auc_pr  # noqa: E402
from .events import alarm_intervals, context_metrics, evaluate_events  # noqa: E402

COLORS = {"lstm_ae": "#e67e22", "tranad": "#8e44ad", "iforest": "#27ae60"}
LABELS = {"lstm_ae": "LSTM-AE", "tranad": "TranAD", "iforest": "Isolation Forest"}


def normal_days(bin_starts, rule, start, end, failures, ph_h, extra=(), frozen_flags=None) -> float:
    """Recorded normal time (days) in (start, end]: non-empty resample bins outside [f.start-PH, maintenance],
    outside any extra (start, end) intervals and, if frozen_flags is given, outside frozen bins."""
    t = pd.to_datetime(pd.Series(bin_starts)).to_numpy()
    m = (t > np.datetime64(start)) & (t <= np.datetime64(end))
    for f in failures.itertuples():
        m &= ~((t >= np.datetime64(f.start - pd.Timedelta(hours=ph_h))) & (t <= np.datetime64(f.maintenance)))
    for a, b in extra:
        m &= ~((t >= np.datetime64(a)) & (t <= np.datetime64(b)))
    if frozen_flags is not None:
        m &= ~np.asarray(frozen_flags, bool)
    return m.sum() * pd.Timedelta(rule).total_seconds() / 86400


def flag_variants(s: pd.DataFrame, cfg) -> dict:
    """Alarm flags per variant: 'flag_<name>' columns if the file has them (trustworthy layer), else score >= thr_<v>."""
    cols = [c for c in s.columns if c.startswith("flag_")]
    if cols:
        return {c[5:]: s[c].astype(bool).to_numpy() for c in cols}
    out = {}
    for v in cfg["thresholds"]["variants"]:
        col = f"thr_{v}" if f"thr_{v}" in s else "threshold"
        out[v] = (s["score"] >= s[col]).to_numpy()
    return out


def evaluate_all(cfg, d, failures, data, episodes, mode="baseline", info_mode="baseline", models=None, info_model=None):
    """models: names to evaluate (default cfg['models']); info_model: model whose run info gives the test window
    (needed for combined detectors, which have no run info of their own)."""
    ev = cfg["evaluation"]
    resampled = data["resampled"]
    ph = pd.Timedelta(hours=ev["predictive_horizon_h"])
    info = pd.read_csv(d["metrics"] / f"{info_mode}_run_info.csv", parse_dates=["cut", "test_end"])
    per_fail, summ = [], []
    for path in sorted((d["scores"] / mode).glob("*/*/seed*.csv")):
        scheme, model, seed = path.parts[-3], path.parts[-2], int(path.stem[4:])
        if model not in (models or cfg["models"]) or scheme not in cfg["splits"]["schemes"]:
            continue
        s = pd.read_csv(path, parse_dates=["decision_time"])
        rows = info[(info.scheme == scheme) & (info.model == (info_model or model)) & (info.seed == seed)]
        t0, t1 = rows["cut"].min(), rows["test_end"].max()
        nd = normal_days(resampled["bin_start"], cfg["data"]["resample"], t0, t1, failures, ev["predictive_horizon_h"])
        nd_clean = normal_days(resampled["bin_start"], cfg["data"]["resample"], t0, t1, failures, ev["predictive_horizon_h"],
                               [(e.start - ph, e.end) for e in episodes.itertuples()], resampled["frozen"])
        seg = s["segment"].astype(str) + "_" + s["part"].astype(str)
        ad1 = {}
        if scheme == "fixed":           # one model, one score scale -> comparable with the benchmark (threshold-free)
            for ref in ev.get("ad1_reference", ["start"]):
                ad1[f"ad1_auc_pr_{ref}"], _ = ad1_auc_pr(s["score"], s["decision_time"], failures,
                                                         ev["predictive_horizon_h"], ev["warning_h"],
                                                         ev["ad1_thresholds"], ref)
        for v, flags in flag_variants(s, cfg).items():
            pf, sm = evaluate_events(s["decision_time"], flags, seg, failures, ev["predictive_horizon_h"],
                                     ev["warning_h"], nd, t0, t1)
            cm = context_metrics(s["decision_time"], flags, seg, failures, episodes, data["frozen"],
                                 ev["predictive_horizon_h"], nd_clean, t0, t1)
            meta = {"scheme": scheme, "model": model, "threshold": v, "seed": seed}
            per_fail.append(pf.assign(**meta)); summ.append({**meta, **sm, **cm, **ad1})
    per_fail = pd.concat(per_fail, ignore_index=True); summ = pd.DataFrame(summ)
    per_fail.to_csv(d["metrics"] / f"{mode}_per_failure.csv", index=False)
    summ.to_csv(d["metrics"] / f"{mode}_summary_per_seed.csv", index=False)
    cols = [c for c in ["detected", "detected_by_chance", "meets_warning", "meets_warning_by_chance",
                        "false_alarms_per_day", "event_precision",
                        "share_flagged_normal", "fa_in_frozen", "fa_in_episodes", "false_alarms_per_day_clean",
                        "episodes_detected", "episodes_detected_by_chance", "episodes_in_test"] if c in summ] \
        + sorted(c for c in summ if c.startswith("ad1_"))
    agg = summ.groupby(["threshold", "scheme", "model"])[cols].agg(["mean", "std"]).round(3)
    agg.to_csv(d["metrics"] / f"{mode}_summary.csv")
    lead = per_fail.groupby(["threshold", "scheme", "model", "failure"]).agg(
        detected_seeds=("detected", "sum"), seeds=("detected", "size"),
        lead_mean_h=("lead_time_h", "mean"), lead_min_h=("lead_time_h", "min"), lead_max_h=("lead_time_h", "max")).round(2)
    lead.to_csv(d["metrics"] / f"{mode}_lead_times.csv")
    return per_fail, summ, agg, lead


def _broken(w):
    """Insert NaN between segments so lines are not drawn across data gaps."""
    seg = w["segment"].astype(str) + "_" + w["part"].astype(str)
    br = seg.ne(seg.shift()).to_numpy()
    y = (w["score"] / w["threshold"]).clip(upper=20).to_numpy(float)
    t = w["decision_time"].to_numpy()
    idx = np.flatnonzero(br)[1:]
    return np.insert(t, idx, np.datetime64("NaT")), np.insert(y, idx, np.nan)


def plot_failures(cfg, d, failures, mode="baseline", seed=None, frozen=None):
    """Score / threshold around each failure for every model (1 = alarm level), one figure per scheme."""
    ph = pd.Timedelta(hours=cfg["evaluation"]["predictive_horizon_h"])
    warn = pd.Timedelta(hours=cfg["evaluation"]["warning_h"])
    seed = seed or cfg["project"]["seeds"][0]
    for scheme in cfg["splits"]["schemes"]:
        fig, axes = plt.subplots(len(failures), 1, figsize=(13, 3 * len(failures)))
        for ax, f in zip(np.atleast_1d(axes), failures.itertuples()):
            for model in cfg["models"]:
                p = d["scores"] / mode / scheme / model / f"seed{seed}.csv"
                if not p.exists():
                    continue
                s = pd.read_csv(p, parse_dates=["decision_time"])
                w = s[(s.decision_time >= f.start - pd.Timedelta(days=7)) & (s.decision_time <= f.maintenance + pd.Timedelta(hours=6))]
                tx, ty = _broken(w)
                ax.plot(tx, ty, lw=.8, color=COLORS[model], label=LABELS[model])
            ax.axhline(1, color="k", ls="--", lw=.8)
            for z in (frozen if frozen is not None else pd.DataFrame()).itertuples():
                if z.end >= f.start - pd.Timedelta(days=7) and z.start <= f.maintenance:
                    ax.axvspan(z.start, z.end, color="grey", alpha=.25, hatch="//", lw=0)
            ax.axvspan(f.start - ph, f.start - warn, color="yellow", alpha=.15)
            ax.axvspan(f.start - warn, f.start, color="orange", alpha=.3)
            ax.axvspan(f.start, f.eval_end, color="red", alpha=.2)
            ax.axvline(f.maintenance, color="green", ls="--", lw=1)
            ax.set(yscale="log", ylabel=f"score / {cfg['thresholds']['primary']} threshold",
                   title=f"{f.name} ({f.severity}) - {scheme} split, seed {seed}")
            ax.legend(fontsize=8, loc="upper left")
        fig.suptitle("yellow = 48 h warning window, orange = last 2 h, red = failure, green = maintenance, "
                     "grey hatched = frozen data; above 1 = alarm")
        fig.tight_layout(); fig.savefig(d["figures"] / f"{mode}_{scheme}_failures.png", dpi=130); plt.close(fig)


def plot_false_alarms(cfg, d, failures, mode="baseline"):
    ph = pd.Timedelta(hours=cfg["evaluation"]["predictive_horizon_h"])
    seed = cfg["project"]["seeds"][0]
    for scheme in cfg["splits"]["schemes"]:
        rows = []
        for model in cfg["models"]:
            p = d["scores"] / mode / scheme / model / f"seed{seed}.csv"
            if not p.exists():
                continue
            s = pd.read_csv(p, parse_dates=["decision_time"])
            iv = alarm_intervals(s.decision_time, s.flag, s.segment.astype(str) + "_" + s.part.astype(str))
            for f in failures.itertuples():
                iv = iv[~((iv.start >= f.start - ph) & (iv.start <= f.maintenance))]
            rows.append(pd.to_datetime(iv.start).dt.to_period("M").value_counts().rename(LABELS[model]))
        if rows:
            tab = pd.concat(rows, axis=1).fillna(0).sort_index()
            tab.plot.bar(figsize=(10, 3.5), color=[COLORS[m] for m in cfg["models"] if LABELS[m] in tab], rot=0,
                         title=f"False-alarm intervals per month ({scheme} split, seed {seed})")
            plt.tight_layout(); plt.savefig(d["figures"] / f"{mode}_{scheme}_false_alarms_by_month.png", dpi=130); plt.close()


def write_summary(cfg, d, agg, lead, info, mode="baseline"):
    lines = [f"# {mode.capitalize()} results", "",
             f"Seeds: {cfg['project']['seeds']} | predictive horizon {cfg['evaluation']['predictive_horizon_h']} h | "
             f"warning requirement {cfg['evaluation']['warning_h']} h | thresholds {list(cfg['thresholds']['variants'])}", "",
             "`detected_by_chance` = detections expected from random alarms at the same false-alarm rate.",
             "`ad1_auc_pr_start/end` = benchmark metric (threshold-free) with the horizon anchored at failure start / end.",
             "`fa_in_frozen` / `fa_in_episodes` = false alarms that start in frozen data / in an unreported fault episode "
             "window; `false_alarms_per_day_clean` = the remaining ones per day of clean normal time;",
             "`episodes_detected` = unreported episodes in the test period with >= 1 alarm (chance level alongside).", "",
             "## Summary (mean, std over seeds)", "", "```", agg.to_string(), "```", "",
             "## Lead time per failure (hours before failure start; negative = after start)", "", "```",
             lead.to_string(), "```", "", "## Runs", "", "```",
             info.groupby(["scheme", "model"])[["n_train", "n_test", "best_epoch", "seconds"]].mean().round(1).to_string(),
             "```"]
    (d["root"] / f"{mode.upper()}_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")