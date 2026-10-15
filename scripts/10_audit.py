"""Step 10 - consistency audit (Phase 5, consistency pillar). Do alarms on clean normal data occur equally often across
months, compressor states and times of day? For the baselines (p95) and the trustworthy operating system (static
conformal, R1-gated, singles and the pair) on the fixed split.
Metric: a group's alarm share / the subject's overall alarm share (1 = as often as on average), with a block bootstrap
over days (decisions within a day are correlated). A group differs only if its 95% interval excludes 1.
Writes results/AUDIT_RESULTS.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline import episode_windows, load_processed  # noqa: E402
from src.utils.config import dirs, failure_table, setup  # noqa: E402
from src.utils.fileio import load_json  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

DIMENSIONS = ["month", "state", "time_of_day"]


def subjects(cfg, d):
    """(name, path, flag function) for every audited subject."""
    ac = cfg["audit"]
    F = f"fa{ac['target']:g}_gated"
    out = []
    for m in cfg["models"]:
        out.append((f"baseline_p95/{m}", d["scores"] / "baseline" / "fixed" / m, lambda s: s["score"] >= s["thr_p95"]))
        out.append((f"conformal_F{ac['target']:g}/{m}", d["scores"] / "conformal_static" / "fixed" / m,
                    lambda s, F=F: s[f"flag_{F}"].astype(bool)))
    for name in ac["combinations"]:
        out.append((f"conformal_F{ac['target']:g}/{name}", d["scores"] / "combined" / "fixed" / name,
                    lambda s, F=F: s[f"flag_{F}"].astype(bool)))
    return out


def bootstrap(counts: pd.DataFrame, group_col: str, n_boot: int, rng) -> pd.DataFrame:
    """counts: rows (date, group, flagged, total). Ratio of each group's share to the overall share, resampling days."""
    dates = counts["date"].unique()
    piv_f = counts.pivot_table(index="date", columns=group_col, values="flagged", aggfunc="sum", fill_value=0)
    piv_t = counts.pivot_table(index="date", columns=group_col, values="total", aggfunc="sum", fill_value=0)
    f, t = piv_f.to_numpy(float), piv_t.to_numpy(float)
    def ratio(fs, ts):
        overall = fs.sum() / max(ts.sum(), 1)
        share = fs.sum(0) / np.maximum(ts.sum(0), 1)
        return share / overall if overall > 0 else np.full(fs.shape[1], np.nan)
    point = ratio(f, t)
    boots = np.array([ratio(f[i], t[i]) for i in (rng.integers(0, len(dates), len(dates)) for _ in range(n_boot))])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    res = pd.DataFrame({"decisions": t.sum(0).astype(int), "alarms": f.sum(0).astype(int),
                        "alarm_share": f.sum(0) / np.maximum(t.sum(0), 1), "ratio": point, "ci_low": lo, "ci_high": hi},
                       index=piv_f.columns)
    res["differs"] = (res["ci_low"] > 1) | (res["ci_high"] < 1)
    return res


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("10_audit", d["logs"])
    ac = cfg["audit"]
    rng = np.random.default_rng(ac["seed"])
    data, fails = load_processed(cfg), failure_table(cfg)
    episodes = episode_windows(cfg, data)
    ph = pd.Timedelta(hours=cfg["evaluation"]["predictive_horizon_h"])
    thr = load_json(d["processed"] / "meta.json")["motor_load_threshold_A"]
    res = data["resampled"].set_index("bin_start")
    state_of_bin = pd.Series(np.where(res["Motor_current"] > thr, "mostly_running", "mostly_idle"), index=res.index)
    tables, summary = {}, []
    for name, folder, flag_fn in subjects(cfg, d):
        parts = []
        for p in sorted(folder.glob("seed*.csv")):
            s = pd.read_csv(p, parse_dates=["decision_time"])
            t = s["decision_time"]
            clean = ~s["frozen"].astype(bool)
            for f in fails.itertuples():
                clean &= ~((t >= f.start - ph) & (t <= f.maintenance))
            for e in episodes.itertuples():
                clean &= ~((t >= e.start - ph) & (t <= e.end))
            s = s[clean].assign(flag=flag_fn(s[clean]).to_numpy())
            b = (s["decision_time"] - pd.Timedelta(seconds=1)).dt.floor(cfg["data"]["resample"])
            s["state"] = state_of_bin.reindex(b).to_numpy()
            s["month"] = s["decision_time"].dt.to_period("M").astype(str)
            h = s["decision_time"].dt.hour
            s["time_of_day"] = [f"{(x // 4) * 4:02d}-{(x // 4) * 4 + 4:02d}h" for x in h]
            s["date"] = s["decision_time"].dt.date.astype(str) + f"_s{p.stem}"
            parts.append(s.dropna(subset=["state"]))
        if not parts:
            log.info(f"skip {name}: no score files")
            continue
        s = pd.concat(parts, ignore_index=True)
        for dim in DIMENSIONS:
            counts = s.groupby(["date", dim]).agg(flagged=("flag", "sum"), total=("flag", "size")).reset_index()
            r = bootstrap(counts, dim, ac["n_boot"], rng).round(3)
            tables[(name, dim)] = r
            summary.append({"subject": name, "dimension": dim, "overall_alarm_share": round(s["flag"].mean(), 4),
                            "min_ratio": r["ratio"].min(), "max_ratio": r["ratio"].max(),
                            "groups_differing": int(r["differs"].sum()), "groups": len(r)})
        log.info(f"{name}: audited {len(s):,} clean decisions")
    sm = pd.DataFrame(summary)
    sm.to_csv(d["metrics"] / "audit_summary.csv", index=False)
    allt = pd.concat({k: v for k, v in tables.items()}, names=["subject", "dimension", "group"])
    allt.to_csv(d["metrics"] / "audit_groups.csv")
    lines = ["# Consistency audit (Phase 5)", "",
             "Clean normal decisions only (outside failures' warning window..maintenance, unreported episodes and frozen "
             "data), fixed split, all seeds. `ratio` = the group's alarm share / the subject's overall alarm share "
             f"(1 = as often as on average); 95% block-bootstrap interval over days ({ac['n_boot']} resamples); "
             "`differs` = interval excludes 1. State = the decision's 10-min bin is mostly running (mean motor current "
             f"> {thr:.2f} A, the R2 load threshold) or mostly idle.", "",
             "## Summary", "", "```", sm.to_string(index=False), "```", ""]
    for dim in DIMENSIONS:
        lines += [f"## By {dim.replace('_', ' ')}", ""]
        for (name, dm), r in tables.items():
            if dm == dim:
                lines += [f"**{name}**", "", "```", r.to_string(), "```", ""]
    (d["root"] / "AUDIT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"wrote {d['root'] / 'AUDIT_RESULTS.md'}")


if __name__ == "__main__":
    main()
