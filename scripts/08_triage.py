"""Step 8 - alarm triage with a guaranteed risk (Phase 5, human oversight; Geifman & El-Yaniv 2017, SGR).
The operating detector raises alarms at a permissive budget; each alarm is confirmed automatically if its evidence
kappa = -log10(min adjusted p) >= theta, otherwise sent to engineer review. theta comes from SGR on the VALIDATION
alarms of the benchmark copies (true = overlaps an injected leak), so that P(share of false confirmed alarms > r*) <
delta. Reported once on the test leaks and on the real data. Writes results/TRIAGE_RESULTS.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline import episode_windows, load_processed  # noqa: E402
from src.trustworthy.triage import KAPPAS, alarm_table, sgr  # noqa: E402
from src.utils.config import dirs, failure_table, setup  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def overlaps(t: pd.DataFrame, windows) -> np.ndarray:
    hit = np.zeros(len(t), bool)
    for a, b in windows:
        hit |= ((t["start"] <= b) & (t["end"] >= a)).to_numpy()
    return hit


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("08_triage", d["logs"])
    tc = cfg["triage"]
    scheme, model = tc["scheme"], tc["detector"]
    flag = f"flag_fa{tc['alarm_target']:g}_gated"
    ph = pd.Timedelta(hours=cfg["evaluation"]["predictive_horizon_h"])
    warn = pd.Timedelta(hours=cfg["evaluation"]["warning_h"])
    data, fails = load_processed(cfg), failure_table(cfg)
    episodes = episode_windows(cfg, data)
    events = pd.read_csv(d["processed"] / "benchmark" / "events.csv", parse_dates=["onset", "fail_start", "fail_end"])
    src = d["scores"] / "combined" / scheme / model
    bsrc = d["scores"] / "combined_bench" / scheme / model
    if not src.exists():
        sys.exit("combined scores missing - run scripts/06_combine.py first")

    # ---- benchmark alarms with labels
    rows = []
    for p in sorted(bsrc.glob("seed*_copy*.csv")):
        seed, c = p.stem.split("_copy")
        seed, c = int(seed[4:]), int(c)
        a = alarm_table(pd.read_csv(p, parse_dates=["decision_time"]), flag)
        if not len(a):
            continue
        ev = events[(events["copy"] == c) & (events["kind"] == "leak")]
        a["real_leak"] = overlaps(a, [(e.onset, e.fail_end) for e in ev.itertuples()])
        rows.append(a.assign(seed=seed, copy=c))
    bench = pd.concat(rows, ignore_index=True)
    val, test = bench[bench["split"] == "val"].copy(), bench[bench["split"] == "test"].copy()
    log.info(f"benchmark alarms: validation {len(val)} ({val['real_leak'].mean():.0%} real), test {len(test)}")

    # ---- real alarms with labels (relevant = failure warning window..maintenance, or an episode window)
    relevant = [(f.start - ph, f.maintenance) for f in fails.itertuples()] + [(e.start - ph, e.end) for e in episodes.itertuples()]
    bins = data["resampled"]["bin_start"]
    real_rows = []
    for p in sorted(src.glob("seed*.csv")):
        s = pd.read_csv(p, parse_dates=["decision_time"])
        a = alarm_table(s, flag)
        t0, t1 = s["decision_time"].min(), s["decision_time"].max()
        days = ((bins >= t0) & (bins <= t1)).sum() * pd.Timedelta(cfg["data"]["resample"]).total_seconds() / 86400
        if len(a):
            a["relevant"] = overlaps(a, relevant)
        real_rows.append((int(p.stem[4:]), a, days))

    # choose the confidence function on the validation alarms: most total coverage over the risk targets
    cov = {}
    for kname, kf in KAPPAS.items():
        cov[kname] = sum(sgr(kf(val), ~val["real_leak"].to_numpy(), r, tc["delta"])["coverage"] for r in tc["risk_targets"])
    kname = max(cov, key=cov.get)
    log.info(f"confidence function chosen on validation alarms: {kname} (summed coverage {cov})")
    for t_ in [val, test]:
        t_["kappa"] = KAPPAS[kname](t_)
    for _, a, _ in real_rows:
        if len(a):
            a["kappa"] = KAPPAS[kname](a)
    out_val, out_test, out_real = [], [], []
    for r_star in tc["risk_targets"]:
        g = sgr(val["kappa"], ~val["real_leak"].to_numpy(), r_star, tc["delta"])
        th = g["theta"]
        out_val.append({"r_star": r_star, "kappa": kname, "theta": th, "guaranteed_bound": round(g["bound"], 3),
                        "val_alarms": g["m"], "val_coverage": round(g["coverage"], 3),
                        "val_risk": None if np.isnan(g["risk"]) else round(g["risk"], 3)})
        conf = test["kappa"] >= th
        tl = events[(events["kind"] == "leak")]
        # leak-level: share of test leaks (per seed and copy) with >= 1 confirmed alarm / with any alarm
        leaks_any, leaks_conf, early_conf = [], [], []
        for (seed, c), a in test.groupby(["seed", "copy"]):
            ev = tl[tl["copy"] == c]
            for e in ev.itertuples():
                ov = ((a["start"] <= e.fail_end) & (a["end"] >= e.onset)).to_numpy()
                if not ov.any():
                    leaks_any.append(False); leaks_conf.append(False); early_conf.append(False)
                    continue
                leaks_any.append(True)
                ca = a[ov & (a["kappa"] >= th).to_numpy()]
                leaks_conf.append(len(ca) > 0)
                early_conf.append(bool(len(ca)) and ca["start"].min() < e.fail_start)
        out_test.append({"r_star": r_star, "test_alarms": len(test), "coverage": round(conf.mean(), 3),
                         "false_share_confirmed": round(float((~test.loc[conf, "real_leak"]).mean()), 3) if conf.any() else None,
                         "false_share_all_alarms": round(float((~test["real_leak"]).mean()), 3),
                         "alarmed_leaks_with_confirmed_alarm": round(np.mean([c_ for a_, c_ in zip(leaks_any, leaks_conf) if a_]), 3)
                         if any(leaks_any) else None,
                         "confirmed_during_buildup": round(np.mean([x for a_, x in zip(leaks_any, early_conf) if a_]), 3)
                         if any(leaks_any) else None})
        for seed, a, days in real_rows:
            if not len(a):
                out_real.append({"r_star": r_star, "seed": seed, "alarms_per_day": 0.0}); continue
            ca = a["kappa"] >= th
            fr = {"r_star": r_star, "seed": seed, "alarms_per_day": len(a) / days,
                  "confirmed_per_day": ca.sum() / days, "review_per_day": (~ca).sum() / days,
                  "false_confirmed_per_day": (ca & ~a["relevant"]).sum() / days,
                  "relevant_alarms_confirmed": float(ca[a["relevant"]].mean()) if a["relevant"].any() else np.nan}
            det, warned = 0, 0
            for f in fails.itertuples():
                hit = a[ca & ((a["start"] <= f.eval_end) & (a["end"] >= f.start - ph))]
                if len(hit):
                    det += 1
                    warned += int(hit["start"].min() <= f.start - warn)
            fr.update(failures_with_confirmed_alarm=det, failures_confirmed_2h_ahead=warned)
            out_real.append(fr)
    vt, tt = pd.DataFrame(out_val), pd.DataFrame(out_test)
    rt = pd.DataFrame(out_real).groupby("r_star").mean(numeric_only=True).drop(columns="seed").round(3)
    for name, t in [("val", vt), ("test", tt)]:
        t.to_csv(d["metrics"] / f"triage_{name}.csv", index=False)
    rt.to_csv(d["metrics"] / "triage_real.csv")
    lines = ["# Alarm triage with a guaranteed risk (Phase 5, human oversight)", "",
             f"Detector `{model}` ({scheme} split), alarms at F = {tc['alarm_target']} false alarms/day (R1-gated). "
             "An alarm is confirmed automatically if its confidence kappa >= theta, otherwise sent to engineer review. "
             f"kappa = `{kname}` (chosen on the validation alarms among {list(KAPPAS)}; 'evidence' = -log10 min "
             "adjusted p, ties at the conformal floor; 'duration' = alarm length, evidence breaks ties). "
             f"theta from SGR (Geifman & El-Yaniv 2017) on the validation alarms of the synthetic "
             f"benchmark: P(share of false confirmed alarms > r*) < delta = {tc['delta']}. theta = inf means no "
             "threshold can carry the guarantee: every alarm goes to review.", "",
             "## Validation (where theta is chosen)", "", "```", vt.to_string(index=False), "```", "",
             "## Synthetic test leaks", "", "`false_share_confirmed` should stay <= r*; compare with "
             "`false_share_all_alarms` (no triage).", "", "```", tt.to_string(index=False), "```", "",
             "## Real data (mean over seeds)", "",
             "`relevant` = alarm overlapping a failure's warning window..maintenance or an unreported episode.", "",
             "```", rt.to_string(), "```"]
    (d["root"] / "TRIAGE_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("\n" + vt.to_string(index=False))
    log.info(f"wrote {d['root'] / 'TRIAGE_RESULTS.md'}")


if __name__ == "__main__":
    main()
