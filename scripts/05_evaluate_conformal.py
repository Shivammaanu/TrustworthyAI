"""Step 5 - evaluate the conformal layer: real failures and episodes (same metrics as the baselines), synthetic
benchmark, empirical validity per month (observed false-alarm share / eps on clean normal decisions) and the
calibration-conditional feasibility. Writes results/CONFORMAL_RESULTS.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.benchmark import benchmark_events_table, summarise  # noqa: E402
from src.evaluation.report import evaluate_all  # noqa: E402
from src.pipeline import episode_windows, load_processed  # noqa: E402
from src.utils.config import dirs, failure_table, setup  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def validity(cfg, d, v, failures, episodes, frozen_periods):
    """Observed share of clean normal decisions with p < eps, divided by eps, per month (1 = exactly as promised)."""
    ph = pd.Timedelta(hours=cfg["evaluation"]["predictive_horizon_h"])
    rows = []
    for path in sorted((d["scores"] / f"conformal_{v}").glob("*/*/seed*.csv")):
        scheme, model, seed = path.parts[-3], path.parts[-2], int(path.stem[4:])
        s = pd.read_csv(path, parse_dates=["decision_time"])
        t = s["decision_time"]
        clean = ~s["frozen"].astype(bool)
        for f in failures.itertuples():
            clean &= ~((t >= f.start - ph) & (t <= f.maintenance))
        for e in episodes.itertuples():
            clean &= ~((t >= e.start - ph) & (t <= e.end))
        s = s[clean]
        for name in [c[5:] for c in s.columns if c.startswith("flag_")]:
            g = s.groupby(s["decision_time"].dt.to_period("M"))
            ratio = g[f"flag_{name}"].mean() / g[f"eps_{name.replace('_gated', '')}"].mean()
            for m, r in ratio.items():
                rows.append({"variant": v, "scheme": scheme, "model": model, "seed": seed, "target": name,
                             "month": str(m), "observed_over_eps": r})
    return pd.DataFrame(rows)


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("05_evaluate_conformal", d["logs"])
    data, fails = load_processed(cfg), failure_table(cfg)
    episodes = episode_windows(cfg, data)
    real_cols = ["false_alarms_per_day_clean", "detected", "detected_by_chance", "meets_warning",
                 "meets_warning_by_chance", "episodes_detected", "episodes_detected_by_chance"]
    real_tabs, bench_tabs, sens_tabs, val_tabs = [], [], [], []
    for v in cfg["conformal"]["variants"]:
        if not (d["scores"] / f"conformal_{v}").exists():
            sys.exit("no conformal scores - run scripts/04_conformal.py first")
        _, summ, _, lead = evaluate_all(cfg, d, fails, data, episodes, mode=f"conformal_{v}", info_mode="baseline")
        real_tabs.append(summ.assign(variant=v))
        ev = benchmark_events_table(cfg, d, f"conformal_{v}_bench")
        if len(ev):
            ev.to_csv(d["metrics"] / f"conformal_{v}_benchmark_events.csv", index=False)
            leaks, sens = summarise(ev)
            bench_tabs.append(leaks.reset_index().assign(variant=v))
            sens_tabs.append(sens.reset_index().assign(variant=v))
        val_tabs.append(validity(cfg, d, v, fails, episodes, data["frozen"]))

    real = pd.concat(real_tabs)
    real["target_F"] = real["threshold"].str.extract(r"fa([0-9.]+)")[0].astype(float)
    rt = real.groupby(["variant", "scheme", "model", "threshold"])[real_cols + ["target_F"]].mean().round(3)
    rt.to_csv(d["metrics"] / "conformal_real_summary.csv")
    lines = ["# Conformal layer results (Phase 5, step 2)", "",
             "Alarm if p < eps, eps = F / decisions per day, F = the operator's accepted false alarms per day "
             "(`threshold` fa0.1 means F = 0.1). Means over seeds.", "",
             "## Real data: failures, unreported episodes, clean false alarms", "",
             "`false_alarms_per_day_clean` should stay near `target_F`; detections only count if clearly above their "
             "chance level.", "", "```", rt.to_string(), "```", ""]
    if bench_tabs:
        bt = pd.concat(bench_tabs)
        bt = bt[bt["sev"] == "all"].set_index(["variant", "scheme", "model", "split", "threshold"]).drop(columns="sev")
        bt.to_csv(d["metrics"] / "conformal_benchmark_leaks.csv")
        st = pd.concat(sens_tabs).set_index(["variant", "scheme", "model", "split", "threshold", "kind"])
        st.to_csv(d["metrics"] / "conformal_benchmark_sensor_faults.csv")
        lines += ["## Synthetic leaks (all severities; n counts each event once per seed)", "", "```",
                  bt.xs("test", level="split").round(3).to_string(), "```", "",
                  "## Sensor faults: share raising a (false) leak alarm, test split", "", "```",
                  st.xs("test", level="split")["false_leak_alarm"].unstack("kind").round(3).to_string(), "```", ""]
    va = pd.concat(val_tabs)
    va.to_csv(d["metrics"] / "conformal_validity_by_month.csv", index=False)
    vt = va.groupby(["variant", "scheme", "model", "target", "month"])["observed_over_eps"].mean().unstack("month").round(2)
    lines += ["## Validity: observed false-alarm share / eps on clean normal decisions, by month", "",
              "1.0 = exactly as promised; > 1 = more false alarms than promised (e.g. drift); < 1 = conservative.", "",
              "```", vt.to_string(), "```", ""]
    fe = pd.read_csv(d["metrics"] / "conformal_feasibility.csv")
    ft = fe.groupby(["scheme", "model"])[["n_cal", "decisions_per_day", "marginal_floor", "min_reachable_F",
                                          "ccv_floor", "guard_eps"] + [c for c in fe if c.startswith("eps_")]].mean()
    lines += ["## Feasibility: which targets can be reached at all", "",
              "A p-value is at least `marginal_floor` = 1/(n+1), so a target F below `min_reachable_F` (= decisions per day "
              "/ (n+1)) can never alarm. With the calibration-conditional guarantee (Bates et al., DKWM, delta = "
              f"{cfg['conformal']['delta']}) an alarm needs eps > `ccv_floor`. `guard_eps` = strict level used by "
              "rolling_guard (smallest reachable eps).", "", "```",
              ft.to_string(float_format=lambda x: f"{x:.2g}"), "```"]
    (d["root"] / "CONFORMAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"wrote {d['root'] / 'CONFORMAL_RESULTS.md'}")


if __name__ == "__main__":
    main()