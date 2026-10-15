"""Step 4 - conformal p-values (Phase 5, uncertainty pillar) for every model, scheme and seed, on the real test data
and on every benchmark copy, in three calibration variants: static, rolling, rolling_guard.
Alarm levels: eps = F / (decisions per day) for each operator target F (false alarms per day) in config.
Also writes the calibration-conditional feasibility table (Bates et al. 2023, DKWM)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline import load_processed  # noqa: E402
from src.trustworthy.conformal import decisions_per_day, dkwm_floor, p_values, rolling_p_values  # noqa: E402
from src.utils.config import dirs, failure_table, setup  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

KEEP = ["decision_time", "score", "segment", "part", "frozen"]


def guard_level(eps: dict, n_cal: int) -> float:
    """Strict alarm level for the guard: the smallest eps that can actually be reached. A p-value is at least
    1/(n+1), so eps <= 1/(n+1) never alarms; the guard needs eps >= 2/(n+1) (a score above all but one calibration
    score). Falls back to the largest eps."""
    feasible = [e for e in sorted(eps.values()) if e >= 2 / (n_cal + 1)]
    return feasible[0] if feasible else max(eps.values())


def conformal_frame(df, cal, eps, variant, cc, fails, ph_h):
    """p-values and flags for one part's decisions (df sorted by time)."""
    if variant == "static":
        p = p_values(cal, df["score"])
    else:
        p = rolling_p_values(cal, df["decision_time"], df["score"], guard_level(eps, len(cal)), cc["lag_h"], fails,
                             ph_h, guard=(variant == "rolling_guard"))
    out = df[[c for c in KEEP + ["split"] if c in df]].copy()
    out["p"] = p
    frozen = out["frozen"].astype(bool).to_numpy() if "frozen" in out else np.zeros(len(out), bool)
    for name, e in eps.items():
        out[f"eps_{name}"] = e
        out[f"flag_{name}"] = p < e
        out[f"flag_{name}_gated"] = (p < e) & ~frozen        # R1 gate: frozen data is a data fault, not a leak
    return out


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("04_conformal", d["logs"])
    cc, ph_h = cfg["conformal"], cfg["evaluation"]["predictive_horizon_h"]
    data, fails = load_processed(cfg), failure_table(cfg)
    bins = data["resampled"]["bin_start"]
    targets = {f"fa{f:g}": f for f in cc["fa_targets"]}
    feas = []
    for scheme in cfg["splits"]["schemes"]:
        for model in cfg["models"]:
            for seed in cfg["project"]["seeds"]:
                base = d["scores"] / "baseline" / scheme / model / f"seed{seed}.csv"
                calf = d["scores"] / "calibration" / scheme / model / f"seed{seed}.csv"
                if not (base.exists() and calf.exists()):
                    log.info(f"skip {scheme}/{model}/seed{seed}: run scripts/02_run_baselines.py first")
                    continue
                test = pd.read_csv(base, parse_dates=["decision_time"]).sort_values("decision_time")
                cal = pd.read_csv(calf, parse_dates=["decision_time"])
                bench = sorted((d["scores"] / "benchmark" / scheme / model).glob(f"seed{seed}_copy*.csv"))
                bench = {int(p.stem.split("_copy")[1]): pd.read_csv(p, parse_dates=["decision_time"]) for p in bench}
                outs = {v: [] for v in cc["variants"]}
                bouts = {v: {c: [] for c in bench} for v in cc["variants"]}
                for part, cpart in cal.groupby("part"):
                    cs = cpart["score"].to_numpy()
                    dpd = decisions_per_day(cpart["decision_time"], bins, cfg["data"]["resample"])
                    eps = {k: f / dpd for k, f in targets.items()}
                    floor = dkwm_floor(len(cs), cc["delta"])
                    feas.append({"scheme": scheme, "model": model, "seed": seed, "part": part, "n_cal": len(cs),
                                 "decisions_per_day": round(dpd, 1), "marginal_floor": 1 / (len(cs) + 1),
                                 "min_reachable_F": round(dpd / (len(cs) + 1), 3), "ccv_floor": floor,
                                 "guard_eps": guard_level(eps, len(cs)),
                                 **{f"eps_{k}": e for k, e in eps.items()},
                                 **{f"reachable_{k}": e > 1 / (len(cs) + 1) for k, e in eps.items()},
                                 **{f"ccv_feasible_{k}": floor < e for k, e in eps.items()}})
                    tp = test[test["part"] == part].sort_values("decision_time")
                    for v in cc["variants"]:
                        if len(tp):
                            outs[v].append(conformal_frame(tp, cs, eps, v, cc, fails, ph_h))
                        for c, b in bench.items():
                            bp = b[b["part"] == part].sort_values("decision_time")
                            if not len(bp):
                                continue
                            val = bp[bp["split"] == "val"]
                            tst = bp[bp["split"] == "test"]
                            # validation events are scored against the static calibration (parameter choice only)
                            parts_ = [conformal_frame(val, cs, eps, "static", cc, fails, ph_h)] if len(val) else []
                            if len(tst):
                                parts_.append(conformal_frame(tst, cs, eps, v, cc, fails, ph_h))
                            bouts[v][c] += parts_
                for v in cc["variants"]:
                    od = d["scores"] / f"conformal_{v}" / scheme / model
                    od.mkdir(parents=True, exist_ok=True)
                    if outs[v]:
                        pd.concat(outs[v]).to_csv(od / f"seed{seed}.csv", index=False)
                    bd = d["scores"] / f"conformal_{v}_bench" / scheme / model
                    bd.mkdir(parents=True, exist_ok=True)
                    for c, fr in bouts[v].items():
                        if fr:
                            pd.concat(fr).to_csv(bd / f"seed{seed}_copy{c}.csv", index=False)
                log.info(f"{scheme}/{model}/seed{seed}: {len(test):,} test decisions, {len(bench)} copies, "
                         f"variants {cc['variants']}")
    fe = pd.DataFrame(feas)
    fe.to_csv(d["metrics"] / "conformal_feasibility.csv", index=False)
    log.info("reachable targets (eps above the marginal floor 1/(n+1)) and calibration-conditional feasibility:\n"
             + fe.groupby(["scheme", "model"])[["min_reachable_F"] + [c for c in fe if c.startswith(("reachable", "ccv_feasible"))]]
             .mean().round(2).to_string())


if __name__ == "__main__":
    main()