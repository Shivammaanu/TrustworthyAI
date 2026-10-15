"""Step 6 - combine detectors with a controlled false-alarm budget (Bonferroni on the rate scale, static conformal
p-values) and compare with the single detectors on the real data and the synthetic benchmark.
Writes results/COMBINED_RESULTS.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.benchmark import benchmark_events_table, summarise  # noqa: E402
from src.evaluation.report import evaluate_all  # noqa: E402
from src.pipeline import episode_windows, load_processed  # noqa: E402
from src.trustworthy.combine import combine  # noqa: E402
from src.utils.config import dirs, failure_table, setup  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

SRC = "conformal_static"


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("06_combine", d["logs"])
    combos = cfg["combine"]["combinations"]
    data, fails = load_processed(cfg), failure_table(cfg)
    episodes = episode_windows(cfg, data)
    real_bins = data["resampled"][["bin_start", "segment", "frozen"]]
    root = d["processed"] / "benchmark"
    copy_bins = {c: pd.read_pickle(root / f"copy{c}" / "resampled.pkl")[["bin_start", "segment", "frozen"]]
                 for c in range(cfg["benchmark"]["n_copies"])} if (root / "events.csv").exists() else {}
    for scheme in cfg["splits"]["schemes"]:
        for seed in cfg["project"]["seeds"]:
            for name, members in combos.items():
                paths = {m: d["scores"] / SRC / scheme / m / f"seed{seed}.csv" for m in members}
                if not all(p.exists() for p in paths.values()):
                    sys.exit("conformal scores missing - run scripts/04_conformal.py first")
                frames = {m: pd.read_csv(p, parse_dates=["decision_time"]) for m, p in paths.items()}
                out = d["scores"] / "combined" / scheme / name
                out.mkdir(parents=True, exist_ok=True)
                combine(frames, real_bins, cfg["data"]["resample"]).to_csv(out / f"seed{seed}.csv", index=False)
                bout = d["scores"] / "combined_bench" / scheme / name
                bout.mkdir(parents=True, exist_ok=True)
                for c, cb in copy_bins.items():
                    bp = {m: d["scores"] / f"{SRC}_bench" / scheme / m / f"seed{seed}_copy{c}.csv" for m in members}
                    if all(p.exists() for p in bp.values()):
                        bf = {m: pd.read_csv(p, parse_dates=["decision_time"]) for m, p in bp.items()}
                        combine(bf, cb, cfg["data"]["resample"]).to_csv(bout / f"seed{seed}_copy{c}.csv", index=False)
            log.info(f"{scheme}/seed{seed}: combinations {list(combos)} built")

    # evaluate the combinations; single detectors (static conformal) come from step 05
    names = list(combos)
    _, summ, _, _ = evaluate_all(cfg, d, fails, data, episodes, mode="combined", info_mode="baseline", models=names,
                                 info_model=cfg["combine"]["window_model"])
    singles = pd.read_csv(d["metrics"] / f"{SRC}_summary_per_seed.csv")
    real = pd.concat([summ, singles], ignore_index=True)
    keep = [f"fa{f:g}" for f in cfg["combine"]["report_targets"]]
    keep += [k + "_gated" for k in keep]
    real = real[real["threshold"].isin(keep)]
    cols = ["false_alarms_per_day_clean", "detected", "detected_by_chance", "meets_warning", "meets_warning_by_chance",
            "episodes_detected", "episodes_detected_by_chance", "fa_in_frozen"]
    rt = real.groupby(["scheme", "threshold", "model"])[cols].mean().round(2)
    rt.to_csv(d["metrics"] / "combined_real_summary.csv")
    lines = ["# Combined detectors (Phase 5, step 3)", "",
             "Bonferroni on the rate scale: budget F split evenly over the m detectors, detector k alarms when "
             "p_k < eps_k(F)/m; the union keeps at most F false alarms per day. Static conformal p-values. "
             f"Combinations: {combos}. Single detectors (static conformal) shown for comparison. Means over seeds. "
             "`_gated` = with the R1 data-quality gate: alarms on frozen data are reported as data faults, not leaks. "
             "`meets_warning_by_chance` = >= 2 h warnings expected from random alarms at the same rate.", ""]
    fe = d["metrics"] / "conformal_feasibility.csv"
    if fe.exists():
        fe = pd.read_csv(fe)
        reach = fe.groupby(["scheme", "model"])["min_reachable_F"].max()
        lines += ["**Reachability:** each detector gets F/m, so a combination can only alarm if F/m is above every "
                  "member's smallest reachable target (max over parts and seeds, false alarms per day):", "", "```"]
        for name, members in combos.items():
            for scheme in cfg["splits"]["schemes"]:
                need = len(members) * max(reach.get((scheme, m), 0) for m in members)
                lines.append(f"{name:12s} {scheme:8s} needs F >= {need:.2f}")
        lines += ["```", ""]
    lines += ["## Real data", "", "```", rt.to_string(), "```", ""]
    ev = benchmark_events_table(cfg, d, "combined_bench", models=names)
    single_ev = d["metrics"] / f"{SRC}_benchmark_events.csv"
    if len(ev) and single_ev.exists():
        ev = pd.concat([ev, pd.read_csv(single_ev)], ignore_index=True)
        ev = ev[ev["threshold"].isin(keep)]
        ev.to_csv(d["metrics"] / "combined_benchmark_events.csv", index=False)
        leaks, sens = summarise(ev)
        lk = leaks.xs("all", level="sev").xs("test", level="split").reorder_levels(["scheme", "threshold", "model"]).sort_index()
        lines += ["## Synthetic leaks (test split, all severities)", "", "```",
                  lk.drop(columns=["n"]).round(3).to_string(), "```", "",
                  "## Sensor faults: share raising a (false) leak alarm (test split)", "", "```",
                  sens.xs("test", level="split")["false_leak_alarm"].unstack("kind")
                  .reorder_levels(["scheme", "threshold", "model"]).sort_index().round(3).to_string(), "```"]
    (d["root"] / "COMBINED_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"wrote {d['root'] / 'COMBINED_RESULTS.md'}")


if __name__ == "__main__":
    main()