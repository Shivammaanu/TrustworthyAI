"""Step 3b - evaluate every model on the synthetic benchmark (leak detection by severity, lead time, chance level,
false leak alarms during sensor faults). Writes results/BENCHMARK_RESULTS.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.benchmark import benchmark_events_table, summarise  # noqa: E402
from src.utils.config import dirs, failure_table, setup  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("03b_evaluate_benchmark", d["logs"])
    ev_path = d["processed"] / "benchmark" / "events.csv"
    if not ev_path.exists():
        sys.exit("no benchmark - run scripts/01b_make_benchmark.py and scripts/02_run_baselines.py first")
    events = pd.read_csv(ev_path, parse_dates=["onset", "fail_start", "fail_end"])
    ph_h, warn = cfg["evaluation"]["predictive_horizon_h"], cfg["evaluation"]["warning_h"]
    real = [(f.start - pd.Timedelta(hours=ph_h), f.maintenance) for f in failure_table(cfg).itertuples()]
    ev = benchmark_events_table(cfg, d, "benchmark")
    if not len(ev):
        sys.exit("no benchmark scores - run scripts/02_run_baselines.py after creating the benchmark")
    ev.to_csv(d["metrics"] / "benchmark_events.csv", index=False)
    leaks, sens = summarise(ev)
    leaks.to_csv(d["metrics"] / "benchmark_leaks.csv")
    sens.to_csv(d["metrics"] / "benchmark_sensor_faults.csv")
    n_ev = events.groupby("kind").size().to_dict()
    lines = ["# Synthetic benchmark results", "",
             f"{len(events)} injected events in {cfg['benchmark']['n_copies']} copies: {n_ev}. `n` counts each event once "
             "per seed (distinct events = n / number of seeds). "
             "Parameters may only be chosen on split = val; split = test is for reporting.", "",
             "`detected` = share of leaks with >= 1 alarm between onset and the end of the failure phase; `chance` = "
             "same share expected from random alarms at the background rate; `early` = alarm during the build-up; "
             "`meets_warning` = first alarm >= 2 h before the failure phase.", "",
             "## Leaks (by severity)", "", "```", leaks.to_string(), "```", "",
             "## Sensor faults (should NOT raise leak alarms)", "",
             "`false_leak_alarm` = share of faults with >= 1 leak alarm; `r1_frozen` = share of the fault's decisions "
             "marked frozen by R1.", "", "```", sens.to_string(), "```"]
    (d["root"] / "BENCHMARK_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("\n" + leaks.xs("all", level="sev").to_string())
    log.info(f"wrote {d['root'] / 'BENCHMARK_RESULTS.md'}")


if __name__ == "__main__":
    main()