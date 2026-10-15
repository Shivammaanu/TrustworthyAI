"""Step 1b - synthetic benchmark (Phase 5): copies of the real data with injected air leaks and sensor faults,
placed only in normal time, processed exactly like the real data. Output: data/processed/benchmark/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.benchmark.inject import make_copy, plan_events  # noqa: E402
from src.data import quality as q  # noqa: E402
from src.data.gaps import block_ids  # noqa: E402
from src.data.load import load_raw  # noqa: E402
from src.data.process import representations  # noqa: E402
from src.utils.config import dirs, failure_table, setup  # noqa: E402
from src.utils.fileio import load_json  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("01b_make_benchmark", d["logs"])
    dc, bc, ph = cfg["data"], cfg["benchmark"], pd.Timedelta(hours=cfg["evaluation"]["predictive_horizon_h"])
    meta = load_json(d["processed"] / "meta.json")
    df = load_raw(cfg["paths"]["raw_csv"])
    block = block_ids(df["timestamp"], meta["gap_threshold_s"])
    thr = meta["motor_load_threshold_A"]
    running = q.running_mask(df, "comp_and_motor", dc["running_signal"], dc["running_value"], thr)

    # never touch reported failures, unreported episodes (major: with PH; minor runs) or frozen data
    fails = failure_table(cfg)
    forbidden = [(f.start - ph, f.maintenance) for f in fails.itertuples()]
    win = pd.read_csv(d["processed"] / "event_windows.csv", parse_dates=["start", "end"])
    forbidden += [(w.start - ph, w.end) for w in win[win["source"] != "failure"].itertuples()]
    hdv = pd.read_csv(d["processed"] / "high_dv_runs.csv", parse_dates=["start", "end"])
    forbidden += [(r.start - pd.Timedelta(hours=6), r.end + pd.Timedelta(hours=6)) for r in hdv.itertuples()]
    fz = pd.read_csv(d["processed"] / "frozen_periods.csv", parse_dates=["start", "end"])
    forbidden += [(z.start - pd.Timedelta(hours=1), z.end + pd.Timedelta(hours=1)) for z in fz.itertuples()]

    out_dir = d["processed"] / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_ev = []
    for c in range(bc["n_copies"]):
        rng = np.random.default_rng(bc["seed"] + c)
        ev = plan_events(df["timestamp"], forbidden, rng, bc["spacing_h"])
        ev["kind"] = ev["kind"].astype(str)
        ev["sensor"] = ev["sensor"].astype(str)
        ev.insert(0, "copy", c)
        ev.insert(1, "event", [f"c{c}_{i:03d}" for i in range(len(ev))])
        copy_df = make_copy(df, ev, running, thr, block, bc["seed"] + 1000 + c)
        reps = representations(copy_df, block, thr, dc)
        cdir = out_dir / f"copy{c}"
        cdir.mkdir(exist_ok=True)
        for name, frame in reps.items():
            frame.to_pickle(cdir / f"{name}.pkl")
        all_ev.append(ev)
        log.info(f"copy {c}: {len(ev)} events {ev['kind'].value_counts().to_dict()} | "
                 f"{len(reps['cycles_comp']):,} cycles (comp), {int(reps['resampled']['frozen'].sum())} frozen bins")
    events = pd.concat(all_ev, ignore_index=True)
    events.to_csv(out_dir / "events.csv", index=False)
    log.info(f"{len(events)} events in {bc['n_copies']} copies -> {out_dir}")


if __name__ == "__main__":
    main()