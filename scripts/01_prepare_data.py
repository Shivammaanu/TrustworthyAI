"""Step 1 - read the raw CSV once and derive everything from the data:
gaps, frozen periods (R1), the load-current threshold (R2), compressor cycles under both running rules,
the 44 cycle features, the 10-minute bins, unreported fault episodes (R3) and the event-window table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import quality as q  # noqa: E402
from src.data.cycles import cycle_features, segment_cycles  # noqa: E402
from src.data.gaps import block_ids, gap_threshold  # noqa: E402
from src.data.load import load_raw  # noqa: E402
from src.data.resample import resample  # noqa: E402
from src.utils.config import PROJECT_ROOT, dirs, failure_table, setup  # noqa: E402
from src.utils.fileio import save_json  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

RULES = ["comp", "comp_and_motor"]


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("01_prepare_data", d["logs"])
    dc, ph = cfg["data"], cfg["evaluation"]["predictive_horizon_h"]
    log.info(f"reading {cfg['paths']['raw_csv']}")
    df = load_raw(cfg["paths"]["raw_csv"])
    log.info(f"{len(df):,} rows, {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    fails = failure_table(cfg)

    gap_s = gap_threshold(df["timestamp"], dc["gap_threshold_s"])
    block = block_ids(df["timestamp"], gap_s)
    log.info(f"gap threshold {gap_s:.0f} s -> {block.max() + 1:,} gap-free blocks")

    # R1 frozen data
    fz_bins = q.frozen_bins(df, dc["frozen_bin"])
    frozen = q.frozen_rows(df, fz_bins, dc["frozen_bin"])
    fz_per = q.frozen_periods(fz_bins, dc["frozen_bin"])
    log.info(f"R1 frozen: {int(fz_bins.sum())} of {len(fz_bins)} bins ({fz_bins.mean():.2%}) in {len(fz_per)} periods, "
             f"{fz_per['hours'].sum():.0f} h")

    # R2 running rule
    motor_thr = q.loaded_threshold(df, dc["running_signal"], dc["running_value"])
    log.info(f"R2 load-current threshold (Otsu over rows with {dc['running_signal']} == {dc['running_value']}): {motor_thr:.2f} A")

    meta = {"gap_threshold_s": gap_s, "n_rows": len(df), "start": df["timestamp"].iloc[0], "end": df["timestamp"].iloc[-1],
            "n_blocks": int(block.max() + 1), "motor_load_threshold_A": motor_thr,
            "frozen_hours": float(fz_per["hours"].sum()), "resample": dc["resample"]}
    runs_by_rule = {}
    for rule in RULES:
        running = q.running_mask(df, rule, dc["running_signal"], dc["running_value"], motor_thr)
        runs_by_rule[rule] = q.runs_table(df, running, block, frozen)
        cid, cycles = segment_cycles(df, running, block)
        feats = cycle_features(df, cid, cycles, running, dc["cycle_bins"])
        cyc = cycles.set_index("cycle").join(feats).reset_index()
        m = cid >= 0
        cyc["frozen"] = pd.Series(frozen[m]).groupby(cid[m]).max().reindex(cyc["cycle"]).to_numpy(bool)
        cyc.to_pickle(d["processed"] / f"cycles_{rule}.pkl")
        meta["cycle_features"] = list(feats.columns)
        meta[f"n_cycles_{rule}"] = len(cyc)
        log.info(f"running rule '{rule}': {len(cyc):,} cycles ({int(cyc['frozen'].sum())} touch frozen data) | "
                 f"run median {np.median(feats['T_run']) / 60:.1f} min, idle median {np.median(feats['T_idle']) / 60:.1f} min")
        for f in fails.itertuples():
            ov = cyc[(cyc["start"] <= f.maintenance) & (cyc["decision_time"] >= f.start)]
            longest = ov["T_run"].max()
            log.info(f"   {f.name}: {len(ov)} cycles overlap failure->maintenance; longest run "
                     f"{'n/a (no complete cycle)' if np.isnan(longest) else f'{longest / 3600:.1f} h'}")

    res = resample(df, dc["resample"])
    res["frozen"] = fz_bins.reindex(res["bin_start"]).fillna(False).to_numpy(bool)
    res.to_pickle(d["processed"] / "resampled.pkl")
    log.info(f"{len(res):,} non-empty {dc['resample']} bins in {res['segment'].max() + 1:,} segments "
             f"({int(res['frozen'].sum())} frozen)")

    # R3 unreported episodes (always derived with the R2 rule: COMP flickers are not runs)
    episodes, breaks = q.derive_episodes(runs_by_rule["comp_and_motor"], fails, ph)
    meta.update(breaks)
    log.info(f"R3 stage 1: fragment break {breaks['fragment_break_min']:.2f} min -> {breaks['n_runs_above_fragment_break']:,} runs")
    log.info(f"R3 stage 2: long-run break {breaks['run_length_break_min']:.1f} min, DV pressure break "
             f"{breaks['dv_pressure_break_bar']:.2f} bar -> {len(episodes)} high-DV long runs")
    log.info(f"R3 stage 3: major-episode break {breaks['major_episode_break_min'] / 60:.1f} h -> "
             f"{breaks['n_major']} major episodes (event windows), {breaks['n_minor']} minor high-DV runs")
    for e in episodes[episodes["major"]].itertuples():
        log.info(f"   {e.name}: {e.start} -> {e.end} ({e.minutes / 60:.1f} h, DV {e.dv_mean:.2f} bar)")
    if (~episodes["major"]).any():
        mn = episodes[~episodes["major"]]
        log.info(f"   minor runs: {mn['start'].min()} -> {mn['end'].max()}, {mn['minutes'].min():.0f}-{mn['minutes'].max():.0f} min each; "
                 f"per month {mn['start'].dt.to_period('M').value_counts().sort_index().to_dict()}")
    episodes.to_csv(d["processed"] / "high_dv_runs.csv", index=False)
    meta["n_episodes_derived"] = breaks["n_major"]

    company = PROJECT_ROOT / cfg["data_quality"]["company_windows"]
    windows = q.event_windows(fails, episodes, company if company.exists() else None)
    log.info(f"event windows: {windows['source'].value_counts().to_dict()}"
             + ("" if company.exists() else f" (no {company.name} yet: derived episodes are used)"))

    fz_per.to_csv(d["processed"] / "frozen_periods.csv", index=False)
    windows.to_csv(d["processed"] / "event_windows.csv", index=False)
    save_json(meta, d["processed"] / "meta.json")
    log.info(f"saved to {d['processed']}")


if __name__ == "__main__":
    main()