"""Step 7 - EWMA score smoothing before the conformal step (Phase 5, step 4; Hundman et al. 2018).
1. For every span H in the grid (hours), smooth calibration and validation-copy scores and measure >= 2 h warnings
   on the VALIDATION leaks only, at the selection target F.
2. Choose H per scheme and model: most validation warnings ABOVE CHANCE (meets_warning - chance level of the
   same alarm rate; ties -> shorter span). The real failures and test-window leaks are never used for the choice.
3. Apply the chosen span: static conformal p-values on smoothed scores (with R1-gated variants), real data and
   benchmark copies; rebuild the combinations from the smoothed members.
4. Evaluate and compare with the unsmoothed results. Writes results/SMOOTHING_RESULTS.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.benchmark import benchmark_events_table, evaluate_copy, summarise  # noqa: E402
from src.evaluation.report import evaluate_all  # noqa: E402
from src.pipeline import episode_windows, load_processed  # noqa: E402
from src.trustworthy.combine import combine  # noqa: E402
from src.trustworthy.conformal import decisions_per_day, p_values  # noqa: E402
from src.trustworthy.smoothing import ewma  # noqa: E402
from src.utils.config import dirs, failure_table, setup  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

KEEP = ["decision_time", "score", "segment", "part", "frozen"]


def frame(df, cal_s, score_s, eps):
    out = df[[c for c in KEEP + ["split"] if c in df]].copy()
    out["score_raw"], out["score"] = out["score"], score_s
    p = p_values(cal_s, score_s)
    out["p"] = p
    fz = out["frozen"].astype(bool).to_numpy()
    for name, e in eps.items():
        out[f"eps_{name}"] = e
        out[f"flag_{name}"] = p < e
        out[f"flag_{name}_gated"] = (p < e) & ~fz
    return out


def load_run(d, scheme, model, seed):
    cal = pd.read_csv(d["scores"] / "calibration" / scheme / model / f"seed{seed}.csv", parse_dates=["decision_time"])
    test = pd.read_csv(d["scores"] / "baseline" / scheme / model / f"seed{seed}.csv", parse_dates=["decision_time"])
    bench = {int(p.stem.split("_copy")[1]): pd.read_csv(p, parse_dates=["decision_time"])
             for p in sorted((d["scores"] / "benchmark" / scheme / model).glob(f"seed{seed}_copy*.csv"))}
    return cal.sort_values("decision_time"), test.sort_values("decision_time"), bench


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("07_smoothing", d["logs"])
    sc, ph_h, warn = cfg["smoothing"], cfg["evaluation"]["predictive_horizon_h"], cfg["evaluation"]["warning_h"]
    data, fails = load_processed(cfg), failure_table(cfg)
    episodes = episode_windows(cfg, data)
    bins = data["resampled"]["bin_start"]
    targets = {f"fa{f:g}": f for f in cfg["conformal"]["fa_targets"]}
    sel_name = f"fa{sc['select_target']:g}"
    root = d["processed"] / "benchmark"
    events = pd.read_csv(root / "events.csv", parse_dates=["onset", "fail_start", "fail_end"])
    copy_bins = {c: pd.read_pickle(root / f"copy{c}" / "resampled.pkl") for c in range(cfg["benchmark"]["n_copies"])}
    real_win = [(f.start - pd.Timedelta(hours=ph_h), f.maintenance) for f in fails.itertuples()]

    # ---- 1-2: choose the span on validation leaks
    curve = []
    for scheme in cfg["splits"]["schemes"]:
        for model in cfg["models"]:
            for seed in cfg["project"]["seeds"]:
                cal, _, bench = load_run(d, scheme, model, seed)
                for part, cp in cal.groupby("part"):
                    dph = decisions_per_day(cp["decision_time"], bins, cfg["data"]["resample"]) / 24
                    eps = {sel_name: targets[sel_name] / (dph * 24)}
                    for H in sc["spans_h"]:
                        span = H * dph
                        cal_s = ewma(cp, span)
                        for c, b in bench.items():
                            bv = b[(b["part"] == part) & (b["split"] == "val")].sort_values("decision_time")
                            if not len(bv):
                                continue
                            f = frame(bv, cal_s, ewma(bv, span), eps)
                            r = evaluate_copy(f, f[f"flag_{sel_name}_gated"].to_numpy(), events[events["copy"] == c],
                                              ph_h, warn, real_win, copy_bins[c]["bin_start"], cfg["data"]["resample"])
                            if r.empty:                  # no validation event of this copy in this part
                                continue
                            r = r[(r["kind"] == "leak") & (r["split"] == "val")]
                            if len(r):
                                curve.append(r.assign(scheme=scheme, model=model, seed=seed, span_h=H))
    curve = pd.concat(curve, ignore_index=True)
    cv = curve.groupby(["scheme", "model", "span_h"]).agg(
        n=("event", "size"), detected=("detected", "mean"), early=("early", "mean"),
        meets_warning=("meets_warning", "mean"), chance=("chance", "mean"))
    cv["excess_over_chance"] = cv["meets_warning"] - cv["chance"]
    cv = cv.round(3)
    cv.to_csv(d["metrics"] / "smoothing_validation_curve.csv")
    best, adopted = {}, {}
    for (scheme, model), g in cv.reset_index().groupby(["scheme", "model"]):
        g = g.sort_values(["excess_over_chance", "span_h"], ascending=[False, True])
        best[(scheme, model)] = float(g.iloc[0]["span_h"])
        top = float(g.iloc[0]["excess_over_chance"])
        base = float(g.loc[g["span_h"] == 0, "excess_over_chance"].iloc[0]) if (g["span_h"] == 0).any() else -np.inf
        # adopt smoothing only if it gives warnings ABOVE chance and beats no smoothing; otherwise 0 h
        adopted[(scheme, model)] = best[(scheme, model)] if (top > 0 and top > base) else 0.0
    log.info("best spans (h) on validation leaks: " + str({f"{k[0]}/{k[1]}": v for k, v in best.items()}))
    log.info("adopted spans (h) (0 unless above chance and better than none): "
             + str({f"{k[0]}/{k[1]}": v for k, v in adopted.items()}))

    # ---- 3: apply the chosen span to real test data and all benchmark rows
    for scheme in cfg["splits"]["schemes"]:
        for model in cfg["models"]:
            H = best[(scheme, model)]
            for seed in cfg["project"]["seeds"]:
                cal, test, bench = load_run(d, scheme, model, seed)
                outs, bouts = [], {c: [] for c in bench}
                for part, cp in cal.groupby("part"):
                    dpd = decisions_per_day(cp["decision_time"], bins, cfg["data"]["resample"])
                    eps = {k: f / dpd for k, f in targets.items()}
                    span = H * dpd / 24
                    cal_s = ewma(cp, span)
                    tp = test[test["part"] == part]
                    if len(tp):
                        outs.append(frame(tp, cal_s, ewma(tp, span), eps))
                    for c, b in bench.items():
                        bp = b[b["part"] == part].sort_values("decision_time")
                        for sp in ("val", "test"):
                            x = bp[bp["split"] == sp]
                            if len(x):
                                bouts[c].append(frame(x, cal_s, ewma(x, span), eps))
                od = d["scores"] / "conformal_ewma" / scheme / model
                od.mkdir(parents=True, exist_ok=True)
                pd.concat(outs).to_csv(od / f"seed{seed}.csv", index=False)
                bd = d["scores"] / "conformal_ewma_bench" / scheme / model
                bd.mkdir(parents=True, exist_ok=True)
                for c, fr in bouts.items():
                    if fr:
                        pd.concat(fr).to_csv(bd / f"seed{seed}_copy{c}.csv", index=False)
            log.info(f"{scheme}/{model}: smoothed with span {H} h")
    combos = cfg["combine"]["combinations"]
    real_bins = data["resampled"][["bin_start", "segment", "frozen"]]
    for scheme in cfg["splits"]["schemes"]:
        for seed in cfg["project"]["seeds"]:
            for name, members in combos.items():
                fr = {m: pd.read_csv(d["scores"] / "conformal_ewma" / scheme / m / f"seed{seed}.csv",
                                     parse_dates=["decision_time"]) for m in members}
                out = d["scores"] / "combined_ewma" / scheme / name
                out.mkdir(parents=True, exist_ok=True)
                combine(fr, real_bins, cfg["data"]["resample"]).to_csv(out / f"seed{seed}.csv", index=False)
                bout = d["scores"] / "combined_ewma_bench" / scheme / name
                bout.mkdir(parents=True, exist_ok=True)
                for c, cb in copy_bins.items():
                    paths = {m: d["scores"] / "conformal_ewma_bench" / scheme / m / f"seed{seed}_copy{c}.csv" for m in members}
                    if all(p.exists() for p in paths.values()):
                        bf = {m: pd.read_csv(p, parse_dates=["decision_time"]) for m, p in paths.items()}
                        combine(bf, cb[["bin_start", "segment", "frozen"]], cfg["data"]["resample"]).to_csv(
                            bout / f"seed{seed}_copy{c}.csv", index=False)

    # ---- 4: evaluate and compare (gated variants, reported targets)
    names = list(combos)
    keep = [f"fa{f:g}_gated" for f in cfg["combine"]["report_targets"]]
    _, s_single, _, _ = evaluate_all(cfg, d, fails, data, episodes, mode="conformal_ewma", info_mode="baseline")
    _, s_comb, _, _ = evaluate_all(cfg, d, fails, data, episodes, mode="combined_ewma", info_mode="baseline",
                                   models=names, info_model=cfg["combine"]["window_model"])
    new = pd.concat([s_single, s_comb]).assign(smoothing="ewma")
    old = pd.concat([pd.read_csv(d["metrics"] / "conformal_static_summary_per_seed.csv"),
                     pd.read_csv(d["metrics"] / "combined_summary_per_seed.csv")]).assign(smoothing="none")
    real = pd.concat([old, new])
    real = real[real["threshold"].isin(keep)]
    cols = ["false_alarms_per_day_clean", "detected", "detected_by_chance", "meets_warning", "meets_warning_by_chance",
            "episodes_detected", "episodes_detected_by_chance"]
    rt = real.groupby(["scheme", "threshold", "model", "smoothing"])[cols].mean().round(2)
    rt.to_csv(d["metrics"] / "smoothing_real_summary.csv")
    ev_new = pd.concat([benchmark_events_table(cfg, d, "conformal_ewma_bench"),
                        benchmark_events_table(cfg, d, "combined_ewma_bench", models=names)]).assign(smoothing="ewma")
    ev_old = pd.read_csv(d["metrics"] / "combined_benchmark_events.csv").assign(smoothing="none")
    ev = pd.concat([ev_old, ev_new])
    ev = ev[ev["threshold"].isin(keep) & (ev["kind"] == "leak") & (ev["split"] == "test")]
    bt = ev.groupby(["scheme", "threshold", "model", "smoothing"]).agg(
        detected=("detected", "mean"), chance=("chance", "mean"), early=("early", "mean"),
        meets_warning=("meets_warning", "mean"), lead_median_h=("lead_h", "median")).round(3)
    bt.to_csv(d["metrics"] / "smoothing_benchmark_leaks.csv")
    span_tab = pd.DataFrame({"best span on validation (h), evaluated below": {f"{k[0]}/{k[1]}": v for k, v in best.items()},
                             "adopted span (h)": {f"{k[0]}/{k[1]}": v for k, v in adopted.items()}})
    lines = ["# Score smoothing before the conformal step (Phase 5, step 4)", "",
             f"Causal EWMA within data segments; span grid {sc['spans_h']} h; chosen per scheme and model on the "
             f"VALIDATION leaks only (most >= 2 h warnings above chance at F = {sc['select_target']}, R1-gated; ties -> "
             "shorter span). Static conformal calibration on smoothed scores. All rows below are R1-gated. Means over seeds.", "",
             "## Spans", "",
             "The comparison below uses the best validation span so the effect of smoothing is visible; a span is "
             "adopted for the operating system only if it gives >= 2 h warnings above chance on the validation leaks "
             "and beats no smoothing (otherwise 0 h).", "", "```", span_tab.to_string(), "```", "",
             "## Validation curve (synthetic validation leaks; the only data used for the choice)", "", "```",
             cv.to_string(), "```", "",
             "## Real data: unsmoothed (none) vs smoothed (ewma)", "", "```", rt.to_string(), "```", "",
             "## Synthetic test leaks: unsmoothed vs smoothed", "", "```", bt.to_string(), "```"]
    (d["root"] / "SMOOTHING_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"wrote {d['root'] / 'SMOOTHING_RESULTS.md'}")


if __name__ == "__main__":
    main()