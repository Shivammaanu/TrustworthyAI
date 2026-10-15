"""Step 9 - per-alarm explanations (Phase 5, transparency) for the fixed-split operating detectors.
Re-trains each model exactly as step 02 (same seed -> same model; the per-signal errors were not saved then),
explains every alarm at its peak decision (TranAD per-signal error, LSTM-AE per-feature error, Isolation Forest DIFFI),
groups attributions by physical sensor and tests them:
  physics   - alarms during real failures should point at the air-leak signature groups (Phase 0 data)
  stability - the three seeds' models should give the same top group for the same alarm
  frozen    - alarms on frozen data show what the detector reacts to there (why the R1 gate is needed)
Writes results/EXPLANATIONS_RESULTS.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.pipeline import episode_windows, load_processed, train_fixed  # noqa: E402
from src.trustworthy.explain import GROUPS, LEAK_GROUPS, diffi_local, group_shares  # noqa: E402
from src.trustworthy.triage import alarm_table  # noqa: E402
from src.utils.config import dirs, failure_table, resolve_device, setup  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def attributions(model, det, w, x, idx):
    if model == "iforest":
        return diffi_local(det.model, x[idx])
    return det.explain_dims(w[idx])


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("09_explain", d["logs"])
    ec = cfg["explain"]
    flag = f"flag_fa{ec['alarm_target']:g}"            # ungated: frozen-data alarms are explained too
    ph = pd.Timedelta(hours=cfg["evaluation"]["predictive_horizon_h"])
    data, fails, device = load_processed(cfg), failure_table(cfg), resolve_device(cfg)
    episodes = episode_windows(cfg, data)
    rows, stab = [], []
    for model in cfg["models"]:
        trained = {seed: train_fixed(cfg, model, seed, data, fails, device, log) for seed in cfg["project"]["seeds"]}
        ref_seed = cfg["project"]["seeds"][0]
        _, _, _, times, _, _, names = trained[ref_seed]
        pos = pd.Series(np.arange(len(times)), index=times.values)
        pos = pos[~pos.index.duplicated()]
        peaks_ref = None
        for seed, (det, w, x, _, _, frozen, _) in trained.items():
            cs = pd.read_csv(d["scores"] / "conformal_static" / "fixed" / model / f"seed{seed}.csv", parse_dates=["decision_time"])
            al = alarm_table(cs.assign(p_adj=cs["p"]), flag, "p_adj")
            if not len(al):
                continue
            # peak decision of each alarm = highest score inside it
            cs_idx = cs.set_index("decision_time")["score"]
            peaks = [cs_idx.loc[a.start:a.end].idxmax() for a in al.itertuples()]
            idx = pos.reindex(peaks).dropna().astype(int).to_numpy()
            pk = times.iloc[idx].to_numpy()
            gs = group_shares(attributions(model, det, w, x, idx), names)
            ctx = np.full(len(idx), "false", dtype=object)
            fz = frozen[idx]
            for e in episodes.itertuples():
                ctx[(pk >= np.datetime64(e.start - ph)) & (pk <= np.datetime64(e.end))] = "episode"
            for f in fails.itertuples():
                ctx[(pk >= np.datetime64(f.start - ph)) & (pk <= np.datetime64(f.maintenance))] = "failure"
            ctx[fz] = "frozen"
            frame = pd.DataFrame({"model": model, "seed": seed, "peak_time": pk, "context": ctx, **gs})
            frame["top_group"] = frame[list(GROUPS)].idxmax(axis=1)
            frame["leak_share"] = frame[LEAK_GROUPS].sum(axis=1)
            rows.append(frame)
            if seed == ref_seed:
                peaks_ref = idx
        # stability: explain the reference seed's alarm peaks with every seed's model
        if peaks_ref is not None and len(peaks_ref):
            per_seed = {s: attributions(model, det, w, x, peaks_ref) for s, (det, w, x, *_) in trained.items()}
            seeds = list(per_seed)
            rho, agree = [], []
            for i in range(len(peaks_ref)):
                vals = [per_seed[s][i] for s in seeds]
                for a in range(len(seeds)):
                    for b in range(a + 1, len(seeds)):
                        r = spearmanr(vals[a], vals[b]).correlation
                        rho.append(0.0 if np.isnan(r) else r)
                tops = [max(GROUPS, key=lambda g, s=s: group_shares(per_seed[s][i:i + 1], names)[g][0]) for s in seeds]
                agree.append(len(set(tops)) == 1)
            stab.append({"model": model, "alarms": len(peaks_ref), "mean_pairwise_spearman": round(float(np.mean(rho)), 3),
                         "top_group_same_in_all_seeds": round(float(np.mean(agree)), 3)})
        log.info(f"{model}: explained {sum(len(r) for r in rows if r['model'].iloc[0] == model)} alarm peaks")
    ex = pd.concat(rows, ignore_index=True)
    ex.to_csv(d["metrics"] / "explanations_alarms.csv", index=False)
    gm = ex.groupby(["model", "context"])[list(GROUPS) + ["leak_share"]].mean().round(3)
    gm.insert(0, "alarms", ex.groupby(["model", "context"]).size())
    top = ex.groupby(["model", "context"])["top_group"].agg(lambda s: s.value_counts(normalize=True).round(2).to_dict())
    base = len(LEAK_GROUPS) / len(GROUPS)
    fail_rows = ex[ex["context"] == "failure"].copy()
    fail_rows["failure"] = ""
    for f in fails.itertuples():
        m = (fail_rows["peak_time"] >= f.start - ph) & (fail_rows["peak_time"] <= f.maintenance)
        fail_rows.loc[m, "failure"] = f.name
    per_fail = fail_rows.groupby(["failure", "model"])["top_group"].agg(lambda s: s.value_counts().index[0])
    lines = ["# Per-alarm explanations (Phase 5, transparency)", "",
             f"Fixed split, alarms at F = {ec['alarm_target']} (ungated, so frozen-data alarms are explained too), each "
             "explained at its peak decision: TranAD per-signal error, LSTM-AE per-feature error, Isolation Forest "
             "DIFFI. Attribution shares summed into sensor groups "
             f"{ {g: v for g, v in GROUPS.items()} }. Leak signature groups (Phase 0 data): {LEAK_GROUPS}.", "",
             "## Mean attribution share by group and alarm context", "",
             f"`leak_share` = share on the leak-signature groups; {base:.2f} if attribution were spread evenly over "
             "the groups. `failure` = peak inside a failure's warning window..maintenance; `episode` = unreported "
             "episode window; `frozen` = frozen data; `false` = everything else.", "",
             "```", gm.to_string(), "```", "",
             "## Most frequent top group", "", "```", top.to_string(), "```", "",
             "## Top group of the alarms during each real failure (majority over alarms and seeds)", "", "```",
             per_fail.unstack("model").to_string() if len(per_fail) else "(no alarms during failures)", "```", "",
             "## Stability across seeds (same alarm peaks explained by each seed's model)", "", "```",
             pd.DataFrame(stab).to_string(index=False), "```"]
    (d["root"] / "EXPLANATIONS_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"wrote {d['root'] / 'EXPLANATIONS_RESULTS.md'}")


if __name__ == "__main__":
    main()
