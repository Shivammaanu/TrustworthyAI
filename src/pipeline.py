"""Train / score / threshold one detector for one split scheme and seed."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data.load import SIGNALS
from .data.splits import exclusion_intervals, in_intervals, make_parts, masks
from .data.windows import make_windows, window_touches
from .models.factory import build
from .thresholds.baseline import threshold
from .utils.seed import set_seed


def load_processed(cfg) -> dict:
    from pathlib import Path
    from .utils.fileio import load_json
    d = Path(cfg["paths"]["processed_dir"])
    if not (d / "event_windows.csv").exists():
        raise FileNotFoundError("data/processed is missing or out of date - run scripts/01_prepare_data.py first")
    rule = cfg["data_quality"]["running_rule"]
    return {"cycles": pd.read_pickle(d / f"cycles_{rule}.pkl"), "resampled": pd.read_pickle(d / "resampled.pkl"),
            "meta": load_json(d / "meta.json"),
            "frozen": pd.read_csv(d / "frozen_periods.csv", parse_dates=["start", "end"]),
            "windows": pd.read_csv(d / "event_windows.csv", parse_dates=["start", "end"])}


def episode_windows(cfg, data) -> pd.DataFrame:
    """Unreported fault episodes: the operator's list if available ('company'), else the derived ones (R3)."""
    w = data["windows"]
    src = cfg["evaluation"].get("episode_source", "auto")
    if src == "auto":
        src = "company" if (w["source"] == "company").any() else "derived"
    return w[w["source"] == src].reset_index(drop=True)


def representation(data: dict, kind: str):
    """Returns (features DataFrame, decision times, segment ids, frozen flag)."""
    if kind == "cycles":
        df = data["cycles"]
        return df[data["meta"]["cycle_features"]], df["decision_time"], df["block"].to_numpy(), df["frozen"].to_numpy(bool)
    df = data["resampled"]
    return df[SIGNALS], df["decision_time"], df["segment"].to_numpy(), df["frozen"].to_numpy(bool)


def fit_scaler(x_train: np.ndarray, kind):
    """Scaling learned on training samples only; returns a function applied to any data (real or benchmark)."""
    if kind is None:
        return lambda z: z
    if kind == "zscore":
        mu, sd = x_train.mean(0), x_train.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        return lambda z: (z - mu) / sd
    if kind == "minmax":                      # TranAD eq. 1
        lo, hi = x_train.min(0), x_train.max(0)
        return lambda z: (z - lo) / (hi - lo + 1e-4)
    raise ValueError(kind)


def scale(x: np.ndarray, train: np.ndarray, kind):
    return fit_scaler(x[train], kind)(x)


def load_benchmark(cfg, kind: str):
    """Representations of every benchmark copy for one input kind ('cycles' or 'resampled')."""
    from pathlib import Path
    root = Path(cfg["paths"]["processed_dir"]) / "benchmark"
    if not (root / "events.csv").exists():
        return []
    rule = cfg["data_quality"]["running_rule"]
    out = []
    for c in range(cfg["benchmark"]["n_copies"]):
        if kind == "cycles":
            df = pd.read_pickle(root / f"copy{c}" / f"cycles_{rule}.pkl")
        else:
            df = pd.read_pickle(root / f"copy{c}" / "resampled.pkl")
        out.append(df)
    return out


def _bench_arrays(df, kind, meta):
    if kind == "cycles":
        return df[meta["cycle_features"]].to_numpy(np.float32), df["decision_time"], df["block"].to_numpy(), df["frozen"].to_numpy(bool)
    return df[SIGNALS].to_numpy(np.float32), df["decision_time"], df["segment"].to_numpy(), df["frozen"].to_numpy(bool)


def run_scheme(cfg, model_name, scheme, seed, data, failures, device, log, bench=None):
    """Returns (test scores, run info, validation scores, {copy: benchmark scores})."""
    ph = cfg["evaluation"]["predictive_horizon_h"]
    probe = build(model_name, cfg, seed, device)
    feats, times, segment, frozen = representation(data, probe.representation)
    x = feats.to_numpy(np.float32)
    dq = cfg["data_quality"]
    excl = exclusion_intervals(failures, ph)
    if dq.get("exclude_episodes_from_training"):          # R3 as a training step
        excl += [(e.start - pd.Timedelta(hours=ph), e.end) for e in episode_windows(cfg, data).itertuples()]
    excl_rows = in_intervals(times, excl)
    if dq.get("exclude_frozen_from_training"):            # R1 as a training step
        excl_rows = excl_rows | frozen
    k = probe.window
    invalid = window_touches(excl_rows, segment, k) if k > 1 else excl_rows
    data_end = pd.to_datetime(times).max()
    outs, info, prev, vals, bouts = [], [], None, [], {}
    for part in make_parts(scheme, failures, ph, data_end):
        set_seed(seed)
        t0 = time.time()
        tr, va, te = masks(times, part, excl, cfg["splits"]["val_fraction"], invalid)
        log.info(f"{scheme}/{model_name}/seed{seed}/part{part['part']}: train {tr.sum():,} val {va.sum():,} "
                 f"test {te.sum():,} (cut {part['cut']})")
        if prev is not None and np.array_equal(tr, prev["tr"]) and np.array_equal(va, prev["va"]):
            det, w, hist = prev["det"], prev["w"], prev["hist"]      # identical training data: reuse the model
            log.info("  same training data as the previous part - model reused")
        else:
            det = build(model_name, cfg, seed, device)
            scaler = fit_scaler(x[tr], det.scaler)
            xs = scaler(x).astype(np.float32)
            w = make_windows(xs, segment, k) if k > 1 else xs
            hist = det.fit(w[tr], w[va], log)
        if prev is not None and prev["det"] is det:
            scaler = prev["scaler"]
        prev = {"tr": tr, "va": va, "det": det, "w": w, "hist": hist, "scaler": scaler}
        score_idx = tr | va | te
        scores = np.full(len(x), np.nan)
        scores[score_idx] = det.score(w[score_idx])
        ths = {v: threshold(scores[tr], spec) for v, spec in cfg["thresholds"]["variants"].items()}
        th = ths[cfg["thresholds"]["primary"]]
        out = pd.DataFrame({"decision_time": pd.to_datetime(times)[te].to_numpy(), "score": scores[te],
                            "threshold": th, "flag": scores[te] >= th, "segment": segment[te], "part": part["part"],
                            "frozen": frozen[te]})
        for v, val in ths.items():
            out[f"thr_{v}"] = val
        outs.append(out)
        # clean validation scores = calibration set for the conformal layer
        vdf = pd.DataFrame({"decision_time": pd.to_datetime(times)[va].to_numpy(), "score": scores[va],
                            "segment": segment[va], "part": part["part"], "frozen": frozen[va]})
        for v, val in ths.items():
            vdf[f"thr_{v}"] = val
        vals.append(vdf)
        # benchmark copies: same model, scaler and thresholds; rows in this part's validation or test window
        if bench:
            tt = pd.to_datetime(times)
            v0 = tt[va].min() if va.any() else part["cut"]
            for c, bdf in enumerate(bench):
                bx, bt, bseg, bfz = _bench_arrays(bdf, probe.representation, data["meta"])
                btt = pd.to_datetime(bt).to_numpy()
                in_val = (btt >= np.datetime64(v0)) & (btt <= np.datetime64(part["cut"]))
                in_te = (btt > np.datetime64(part["cut"])) & (btt <= np.datetime64(part["test_end"]))
                sel = in_val | in_te
                if not sel.any():
                    continue
                bxs = scaler(bx).astype(np.float32)
                bw = make_windows(bxs, bseg, k) if k > 1 else bxs
                bs = det.score(bw[sel])
                b = pd.DataFrame({"decision_time": btt[sel], "score": bs, "segment": bseg[sel], "part": part["part"],
                                  "frozen": bfz[sel], "split": np.where(in_val[sel], "val", "test")})
                for v, val in ths.items():
                    b[f"thr_{v}"] = val
                bouts.setdefault(c, []).append(b)
        info.append({"scheme": scheme, "model": model_name, "seed": seed, "part": part["part"],
                     "cut": part["cut"], "test_end": part["test_end"], "n_train": int(tr.sum()),
                     "n_val": int(va.sum()), "n_test": int(te.sum()), **{f"thr_{v}": val for v, val in ths.items()},
                     "best_epoch": hist.get("best_epoch"), "seconds": round(time.time() - t0, 1)})
    return (pd.concat(outs, ignore_index=True), info, pd.concat(vals, ignore_index=True),
            {c: pd.concat(v, ignore_index=True) for c, v in bouts.items()})


def train_fixed(cfg, model_name, seed, data, failures, device, log):
    """Re-train the fixed-split model exactly as run_scheme does (same seed, masks, scaler, windows).
    Returns the detector, windows/inputs for every decision, raw inputs, times, segments, frozen flags, features."""
    ph = cfg["evaluation"]["predictive_horizon_h"]
    probe = build(model_name, cfg, seed, device)
    feats, times, segment, frozen = representation(data, probe.representation)
    x = feats.to_numpy(np.float32)
    dq = cfg["data_quality"]
    excl = exclusion_intervals(failures, ph)
    if dq.get("exclude_episodes_from_training"):
        excl += [(e.start - pd.Timedelta(hours=ph), e.end) for e in episode_windows(cfg, data).itertuples()]
    excl_rows = in_intervals(times, excl)
    if dq.get("exclude_frozen_from_training"):
        excl_rows = excl_rows | frozen
    k = probe.window
    invalid = window_touches(excl_rows, segment, k) if k > 1 else excl_rows
    part = make_parts("fixed", failures, ph, pd.to_datetime(times).max())[0]
    set_seed(seed)
    tr, va, _ = masks(times, part, excl, cfg["splits"]["val_fraction"], invalid)
    det = build(model_name, cfg, seed, device)
    scaler = fit_scaler(x[tr], det.scaler)
    xs = scaler(x).astype(np.float32)
    w = make_windows(xs, segment, k) if k > 1 else xs
    det.fit(w[tr], w[va], log)
    return det, w, x, pd.to_datetime(times).reset_index(drop=True), segment, frozen, list(feats.columns)