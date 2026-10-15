"""
Per-alarm explanations (Phase 5, transparency pillar).

* TranAD   : its own per-signal phase-2 reconstruction error at the alarm decision (official code's diagnosis signal).
* LSTM-AE  : per-feature reconstruction error of the decision cycle (44 cycle features), grouped by sensor.
* IForest  : DIFFI local importance (Carletti et al.): for every tree, along the path of x, each split on feature f
             adds 1/h_t(x) - 1/h_max to I_f and 1 to C_f (h_t = depth reached, h_max = ceil(log2 psi));
             LFI_f = I_f / C_f.
Attributions are normalised to shares (sum 1) and aggregated into physical sensor GROUPS, so the three detectors (15
signals or 44 cycle features) become comparable and redundant sensors (TP3 = Reservoirs; COMP ~ MPG ~ not DV_electric)
do not split credit. LEAK_GROUPS = the air-leak signature found in the Phase 0 data (continuous running, DV pressure
~2 bar, oil temperature up, TP3/Reservoirs down).
"""
from __future__ import annotations

import numpy as np

GROUPS = {
    "running": ["COMP", "MPG", "DV_electric", "Motor_current", "T_run", "T_idle"],
    "DV_pressure": ["DV_pressure"],
    "oil": ["Oil"],
    "TP3_reservoir": ["TP3", "Reservoirs"],
    "TP2_H1": ["TP2", "H1"],
    "other_digital": ["Towers", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"],
}
LEAK_GROUPS = ["running", "DV_pressure", "oil", "TP3_reservoir"]


def group_of(feature: str) -> str:
    """Map a signal or cycle-feature name to its sensor group (prefix match; N_<signal> counts of digital signals)."""
    name = feature[2:] if feature.startswith("N_") else feature
    for g, prefixes in GROUPS.items():
        for p in prefixes:
            if name == p or name.startswith(p + "_") or name.startswith(p):
                if p == "Oil" and name.startswith("Oil_level"):
                    continue
                return g
    return "other_digital" if name.startswith("Oil_level") else "other"


def shares(a: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(a, float), 0, None)
    s = a.sum(axis=1, keepdims=True)
    return np.where(s > 0, a / np.where(s > 0, s, 1), 1.0 / a.shape[1])


def group_shares(a: np.ndarray, feature_names) -> dict:
    """(n, features) shares -> {group: (n,) share}."""
    sh = shares(a)
    out = {g: np.zeros(len(sh)) for g in GROUPS}
    for j, f in enumerate(feature_names):
        g = group_of(f)
        if g in out:
            out[g] += sh[:, j]
    return out


def diffi_local(forest, X: np.ndarray) -> np.ndarray:
    """DIFFI local feature importance for each row of X (sklearn IsolationForest)."""
    X = np.asarray(X, np.float32)
    n, p = X.shape
    I = np.zeros((n, p))
    C = np.zeros((n, p))
    h_max = int(np.ceil(np.log2(max(forest.max_samples_, 2))))
    for est, feats in zip(forest.estimators_, forest.estimators_features_):
        tree = est.tree_
        path = est.decision_path(X[:, feats]).tocsr()
        for i in range(n):
            nodes = path.indices[path.indptr[i]:path.indptr[i + 1]]
            depth = len(nodes) - 1
            inc = 1.0 / max(depth, 1) - 1.0 / h_max
            for node in nodes[:-1]:                       # internal nodes on the path
                f = feats[tree.feature[node]]
                I[i, f] += inc
                C[i, f] += 1
    return np.where(C > 0, I / np.where(C > 0, C, 1), 0.0)
