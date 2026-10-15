"""Baseline ("without") thresholds, each from the source of its detector.

percentile: q-th percentile of the training scores (Alhammadi et al. 2025 use the 95th).
pot:        Peaks-Over-Threshold / SPOT (Siffer et al., KDD 2017), as used by the TranAD code:
            initial threshold t = level-quantile of the training scores, a Generalised Pareto
            distribution is fitted to the excesses s - t, and the final threshold is the score whose
            exceedance probability is `risk`:
                z = t + (sigma/gamma) * ((risk * n / N_t)^(-gamma) - 1)      (gamma != 0)
                z = t - sigma * ln(risk * n / N_t)                          (gamma == 0)
            The GPD is fitted by maximum likelihood (scipy) instead of Grimshaw's procedure,
            which is a numerical method for the same maximum-likelihood estimate.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


def percentile_threshold(train_scores, q=95):
    return float(np.percentile(train_scores, q))


def pot_threshold(train_scores, risk=1e-5, level=0.99, multiplier=1.0):
    s = np.asarray(train_scores, float)
    t = float(np.quantile(s, level))
    peaks = s[s > t] - t
    if len(peaks) < 10:                       # too few excesses for a GPD fit
        return float(np.max(s)) * multiplier
    gamma, _, sigma = genpareto.fit(peaks, floc=0)
    r = risk * len(s) / len(peaks)
    z = t - sigma * np.log(r) if abs(gamma) < 1e-8 else t + (sigma / gamma) * (r ** (-gamma) - 1)
    return float(z) * multiplier


def threshold(train_scores, spec: dict) -> float:
    if spec["method"] == "percentile":
        return percentile_threshold(train_scores, spec["q"])
    if spec["method"] == "pot":
        return pot_threshold(train_scores, spec["risk"], spec["level"], spec.get("multiplier", 1.0))
    raise ValueError(spec["method"])
