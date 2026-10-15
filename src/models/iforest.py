"""Isolation Forest (Liu, Ting & Zhou, ICDM 2008), scikit-learn implementation.
sklearn's score_samples returns -s(x, n); the anomaly score here is s(x, n) = 2^(-E[h(x)]/c(n))."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import Detector


class IForestDetector(Detector):
    name = "iforest"
    representation = "resampled"
    window = 1
    scaler = None                    # trees are invariant to monotone rescaling

    def __init__(self, cfg: dict, seed: int, device: str = "cpu"):
        self.p, self.seed = cfg, seed

    def fit(self, x_train, x_val, log=None):
        x = x_train.reshape(len(x_train), -1)
        self.model = IsolationForest(n_estimators=self.p["n_estimators"],
                                     max_samples=min(self.p["max_samples"], len(x)),
                                     max_features=self.p["max_features"], random_state=self.seed, n_jobs=-1).fit(x)
        if log:
            log.info(f"  iforest: {len(x):,} training samples, psi={min(self.p['max_samples'], len(x))}")
        return {"n_train": len(x)}

    def score(self, x):
        return -self.model.score_samples(x.reshape(len(x), -1))
