"""Common interface: every detector maps samples to one anomaly score per sample (higher = more anomalous)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    name = "base"
    representation = "resampled"     # 'cycles' or 'resampled'
    window = 1                       # samples per input window (1 = point-based)
    scaler = None                    # 'zscore', 'minmax' or None (fitted on training samples only)

    @abstractmethod
    def fit(self, x_train: np.ndarray, x_val: np.ndarray, log=None) -> dict: ...

    @abstractmethod
    def score(self, x: np.ndarray) -> np.ndarray: ...

    def explain_dims(self, x: np.ndarray):
        """Optional per-feature score (for later explanation analyses)."""
        return None
