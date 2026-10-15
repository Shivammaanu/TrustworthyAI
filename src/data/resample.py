"""10-minute means of the 15 signals: the benchmark input for TranAD and Isolation Forest
(Papadopoulos et al., SIGMOD 2026: mean aggregation, empty bins removed)."""
from __future__ import annotations

import pandas as pd

from .load import SIGNALS


def resample(df: pd.DataFrame, rule: str = "10min") -> pd.DataFrame:
    r = df.set_index("timestamp")[SIGNALS].resample(rule).mean().dropna()
    step = pd.Timedelta(rule)
    r["decision_time"] = r.index + step          # a bin is complete (and scorable) at its end
    r["n_readings"] = df.set_index("timestamp")["TP2"].resample(rule).count().reindex(r.index).to_numpy()
    r["segment"] = (r.index.to_series().diff() > step).cumsum().to_numpy()   # consecutive bins only
    return r.reset_index().rename(columns={"timestamp": "bin_start"})
