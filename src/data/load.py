"""Reading the raw MetroPT-3 CSV."""
from __future__ import annotations

import pandas as pd

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_electric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
SIGNALS = ANALOG + DIGITAL


def load_raw(path) -> pd.DataFrame:
    """Returns rows sorted by time with canonical column names ('DV_eletric' in the file -> 'DV_electric')."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])
    df = df.rename(columns={c: "DV_electric" for c in df.columns if c.lower() == "dv_eletric"})
    missing = [c for c in ["timestamp"] + SIGNALS if c not in df.columns]
    if missing:
        raise ValueError(f"columns missing from {path}: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)[["timestamp"] + SIGNALS]
