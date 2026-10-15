"""Configuration loading, path resolution, device choice and the shared CLI."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=None) -> dict:
    with open(Path(path) if path else DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


def failure_table(cfg: dict) -> pd.DataFrame:
    """Failures as a DataFrame of Timestamps; 'eval_end' applies the F2 end-time choice."""
    rows = []
    for f in cfg["failures"]:
        end = f.get("end_alternative") if (cfg.get("f2_use_alternative_end") and f.get("end_alternative")) else f["end"]
        rows.append({"name": f["name"], "start": pd.Timestamp(f["start"]), "end": pd.Timestamp(f["end"]),
                     "eval_end": pd.Timestamp(end), "maintenance": pd.Timestamp(f["maintenance"]),
                     "severity": f["severity"]})
    return pd.DataFrame(rows)


def resolve_device(cfg: dict) -> str:
    import torch
    d = cfg["project"].get("device", "auto")
    return ("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else d


def dirs(cfg: dict) -> dict:
    root = Path(cfg["paths"]["results_dir"])
    out = {"root": root, "processed": Path(cfg["paths"]["processed_dir"])}
    for k in ("scores", "models", "metrics", "figures", "logs"):
        out[k] = root / k
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    return out


def setup(description: str, argv=None, extra=None):
    """Common CLI: --config, --models, --schemes, --seeds, --csv, --quick."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--schemes", nargs="+", default=None, choices=["fixed", "rolling"])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--csv", default=None, help="override paths.raw_csv")
    p.add_argument("--quick", action="store_true", help="1 seed, few epochs: checks the pipeline runs, results meaningless")
    if extra:
        extra(p)
    args = p.parse_args(argv)
    cfg = copy.deepcopy(load_config(args.config))
    if args.models:
        cfg["models"] = args.models
    if args.schemes:
        cfg["splits"]["schemes"] = args.schemes
    if args.seeds:
        cfg["project"]["seeds"] = args.seeds
    if args.csv:
        cfg["paths"]["raw_csv"] = args.csv
    if args.quick:
        cfg["project"]["seeds"] = cfg["project"]["seeds"][:1]
        cfg["lstm_ae"]["epochs"] = 2
        cfg["tranad"]["epochs"] = 2
    return cfg, args
