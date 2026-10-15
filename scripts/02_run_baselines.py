"""Step 2 - train, score and threshold every baseline detector for every split scheme and seed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.pipeline import load_processed, run_scheme  # noqa: E402
from src.utils.config import dirs, failure_table, resolve_device, setup  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("02_run_baselines", d["logs"])
    data, fails, device = load_processed(cfg), failure_table(cfg), resolve_device(cfg)
    log.info(f"device {device} | models {cfg['models']} | schemes {cfg['splits']['schemes']} | seeds {cfg['project']['seeds']}")
    from src.models.factory import build  # noqa: E402
    from src.pipeline import load_benchmark  # noqa: E402
    bench = {m: load_benchmark(cfg, build(m, cfg, 0, "cpu").representation) for m in cfg["models"]}
    log.info(f"benchmark copies loaded: {len(next(iter(bench.values()), []))}"
             + ("" if any(bench.values()) else " (run scripts/01b_make_benchmark.py to create them)"))
    infos = []
    for scheme in cfg["splits"]["schemes"]:
        for model in cfg["models"]:
            for seed in cfg["project"]["seeds"]:
                scores, info, val, bench_scores = run_scheme(cfg, model, scheme, seed, data, fails, device, log,
                                                             bench.get(model))
                for sub, frame in [("baseline", scores), ("calibration", val)]:
                    out = d["scores"] / sub / scheme / model
                    out.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(out / f"seed{seed}.csv", index=False)
                bdir = d["scores"] / "benchmark" / scheme / model
                bdir.mkdir(parents=True, exist_ok=True)
                for c, frame in bench_scores.items():
                    frame.to_csv(bdir / f"seed{seed}_copy{c}.csv", index=False)
                infos += info
    info_path = d["metrics"] / "baseline_run_info.csv"
    new = pd.DataFrame(infos)
    if info_path.exists():                      # keep earlier runs of other models / schemes
        new = pd.concat([pd.read_csv(info_path), new]).drop_duplicates(["scheme", "model", "seed", "part"], keep="last")
    new.to_csv(info_path, index=False)
    log.info("done")


if __name__ == "__main__":
    main()