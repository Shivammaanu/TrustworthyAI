"""Step 3 - event-level and benchmark metrics, figures and BASELINE_RESULTS.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.report import evaluate_all, plot_failures, plot_false_alarms, write_summary  # noqa: E402
from src.pipeline import episode_windows, load_processed  # noqa: E402
from src.utils.config import dirs, failure_table, setup  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def main(argv=None):
    cfg, _ = setup(__doc__, argv)
    d = dirs(cfg)
    log = get_logger("03_evaluate", d["logs"])
    fails = failure_table(cfg)
    data = load_processed(cfg)
    episodes = episode_windows(cfg, data)
    log.info(f"episode windows for the candidate-aware metrics: {len(episodes)} "
             f"({episodes['source'].iloc[0] if len(episodes) else 'none'})")
    per_fail, summ, agg, lead = evaluate_all(cfg, d, fails, data, episodes)
    plot_failures(cfg, d, fails, frozen=data["frozen"])
    plot_false_alarms(cfg, d, fails)
    info = pd.read_csv(d["metrics"] / "baseline_run_info.csv")
    write_summary(cfg, d, agg, lead, info)
    log.info("\n" + agg.to_string())
    log.info("\n" + lead.to_string())
    log.info(f"wrote {d['root'] / 'BASELINE_RESULTS.md'}")


if __name__ == "__main__":
    main()