"""Run the pipeline: 01 prepare data -> 01b synthetic benchmark -> 02 run baselines -> 03 evaluate -> 03b benchmark -> 04 conformal -> 05 evaluate conformal -> 06 combine detectors -> 07 smoothing -> 08 triage -> 09 explanations -> 10 audit.
    python run_all.py                  # full run
    python run_all.py --quick          # 1 seed, 2 epochs: checks everything runs (results meaningless)
    python run_all.py --from 3         # resume at step 3 (skip data preparation and the benchmark)
Other flags (--models, --schemes, --seeds, --csv, --config) are passed to every step.

Step numbers for --from:
     1 01_prepare_data         2 01b_make_benchmark      3 02_run_baselines (~35 min)
     4 03_evaluate             5 03b_evaluate_benchmark  6 04_conformal
     7 05_evaluate_conformal   8 06_combine              9 07_smoothing
    10 08_triage              11 09_explain (~10 min)   12 10_audit"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["01_prepare_data.py", "01b_make_benchmark.py", "02_run_baselines.py", "03_evaluate.py",
         "03b_evaluate_benchmark.py", "04_conformal.py", "05_evaluate_conformal.py", "06_combine.py",
         "07_smoothing.py", "08_triage.py", "09_explain.py",
         "10_audit.py"]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="start", type=int, default=1)
    args, rest = p.parse_known_args()
    for i, step in enumerate(STEPS, 1):
        if i < args.start:
            continue
        print(f"\n===== STEP {i}: {step} =====", flush=True)
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / step), *rest]).returncode
        if rc:
            why = f"killed by signal {-rc} (11 = segmentation fault, 9 = out of memory)" if rc < 0 else f"exit code {rc}"
            sys.exit(f"step {i} failed ({why}) - see the messages above; resume with --from {i}")
    print("\nDone. Reports in results/: BASELINE_RESULTS.md, BENCHMARK_RESULTS.md, CONFORMAL_RESULTS.md, "
          "COMBINED_RESULTS.md, SMOOTHING_RESULTS.md, TRIAGE_RESULTS.md, EXPLANATIONS_RESULTS.md, AUDIT_RESULTS.md; "
          "figures in results/figures/. Conclusions: FINDINGS.md")