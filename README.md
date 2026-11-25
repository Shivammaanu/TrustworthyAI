# trustworthy-apu-metropt3

Code for an unsupervised air-leak early-warning system for the Air Production Unit (APU) of a Metro do Porto
train, evaluated on the public **MetroPT-3** dataset. The repository rebuilds three published detectors and adds a
trustworthy layer (data-quality gate, calibrated alarms, safe detector fusion, triage, explanations and an audit).

> **Paper.** The method, the analysis and the full discussion are in our paper *Trustworthy Air-Leak Detection for a
> Metro Air Production Unit: Chance-Aware Evaluation, Data-Quality Rules and a Six-Pillar Trust Layer on MetroPT-3*
> (under review). This README only documents the code. Please cite the paper if you use it (see [Citation](#citation)).

---

## What the code does

| Module | Purpose |
|---|---|
| Data preparation | Detects logging gaps, derives three data-quality rules (R1 frozen logger, R2 running state, R3 unreported fault episodes) and builds two inputs: per-cycle features and 10-minute means |
| Baselines | LSTM autoencoder (Davari et al., SAC 2024), TranAD (official code, Tuli et al., VLDB 2022), Isolation Forest (Liu et al., ICDM 2008), each with p95 and POT thresholds |
| Synthetic benchmark | Injects labelled air leaks and sensor faults (frozen, stuck, noise) into 10 copies of the data |
| Trustworthy layer | Conformal p-values with a false-alarm budget per day, R1 gate, Bonferroni combination, EWMA smoothing, SGR triage, per-alarm explanations, consistency audit |
| Evaluation | Event-level metrics reported next to their chance level, plus the AD1 benchmark metric |

## Headline numbers (fixed split, mean of three seeds)

| System | Failures detected (chance) | Unreported episodes (chance) | Clean false alarms / day |
|---|---|---|---|
| TranAD, published p95 threshold | 4.0 / 4 (4.00) | 2 / 2 (2.00) | 11.5 |
| TranAD + LSTM-AE, conformal budget F = 0.5, R1 gate | **4.0 / 4 (0.98)** | **2 / 2 (0.43)** | **0.13** |

Other key outputs: frozen-data faults reported as leaks fall from 33% to 4% with the R1 gate; the three baselines at
their published thresholds detect no more failures than random alarms at the same rate. Details are in the paper and
in the reports the pipeline writes (`results/*_RESULTS.md`).

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch                                       # CUDA build: see pytorch.org
pip install -r requirements.txt

# put the dataset CSV here (download from the UCI Machine Learning Repository, "MetroPT-3 Dataset")
#   data/raw/MetroPT3(AirCompressor).csv

python run_all.py --quick     # smoke test (1 seed, 2 epochs, results not meaningful)
python run_all.py             # full run: 3 seeds, fixed and rolling splits, about 1 h on a laptop CPU
python run_all.py --from 6    # resume from a given step
```

Every script accepts `--models`, `--schemes`, `--seeds`, `--csv`, `--config` and `--quick`.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `scripts/01_prepare_data.py` | `data/processed/` (cycles under both running rules, 10-min bins, frozen periods, event windows) |
| 2 | `scripts/01b_make_benchmark.py` | `data/processed/benchmark/` (10 copies, 616 labelled events) |
| 3 | `scripts/02_run_baselines.py` | `results/scores/` (test, calibration and benchmark scores per model, split and seed) |
| 4 | `scripts/03_evaluate.py` | `results/BASELINE_RESULTS.md` |
| 5 | `scripts/03b_evaluate_benchmark.py` | `results/BENCHMARK_RESULTS.md` |
| 6 | `scripts/04_conformal.py` | conformal p-values and gated flags in `results/scores/` |
| 7 | `scripts/05_evaluate_conformal.py` | `results/CONFORMAL_RESULTS.md` |
| 8 | `scripts/06_combine.py` | `results/COMBINED_RESULTS.md` |
| 9 | `scripts/07_smoothing.py` | `results/SMOOTHING_RESULTS.md` |
| 10 | `scripts/08_triage.py` | `results/TRIAGE_RESULTS.md` |
| 11 | `scripts/09_explain.py` | `results/EXPLANATIONS_RESULTS.md` (retrains the fixed-split models exactly) |
| 12 | `scripts/10_audit.py` | `results/AUDIT_RESULTS.md` |

All tables are also written as CSV to `results/metrics/`, figures to `results/figures/` and one log per step to
`results/logs/`, including every threshold derived from the data. Generated data and results are git-ignored.

## Configuration

All settings are in `configs/config.yaml`. Each value carries a tag: `[data]` (derived from the data), `[paper]`
(taken from the cited source) or `[ours]` (a choice the sources leave open).

**Data-quality switches.** With all switches off the baselines run exactly as published. They are used for the
training ablation:

```yaml
data_quality:
  running_rule: comp                     # comp (published) | comp_and_motor (rule R2)
  exclude_frozen_from_training: false    # rule R1
  exclude_episodes_from_training: false  # rule R3
  company_windows: configs/company_windows.csv   # optional operator fault list (name,start,end)
```

**Operator fault list.** If `configs/company_windows.csv` exists, its windows replace the episodes derived by R3 in
the evaluation. Re-run from step 1 after adding it.

## Where each component is implemented

| Component | File |
|---|---|
| Gap threshold (Otsu) | `src/data/gaps.py` |
| Rules R1, R2, R3 | `src/data/quality.py` |
| Cycle features (44) | `src/data/cycles.py` |
| 10-minute means, segments | `src/data/resample.py` |
| Chronological splits | `src/data/splits.py` |
| LSTM-AE, TranAD, Isolation Forest | `src/models/` |
| p95 and POT thresholds | `src/thresholds/baseline.py` |
| Event metrics and chance levels | `src/evaluation/events.py` |
| Synthetic fault injection | `src/benchmark/inject.py` |
| Conformal p-values, reachability | `src/trustworthy/conformal.py` |
| R1 gate, Bonferroni combination | `src/trustworthy/combine.py`, `scripts/04_conformal.py` |
| EWMA smoothing | `src/trustworthy/smoothing.py` |
| SGR triage | `src/trustworthy/triage.py` |
| Sensor groups, DIFFI | `src/trustworthy/explain.py` |
| Block-bootstrap audit | `scripts/10_audit.py` |

## Repository structure

```
.
├── README.md
├── requirements.txt
├── run_all.py
├── configs/config.yaml
├── data/raw/                 dataset CSV goes here (git-ignored)
├── scripts/                  the 12 pipeline steps listed above
├── src/
│   ├── pipeline.py           train and score one detector / split / seed
│   ├── utils/                config, logging, file I/O
│   ├── data/                 load, gaps, quality, cycles, resample, splits, windows, process
│   ├── models/               base, lstm_ae, tranad, iforest, factory
│   ├── thresholds/           baseline (p95, POT)
│   ├── evaluation/           events, ad1, benchmark, report
│   ├── benchmark/            inject
│   └── trustworthy/          conformal, combine, smoothing, triage, explain
└── third_party/TranAD_LICENSE.txt
```

## Reproducibility

- Seeds 42, 43 and 44; every result is the mean over the three seeds.
- Training runs on CPU and is deterministic: retraining with the same seed reproduces the scores to within 1e-13.
- No parameter is chosen on the four reported failures. Thresholds come from the data, the cited papers, validation
  data or the operator budget.
- Full run: about one hour on a laptop CPU.

## Data and licences

- **MetroPT-3** (Davari, Veloso, Ribeiro, Gama; UCI Machine Learning Repository, 2023), CC BY 4.0. The data are not
  redistributed here.
- **TranAD** network classes are adapted from the official implementation (BSD-3-Clause); see
  `third_party/TranAD_LICENSE.txt`.

## Citation

```bibtex
@unpublished{2026trustworthy,
  author = {Anant Jain, Karmveer Kumar, Shivam Gangwar and Shweta Saharan},
  title  = {Trustworthy Air-Leak Detection for a Metro Air Production Unit: Chance-Aware Evaluation, Data-Quality Rules and a Six-Pillar Trust Layer on {MetroPT-3}},
  note   = {Manuscript under review},
  year   = {2026}
}
```

Please also cite the dataset (Veloso et al., *Scientific Data* 9, 764, 2022; Davari et al., UCI Machine Learning
Repository, 2023) and the baseline papers listed in the table above.

## Acknowledgements

Minor project, Department of AI & Data Science, Gati Shakti Vishwavidyalaya, Vadodara.