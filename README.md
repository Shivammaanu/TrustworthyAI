# Trustworthy Air-Leak Detection for the Metro do Porto APU (MetroPT-3)

Unsupervised early detection of air-leak failures in a metro train's **Air Production Unit (APU)**, built on three
published anomaly detectors and extended with a **trustworthy-AI layer** covering six pillars: robustness, calibrated
uncertainty, safety, human oversight, transparency and consistency.

> **Headline result.** At their published thresholds the three baseline detectors find the four reported failures
> only as often as random alarms would. The trustworthy operating system detects **4/4 failures (chance 0.98)** and
> **2/2 unreported fault episodes (chance 0.43)** at **0.13 false alarms per day**, against 10.8 per day for the
> baseline TranAD.

## Highlights

- **Two hidden traps found in the data:** the data logger *freezes* (9 periods, ~170 h) and the "normal" data contain
  unreported fault episodes. Both distort training and evaluation if ignored.
- **The published baselines detect at chance level** once every result is reported against what random alarms at the
  same false-alarm rate would achieve.
- **A published result explained:** the 15.7 h early warning for failure F1 reported by Davari et al. (SAC 2024) falls
  almost entirely inside a 14.8 h frozen-logger period that ends exactly when F1 starts.
- **Trustworthy layer:** conformal p-values give the operator a false-alarm budget, a data-quality gate separates logger
  faults from leaks, a Bonferroni combination merges detectors safely, alarm triage carries a statistical risk
  guarantee, and every alarm comes with a sensor-level explanation.
- **Honest negative results:** online recalibration and score smoothing were tested and rejected, and seasonal drift
  is shown to break distribution-free guarantees.
- **Nothing is tuned on the four real failures.** Every rule is derived from the data or taken from a cited source;
  every parameter is chosen on validation data or set by the operator.

## Contents

1. [Overview](#1-overview)
2. [Getting started](#2-getting-started)
3. [Repository structure](#3-repository-structure)
4. [Configuration](#4-configuration)
5. [The data and what it revealed](#5-the-data-and-what-it-revealed)
6. [Data pipeline and splits](#6-data-pipeline-and-splits)
7. [Baseline detectors](#7-baseline-detectors)
8. [Evaluation](#8-evaluation)
9. [Baseline results](#9-baseline-results)
10. [The trustworthy layer](#10-the-trustworthy-layer)
11. [Final operating system](#11-final-operating-system)
12. [Limitations and future work](#12-limitations-and-future-work)
13. [References](#13-references)

---

## 1. Overview

The APU supplies compressed air to a train's brakes, suspension and doors. An air leak makes the compressor run longer
and more often, raises the pressure at the air-dryer discharge (DV pressure) and heats the oil. Caught late, it takes
the train out of service.

The public **MetroPT-3** dataset (Veloso et al., 2022) records 15 APU signals roughly every 10 seconds from February
to September 2020, but contains **only four reported failures**. That is far too few labels for a supervised model,
so detection is **unsupervised**: each model learns what normal operation looks like and flags deviations.

**Goals**

1. Detect leaks, ideally at least **2 hours before** a failure (the operator's requirement).
2. Keep false alarms low and, more importantly, *known*.
3. Make the system trustworthy across six pillars.

**Baselines**

| Detector | Source | Input |
|---|---|---|
| LSTM autoencoder | Davari et al., SAC 2024 | 44 features per compressor cycle |
| TranAD | Tuli et al., VLDB 2022 (official code) | 10-minute means of the 15 signals |
| Isolation Forest | Liu et al., ICDM 2008 | 10-minute means of the 15 signals |

**Pipeline at a glance**

```
raw data ─► data-quality rules (R1–R3) ─► cycles / 10-min bins ─► 3 baseline detectors
        ─► conformal p-values ─► R1 gate ─► Bonferroni combination ─► alarm triage ─► explanations
        ─► evaluation on real failures + synthetic benchmark ─► consistency audit
```

---

## 2. Getting started

### Requirements

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install torch            # GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Data

Download the MetroPT-3 dataset (UCI Machine Learning Repository) and place the CSV at:

```
data/raw/MetroPT3(AirCompressor).csv
```

The path is set in `configs/config.yaml` (`paths.raw_csv`); change it there or pass `--csv <path>` to any step. The
raw data and everything the pipeline generates are git-ignored.

### Run everything

```bash
python run_all.py --quick     # smoke test: 1 seed, 2 epochs (results meaningless)
python run_all.py             # full run: 3 seeds, fixed + rolling splits (~1 hour on a laptop CPU)
python run_all.py --from 3    # resume at a given step; --help lists the step numbers
```

### Run step by step

| # | Script | What it does | Writes |
|---|---|---|---|
| 1 | `scripts/01_prepare_data.py` | Gaps, frozen data (R1), cycles under both running rules (R2), 44 cycle features, 10-min bins, unreported episodes (R3), event windows | `data/processed/` |
| 2 | `scripts/01b_make_benchmark.py` | Synthetic leaks and sensor faults in 10 copies of the data | `data/processed/benchmark/` |
| 3 | `scripts/02_run_baselines.py` | Trains, scores and thresholds every model, split and seed; scores calibration data and benchmark copies (~35 min) | `results/scores/` |
| 4 | `scripts/03_evaluate.py` | Baseline metrics and figures | `BASELINE_RESULTS.md` |
| 5 | `scripts/03b_evaluate_benchmark.py` | Baselines on the synthetic benchmark | `BENCHMARK_RESULTS.md` |
| 6 | `scripts/04_conformal.py` | Conformal p-values: static, rolling and guarded-rolling calibration | `results/scores/` |
| 7 | `scripts/05_evaluate_conformal.py` | Budgets, detections, monthly validity, feasibility | `CONFORMAL_RESULTS.md` |
| 8 | `scripts/06_combine.py` | Bonferroni combination of detectors, with and without the R1 gate | `COMBINED_RESULTS.md` |
| 9 | `scripts/07_smoothing.py` | EWMA score smoothing, span chosen on validation leaks | `SMOOTHING_RESULTS.md` |
| 10 | `scripts/08_triage.py` | Alarm triage with a guaranteed risk (SGR) | `TRIAGE_RESULTS.md` |
| 11 | `scripts/09_explain.py` | Per-alarm explanations; re-trains the fixed-split models exactly (~10 min) | `EXPLANATIONS_RESULTS.md` |
| 12 | `scripts/10_audit.py` | Consistency audit by month, compressor state and time of day | `AUDIT_RESULTS.md` |

Every step accepts `--models lstm_ae tranad iforest`, `--schemes fixed rolling`, `--seeds 42 43 44`, `--csv <path>`,
`--config <path>` and `--quick`.

### What a run produces (locally, not in the repository)

| Location | Content |
|---|---|
| `data/processed/` | Cycles, 10-min bins, frozen periods, event windows (`event_windows.csv`), minor high-DV runs (`high_dv_runs.csv`), benchmark copies and events |
| `results/*_RESULTS.md` | One human-readable report per evaluation step (table above) |
| `results/metrics/` | Every table as CSV, per seed and aggregated |
| `results/scores/` | Every decision's score and alarm flags, per step, split, model and seed |
| `results/figures/` | Scores around each failure; false alarms per month |
| `results/logs/` | One log per step, including every data-derived threshold |

---

## 3. Repository structure

```
.
├── README.md
├── requirements.txt
├── run_all.py                    runs all 12 steps in order
├── .gitignore
├── configs/
│   └── config.yaml               every setting, tagged [data], [paper] or [ours]
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 01b_make_benchmark.py
│   ├── 02_run_baselines.py
│   ├── 03_evaluate.py
│   ├── 03b_evaluate_benchmark.py
│   ├── 04_conformal.py
│   ├── 05_evaluate_conformal.py
│   ├── 06_combine.py
│   ├── 07_smoothing.py
│   ├── 08_triage.py
│   ├── 09_explain.py
│   └── 10_audit.py
├── src/
│   ├── pipeline.py               train / score one detector × split × seed; exact retraining helper
│   ├── utils/                    
│   │   ├── config.py             config loading
│   │   ├── fileio.py             File I/O (JSON)
│   │   ├── logger.py             logger file
│   │   └── seed.py               seed setter
│   ├── data/
│   │   ├── load.py               raw CSV loader
│   │   ├── gaps.py               Otsu gap threshold, gap-free blocks
│   │   ├── quality.py            R1 frozen data, R2 running rule, R3 unreported episodes
│   │   ├── cycles.py             compressor cycles and the 44 cycle features
│   │   ├── resample.py           10-minute means and segments
│   │   ├── splits.py             fixed and rolling chronological splits, failure buffers
│   │   ├── windows.py            segment-aware sliding windows
│   │   └── process.py            full processing of benchmark copies
│   ├── models/
│   │   ├── base.py               common detector interface
│   │   ├── lstm_ae.py            LSTM autoencoder
│   │   ├── tranad.py             TranAD (official classes, BSD-3)
│   │   ├── iforest.py            Isolation Forest
│   │   └── factory.py            builds a detector from the config
│   ├── thresholds/
│   │   └── baseline.py           p95 and POT thresholds
│   ├── evaluation/
│   │   ├── events.py             event metrics and chance levels
│   │   ├── ad1.py                benchmark AD1 AUC-PR
│   │   ├── benchmark.py          synthetic-benchmark metrics
│   │   └── report.py             result tables, figures, reports
│   ├── benchmark/
│   │   └── inject.py             synthetic leak and sensor-fault injection
│   └── trustworthy/
│       ├── conformal.py          conformal p-values, reachability, recalibration
│       ├── combine.py            Bonferroni combination, R1 gate
│       ├── smoothing.py          causal EWMA
│       ├── triage.py             SGR triage with a risk guarantee
│       └── explain.py            sensor groups, DIFFI
└── third_party/
    └── TranAD_LICENSE.txt
```

---

## 4. Configuration

Every setting lives in `configs/config.yaml` and is tagged by origin:

- **[data]** — derived from the data (e.g. a threshold found with Otsu's method);
- **[paper]** — taken from a cited source;
- **[ours]** — a choice the sources leave open, stated so it can be challenged.

The file is organised in sections: `data`, `data_quality`, `benchmark`, `conformal`, `combine`, `smoothing`, `triage`,
`explain`, `audit`, `splits`, `models`, `thresholds` and `evaluation`.

### Data-quality switches

With every `data_quality` switch off, the baselines run exactly as published. The switches (the R2 running rule, and
excluding frozen data and unreported episodes from training) exist for an ablation of training on cleaned data.

### Using the operator's fault list

The operator reported 21 faulty windows (Davari et al., 2021). To use them instead of the episodes derived from the
data, create `configs/company_windows.csv` and re-run from step 1:

```
name,start,end
W01,2020-03-12 00:00,2020-03-12 12:00
```

The operator's list is preferable for evaluation, because the derived episodes (R3) are defined from the same signals
the detectors learn from. Either list can be used to exclude episodes from training.

---

## 5. The data and what it revealed

The only hand-entered information is the failure table. Everything else in this section was derived from the data.

### 5.1 The four reported failures

| Failure | Start | End | Maintenance | Severity |
|---|---|---|---|---|
| F1 | 18 Apr 2020 00:00 | 18 Apr 23:59 | 19 Apr 01:00 | High |
| F2 | 29 May 23:30 | 29 May 23:59 (data: ≈ 06:00) | 30 May 12:00 | High |
| F3 | 5 Jun 10:00 | 7 Jun 14:30 | 8 Jun 16:00 | High |
| F4 | 15 Jul 14:30 | 15 Jul 19:00 | 16 Jul 00:00 | Medium |

Source: Davari et al., SAC 2024, Table 1. After 23:59 on 29 May the compressor keeps running 97% of the time until
about 05:00, so F2 is evaluated with the UCI end time of 06:00 (configurable).

### 5.2 Data facts

| Fact | Value | Consequence |
|---|---|---|
| Size and period | 1,516,948 rows, 1 Feb 2020 → 1 Sep 2020 03:59 | Matches SAC 2024; the UCI "15 M rows at 1 Hz" is wrong |
| Sampling | Median 10 s (9–13 s jitter) | 10 s confirmed |
| Gaps **[data]** | Otsu break at **194 s**; 311 gaps, **37.9 days** missing, mostly overnight (≈ 00:30–04:40, train parked) | Windows and cycles never cross a gap; normal time is counted only when recorded |
| Coverage | 77–86% per month | False alarms are normalised by *recorded* days |
| Signal types **[data]** | 7 continuous + 8 binary, as described | — |
| Compressor states **[data]** | Gaussian mixture, BIC → 6 components: off ≈ 0.04 A (54%), offloaded ≈ 3.7–3.9 A, transition ≈ 4.3 A, loaded ≈ 5.6–6.0 A | The description's 7 A / 9 A do not appear; loaded is ≈ 6 A |
| Running signal **[data]** | COMP = 0 agrees with the loaded states in 98.6% of rows | Basis of the published cycle segmentation |
| Cycles **[data]** | Run median 2.0 min (p95 3.1), idle median 12.7–16.2 min | ≈ 18-min cycles; 10 cycles ≈ 3 h |
| Redundancy **[data]** | TP3 ≡ Reservoirs (r = 1.000); COMP ≈ MPG (99.6% agreement); DV_electric ≈ NOT COMP | Explanations would split credit between these |
| Failure signature **[data]** | Compressor running 95–100% during every failure (normally 6–27%); DV pressure ≈ 2 bar | Core air-leak signature |
| Precursors **[data]** | F1: COMP, MPG, Towers, Oil_level deviate from −48 h (z ≈ −6), strongly in the last 2 h (z ≈ −22 to −30). F4: oil temperature z ≈ 3.3 in the last 2 h. F2, F3: weak | Early warning is realistic for F1 and F4, hard for F2 and F3 |
| F4 precursor **[data]** | From ~14 h before: compressor runs about twice as often (every ~8 min instead of 15–25), runs lengthen 2.8 → 8.6 min, oil starts hotter (69–76 °C vs 58–65 °C) | A duty-cycle precursor |
| Drift **[data]** | Duty cycle 5.8% (Feb) → 26.6% (Jun); oil temperature in the off state 55.9 → 64.0 °C (Feb → Jul) | Models trained in early months see a different regime later |
| Reference data | Only ≈ 4 days (28,938 rows) of normal data between F2's maintenance and F3 − 48 h | The third rolling model trains mostly on older data |

### 5.3 Data-quality rules R1–R3

None of these rules appears in the baseline papers, so **the baselines run without them**. R1 and R2 belong to the
trustworthy layer (robustness pillar); R3 is used in the reporting of every model. All breaks are found with Otsu's
method and printed by step 1.

| Rule | Definition | Evidence |
|---|---|---|
| **R1 — frozen data** | A 10-min bin in which all 7 analog signals show exactly zero variation | 9 periods, 7 of them far from any failure |
| **R2 — running** | COMP = 0 **and** motor current above **2.85 A** (Otsu over the rows with COMP = 0) | 6,171 COMP "runs" with the motor at 0.04 A (flicker inside frozen data), plus COMP–motor disagreements |
| **R3 — unreported episodes** | Three Otsu stages over real runs, outside the reported failures' windows (below) | Same signature as the four failures; the operator reported 21 faulty windows, only 4 of them failures |

#### R1: what a "freeze" is

During a freeze the data logger keeps repeating **stale analog values**: all 7 analog signals stay exactly constant
for hours (oil temperature fixed at 49.450 °C, for example), while COMP and Towers **flicker** (~20% zeros) and the
motor current sits at 0.04 A. A parked train's pressure would slowly fall and its oil would cool, so this is
physically impossible: a **logger fault**, not a compressor fault. Data were sent to the server by GSM every
5 minutes; failed transmission is a plausible cause (our hypothesis).

In total, 1,022 of 25,521 bins (4.00%) are frozen, in 9 periods totalling ~170 h. The 62 h on 22–25 June, with COMP
stuck at 0, looked like a long leak until R1 identified it.

<details>
<summary>All nine frozen periods</summary>

| Period | Hours | Near a failure? |
|---|---|---|
| 11 Mar 18:20 → 19:00 | 0.7 | — |
| 13 Apr 18:30 → 19:30 | 1.0 | 4 days before F1 |
| **17 Apr 09:30 → 18 Apr 00:20** | **14.8** | **ends exactly at F1's start** |
| 20 Apr 04:50 → 21 Apr 01:20 | 20.5 | — |
| **26 May 09:20 → 28 May 03:20** | **42.0** | ends 44 h before F2 |
| 12 Jun 02:10 → 17:10 | 15.0 | — |
| **22 Jun 15:10 → 25 Jun 05:10** | **62.0** | — (COMP stuck at 0) |
| 21 Jul 13:50 → 22:00 | 8.2 | — |
| 22 Jul 06:50 → 13:00 | 6.2 | — |

</details>

**The link to the published F1 result.** Davari et al. describe an "abnormal window" before F1 in which "TP3 values
remain constant … the compressor operates more frequently than usual". This is the frozen period of 17 Apr 09:30 →
18 Apr 00:20; the "more frequent operation" is COMP flickering while the motor is off. Their LSTM-AE alarm for F1
comes at 17 Apr 08:16, about an hour before the freeze starts, and their 15.7 h lead time spans almost all of it.
Davari et al. also define T_run as the time the compressor pumps air into the main reservoir and T_idle as compressor
off, without naming a signal, and use a 70-10-20 split whose chronological order is not stated.

#### R2: effect on the published segmentation

With running defined by COMP alone, **5,810 of 16,673 cycles (35%) touch frozen data**, so the LSTM-AE baseline is
trained largely on frozen-data artefacts. R2 removes 4,355 fake cycles (12,318 remain, 1,462 of them touching frozen
data). Under R2 no complete cycle overlaps F1, and F4's longest run drops from 4.5 h to 2.9 h because brief motor dips
split it.

#### R3: the three stages

1. **Fragments.** Otsu on the log run length of all real runs (R2, not frozen) separates short fragments from runs:
   break 0.84 min, leaving 10,685 runs.
2. **Long, high-DV runs.** On those runs, a long-run break (5.4 min) and a DV-pressure break (0.34 bar) give 48 runs
   that are both long and high-DV.
3. **Major vs minor.** Their durations split cleanly at 1.46 h into **6 major episodes (4.4–21.4 h)** and 42 minor
   runs (0.1–1.5 h). Only the major episodes become event windows.

| Major episode | Duration | DV pressure |
|---|---|---|
| E09: 12 Mar 00:16 → 11:49 | 11.6 h | 0.73 bar |
| E14: 27 Mar 07:12 → 11:38 | 4.4 h | 0.59 bar |
| E18: 28 Mar 23:05 → 29 Mar 19:49 | 20.7 h | 0.66 bar |
| E23: 12 Apr 11:50 → 23:36 | 11.8 h | 1.70 bar |
| E36: 13 May 13:44 → 14 May 00:56 | 11.2 h | 2.32 bar |
| E42: 19 May 22:22 → 20 May 19:48 | 21.4 h | 1.99 bar |

The reported failures' runs all belong to the same hours-long class (4.5–30 h), and the six major episodes are
exactly those found by hand. Two of them (13–14 May, 19–20 May) fall in the test period.

The **42 minor high-DV runs** (DV 0.36–1.83 bar in ordinary 6–90-minute runs, from March to 1 June, none after F3's
repair on 8 June) indicate a **prolonged degraded state** rather than separate faults. They are saved for the
drift and consistency analysis.

> **Development note.** A first version of R3 had no stage 3. Its 48 windows, each with a 48 h horizon, covered most of
> March–June and made the episode metrics uninformative (detections exactly at chance).

---

## 6. Data pipeline and splits

### 6.1 Cycles (LSTM-AE input) — Davari et al. 2024, Table 2

A **cycle** is one run period plus the following idle period, inside one gap-free block. A cycle is complete only when
the next run starts, so that moment is its **decision time**: no future data are used.

| Features (44) | Count |
|---|---|
| Mean of TP2, TP3, H1 and motor current over 7 time bins | 28 |
| Oil temperature: min, max, mean during the run (MA1), mean during idle (MA2) | 4 |
| Median DV pressure, median Reservoirs | 2 |
| Run duration, idle duration | 2 |
| Count of ones for each of the 8 digital signals | 8 |

The paper does not define its "7 bins" **[ours]**: each cycle is split into 7 equal-time bins, and empty bins of very
short cycles are filled from neighbouring bins.

### 6.2 10-minute means (TranAD and Isolation Forest input) — Papadopoulos et al. 2026, App. A.1.5

The mean of each signal per 10-minute bin, with empty bins dropped (25,521 bins in 213 segments). The decision time is
the end of the bin. Consecutive bins form a segment; a missing bin starts a new one.

### 6.3 Windows and scaling

- **Windows** never cross a segment. Positions before a segment's start repeat its first sample (replication padding,
  as in the TranAD code). Window length: 10 cycles for the LSTM-AE **[ours, paper silent]**, 10 bins for TranAD **[paper]**.
- **Scaling** is fitted on training samples only: z-score for the LSTM-AE **[ours, paper silent]**, min-max for TranAD
  **[paper, eq. 1]**, none for Isolation Forest (trees are invariant to monotone rescaling).

### 6.4 Chronological splits

Splits are **time-based only**, because adjacent windows are near-duplicates and a random split would leak test data
into training. From 48 h before each failure (the benchmark's predictive horizon) until its maintenance, data are
**never used for training**, and training windows reaching into such a period are removed too.

- **Fixed:** one model trained on everything before F1 − 48 h (1 Feb → 16 Apr) is tested on everything after.
- **Rolling** (walk-forward, as in Laxhammar & Falkman's update trials): model *k* is trained on all normal data before
  cut *k* and tested on (cut *k*, cut *k+1*]. The cuts are F1 − 48 h, then each previous maintenance time. The first two
  models share their training data, since nothing normal lies between F1 − 48 h and F1's maintenance.
- The last 20% (by time) of each training period is the **validation** set, used for early stopping and calibration.
  **The four failures are never used to choose anything.**

| Split / part | LSTM-AE (cycles) train / val / test | TranAD & IF (10-min bins) train / val / test |
|---|---|---|
| Fixed | 2,862 / 715 / 13,096 | 7,318 / 1,829 / 16,373 |
| Rolling part 0 (tests F1) | 2,862 / 715 / 829 | 7,318 / 1,829 / 406 |
| Rolling part 1 (model reused) | 2,862 / 715 / 5,153 | 7,318 / 1,829 / 4,648 |
| Rolling part 2 | 6,690 / 1,673 / 349 | 10,753 / 2,688 / 1,071 |
| Rolling part 3 | 6,855 / 1,714 / 6,765 | 11,131 / 2,783 / 10,248 |

All results use 3 seeds (42, 43, 44). Training on CPU is deterministic: retraining reproduces the scores to within
10⁻¹³.

---

## 7. Baseline detectors

### 7.1 LSTM autoencoder — Davari et al., SAC 2024

- Encoder LSTMs 256-200-140-100-60, bottleneck 20, dropout 0.2, batch 50, up to 100 epochs, Adam **[paper, Table 3]**.
- Decoder mirrors the encoder, with the bottleneck state repeated over the sequence **[ours]**.
- Early stopping on validation loss, patience 10 **[ours]**.
- Score = RMSE of the reconstruction of the last (decision) cycle **[ours]**; the paper plots an RMSE over time without
  defining it.

### 7.2 TranAD — Tuli et al., VLDB 2022

The network classes are copied from the official repository (BSD-3; licence in `third_party/`). The **code** is
followed rather than the paper, because the code produced the published MetroPT-3 benchmark results.

<details>
<summary>Paper vs official code</summary>

| Element | Paper | Official code (used) |
|---|---|---|
| Loss weights | ε⁻ⁿ (ε described inconsistently) | 1/n and 1 − 1/n, n = epoch |
| Adversarial min-max | Two decoder losses with opposite signs | Not implemented: one combined loss |
| MAML meta-step | End of each epoch | Not implemented |
| Score | ½ phase-1 + ½ phase-2 error | Phase-2 error only, averaged over features |
| Target | Whole window | Last element of the window |
| Size | "64 hidden units" | Feed-forward 16, model width 2 × features, heads = features |
| Schedule | Step 0.5 | AdamW, weight decay 1e-5, × 0.9 every 5 epochs |
| Activation | — | `nn.LeakyReLU(True)` sets the negative slope to 1 (the identity); kept as in the code |

</details>

Our additions: early stopping on validation loss (patience 5); the single encoder/decoder layer is called directly
rather than through `torch.nn.TransformerEncoder`, which no longer accepts the custom layers; learning rate 0.001 and up
to 25 epochs, inside the benchmark's grid.

### 7.3 Isolation Forest — Liu et al., ICDM 2008

scikit-learn with t = 100 trees, ψ = 256 and all features **[paper defaults]**. Score s(x, n) = 2^(−E[h(x)]/c(n)),
i.e. −`score_samples`.

### 7.4 Baseline thresholds

The first run showed that TranAD's own POT threshold sat far above every score, so TranAD never alarmed — which says
nothing about the detector itself. For a fair comparison, **every model is evaluated under every threshold rule**:

- **p95** — the 95th percentile of the training scores (Alhammadi et al. 2025); the primary rule in figures.
- **POT** — peaks-over-threshold (SPOT, Siffer et al. 2017) on training scores, risk 1e-5, initial level 0.99, GPD fitted
  by maximum likelihood. This is TranAD's published rule; its code tunes the level per dataset, often on test data, so it
  is fixed here.

Both rules are uncalibrated: neither says what false-alarm rate it implies on new months. That is what the trustworthy
layer adds.

---

## 8. Evaluation

### 8.1 Event-level metrics

Following the problem statement and the operator's protocol (Veloso et al. 2022):

- **Alarm interval:** a run of consecutive flagged decisions inside one segment.
- **Detected:** some flagged decision between failure start − 48 h and failure end.
- **Lead time:** failure start − first such decision. It **meets the requirement** if ≥ 2 h.
- **False alarm:** an alarm interval starting outside every [failure start − 48 h, maintenance]. Alarms between failure
  end and maintenance are ignored, since the fault is known and being repaired.
- **False alarms per day:** divided by **recorded** normal days (non-empty 10-minute bins).
- **Event precision:** alarm intervals touching a failure window / all counted alarm intervals.

### 8.2 Chance level

A 48 h warning window is long. A random alarm process firing at a model's own false-alarm rate *r* (per day) lands in
it with probability **1 − exp(−2r)** — 99% at 2.5 false alarms per day. Every detection count is therefore reported
next to `detected_by_chance`, and every ≥ 2 h warning count next to `meets_warning_by_chance` (a random alarm in
[start − 48 h, start − 2 h]). Only detections clearly above chance are evidence.

This was found in the first run, where Isolation Forest "detected" all four failures about 48 h ahead — exactly at the
start of the window.

### 8.3 Where false alarms fall

Reported for every model:

- `fa_in_frozen` — false alarms starting in frozen data;
- `fa_in_episodes` — false alarms inside an unreported-episode window [start − 48 h, end];
- `false_alarms_per_day_clean` — the remaining ones per day of clean normal time (no failure, episode or frozen time);
- `episodes_detected` — unreported episodes in the test period with ≥ 1 alarm, next to its chance level.

### 8.4 Benchmark AD1 metric (Papadopoulos et al. 2026)

For comparison with the benchmark's medians (TranAD 0.510, Isolation Forest 0.580):

- Predictive horizon PH = [f − 50 h, f − 2 h); alarms in the last 2 h are ignored.
- Precision = flagged decisions in PH / flagged decisions; recall = failures with ≥ 1 flag in PH / failures.
- **AUC-PR** over 200 score quantiles — threshold-free.

AD1 is computed for the fixed split only (one model, one score scale). Our scores cover only the test period, and
decisions from failure start to maintenance are excluded. The benchmark's failure file is not in its repository, so it
is unknown whether its failure time is the start or the end; AD1 is reported with both anchors (`ad1_auc_pr_start`,
`ad1_auc_pr_end`). Anchoring at the end puts the failure itself inside the horizon, which favours detectors that fire
*during* the failure. The benchmark's best TranAD score (0.846) is the best of 50 configurations, chosen on the test
failures; its medians are the fair reference.

### 8.5 Statistical honesty

With four failures, detection rates move in steps of 25 points, so results are reported per failure, over 3 seeds and
against chance. Statistical comparisons use the synthetic benchmark (section 10.1).

---

## 9. Baseline results

Full run: 3 seeds, fixed and rolling splits, before the data-quality rules.

| Fixed split, p95 | Failures detected | By chance | False alarms / day | AD1 (start / end anchor) |
|---|---|---|---|---|
| Isolation Forest | 4.00 | 3.93 | 2.09 | 0.052 / 0.159 |
| LSTM-AE | 3.67 | 3.79 | 1.47 | **0.531** / 0.514 |
| TranAD | 4.00 | 4.00 | 10.76 | 0.021 / 0.426 |

- **Every detection is at chance level.** Lead times of ~48 h are the start of the window: background noise.
- **POT never fires** (risk 1e-5 on ~7,000 training scores ≈ one exceedance every two years of 10-min bins).
- **Threshold-free**, the LSTM-AE has the best early signal (AD1 std 0.27 across seeds); TranAD detects failures *while
  they happen*, and its end-anchored AD1 is close to the benchmark median (0.510), suggesting the benchmark anchors at or
  after the failure; Isolation Forest stays below the benchmark (0.580).
- **On the synthetic benchmark** (section 10.1) the picture repeats: Isolation Forest 100% (chance 92%), LSTM-AE 77%
  (chance 81%), TranAD 100% (chance 99.9%).
- **Frozen data, not precursors.** TranAD's long, strong alarms from 17 Apr 09:30 and on 26–28 May are frozen data:
  TranAD detects the logger freezing.
- **Drift.** Fixed TranAD's false alarms grow from 45 (April) to 600 (July), flagging 32% of normal decisions under a 5%
  rule; rolling retraining lowers false alarms per day (TranAD 10.8 → 6.5, Isolation Forest 2.1 → 1.6).
- **LSTM-AE.** It cannot score F1 (no cycle closes during it); it has the best early signal but is unstable across
  seeds. Its rolling validation loss collapsed (0.0003, 0.07) when the 26–28 May frozen stretch was in its training
  data, because COMP flicker inside frozen data creates many identical fake cycles (fixed by R2).
- Some "false alarms" are correct detections of unreported episodes (R3); they are reported separately.

**Why these differ from Davari et al.** (recall 100%, precision 95%, F1 97.4%, lead times 15.7 / 1.6 / 1.3 / 8.3 h):
their split is 70-10-20 with no stated chronological order, their 21 operator-labelled windows are not public, their
F1 warning overlaps the frozen period, and they do not report against chance.

**Known limitations of the baselines** (targets for the trustworthy layer):

1. **Late cycle closure (LSTM-AE).** During a leak the compressor runs almost continuously, so a cycle may not close
   for hours and cannot be scored until then. This follows Davari et al.'s segmentation faithfully.
2. **Uncalibrated thresholds.** p95 flags ~5% of normal decisions by construction; POT's level is a free parameter.
3. **Seasonal drift.** The fixed model is trained on February–mid April, when the duty cycle is lowest.
4. **Short reference before F3** (≈ 4 days of new normal data for the third rolling model).

---

## 10. The trustworthy layer

| Pillar | Technique | Key equation | Source | Parameters chosen on |
|---|---|---|---|---|
| Robustness | Data-quality gate (R1–R3); sensor-fault tests | Section 5.3 | Our data analysis | Otsu breaks (data) |
| Uncertainty | Conformal p-values | p = (1 + #{cal ≥ s}) / (n + 1) | Laxhammar & Falkman 2014; Bates et al. 2023 | Operator budget F |
| | Calibration-conditional check | b_i = min(i/n + √(log(2/δ)/2n), 1) | Bates et al. 2023 | δ = 0.1 |
| Safety | Bonferroni combination | p_k < ε_k(F)/m | Bates et al. 2023 | Operator budget F |
| | EWMA smoothing | s_t = a·x_t + (1 − a)·s_{t−1} | Hundman et al. 2018 | Validation leaks |
| Oversight | Triage with guaranteed risk (SGR) | Binomial tail bound B*, δ = 0.001 | Geifman & El-Yaniv 2017 | Validation alarms |
| Transparency | Per-alarm explanations | Per-signal error; DIFFI LFI = I_f / C_f | Tuli et al.; Carletti et al. 2023 | — |
| Consistency | Audit; recalibration | Block bootstrap; sliding window | Gama et al. 2014 | Normal data only |

All conformal guarantees assume exchangeable data. Decisions are correlated in time, so the guarantees are **checked
empirically** (observed monthly false-alarm share vs ε). No parameter is chosen on the four real failures.

### 10.1 Synthetic benchmark (step 2)

Four failures cannot support statistics, and nothing may be tuned on them. The benchmark therefore makes **10 copies**
of the real data and injects labelled events, then processes each copy exactly like the real data (both running rules,
44 features, 10-min bins, R1 flags) and scores it with the same trained models, scalers and thresholds.

Events are placed only in normal time — never within 48 h of a reported failure or major episode, within 6 h of a minor
high-DV run or within 1 h of frozen data — and at least 48 h apart.

- **Leaks (60%).** A **build-up of 6–24 h** in which the compressor cycle is rebuilt from the host's own cycles (runs
  × (1 + 3·sev), idle × (1 − 0.5·sev)), oil + 8·sev °C, TP3/Reservoirs − 0.8·sev bar and DV pressure + 0.6·sev bar
  during runs, where sev(t) rises linearly to the event's severity S ~ U(0.2, 1). Then a **failure phase of 1–12 h**
  of continuous loaded running (from the host's loaded rows) with DV pressure 0.6 + 1.7·S bar, TP3 lowered by
  0.3 + 0.7·S bar and oil rising by 4 + 8·S °C. The ranges span the training-period episodes (E09–E23: 4.4–20.7 h,
  DV 0.59–1.70 bar), the build-up seen before E18 (26–28 Mar) and the reported failures (DV ≈ 2 bar).
- **Sensor faults (40%), 1–12 h:** *frozen* (all analog signals held, COMP flickering ~20% zeros, as in the real
  periods), *stuck* (one analog signal held) and *noise* (one analog signal + N(0, (3 sd)²)).
- **Splits:** events inside a model's validation window choose parameters; test-window events are only reported.

| Benchmark | Value |
|---|---|
| Events | 616 in 10 copies: 379 leaks, 97 frozen, 78 stuck, 62 noise |
| Validation leaks | 25 distinct (fixed split), 49 (rolling) |
| Test leaks | 232 distinct |

**Checks.** The injected signature behaves as intended (by the end of the build-up the running share has risen, oil is
7 °C warmer, TP3 0.7 bar lower and DV pressure 0.5 bar higher; in the failure phase DV pressure is 2.25 bar), and away
from the events the copies' scores equal the real data's exactly.

> **Limitation.** The leak magnitudes are informed by the reported failures' statistics, so the benchmark resembles
> them; parameters are still never fitted on the failures themselves.

### 10.2 Uncertainty — conformal p-values (steps 6–7)

- **Calibration set:** each model's clean validation scores, per split part.
- **p-value:** p = (1 + #{cal ≥ s}) / (n + 1); alarm if **p < ε**, with **ε = F / decisions per day** for an operator
  budget F ∈ {0.05, 0.1, 0.25, 0.5, 1} false alarms per day. F is a requirement, not a tuned parameter. At F = 0.1 a
  random alarm lands in a 48 h window only 18% of the time, so detections become informative.

**Results.** With static calibration, TranAD's clean false alarms drop from 10.8 per day to **0.05 / 0.12 / 0.18** at
F = 0.1 / 0.25 / 0.5. At F = 0.1, TranAD detects **3.33/4 real failures (chance 0.62), 2/2 unreported episodes (chance
0.18) and 51% of synthetic leaks (chance 18%)** — the first detections clearly above chance. But it detects them *as the
failure happens*: only 0.67/4 failures with ≥ 2 h warning, 5% of synthetic leaks caught in the build-up, median lead
−0.2 h. Early warning, where present, comes from the LSTM-AE (F = 0.5: 3/4 real failures warned ≥ 2 h ahead, chance
1.4), but not on the synthetic leaks.

**Two limits.**

- **Reachability.** A p-value is at least 1/(n + 1), so no budget below *decisions per day / (n + 1)* ≈ **0.08 per
  day** can ever alarm; F = 0.05 gave zero alarms by construction. A combination of m detectors needs m times that.
- **Calibration-conditional guarantees are infeasible.** Bates et al.'s DKWM bound gives a smallest conditional p-value
  of b₁ = 1/n + √(log(2/δ)/2n) ≈ **0.029** for n ≈ 1,800 (δ = 0.1), far above the ε ≤ 0.016 needed for F ≤ 1 per day.
  It would take tens of thousands of calibration points. Alarms therefore use the marginal p-value, and validity is
  checked month by month.

**Online recalibration — rejected.** Three calibration variants were compared (Gama et al.: blind sliding window;
Laxhammar & Falkman: semi-offline ICAD):

| Variant | Rule | Outcome |
|---|---|---|
| **static** | Calibration fixed | False alarms controlled, detections above chance — **used from here on** |
| rolling | Each decision joins a FIFO window 48 h after it was made, unless it lies in a logged failure's window | Absorbs abnormal behaviour as "normal": TranAD synthetic detection 51% → 19% |
| rolling_guard | Additionally excluded if a strict alarm followed within 48 h | Selection-bias loop: false alarms at F = 0.1 rise to 0.40 (TranAD), 0.63 (IF) and 1.99 per day (LSTM-AE); detections fall to chance |

The guard's loop: decisions followed by a strict alarm are the elevated ones, so withholding them fills the window with
the lowest scores; its tail shrinks, alarms increase and block still more decisions. Neither online variant improved
the monthly validity. Drift needs another mechanism, such as scheduled recalibration after confirmed maintenance.

> **Development notes.** A unit test with a +0.6 σ drift showed the guard slows adaptation. The first guard also used
> an unreachable alarm level (F = 0.05) and was never active; it was fixed to use the smallest reachable ε
> (≥ 2/(n + 1)) before the results above.

### 10.3 Robustness — the R1 gate (step 8)

Alarms on frozen data are reported as **"sensor data frozen"** (a data fault), not as an air leak. Every alarm level is
evaluated with and without the gate.

| Synthetic sensor fault (TranAD, F = 0.5) | Leak alarm without gate | With gate |
|---|---|---|
| Frozen data | 33% | **4%** |
| Stuck sensor | 5% | 5% |
| Noise burst | 14% | 14% |

Leak detection is unchanged, and every real detection count stays the same: **F1 is detected through the failure
itself** (continuous running after the freeze), not the freeze — even though the frozen stretch lies inside F1's 48 h
window. Stuck and noisy single sensors are not caught by R1.

### 10.4 Safety — combining detectors (step 8)

TranAD detects failures reliably; the LSTM-AE may warn earlier. They are combined without breaking the budget:

- **Bonferroni on the rate scale.** The budget F is split evenly over m detectors; detector *k* alarms when
  p_k < ε_k(F) / m, where ε_k(F) = F / decisions per day of *k*. Each contributes at most F/m false alarms per day, so
  the union contributes at most F, **under any dependence**. (Fisher's combination is invalid for conformal p-values,
  Bates et al. 2023.)
- **The union is formed on the common 10-minute grid:** a bin is in alarm if any member flagged a decision in it.
- **Reachability:** the pair needs F ≥ 0.17 per day and all three detectors F ≥ 0.26 (fixed split); on the rolling
  split, 1.67 and 2.51.

| Fixed split, R1-gated | F | Clean FA / day | Failures detected (chance) | ≥ 2 h warning (chance) | Episodes (chance) |
|---|---|---|---|---|---|
| **TranAD + LSTM-AE** | **0.5** | **0.13** | **4.0 (0.98)** | 2.0 (0.95) | **2 (0.43)** |
| TranAD + LSTM-AE | 1 | 0.40 | 4.0 (2.20) | 3.0 (2.14) | 2 (1.06) |
| TranAD | 0.5 | 0.18 | 3.67 (1.14) | 1.67 (1.11) | 2 (0.52) |
| TranAD | 1 | 3.01 (budget broken) | 3.67 (1.76) | 1.67 (1.74) | 2 (0.82) |
| LSTM-AE | 0.5 | 0.22 | 3.0 (1.44) | **3.0 (1.39)** | 1 (0.72) |

The pair keeps its budget where TranAD alone breaks it (summer tail drift). Without the R1 gate the detection counts
are identical; only the chance levels are higher (1.10 at F = 0.5, 2.27 at F = 1), because frozen-data alarms count
as false alarms. On the synthetic leaks, detection during the
build-up rises from 19% (TranAD) to 32% (pair) at F = 1.

> **Development note.** A first union interleaved the members' decisions on one timeline, so the LSTM-AE's flicker
> cycles in frozen data split one TranAD alarm into many intervals and inflated the chance levels. Forming the union
> on the common grid fixed it.

### 10.5 Safety — score smoothing (step 9): rejected

A causal EWMA, s_t = a·x_t + (1 − a)·s_{t−1} with a = 2/(span + 1), restarted at every data gap, was applied before the
conformal step (Hundman et al. 2018), with calibration scores smoothed the same way. The idea: accumulate a build-up's
weak but persistent evidence while damping isolated spikes. The span H ∈ {0, 0.5, 1, 2, 4, 8} h was chosen on the
**validation leaks only**, by the most ≥ 2 h warnings above chance at F = 0.5, and adopted only if positive and better
than no smoothing.

**Result.** Every span lowered validation detection and warnings (Isolation Forest 83% / 23% at 0 h → 4% / 0% at 8 h;
TranAD 0% warnings at every span), so the adopted span is **0 h for all detectors**. Forced on anyway, smoothing removed
the LSTM-AE's real early warnings (F = 0.5: 3/4 → 0/4) and tripled the pair's false alarms (0.13 → 0.39 per day).

**Why.** The smoothed calibration tail consists of normal slow excursions (season, long runs, morning restarts) as large
as a smoothed build-up, while the LSTM-AE's real warnings were short, sharp spikes that averaging removes. **Early
warning is limited by what the detectors' scores separate, not by post-processing.**

> **Development note.** A first selection rule picked the span with the lowest chance level (8 h for the LSTM-AE) even
> though no span produced any warning; the adoption rule above fixed this.

### 10.6 Human oversight — triage with a guaranteed risk (step 10)

The operating detector (TranAD + LSTM-AE, fixed split) raises alarms at a permissive budget F = 1 per day (R1-gated).
Each alarm is **confirmed automatically** if its confidence κ ≥ θ, and otherwise **sent to engineer review**.

- **SGR** (Geifman & El-Yaniv 2017, Algorithm 1) finds θ by bisection over the sorted κ in ⌈log₂ m⌉ steps, using the
  exact binomial tail bound B* = beta.ppf(1 − δ/⌈log₂ m⌉, errors + 1, n − errors), so that P(share of false confirmed
  alarms > r\*) < δ = 0.001, for r\* ∈ {0.05, 0.1, 0.2, 0.3}. θ = ∞ means no threshold can carry the guarantee.
- θ is chosen on the **validation alarms** of the benchmark copies (true = overlaps an injected leak).
- **Confidence function.** κ = −log₁₀(min adjusted p) ties at the conformal floor 1/(n + 1) for every strong alarm and
  cannot separate anything (zero coverage). The alarm's **duration** (a leak keeps the score up for hours; evidence
  breaks ties) is also offered, and the validation alarms chose it.
- **Sample-size limit:** even a confirmed set with zero errors needs ≈ 90 alarms before its bound falls below 10%
  (bound for 0 errors in 92 alarms: 0.095).
- **Checked by simulation:** no violations in 500 trials; with weakly separated κ, SGR correctly confirms nothing; with
  well-separated κ it confirms 35% of alarms at a true risk of 3.9% (bound 10%).

**Results.** Only r\* = 0.3 is reachable (408 validation alarms): θ ≈ 0.34 h, so alarms lasting at least ~20 minutes are
confirmed (validation coverage 9%, bound 0.27).

- **Real data:** 0.51 alarms per day → **0.09 per day confirmed** (false share 22%, within 0.3) and **0.42 per day to
  review**. 3 of 4 failures get a confirmed alarm, none ≥ 2 h ahead: long alarms come from the failure itself.
- **Synthetic test leaks: the guarantee is violated** — the false share among confirmed alarms is 0.34 > 0.3 (0.56
  without triage). The test alarms are not drawn like the validation alarms: April validation vs April–September test,
  with 26% of test alarms passing the duration threshold against 9% in validation, as false alarms get longer in summer.

**Central finding across the uncertainty and oversight pillars:** distribution-free guarantees (conformal p-values,
SGR) hold only under exchangeability, and the seasonal drift of a time-split deployment breaks it. Static calibration
was the safest choice tested; online recalibration was unsafe in both directions; periodic recalibration on
confirmed-normal data after maintenance is the practical remedy.

### 10.7 Transparency — per-alarm explanations (step 11)

The fixed-split models are re-trained exactly as in step 3 (verified: scores reproduced to within 10⁻¹³), because the
per-signal errors were not saved then. Every alarm at F = 1 (ungated, so frozen-data alarms are explained too) is
explained at its peak decision:

- **TranAD:** its per-signal phase-2 reconstruction error;
- **LSTM-AE:** the per-feature reconstruction error of the decision cycle;
- **Isolation Forest:** DIFFI local importance (Carletti et al.): along the path of x in every tree, each split on
  feature f adds 1/h_t(x) − 1/h_max to I_f and 1 to C_f, with h_max = ⌈log₂ ψ⌉; LFI_f = I_f / C_f. (Checked: an
  outlier in one feature gets that feature as most important.)

Attribution shares are summed into physical **sensor groups** — running (COMP, MPG, DV_electric, motor current,
run/idle times), DV pressure, oil, TP3/Reservoirs, TP2/H1 and other digital signals — so the detectors (15 signals vs
44 cycle features) become comparable and redundant sensors do not split credit. The leak signature is running, DV
pressure, oil and TP3 (4 of 6 groups, so 0.67 if attribution were spread evenly).

| TranAD alarms (F = 1) | Count | DV pressure share | Other digital share | Leak-signature share |
|---|---|---|---|---|
| During real failures | 24 | **0.82** (top group in 92%) | 0.11 | **0.88** |
| During unreported episodes | 11 | 0.70 | 0.11 | 0.86 |
| False alarms | 823 | 0.007 | **0.93** (top group in 99%) | 0.05 |
| Frozen data | 6 | 0.002 | 0.58 (running 0.35) | 0.36 |

- **TranAD's explanations separate real from false alarms.** Its false alarms come from brief state changes of binary
  signals (Towers, LPS, pressure switch, oil level, caudal impulses) — the source of its baseline false alarms.
- **Physics check per failure:** TranAD and the LSTM-AE point at DV pressure for F1–F3; F4 points at *running*
  (TranAD), matching the duty-cycle precursor found in the data.
- **Frozen data:** TranAD reacts to the flickering Towers/COMP signals — the model's own reason for the R1 gate.
- **Isolation Forest / DIFFI** ranks the other digital signals first in every context (95% of failure alarms, 97% of
  false ones): uninformative and least stable, consistent with its weak detection. Rare binary values isolate quickly,
  which DIFFI rewards.
- **Stability across seeds** (same alarms, same top group): LSTM-AE 100%, TranAD 99%, Isolation Forest 81%; mean
  pairwise Spearman 0.91 / 0.60 / 0.52.

### 10.8 Consistency — audit (step 12)

Do alarms on clean normal data occur equally often across **months**, **compressor states** (the decision's 10-min bin
mostly running — mean motor current above 2.85 A — or mostly idle) and **times of day** (4-hour blocks)? The audit
covers the three baselines at p95 and the trustworthy system (static conformal, F = 0.5, R1-gated: each detector and
the pair). The metric is a group's alarm share divided by the overall share (1 = average), with a 95% **block bootstrap
over days** (1,000 resamples; whole days are resampled because decisions within a day are correlated). A group differs
only if its interval excludes 1. Gama et al. (2014) motivate the month dimension; state and time of day test for
operating-regime effects.

| Clean alarm decisions | Apr | May | Jun | Jul | Aug |
|---|---|---|---|---|---|
| TranAD, baseline p95 | 196 | 609 | 902 | 2,533 | **6,543** (59% of decisions) |
| TranAD, conformal F = 0.5 | 21 | 18 | 3 | 3 | 41 |
| Pair, conformal F = 0.5 | 20 | 18 | 2 | 6 | 24 |

The trustworthy layer removes the *magnitude* of uneven alarming (TranAD: 27% → 0.22% of decisions) but not its
*shape*:

- **Month:** the August rise remains (ratio 1.7) — the summer tail drift also seen in the monthly validity and the
  triage violation.
- **Compressor state:** every subject alarms more in running bins, and calibration strengthens this for TranAD (ratio
  1.17 → 2.12; pair 2.22), because it keeps the extreme tail, which lives in running periods. The guarantee holds on
  average, not per operating state.
- **Time of day:** 32 of the pair's 70 clean alarms (ratio 3.2) fall in 00–04 h — the last half-hour before overnight
  parking (gaps ≈ 00:30–04:40); Isolation Forest likewise (2.5). Shutdown sequences switch the digital signals that
  drive false alarms, consistent with the explanations.

---

## 11. Final operating system

```
static conformal p-values → R1 gate → TranAD + LSTM-AE (Bonferroni, F = 0.5) → SGR triage → per-alarm explanations
```

| Metric (fixed split) | Baseline TranAD (p95) | Trustworthy system |
|---|---|---|
| Failures detected (chance) | 4.0 (4.00) | **4.0 (0.98)** |
| Unreported episodes detected (chance) | 2 (2.00) | **2 (0.43)** |
| Clean false alarms per day | 11.5 | **0.13** |
| ≥ 2 h warnings (chance) | at chance | 2.0 (0.95) |
| Frozen-data alarms | Reported as leaks | Reported as data faults |
| Explanation | — | DV pressure for real leaks, digital transients for most false alarms |

- The LSTM-AE alone reaches 3/4 failures warned ≥ 2 h ahead at 0.22 false alarms per day.
- At F = 1 the system raises about one alarm every two days; a fifth of them are confirmed automatically, each with an
  explanation.

---

## 12. Limitations and future work

### Limitations

- **Four failures.** Real-data detections are counts out of four; the synthetic benchmark (232 test leaks) carries the
  statistics, but its magnitudes are informed by the failures' statistics, and its build-up model may differ from real
  ones (the LSTM-AE's early warnings do not transfer to it).
- **Rolling split.** Small validation sets raise the reachable budget (combinations need F ≥ 1.7–2.5 per day), so the
  combination and triage results hold for the fixed split only.
- **Exchangeability.** Every guarantee (conformal, SGR) assumes it, and seasonal drift breaks it.
- **R1 covers frozen data only;** stuck and noisy single sensors still raise leak alarms (5–14%).
- **The episode windows (R3) are derived from the data,** not from the operator's list of 21 windows.
- **Not yet run:** bootstrap intervals for the main result tables, and the data-quality ablation (training with R2
  and with frozen data and episodes excluded); the switches are in place.

### Future work

1. **Build-up features** aimed at the precursor itself (duty-cycle and run-length trends, as before F4) — the bottleneck
   identified by the smoothing result.
2. **State-conditional calibration** (separate calibration for running and idle) to restore per-state guarantees.
3. **Recalibration after confirmed maintenance**, a safe alternative to online recalibration under drift.
4. **Explanation-aware triage** — confirm leak-signature alarms, deprioritise digital-driven ones. The idea came from
   the real failures' explanations (test data), so it is not evaluated here; on the synthetic leaks it would be
   favoured by construction, since they include a DV-pressure rise.
5. **Shutdown-transient handling** for the 00–04 h false alarms, and checks for stuck and noisy single sensors.
6. **The operator's 21 faulty windows** (Davari et al. 2021) as ground truth in place of the derived episodes.

---

## 13. References

- Veloso, B., Ribeiro, R. P., Gama, J., Pereira, P. M. (2022). The MetroPT dataset for predictive maintenance. *Scientific Data*.
- Davari, N., Veloso, B., Ribeiro, R. P., Gama, J. (2024). Fault forecasting using data-driven modeling: a case study for Metro do Porto data set. *ACM SAC*.
- Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., Gama, J. (2021). Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry. *IEEE DSAA*.
- Papadopoulos, A., Giannoulidis, A., Gounaris, A., Paparrizos, J. (2026). Time-series anomaly detection for predictive maintenance (benchmark). *ACM SIGMOD*.
- Alhammadi, Zaiter, Jaradat (2025). Autoencoder-based anomaly detection on MetroPT-3 (p95 threshold). *IEEE*.
- Tuli, S., Casale, G., Jennings, N. R. (2022). TranAD: Deep transformer networks for anomaly detection in multivariate time series data. *VLDB*.
- Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). Isolation Forest. *IEEE ICDM*.
- Laxhammar, R., Falkman, G. (2014). Inductive conformal anomaly detection for sequential detection of anomalous sub-trajectories. *Annals of Mathematics and Artificial Intelligence*.
- Bates, S., Candès, E., Lei, L., Romano, Y., Sesia, M. (2023). Testing for outliers with conformal p-values. *Annals of Statistics*.
- Hundman, K., Constantinou, V., Laporte, C., Colwell, I., Soderstrom, T. (2018). Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. *ACM KDD*.
- Geifman, Y., El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*.
- Carletti, M., Terzi, M., Susto, G. A. (2023). Interpretable anomaly detection with DIFFI: depth-based feature importance of Isolation Forest. *Engineering Applications of Artificial Intelligence*.
- Isgandarov et al. (2025). Isolation Forest with DIFFI for bike-sharing anomalies. *IFAC-PapersOnLine*.
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*.
- Siffer, A., Fouque, P.-A., Termier, A., Largouët, C. (2017). Anomaly detection in streams with extreme value theory. *ACM KDD*.
- Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*.

---

## 14. Authors

- Anant Jain
- Karmveer Kumar
- Shivam Gangwar