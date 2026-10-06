# Reproduction guide

All commands run from the project root
(`C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 39`) with the
environment in `requirements.txt` installed (`pip install -r requirements.txt`).
Read `FINAL_RESEARCH_SUMMARY.md` first for the overall narrative; each artifact below links back
to the task-spec section it satisfies.

## 0. Source-paper audit and reproduction (no ML training required)
```
python scripts/reproduce_14bus.py      # -> results/raw/reproduce_14bus.json, feeds REPRODUCTION_REPORT.md
python scripts/crypto_repro.py         # -> results/raw/crypto_repro.json, real secp256k1 ECDSA/Schnorr timings
```

## 1. Data pipeline (GEFCom2014 price-recovery track)
```
python scripts/data_gefcom.py          # -> DATA_SPLIT_MANIFEST.csv, asserts no cross-split leakage
```
Requires `C:\Users\MMASSAOUDI\Desktop\Data\Load Data\GEFCom2014_Dataset\...` (see DATA_AUDIT.md
for the exact path and provenance). Adjust `TASK15`/`SOLUTION` constants at the top of
`scripts/data_gefcom.py` if the data folder is relocated.

## 2. Stage 1/2 benchmark screening (TRAIN/VAL only, clean data)
```
cd scripts
python benchmark_regression.py --mode screen --seeds 0 1 2 --out ../results/raw/benchmark_screen.csv
```

## 3. Stage 2b/3 candidate screening + tuning (TRAIN/VAL only, simulated attacks)
```
python screen_candidates.py            # -> ../results/raw/candidate_screening.csv
python tune_candidateA.py              # -> ../results/raw/tuning_candidateA.json (Optuna TPE, 30 trials)
```
Results feed `MODEL_SELECTION_REPORT.md`.

## 3b. Post-review model enhancement (TRAIN/VAL only — v2 revision cycle)
```
python enhance_gate.py                 # -> ../results/raw/gate_enhancement_screen.csv (diagnoses + fixes the Byzantine underfit)
python gate_robustness.py              # -> ../results/raw/gate_robustness_screen.csv (validates label-noise-robust training)
python backbone_generalization.py      # -> ../results/raw/backbone_generalization.csv (CatBoost vs. TCN + gate)
```
Results feed `FINAL_MODEL_CONFIG.yaml` (v2) and `REVISION_RESPONSE.md`. **Do not edit
`FINAL_MODEL_CONFIG.yaml` after this point without re-running the relevant validation-only stage
from scratch.**

## 4. Final, single TEST evaluation (frozen v2 config, 10 seeds)
```
python final_test_eval.py              # -> ../results/raw/final_test.csv (~20-30 min, CPU-bound)
python build_ablation_table.py         # -> ../results/aggregate/ablation_table.csv
python stats_tests.py                  # -> ../results/aggregate/{rmse,effloss}_summary.csv, wilcoxon_{rmse,effloss}.csv
python build_wtl.py                    # -> ../BENCHMARK_WTL.csv (both RMSE and efficiency-loss)
python network_validation.py           # -> ../results/raw/network_validation*.csv, ../results/aggregate/network_validation_summary.csv
```

## 5. Robustness sweeps (5 seeds, frozen final model, TEST split)
```
python robustness.py                   # -> ../results/raw/robustness.csv (incl. gate-misclassification and out-of-taxonomy sweeps)
```

## 6. Figures and manuscript
```
python make_figures.py                 # -> ../figures/fig{3,4,5,6,7,8}_*.pdf (from results/ CSVs only)
python make_tables.py                  # -> ../tables/*.tex
cd ../manuscript
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # 2-3 passes for cross-references
```

## Directory map
- `SOURCE_PAPER_AUDIT.md`, `REPRODUCTION_REPORT.md`, `SOURCE_METHOD_REPRODUCTION/` — Sec. 2-4
- `SOURCE_WEAKNESS_ANALYSIS.md` — Sec. 5
- `MODEL_CANDIDATES.md`, `MODEL_SELECTION_REPORT.md` — Sec. 6-10
- `DATA_AUDIT.md`, `DATA_SPLIT_MANIFEST.csv` — Sec. 12-13
- `FINAL_MODEL_CONFIG.yaml` — Sec. 11 (v2: post-review revision)
- `BENCHMARK_WTL.csv` — Sec. 30 (both RMSE and efficiency-loss)
- `PEER_REVIEW_DECISION.md` — simulated 5-reviewer peer review + editorial decision
- `REVISION_RESPONSE.md` — point-by-point response to the review, v1 -> v2
- `scripts/` — all code (documented per-file, see docstrings)
- `results/raw/` — per-seed raw results; `results/aggregate/` — summarized CSVs
- `figures/`, `tables/` — generated, not hand-edited
- `manuscript/main.tex` — IEEE Transactions-format manuscript (v2, resubmission-ready)
- `FINAL_RESEARCH_SUMMARY.md` — top-level narrative summary (v2)
