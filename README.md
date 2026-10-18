
## 1. Data pipeline (GEFCom2014 price-recovery track)
```
python scripts/data_gefcom.py          # -> DATA_SPLIT_MANIFEST.csv, asserts no cross-split leakage
```


## 1. Stage 1/2 benchmark screening (TRAIN/VAL only, clean data)
```
cd scripts
python benchmark_regression.py --mode screen --seeds 0 1 2 --out ../results/raw/benchmark_screen.csv
```

## 2. Stage 2b/3 candidate screening + tuning (TRAIN/VAL only, simulated attacks)
```
python screen_candidates.py            # -> ../results/raw/candidate_screening.csv
python tune_candidateA.py              # -> ../results/raw/tuning_candidateA.json (Optuna TPE, 30 trials)
```
Results feed `MODEL_SELECTION_REPORT.md`.

## 3. Post-review model enhancement (TRAIN/VAL only — v2 revision cycle)
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

