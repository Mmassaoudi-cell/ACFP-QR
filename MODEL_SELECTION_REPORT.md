# MODEL_SELECTION_REPORT.md

Internal development record. TEST was never touched in this document — all numbers below are
TRAIN/VAL only, per Sec. 9-11 policy. The final manuscript does not reproduce these tables; it
reports only the single selected model's frozen-config TEST results.

## Stage 1 — Smoke test
15 models (11 classical/boosting + 4 deep) trained on TRAIN, evaluated on clean VAL, 1 seed
(`results/raw/benchmark_screen_smoke.csv`). All converged; none collapsed. The `Transformer`
baseline was **eliminated** at this stage: worst RMSE of all 15 models (6.58 vs. best 5.00), no
mechanism-specific justification to carry forward (Candidate C, MODEL_CANDIDATES.md).

## Stage 2 — Validation screening (3 seeds, clean VAL)
Full 3-seed run (`results/raw/benchmark_screen.csv`). CatBoost has the lowest mean RMSE
(5.020 vs. XGBoost 5.118, LightGBM 5.159, HistGB 5.185; the source paper's winner, LightGBM, is
3rd here — a real, reported discrepancy attributable to real vs. synthetic data, see
REPRODUCTION_REPORT.md). **CatBoost selected as the backbone** for all hybrid candidates.

Notably, RMSE ranking and mean economic-efficiency-loss ranking **disagree**: Ridge/Linear
Regression have the *lowest* mean efficiency loss (4.28-4.32%) despite mediocre RMSE (5.68-5.69),
while several boosting models with lower RMSE show higher efficiency loss (e.g. RandomForest RMSE
5.60 but effloss 6.00%). This motivates reporting both metrics throughout the rest of this study
rather than optimizing RMSE alone.

## Stage 2b — Candidate screening under simulated deception attacks (3 seeds, contaminated VAL)
`results/raw/candidate_screening.csv`, `scripts/screen_candidates.py`. Backbone fit on the first
80% of TRAIN (chronological); attack-gate/residual correctors fit on the held-out last 20% of
TRAIN (contaminated); evaluated on a separately-contaminated copy of VAL. Aggregate (all attack
types + clean mixed, 20% contamination rate):

| Candidate | RMSE | MAE | R² | MAPE |
|---|---:|---:|---:|---:|
| **A_ACFP-QR (proposed)** | **6.716** | **3.568** | **0.9363** | **6.05%** |
| B_GBM+Res-GRU | 7.032 | 3.914 | 0.9302 | 6.87% |
| D_QR-GBM (= Baseline point-pred) | 7.069 | 3.945 | 0.9294 | 6.93% |
| Baseline-attack-agnostic-GBM | 7.069 | 3.945 | 0.9294 | 6.93% |

Per-attack-type breakdown for Candidate A vs. baseline (the mechanism's actual target):

| Attack type | Baseline RMSE | Candidate A RMSE | Baseline MAPE | Candidate A MAPE |
|---|---:|---:|---:|---:|
| FDI | 14.03 | **11.25** (−19.8%) | 19.36% | **9.84%** (−49.2%) |
| Replay | 11.12 | **10.62** (−4.5%) | 12.34% | **11.21%** (−9.2%) |
| Byzantine | 9.86 | 10.30 (+4.5%, worse) | 13.10% | **10.65%** (−18.7%) |
| Clean | 5.241 | 5.241 (identical — gate never fires) | 4.92% | 4.92% (identical) |

**Interpretation (reported honestly, including the mixed result):** Candidate A's gain is
concentrated where its mechanism directly applies (FDI: large, unambiguous win; Replay: modest
win). On Byzantine contamination, RMSE is slightly *worse* than the baseline (a gradual, spread-out
contamination is harder for a shallow per-type corrector trained on only ~20% of one held-out
TRAIN slice to characterize) even though MAE and MAPE both improve — i.e., the corrector reduces
typical-case error but is more exposed to occasional larger misses on this attack type. This is
retained as a known limitation (see ablation/robustness sections of the final results), not
hidden. Feasibility projection is verified to reduce the mean power-balance violation from ~16-17%
(baseline, pre-projection) / ~14-14.5% (Candidate A, pre-projection — lower than baseline simply
because its price predictions are more accurate) down to ~1e-14% (post-projection, machine
precision) in all 3 screening seeds — this component has no accuracy trade-off to weigh, it is a
strict improvement (`results/raw/candidate_screening.csv`,
`feasibility_violation_pct_{before,after}_projection` columns). Demand for this check is defined
as the synthetic fleet's own dispatch at the *true* price (`economic_impact.implied_demand`),
making the true-price dispatch exactly feasible by construction, so any measured violation is
attributable only to price-prediction error, not to an arbitrary demand-scaling artifact.

Candidate B (GBM+Res-GRU) produces only a marginal improvement over the plain backbone
(RMSE 7.032 vs. 7.069, ~0.5%) — consistent with the Rank-4 null-result prediction in
`SOURCE_WEAKNESS_ANALYSIS.md`: most exploitable temporal structure is already captured by the
hand-engineered lag features, so a temporal residual corrector adds negligible value here. **Not
selected** as the final model, but its ablation-relevant finding is retained.

## Stage 3 — Optuna TPE tuning of the top candidate (validation only)
`scripts/tune_candidateA.py`, 30 trials, TPE sampler, objective = mean RMSE across 2 contamination
seeds on contaminated VAL (TRAIN-only fitting). Best: RMSE 6.668 (vs. 6.716 default params, a
further 0.7% improvement) at `depth=9, learning_rate=0.0223, iterations=800, gate_depth=3,
gate_iterations=100` (`results/raw/tuning_candidateA.json`).

## Final selection
**Candidate A (ACFP-QR)**, tuned hyperparameters above, is selected as the final proposed model
per the Sec. 10 rule: strongest validation performance among survivors, no seed collapse, clear
and mechanistically-explained novelty (exploits the crypto layer's free attack-type signal),
negligible efficiency overhead vs. the backbone alone, and a documented (not hidden) limitation
on one of three attack types. Frozen in `FINAL_MODEL_CONFIG.yaml` before any TEST evaluation.
