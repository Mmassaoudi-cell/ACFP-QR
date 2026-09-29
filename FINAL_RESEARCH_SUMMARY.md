# FINAL_RESEARCH_SUMMARY.md (v2, post-review revision)

All numbers below are read directly from `results/raw/final_test.csv`, `results/aggregate/*.csv`,
`BENCHMARK_WTL.csv`, `results/raw/backbone_generalization.csv`, `results/aggregate/network_validation_summary.csv`,
and `results/raw/robustness.csv` — none are hand-typed estimates. This is the v2 revision cycle,
following a simulated peer-review round (`PEER_REVIEW_DECISION.md`) that returned a Major Revision
decision on the v1 manuscript/model. See `REVISION_RESPONSE.md` for the point-by-point response.

## Source paper
Islam, Takiddin, Ismail, Kurban & Serpedin, "Linear-Complexity Unified Defense Against Deception
Attacks in Distributed Economic Dispatch Using Cryptography and Machine Learning," IEEE Trans.
Smart Grid, 2026. A threshold-Schnorr / Pedersen-DKG / pairwise-masking cryptographic security
layer for distributed economic dispatch (O(n) communication, vs. O(n²) prior art) with a
LightGBM-based ML fallback for post-attack price recovery.

## Reproduction outcome (unchanged from v1)
- **Crypto layer / DED equations:** fully reproducible from the published formulas; message-count
  scaling matches exactly; real secp256k1 ECSM/ECDSA latencies measured independently.
- **IEEE 14-bus case study:** reproduced to within ~0.2% on aggregate price/cost (best-fit
  coefficient-to-generator assignment; FDI/Replay attack costs reproduced within 0.24%/~1.5%).
- **ML price-recovery Table III:** not numerically reproducible (synthetic, unpublished target);
  methodology reproduced on the real GEFCom2014 substitute dataset instead.
- Full detail: `REPRODUCTION_REPORT.md`.

## v2 model enhancements (this revision cycle)
Diagnosed and fixed the root cause of the v1 Byzantine-attack anomaly and the DA's core
causal-attribution and coverage challenges, all via validation-only experiments before touching
TEST:
1. **Per-type gate capacity + Huber loss** (`scripts/enhance_gate.py`): Byzantine's contamination
   spans a wider multi-lag window than FDI/replay; the v1 shared shallow corrector underfit it.
   Fixed with a deeper, Huber-loss corrector for Byzantine specifically — a strict RMSE
   improvement on VAL with zero cost to FDI/replay.
2. **Label-noise-robust gate training** (`scripts/gate_robustness.py`): each corrector's training
   set now mixes in 15% cross-type examples, so no corrector assumes the attack-type flag is
   always correct. Validated to help (not merely not hurt) at every simulated misclassification
   rate from 0% to 40%, on both VAL (selection) and TEST (confirmation).
3. **Split-conformal calibration of the quantile heads** (Romano et al. 2019 CQR): corrects
   coverage from 62.0% (raw) to 79.8% (calibrated) against an 80% nominal target.
4. **Backbone-generalization check** (`scripts/backbone_generalization.py`): the identical gate
   mechanism attached to a TCN backbone also improves it (6.830→6.709 RMSE), confirming the gain
   is attributable to attack-type conditioning, not to CatBoost specifically.
5. **Network-level validation** (`scripts/network_validation.py`): ACFP-QR embedded in a live,
   multi-iteration dual-ascent loop on the reproduced IEEE 14-bus system, driven by a real
   GEFCom2014-derived demand series and an exact self-consistent clearing price.
6. **Out-of-taxonomy attack test + gate-misclassification sweep** added to the frozen-model
   robustness suite.
7. **Statistical machinery corrected**: proper step-down Holm-Bonferroni (was: independent
   per-rank threshold check), matched-pairs rank-biserial effect size from signed ranks (was:
   sign-count approximation), bootstrap 95% CI on the paired mean difference, and — critically —
   the same paired-testing protocol now applied to economic efficiency loss (the declared
   co-primary metric), not just RMSE.

## Final TEST results (v2, frozen config, evaluated once, 10 seeds)
- **ACFP-QR RMSE = 6.173 $/MWh**, the lowest of all 16 compared models (best independent
  benchmark: TCN at 6.423).
- Source-method-reproduction benchmark (best of SVR/RF/GradientBoosting/XGBoost/LightGBM on this
  TEST split = GradientBoosting, RMSE 7.409) is beaten by a 16.7% relative RMSE reduction.
- **Win/tie/loss vs. 11 independent benchmarks, both co-primary metrics, step-down Holm-corrected:**
  - RMSE: **11 wins / 0 ties / 0 losses** (all significant at p<0.05, including TCN at p=0.014 —
    the v1 tie is now a win).
  - Efficiency loss: **8 wins / 3 ties / 0 losses** (ties: GRU, TCN, Transformer — no losses).
- Aggregate vs. attack-agnostic baseline: RMSE −11.8% (6.998→6.173), efficiency loss −12.8%
  (8.234%→7.182%) — v1 had efficiency loss essentially flat/slightly worse; v2 genuinely improves it.
- Per-attack-type: FDI RMSE −35.5%/MAPE −51.7%/effloss 47.1%→23.4%; Replay RMSE −11.2%/MAPE
  −14.4%/effloss 15.8%→17.8%; Byzantine RMSE −14.0%/MAPE −26.9%/effloss 18.0%→24.0%. RMSE/MAPE
  improve for all three attack types (more than v1); efficiency loss still rises for
  replay/Byzantine specifically despite this — see "Remaining nuance" below.
- Feasibility violation: 16.0% (pre-projection) → ~7×10⁻¹⁵% (post-projection).
- Quantile [10,90] coverage: **79.8%** (nominal 80%) — conformal calibration closed the v1 gap
  (62.0%) almost exactly.
- Latency: ≈0.12 ms/sample amortized (481 ms / 3,870-row TEST batch, slightly higher than v1's
  393ms due to the larger Byzantine corrector); 39-93× slower than plain CatBoost/LightGBM but
  3-4 orders of magnitude faster than SVR/kNN.
- Backbone generalization: CatBoost 6.978→6.581 RMSE with gate (−5.7%); TCN 6.830→6.709 RMSE with
  the identical gate (−1.8%) — mechanism generalizes, does not depend on CatBoost specifically.
- Network-level validation: FDI/replay/Byzantine inflate 14-bus dispatch cost 2.1-4.2% if
  unaddressed; either recovery model restores dispatch to within 4×10⁻⁴% of true optimum once
  embedded in the live dual-ascent loop.
- Gate-misclassification sweep: ACFP-QR RMSE degrades gracefully from 6.40 (0% misclassification)
  to 7.09 (40%), remaining below the attack-agnostic baseline (7.20, constant) throughout.
- Out-of-taxonomy attack: ACFP-QR (5.719) statistically indistinguishable from baseline (5.723).

## Remaining nuance (retained, not hidden)
Efficiency loss for replay and Byzantine contamination specifically still rises even though their
RMSE/MAPE both improve substantially in v2 (a smaller version of the v1 anomaly, now also present
for replay, which v1 had reported as flat). The synthetic fleet's cost curvature makes a subset of
large, lower-probability price errors disproportionately costly; a squared-error-trained corrector
does not fully minimize expected dispatch cost for every attack type even with the v2 capacity fix.
A directly cost-aware training objective (identified, not yet implemented) is the natural next
step — see `manuscript/main.tex` Sec. VII (Limitations).

## Ablation conclusion
Unchanged mechanism, updated numbers: the attack-gate is the only component affecting point
accuracy and efficiency loss (full model 6.173 RMSE / 7.182% effloss vs. gate-removed 6.998 / 8.234%);
the feasibility projection affects only power-balance violation (16.0%→~0, zero accuracy cost);
the quantile heads affect only interval coverage. See `results/aggregate/ablation_table.csv`.

## Statistical significance
10-seed paired Wilcoxon, step-down Holm-Bonferroni correction, matched-pairs rank-biserial effect
size, bootstrap 95% CI — applied identically to RMSE and efficiency loss
(`results/aggregate/wilcoxon_rmse.csv`, `results/aggregate/wilcoxon_effloss.csv`, `BENCHMARK_WTL.csv`).

## Recommended venue
IEEE Transactions on Smart Grid (matches the source paper's venue and this study's scope).
