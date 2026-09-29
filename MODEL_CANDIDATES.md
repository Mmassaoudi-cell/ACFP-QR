# MODEL_CANDIDATES.md

Four candidates, each targeting a specific ranked weakness from `SOURCE_WEAKNESS_ANALYSIS.md`.
No "throw every mechanism together" combinations; every component must clear its own null-result
bar.

---
## Candidate A — ACFP-QR: Attack-Conditioned, Feasibility-Projected Quantile Recovery
**Fixes:** Rank 1 (attack-blindness), Rank 2 (feasibility), Rank 3 (uncertainty).

**Architecture** (3 stages around a frozen boosting backbone):
1. **Shared backbone** f_0: a boosting regressor (CatBoost/XGBoost/LightGBM — winner of the
   Table-III-style screening) trained once on clean historical data, exactly as in the source
   paper — gives the point estimate λ̂_0.
2. **Attack-type gate** g ∈ {clean, FDI, replay, byzantine}: supplied by the crypto layer's own
   failure-mode classification (Sec. IV-C of the source paper — already computed, zero extra
   cost). Routes to one of 3 lightweight residual correctors h_g (shallow gradient-boosted
   stumps, ≤ 32 leaves), each trained on (contaminated-input, f_0 error) pairs specific to that
   attack's characteristic contamination signature (Rank-1 fix):
   λ̂ = λ̂_0 + h_g(x)   if g ≠ clean, else λ̂ = λ̂_0.
3. **Quantile heads** q_0.1, q_0.9: two extra boosting models with pinball loss on the same
   features, giving a calibrated [P10, P90] band around λ̂ (Rank-3 fix).
4. **Feasibility projection**: after computing P_i = dispatch(λ̂) for every generator, redistribute
   the power-balance mismatch D − ΣP_i proportionally across non-saturated generators (water-
   filling) so the recovered dispatch is *exactly* feasible before its cost is counted (Rank-2 fix).

**Core math.** Water-filling correction: let U = {i : P_i < pmax_i}, mismatch Δ = D − ΣP_i;
update P_i ← P_i + Δ·(1/2a_i)/Σ_{j∈U}(1/2a_j) for i∈U, clipped to [pmin_i, pmax_i], iterated to
convergence (≤ |N| passes, linear in n).

**Why it should outperform the source method:** it targets the exact information asymmetry the
source method leaves on the table (attack type is already known but unused) and removes an
un-modeled infeasibility that directly inflates the paper's own headline metric (efficiency loss).

**Expected efficiency benefit:** inference cost = 1 backbone + ≤1 tiny corrector + water-filling
(O(n) per hour) — negligible overhead vs. the source method's single GBM call.

**Expected weakness:** residual correctors are trained on *synthetically contaminated* inputs
(no real deception-attack telemetry exists for this dataset) — an explicit, documented
assumption, not a hidden one (see `scripts/attack_contamination.py`).

**Required ablation:** −gate (single global corrector, no attack-type routing), −quantile,
−feasibility-projection, and a parameter-matched backbone-only control.

**Estimated cost:** ~3× backbone training time (correctors are tiny), inference ~1.1× backbone.

---
## Candidate B — GBM+Res-GRU: Boosting Backbone with a Temporal Residual Corrector
**Fixes:** Rank 4 (temporal structure), *without* replacing the backbone (since Rank-4's own
null-result screening showed pure sequence models do not beat boosting).

**Architecture:** frozen boosting backbone f_0 (as in Candidate A) + a small GRU (hidden=16)
trained only on the *residual* r = y − f_0(x), fed the 6-step price/load lag sequence already
used in the screening GRU baseline. Final prediction λ̂ = f_0(x) + GRU_residual(seq(x)).

**Why it might outperform:** boosting captures the strong nonlinear tabular mapping; the residual
GRU only has to learn what's left over — a much easier, lower-variance target than the raw price,
which is where sequence models are more likely to add value than when asked to model the full
signal from scratch.

**Expected weakness:** residual signal may be mostly noise (boosting already captures most
autocorrelation via lag features) — a real risk this candidate is designed to test, not assumed
away.

**Required ablation:** GRU-residual vs. linear-residual (AR(1) on residuals) vs. no correction.

**Estimated cost:** + one small GRU training pass (~10s), negligible inference overhead.

---
## Candidate C — End-to-end Tabular Transformer (deliberately simple, no hybrid tricks)
**Fixes:** nothing new — this is the "just use a bigger/more modern architecture" hypothesis,
included specifically to be falsified (Sec. 6: "do not brainstorm every architecture," but a
single clean deep-learning control is required to justify *not* choosing this path).

**Status:** already smoke-tested in `results/raw/benchmark_screen.csv` (the `Transformer` row) —
**worst RMSE of all 15 screened models** (6.43-6.59 vs. best boosting 5.00-5.09, all 3 seeds).
**Eliminated at Stage 1 (smoke test) per Sec. 9** — kept in the candidate list only as a
documented negative result, not carried into Stage 2/3 screening or final tuning.

---
## Candidate D — QR-GBM: Quantile-only Boosting (no attack-gate, no feasibility projection)
**Fixes:** Rank 3 only — an ablation-adjacent, simpler candidate used to isolate how much of
Candidate A's gain (if any) comes from quantile calibration alone vs. the full assembly.

**Architecture:** backbone f_0 + q_0.1/q_0.9 quantile heads, no gate, no water-filling.

**Required use:** internal control for Candidate A's ablation table, not a submission candidate
in its own right.

---
## Ranking (before validation screening, per Sec. 6 criteria)
| Candidate | Expected perf. | Novelty | Efficiency | Data compatibility | Feasibility | Transactions-level contribution |
|---|---|---|---|---|---|---|
| A (ACFP-QR) | High | High (exploits a free signal the source paper ignores) | High | High | High | High |
| B (GBM+Res-GRU) | Uncertain (explicitly a test) | Medium | High | High | High | Medium |
| D (QR-GBM) | Medium (control only) | Low | High | High | High | Low (internal ablation only) |
| C (Transformer) | **Eliminated** | Low (no new mechanism) | Low | High | High | None |

Candidates A, B, D proceed to Stage 2 validation screening (`scripts/screen_candidates.py`).
