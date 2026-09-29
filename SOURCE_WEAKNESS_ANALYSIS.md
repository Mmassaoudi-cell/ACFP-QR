# SOURCE_WEAKNESS_ANALYSIS.md

Ranked by expected impact on the paper's own stated goal (limit welfare loss during deception
attacks). Each item is backed by either a textual claim in the paper, a reproduced experiment in
this study, or a structural gap in the published equations.

## Rank 1 — The ML recovery model is attack-blind despite the crypto layer already knowing the
attack type, for free
The crypto layer *deterministically discriminates attack type* by construction (Sec. IV-C: ECDSA
failure ⇒ FDI, counter/sequence mismatch ⇒ replay, threshold-Schnorr aggregate failure ⇒
Byzantine) — this is a zero-marginal-cost categorical signal. Yet the fallback regressor
(Eq. 39, f_ML(D, x_D^his, x_λ^his)) is a single global model with **no attack-type input** and is
trained once, offline, on attack-free history. It therefore cannot adapt its reliance on
recent-history features to the *specific* contamination pattern each attack type leaves behind
(FDI: an abrupt single-supplier deviation, caught in ~4 extra iterations per Table II; Replay: a
frozen/stale short window; Byzantine: an intentionally gradual, longer-horizon ramp designed to
evade detection, Sec. III-3). This is the single largest, most directly fixable gap, and the
cheapest to exploit since the signal already exists.

## Rank 2 — No feasibility guarantee on recovered dispatch (power-balance violation)
Eq. 39 maps the *predicted* price λ_rec through the same per-generator KKT projection used
online, but nothing enforces Σ P_i^rec = D. Whenever λ_rec ≠ λ_true (which Table III shows happens
~4-5% of the time in relative terms even for the best model), the recovered dispatch is not even
a feasible operating point — a materially stronger requirement than "close in $/MWh" for a power
system. The paper never reports a power-balance-violation metric, only price/cost. Our economic
simulator (`scripts/economic_impact.py`) confirms recovered dispatch under the best off-the-shelf
regressor still carries a measurable mismatch before any correction is applied.

## Rank 3 — No uncertainty quantification on the recovery signal
A single point estimate is used to drive dispatch during exactly the period when the system is
known to be under attack and least trustworthy — the moment calibrated uncertainty would be most
valuable (e.g., to bound worst-case welfare loss or trigger a more conservative fallback). Table
III reports only point-error metrics (RMSE/MAE/R²), with no interval coverage or calibration
check anywhere in the paper.

## Rank 4 — Purely tabular treatment of an explicitly time-indexed signal
Sec. VI-B explicitly plots a hallmark time series (Fig. 3-4) with clear temporal structure
(seasonality, autocorrelation, correlation coefficient 0.857/0.829 train/test), yet every
candidate model in Table III (SVR/RF/GB/XGBoost/LightGBM) is a stateless tabular regressor over
"historical" features engineered by hand; no explicit sequence model is tried. Our own screening
run (`results/raw/benchmark_screen.csv`) shows this is not a free lunch either way: plain
sequence models (GRU/TCN/Transformer) do **not** outperform boosting on this data
(best boosting RMSE ≈ 5.00-5.09 vs. best sequence-model RMSE ≈ 5.16-5.49 on the validation split,
3 seeds) — so naively swapping the backbone for a bigger temporal model is not automatically a
win, and any temporal mechanism added must be justified against this null result, not assumed.

## Rank 5 — Single point of failure at the coordinator (author-acknowledged, Sec. VII)
Not addressed by this study's ML-recovery track (out of scope for a price-regression
contribution), but retained here for completeness since it bounds how far "resilience" claims can
be pushed without also addressing availability.

## Rank 6 — Weak statistical/experimental rigor
No seeds, no CI, no significance test, no ablation, no robustness sweep, single train/test split
(no validation set) anywhere in the paper (Sec. "Missing" items of SOURCE_PAPER_AUDIT.md). This
does not change the mechanism but means every published number in Table II-IV is a single
realization; our own multi-seed reproduction (`results/raw/benchmark_screen.csv`) shows
run-to-run RMSE spread of ~0.05-0.4 $/MWh across seeds for the stochastic models — small, but
non-zero, and unreported in the source paper.

## Rank 7 — Fault-tolerance threshold is *more* restrictive than classical BFT, not less
T > ⌊2n/3⌋ honest suppliers required (i.e., can tolerate < n/3 malicious) — this matches, not
improves on, classical PBFT-style Byzantine fault tolerance; the paper's complexity contribution
is genuine (O(n) vs O(n²)) but the *fault-tolerance* bound is not stronger than prior BFT
literature, despite the framing implying a general security improvement. Not a flaw, but an
oversold-novelty risk worth correcting in how we position our own downstream contribution
(Sec. 27 of the task spec — avoid the same overclaiming pattern).

## Not counted as a weakness
Sections 15 (per-generator privacy proofs), and the O(n) communication-complexity claim are
independently verified by direct implementation (`scripts/crypto_repro.py`) and hold as stated;
we do not attempt to "fix" what is not broken.

---
## What this rules in/out for candidate design (Sec. 6-8 compliance)
- **In scope, high leverage, cheap:** attack-type-conditioned recovery (Rank 1), feasibility
  projection (Rank 2), calibrated uncertainty (Rank 3).
- **In scope but must clear a null-result bar:** any temporal/sequence mechanism (Rank 4) — our
  own screening shows plain GRU/TCN/Transformer backbones do not beat boosting here, so a
  temporal *residual corrector on top of* a boosting backbone (not a replacement) is the only
  temporal design worth carrying into screening.
- **Out of scope for this study:** coordinator single-point-of-failure (Rank 5) — an
  availability/systems problem, not a learning problem; re-deriving BFT bounds (Rank 7) — a
  cryptographic-protocol question, not addressed by a new hybrid ML model.
