# SOURCE_METHOD_REPRODUCTION

Faithful reproduction of the source paper's two independently-testable components. See
`../REPRODUCTION_REPORT.md` for the published-vs-reproduced comparison table and
`../SOURCE_PAPER_AUDIT.md` for what was fully/partially/not specified.

## 1. DED simulator (`../scripts/ded_sim.py`, `../scripts/reproduce_14bus.py`)
Direct implementation of Algorithm 1 / Eqs. (9)-(11): KKT-projected local dispatch
`P_i(λ) = clip((λ-b_i)/(2a_i), Pmin_i, Pmax_i)` and dual (sub-gradient) ascent
`λ_{k+1} = λ_k + α(D - ΣP_i)`, with an exact bisection cross-check (`clearing_price`) used to
verify the dual-ascent fixed point independent of step-size choice. Step size α is auto-selected
per the linearized stability bound α < 2/Σ(1/2a_i) (not published in the source paper).
FDI, Replay, and Byzantine attack injectors implement Sec. III's threat model exactly as
specified by Eq. 23 (replay) and the ranking-based Byzantine strategy.

## 2. Cryptographic layer (`../scripts/crypto_repro.py`)
Real secp256k1 group arithmetic via the `ecdsa` Python package (not a timing simulation):
ECSM, ECDSA sign/verify (Sec. VI-C claims), ECDH+HMAC-SHA256 pairwise mask derivation (Sec.
II-C4), and a real threshold-Schnorr partial-signature + aggregation routine (Eqs. 31-34).
Message-count scaling (Fig. 6: proposed 4n vs. P2P-augmented 2n+n(n-1) vs. fully decentralized
2n(n-1)) is reproduced exactly since it follows deterministically from the published formulas.
Measured on this machine (see `../results/raw/crypto_repro.json`): ECSM 0.336 ms, ECDSA-sign
0.373 ms, ECDSA-verify 1.423 ms, vs. published 0.49/0.50/0.45 ms on an AMD Ryzen 7 4700U — verify
is markedly slower here, consistent with the pure-Python `ecdsa` library doing unoptimized
big-integer modular inversion during verification (a library/hardware difference, not a
methodological one; the *ordering* sign ≈ ECSM < verify does not hold here, which we report
plainly as a real, unresolved discrepancy rather than adjusting the benchmark to match).

## 3. ML price-recovery benchmark (methodology only — see `../DATA_AUDIT.md`)
The paper's Table III benchmark (SVR/RF/GradientBoosting/XGBoost/LightGBM predicting marginal
price from demand/price history, chosen model = LightGBM, R²=0.9851) is reproduced
*methodologically* on a real substitute dataset (GEFCom2014 price track) in
`../scripts/benchmark_regression.py` / `../results/aggregate/benchmark_regression.csv`, because
the paper's own target series is a synthetic, unpublished-parameter construction (see
SOURCE_PAPER_AUDIT.md, "Data-substitution decision"). This benchmark run is also
`SOURCE_METHOD_REPRODUCTION`'s entry in the mandatory 10-benchmark comparison (Sec. 14 of the
task spec): the best of {SVR, RF, GradientBoosting, XGBoost, LightGBM} on our chronological
test split stands in as the source method's benchmark result throughout the rest of this study.
