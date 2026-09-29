# SOURCE_PAPER_AUDIT.md

## Citation
Md. Mainul Islam, Abdulrahman Takiddin, Muhammad Ismail, Hasan Kurban, Erchin Serpedin,
**"Linear-Complexity Unified Defense Against Deception Attacks in Distributed Economic Dispatch
Using Cryptography and Machine Learning,"** *IEEE Transactions on Smart Grid*, accepted 2026,
DOI: 10.1109/TSG.2026.3689997 (author's accepted version, 13 pp.).

## Target problem
Secure **Distributed Economic Dispatch (DED)** — the iterative, Lagrangian-multiplier-based
distributed solution of the economic dispatch (ED) optimal power flow problem — against
**deception attacks** (False Data Injection, Replay, Byzantine manipulation) that corrupt the
broadcast marginal-price signal λ while preserving apparent convergence, plus an eavesdropping
privacy threat against generator cost-coefficient confidentiality.

## Claimed novelty
1. A **unified security layer** for DED built on a modified **threshold Schnorr signature**
   scheme (Pedersen DKG) that gives integrity, freshness (session/iteration binding), and
   Byzantine-fault-tolerant consistency with **O(n) per-iteration communication**, vs. O(n²) for
   existing peer-to-peer cross-verification schemes.
2. **Lightweight pairwise additive masking** (ECDH-derived, PRF-expanded, zero-sum) for
   generator-output/cost-coefficient privacy without the overhead of homomorphic encryption.
3. An **ML-based post-attack recovery mechanism**: once an attack is cryptographically detected,
   affected suppliers fall back to a regression model (trained offline on historical demand/price
   data) that estimates the marginal price and drives the local KKT dispatch update, limiting
   welfare loss until secure coordination resumes.
4. Formal security proofs (data integrity/freshness, consistency/fault tolerance under Byzantine
   coordinator + up to ⌊(n−1)/3⌋ colluding suppliers via Pedersen VSS, collusion resistance,
   generator cost-coefficient privacy) reducing to EUF-CMA / ECDLP hardness.

## Datasets / test systems
- **IEEE 14-bus** system, 5 generators (owned by 5 independent suppliers), quadratic costs
  Cᵢ(Pᵢ)=aᵢPᵢ²+bᵢPᵢ+cᵢ with **explicit published coefficients**: (a,b,c) =
  {(0.01,32,·),(0.05,31,·),(0.04,33,·),(0.01,34,·),(0.02,30,·)} for G1–G5, capacities
  G1∈[0,332.4] MW, G2∈[0,140] MW, G3–G5∈[0,100] MW. c-terms not numerically given (only used as
  constant offset, irrelevant to λ/P dynamics) — **partially specified**.
- **IEEE 118-bus** system, 54 generators, grouped into 4 fuel types (coal/gas/nuclear/wind) with
  "default" cost coefficients per type (values not published — **missing**).
- **Demand series**: real hourly PJM system load 2024–2025 from PJM Data Miner
  (`dataminer2.pjm.com/feed/hrl_load_metered`), rescaled to the IEEE 118-bus load level. Train =
  Jan–Dec 2024 (8,783 hourly rows), test = Jan–Sep 2025 (6,551 rows).
- **Price/fuel-cost series**: historical coal/natural-gas/uranium commodity prices from
  Business Insider commodity pages, min–max normalized to [0,1] (Fig. 2), used to construct
  **time-varying linear cost coefficients** per fuel type (quadratic term "fixed" — exact
  fixed value not published — **missing**). "Ground-truth" hourly clearing prices are then
  obtained by **re-solving ED** with these synthetic time-varying costs — i.e., the ML
  supervision target is itself a heuristic construction, not an observed market price.
- No public repository/code link is given for any of the above pipelines.

## Preprocessing / feature engineering
- ML predictor inputs: **x_D^his** (current + historical demand) and **x_λ^his** (price-derived
  historical features); exact feature list, lag structure, and window length are **not
  specified** (partially specified — dimensionality "R^{Fλ}" is symbolic only).
- Demand and price both reported on raw MW / $/MWh scales in Table III (RMSE≈0.33, MAE≈0.25 on
  what appears to be a normalized/scaled target — normalization method for the regression target
  itself is **not stated**, an internal inconsistency: Fig. 3/4 axes show price in $/MWh (30–46)
  while Table III RMSE of 0.33 is inconsistent with that scale unless the target was normalized).
  → flagged as **partially specified / ambiguous**.

## Simulation framework
Custom Python-style Lagrangian dual-ascent solver implementing Algorithm 1 directly from the KKT
projection formula (Eq. 9) and dual update (Eq. 11); **no named framework (MATPOWER/PandaPower)
is cited** — the ED solver is bespoke. Convergence criterion Eq. 10, fixed step size α (value not
published — **missing**), max iterations K_max (not published — **missing**).

## Model architecture (crypto layer)
- secp256k1 ECC, Schnorr signatures (Eqs. 12–16), Pedersen DKG for T-of-n threshold secret
  sharing (Eqs. 17–18, T > ⌊2n/3⌋ required), pairwise ECDH+HMAC-SHA256-PRF masking on a sparse
  masking graph with degree ⌊(n−1)/3⌋+1 (Eqs. 19–22), per-iteration protocol (Eqs. 25–38, Fig. 1):
  masked output → Schnorr nonce commit → ECDSA-signed uplink → coordinator aggregation/dual
  update → ECDSA-signed downlink → partial Schnorr signatures → threshold aggregation → verified
  broadcast. 4 message rounds/iteration, 4n messages total (n suppliers).
- ML recovery model: tabular regressors compared — **SVR, Random Forest, Gradient Boosting,
  XGBoost, LightGBM** (Table III); LightGBM selected (best R²=0.9851). No hyperparameters,
  library versions, or tuning procedure are given (**missing**).

## Optimization method
Dual ascent (sub-gradient) on the power-balance constraint, Eq. 11, fixed step α. No line search,
no acceleration, no convergence-rate analysis beyond the empirical iteration counts in Tables II/IV.

## Training procedure / loss functions
ML model: standard supervised regression, loss function not stated (assume default per-library
loss — L2 for GBM variants, ε-insensitive for SVR — **assumed, not verified**). No
train/validation split is described within the "training" 2024 data (only train/test, no
model-selection holdout) — **partially specified**, and no early-stopping/regularization
settings given.

## Hyperparameters
None published for any ML model (tree depth, learning rate, n_estimators, SVR kernel/C/γ, etc. —
all **missing**). Threshold T, degree bound ⌊(n−1)/3⌋+1 are structurally given but no concrete n,
T values are used in the 14-bus/118-bus experiments (only "up to ⌊(n−1)/3⌋ malicious suppliers"
tolerance is asserted generically).

## Hardware / software environment
AMD Ryzen 7 4700U @ 2.0 GHz, 16 GB RAM (stated only for the ECSM/ECDSA latency benchmark, Sec.
VI-C). No software/library versions (Python, scikit-learn, XGBoost, LightGBM, crypto library)
are given anywhere — **missing**.

## Train / validation / test protocol
Two-way split only: 2024 (train) vs. Jan–Sep 2025 (test) for the ML regressor; **no validation
set**, **no reported random seed**, **single run per model** (no repetition/statistics) in
Table III. IEEE 14-bus attack case studies (Table II) are single deterministic runs (no
stochastic attacker policy repeated over seeds).

## Baselines
SVR, Random Forest, Gradient Boosting, XGBoost, LightGBM (regression only); crypto layer compared
qualitatively against 9 prior DED security schemes (Table I: FDI/replay/Byzantine coverage,
privacy, integrity, consistency, coordinator need, complexity) — no re-implementation/quantitative
head-to-head of those 9 schemes, only complexity-class comparison.

## Metrics
RMSE, MAE, R² (regression, Table III); marginal price ($/MWh), iterations-to-converge, total
system cost ($/h) (Tables II, IV); per-iteration message count (Fig. 6); ECSM/ECDSA/verify
latency in ms (Sec. VI-C). **No statistical/uncertainty reporting** (no CI, no variance, no
significance test) anywhere in the paper.

## Figures / Tables
Table I (qualitative scheme comparison), Table II (14-bus dispatch under 6 conditions), Table III
(5-model regression benchmark), Table IV (118-bus dispatch, with/without security, 6 conditions);
Fig. 1 (protocol workflow), Fig. 2 (normalized fuel prices), Fig. 3–4 (train/test demand-price
series + scatter/correlation), Fig. 5 (cost-convergence curves), Fig. 6 (message-count scalability).

## Statistical tests
None. No seeds, no CIs, no hypothesis tests reported anywhere.

## Ablations
None. No component-removal study for the crypto layer or the ML recovery layer.

## Robustness tests
Only discrete attack-type/attack-target case studies (Table II/IV conditions); no continuous
sweep of attack strength τ, no noise-injection test, no missing-data test, no cross-system
(14-bus → 118-bus) transfer test of the *same* ML model, no adversarial-perturbation study of
the regressor itself.

## Runtime analysis
Sec. VI-C: ECSM 0.49 ms, ECDSA sign 0.50 ms, ECDSA verify 0.45 ms (measured on stated hardware);
analytical per-iteration latency formula (4n+5)t_sm + 2(n+1)×120/B; Fig. 6 message-count scaling
O(n) vs O(n²) vs O(n(n−1)). No wall-clock training/inference time is reported for the ML
regressors (Table III has no timing column) — **missing**.

## Limitations (author-stated, Sec. VII)
Coordinator is a **single point of failure** (acknowledged explicitly); scheme targets *deception*
attacks specifically, not denial-of-service/disruption; standby-coordinator failover is proposed
as future work, not implemented; extension to security-constrained unit commitment (binary
on/off) left as future work.

---

## Classification of information

### Fully specified (directly reproducible)
- ED problem formulation, KKT projection (Eq. 9), dual update (Eq. 11), convergence test (Eq. 10).
- IEEE 14-bus generator count, capacity limits, and (a,b) cost coefficients.
- Full cryptographic protocol equations (Schnorr, Pedersen DKG, ECDH+PRF masking, threshold
  aggregation) — secp256k1, HMAC-SHA256 PRF are named explicitly.
- Table I attack-condition list and qualitative complexity classes (O(n) vs O(n²)).
- High-level ML benchmark model list (SVR/RF/GB/XGBoost/LightGBM) and which one was selected.
- Table II/IV attack-scenario *definitions* (which suppliers/units are targeted, attack type).

### Partially specified (reasonable assumptions required)
- ML feature set (x_D^his, x_λ^his): we assume lagged demand + lagged price + calendar features
  (hour-of-day, day-of-week) as a standard, defensible construction.
- 118-bus fuel-type cost coefficients: we assume representative merit-order values (wind cheapest
  fixed cost, nuclear/coal mid, gas priciest quadratic term) consistent with the paper's
  qualitative description ("largest coefficients → gas," "smallest → wind").
- Regression target normalization in Table III: we report both raw-scale and min–max-normalized
  metrics to make our numbers comparable regardless of which convention the original used.
- Train/validation split inside the "training" period: we carve an internal chronological
  validation slice out of the training era (not specified in the paper) for model selection,
  since the paper appears to have tuned/selected LightGBM using only train+test.
- 14-bus c-coefficients (constant cost offset): irrelevant to λ, P, or total *marginal* cost
  ranking, but needed for the absolute "$/h" total-cost column; we fit c-terms so that the
  Normal-condition price ($34.19/MWh) and total cost ($8456.25/h) reported in Table II are
  matched as closely as possible, then hold them fixed for all other conditions.

### Missing (cannot be inferred reliably — used only for a transparent substitute)
- Exact PJM-to-118-bus demand rescaling factor; exact commodity-price-to-cost-coefficient mapping
  and which "fixed" quadratic term was used per fuel type; ML hyperparameters and library
  versions for all 5 regression baselines; fixed step size α and K_max for the dual solver;
  random seeds; original PJM/commodity data files (not distributed with the paper). **We do not
  attempt to fabricate these** — instead, Sec. "Data Substitution" below documents a real,
  independently-sourced dataset (GEFCom2014 electricity-price track) used for the ML-recovery
  reproduction and all downstream candidate/benchmark experiments, and we reproduce the DED
  economic-impact simulation (Table II/IV-style experiments) on the paper's own explicitly-given
  14-bus coefficients, which requires no missing information.

## Data-substitution decision (documented per Sec. 2/12 of task spec)
The paper's own "ground-truth" price series for ML training is **not an observed market price**
but a synthetic construction from normalized commodity indices via unpublished coefficients — it
is not more "authentic" than a well-established public alternative, and it cannot be exactly
reproduced (missing scaling constants). We therefore substitute, for the ML-recovery track only:

- **GEFCom2014 Electricity Price Forecasting track** (Hong et al., *Int. J. Forecasting* 2016),
  found at `C:\Users\MMASSAOUDI\Desktop\Data\Load Data\GEFCom2014_Dataset\...\Price\Task 15\`:
  real hourly **zonal load, system load, and zonal LMP** for one PJM-style zone, 2011-01-01 to
  2013-12-17 (25,968 hourly rows, no gaps in the feature columns). This is a genuine market price
  series (unlike the source paper's own synthetic target), a standard peer-reviewed forecasting
  benchmark, and directly analogous in structure (demand → marginal price) to the paper's task.
- The 14-bus/118-bus **DED economic-impact simulation** (Tables II/IV, Fig. 5–6) is reproduced
  on the paper's **own published coefficients** — no substitution needed there.
- This decision is made *before* any candidate-model results are seen (Sec. 12 compliance).
