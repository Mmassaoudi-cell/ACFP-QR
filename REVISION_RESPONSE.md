# Response to Reviewers

Manuscript: "Attack-Type-Conditioned, Feasibility-Projected Quantile Recovery for
Cryptographically Secured Distributed Economic Dispatch" (ACFP-QR)
Decision on prior round: Major Revision (`PEER_REVIEW_DECISION.md`)

We thank the Editor and reviewers for a thorough, constructive review. Below we respond to each
required and suggested revision. Several items led us to genuinely improve the model (not only
its presentation): the gate now uses per-attack-type capacity and label-noise-robust training,
the quantile heads are conformally calibrated, and we added a backbone-generalization check and a
network-level (14-bus, live iterative-dispatch) validation. All changes were made on
TRAIN/VALIDATION only, per our stated protocol; TEST was re-evaluated exactly once after
re-freezing `FINAL_MODEL_CONFIG.yaml`.

## Priority 1 — Required Revisions

**R1. Efficiency-loss significance testing + rebalanced framing.**
The same paired-Wilcoxon, step-down-Holm-corrected protocol used for RMSE is now applied to
economic efficiency loss (`scripts/stats_tests.py`, `scripts/build_wtl.py`). Result: 8 wins / 3
ties / 0 losses against the 11 independent benchmarks (`BENCHMARK_WTL.csv`, manuscript Table II).
Both metrics are now reported with equal prominence throughout the manuscript (abstract,
Sec. VI-A, VI-G).

**R2. TCN near-tie / causal-attribution check.**
We attached the identical gate mechanism to a TCN backbone (`scripts/backbone_generalization.py`,
manuscript Sec. VI-C, Table IV, Fig. 4). The gate improves both CatBoost (−5.7% RMSE) and TCN
(−1.8% RMSE) independently, confirming the mechanism generalizes across backbones. Separately,
the enhanced gate (fixing the Byzantine underfit) reduced overall RMSE enough that TCN's
comparison is now a statistically significant win for ACFP-QR (p=0.014), not a tie.

**R3. Network-level / iterative-loop validation.**
Added (`scripts/network_validation.py`, manuscript Sec. V-B, VI-D, Table V, Fig. 5): ACFP-QR is
trained on a historically-driven 14-bus demand/price series (real GEFCom2014 load shape, exact
self-consistent clearing price) and used to drive live dual-ascent recovery under FDI/replay/
Byzantine attacks over 300 held-out hours. Left unaddressed, attacks inflate dispatch cost
2.1-4.2%; recovery restores dispatch to within 4×10⁻⁴% of the true optimum. We did not add a
result claiming ACFP-QR beats the baseline at the network-cost level specifically — the
coordinator's own iterative feedback compensates for small price-accuracy differences at that
resolution, which we report as found rather than reframe to fit a stronger claim.

**R4. Gate-misclassification stress test.**
Added to the frozen-model robustness suite (`scripts/robustness.py`, `scripts/gate_robustness.py`,
manuscript Sec. VI-H, Fig. 6): under a simulated 0-40% attack-type-flag misclassification rate,
ACFP-QR degrades gracefully (RMSE 6.40→7.09) and remains below the attack-agnostic baseline
(7.20, constant) at every tested rate. This is enabled by a new label-noise-robust training
regime for the gate correctors, validated on VALIDATION before being adopted
(`scripts/gate_robustness.py`).

**R5. Byzantine VAL/TEST reversal.**
Root-caused and fixed rather than merely explained: the shared shallow gate configuration
underfit Byzantine's wider multi-lag contamination pattern relative to FDI/replay's narrower
signatures. A per-type capacity increase with a Huber loss for the Byzantine corrector
specifically (`scripts/enhance_gate.py`, selected via a 4-configuration validation-only
comparison) improves Byzantine RMSE by a further 3.4 points relative to the prior gate and
carries no cost to FDI/replay. The manuscript's Sec. IV describes this design choice and its
rationale directly; the residual efficiency-loss nuance for replay/Byzantine (Sec. VI-G) is
retained and discussed, with a cost-aware training objective identified as the fix (Sec. VII).

**R6. Latency-characterization inconsistency.**
Corrected: the manuscript now states the CatBoost/LightGBM comparison as "40-90× slower in
absolute terms" rather than "a small constant factor" (Sec. VI-I), consistent with the disclosed
magnitude.

## Priority 2 — Suggested Revisions

**S1. Statistical machinery.** Rewrote `scripts/stats_tests.py`: proper step-down Holm-Bonferroni
(previously an incorrect independent-threshold check per rank), a matched-pairs rank-biserial
effect size computed from Wilcoxon signed ranks (previously a degenerate win/loss sign-count
approximation), and a bootstrap 95% CI on the paired RMSE/efficiency-loss mean difference
(manuscript Table II).

**S2. Literature expansion.** Added the five specific references identified by the domain
reviewer: Zhang & Chow (2012), Kar & Hug (2012), Boyd et al. (2011, ADMM), Chu et al. (2023,
load-altering-attack mitigation), and Wood, Wollenberg & Sheblé (classical ED textbook, cited to
correctly calibrate the feasibility-projection novelty claim). Related Work (Sec. II) now opens
with a dedicated paragraph on the DED literature.

**S3. "Distributed" terminology.** Sec. II now explicitly disambiguates the coordinator-based
architecture studied here from fully decentralized consensus/consensus+innovations DED, with
citations, and states this scope choice plainly rather than leaving it implicit.

**S4. Out-of-taxonomy attack test.** Added (`scripts/attack_contamination.py::contaminate_unknown_shape`,
manuscript Sec. VI-H): an oscillating multi-lag contamination shape never used in gate training.
ACFP-QR (RMSE 5.719) is statistically indistinguishable from the baseline (5.723) — the mechanism
neither helps nor meaningfully hurts outside its declared taxonomy.

**S5. Investigate Byzantine-regression cause.** Investigated directly (`scripts/enhance_gate.py`):
underfitting from shared shallow gate hyperparameters, not the Optuna aggregate objective, was
found to be the dominant cause, and was fixed at the source (see R5 above) rather than only
described.

**S6. Quantile-calibration claim.** Fixed at the source via split-conformal calibration (CQR;
Romano et al. 2019) rather than merely caveated: coverage improves from 62.0% (raw) to 79.8%
(calibrated) against an 80% nominal target (manuscript Sec. IV-C, VI-F). The "calibrated
uncertainty" contribution claim (Introduction, item 3) is now fully supported.

**S7. VAL-split distribution shift.** Noted; VAL's mean price (53.73) differs from both TRAIN
(47.43) and TEST (46.05), and the conformal calibration step (fit on a held-out slice of
TRAIN+VAL) is one direct mitigation for the coverage consequence of this shift, now empirically
confirmed effective.

**S8. Per-branch operational recommendation.** The mechanically separable ablation (Table VI)
already supports disabling any one branch independently; given the v2 fixes substantially
improve the Byzantine branch specifically, we did not add a "disable Byzantine" recommendation to
the final manuscript, since the underlying accuracy issue was fixed rather than worked around.
The residual efficiency-loss nuance is instead flagged as a limitation with a concrete next step
(cost-aware objective) rather than a operational workaround.

**S9. Soften source-method-comparator claim.** Manuscript Sec. VI-A now states the cross-split
boosting-variant instability explicitly (LightGBM in the source paper, XGBoost on our validation
split, GradientBoosting on our test split) and notes this is a limitation item (Sec. VII, item 6)
that the gated-CatBoost result is robust to regardless, given the backbone-generalization check.

## Priority 3 — Text and Formatting
All items addressed: table caption grammar, consistent benchmark-count reporting, consistent
latency units, and the missing-signal/operator-interface discussion folded into Sec. VII
(Limitations) rather than left as an isolated aside.

## Author information and presentation
Author block updated per the corresponding author's instruction. The manuscript now reads as an
independent, confident scientific report: results are stated as findings, and every remaining
scope boundary is consolidated into a single Limitations section (Sec. VII) rather than
interleaved through the Results discussion.
