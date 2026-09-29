# Editorial Decision Package
**Manuscript**: "Attack-Type-Conditioned, Feasibility-Projected Quantile Recovery for Cryptographically Secured Distributed Economic Dispatch" (ACFP-QR)
**Simulated venue**: IEEE Transactions on Smart Grid (per field-analyst configuration)
**Review mode**: `academic-paper-reviewer` full mode — EIC + 3 Peer Reviewers + Devil's Advocate, run as 5 independent, isolated reviews (no cross-referencing)

---

## Part 1: Editorial Decision Letter

Dear Author(s),

Thank you for submitting your manuscript to *IEEE Transactions on Smart Grid*. Your manuscript has been reviewed by 4 independent reviewers (Editor-in-Chief, and 3 Peer Reviewers covering methodology, domain expertise, and cross-disciplinary/practical perspective), plus a Devil's Advocate stress-test reviewer.

### Decision: **Major Revision**

### Consensus Analysis

#### Points of Agreement (Consensus)

- **[CONSENSUS-4]** All four primary reviewers (EIC, R1-Methodology, R2-Domain, R3-Perspective) independently recommend **Major Revision** — no reviewer recommends Accept, Minor Revision, or Reject.
- **[CONSENSUS-3]** The paper's title/abstract/Contributions framing ("...for Cryptographically Secured Distributed Economic Dispatch") overreaches the actually-validated scope (a single-zone price-regression benchmark plus a topology-free synthetic generator fleet, with no network-level evaluation of ACFP-QR itself). Raised independently by **EIC** (Weaknesses 1–2, 5), **R2-Domain** (Weaknesses 1–2, Confidence 5), and **R1-Methodology** (Weakness 5, the dangling 14-bus reference that promises results never delivered in Sec. V). R3 did not weigh in on this axis (silent, not dissenting).
- **[CONSENSUS-3]** The Byzantine-attack finding is under-examined and internally inconsistent across the paper's own supporting record. Raised independently by **R1-Methodology** (Weakness 4: unexplained VAL-screening-vs-TEST reversal), **R3-Perspective** (Weakness 3: the operational stakes of shipping a mechanism that is *worse* than baseline on the metric that matters for exactly one of three defended attack types), and reinforced by the **Devil's Advocate** (CRITICAL #3 and #5, MAJOR #7) from a third angle (undisclosed contradiction with the authors' own development record; plausible selection-objective artifact never investigated).

#### Points of Disagreement
None identified. All four primary reviewers' findings are complementary rather than contradictory — each approaches a shared set of underlying concerns (validation scope, statistical completeness, the Byzantine anomaly) from a distinct, non-overlapping angle, exactly as the review-panel design intends. No arbitration between opposing recommendations was required.

### Devil's Advocate Critical Findings (tracked independently — do not count toward CONSENSUS-4/3 above)

Per the panel's IRON RULE, an Editorial Decision of Accept is not possible while CRITICAL findings stand, regardless of the primary panel's recommendation. The DA identified **5 CRITICAL** issues. All are addressed below with corroboration status and required author response.

| # | DA Finding | Corroborated by | EIC/Editor Assessment | Required Author Response |
|---|---|---|---|---|
| 1 | Quantile under-coverage (62% vs. 80%) is attributed in-text to "distribution shift," but the quantile heads are trained only on clean data (Sec. IV-C) while evaluated on a ~20%-contaminated TEST set — a more direct, self-inflicted cause the paper never considers. | Not directly, but R1 independently flags a related, distinct contributor (VAL-split distribution shift affecting hyperparameter selection). | **Valid and specific.** The mechanistic explanation (train/eval regime mismatch for the quantile heads specifically) is more parsimonious than "distribution shift" and is fully checkable from the authors' own Sec. IV-C description. This is not a rebuttable stylistic quibble; it is a logic-chain gap. | Must directly address: either retrain/re-evaluate quantile heads with attack-regime exposure, or revise the explanation in Sec. V.D to name train/eval mismatch as (at least) a contributing cause alongside distribution shift. |
| 2 | TCN — with no access to the attack-type flag $g$ — statistically ties ACFP-QR ($p=0.32$) and wins 4/10 seeds, undermining the core causal claim that gating on $g$ drives the reported gain. | No independent corroboration from the primary panel, though EIC's Q3 (backbone-choice instability) is adjacent. | **Valid and important — the single most consequential DA finding.** This does not invalidate the paper's contribution, but it does mean the current framing (10 wins including a comparator that has no access to the paper's central mechanism) overstates how much of the gain is attributable to attack-type conditioning specifically, versus plain model capacity. | Must directly address: either demonstrate the gate mechanism attached to a TCN-strength backbone still yields a significant gain over that backbone alone, or explicitly temper claims that attribute the overall margin to the attack-type-gating mechanism per se. |
| 3 | The Byzantine RMSE result reverses sign between the authors' own VAL-screening record (worse than baseline) and the final TEST result (better than baseline), undisclosed in the manuscript body. | Yes — R1 (Weakness 4) and R3 (Weakness 3) independently converge on the Byzantine finding being unstable/under-examined, from different angles. | **Valid.** This is precisely the kind of discrepancy the paper's own stated ethic ("negative results retained... not hidden") commits it to disclosing. Its absence from the manuscript body (present only in a supplementary file) is a real gap, not a fatal flaw. | Must directly address: add a sentence reconciling the VAL-vs-TEST reversal, with a specific hypothesis (Optuna-objective effect vs. small-sample corrector variance) and, ideally, a supporting check. |
| 4 | Sec. V.F characterizes a disclosed 40–80× inference-latency overhead as "a small constant factor," which contradicts the authors' own FINAL_RESEARCH_SUMMARY.md characterization of the same number. | No independent corroboration, though EIC's Minor Issues flag latency-reporting friction adjacent to this point. | **Valid, and an easy fix.** This is an internal-consistency defect, not a fundamental flaw — the underlying number is disclosed and honest; only the rhetorical framing in one location is inconsistent with the authors' own framing elsewhere. | Must directly address: correct the in-text characterization to match the disclosed magnitude consistently in all locations. |
| 5 | The headline "10 wins, $p<0.01$" claim is built entirely on RMSE, while the paper's own declared co-primary economic-efficiency-loss metric is not uniformly improved (worse for Byzantine, flat in aggregate) and is never given the same significance-testing treatment. | Yes — directly and independently corroborated by **R1-Methodology** (Weakness 1: efficiency loss is a declared co-primary metric per `FINAL_MODEL_CONFIG.yaml` but receives no paired/Holm testing) and by **R3-Perspective** (Weakness 3, from the deployability angle: the metric that matters operationally is not what the headline claim is built on). | **Valid, and the highest-priority item in this decision.** Three independent reviewers (DA + R1 + R3), from three different methodological angles, converged on the same underlying gap without seeing each other's reports. This level of independent triangulation should be treated as decisive, not as three redundant restatements of the same shallow observation. | Must directly address: run the same statistical protocol on efficiency loss and report it honestly (Priority 1, Item 3 below), and rebalance the abstract/headline framing accordingly. |

**Editorial note on DA findings 6–9 (MAJOR) and 10–12 (MINOR)**: these are valid, specific, and actionable but do not independently block a path to acceptance; they are folded into the Revision Roadmap below (Priority 2 and 3) rather than itemized here.

### Decision Rationale
All four primary reviewers, working independently and without seeing each other's reports, converged on Major Revision. Confidence scores are consistently high (4, 4, 5, 4), indicating the panel is not simply uncertain — this is a considered judgment from reviewers who found real merit in the work (all four explicitly praised the reproduction discipline, the statistical protocol relative to the source paper, and the paper's unusual candor about its own limitations) alongside real, specific, fixable gaps. The Devil's Advocate's 5 CRITICAL findings independently preclude an Accept or Minor Revision decision under this panel's operating rules, and three of those five findings (the efficiency-loss significance gap, the Byzantine reversal, and by extension the general question of whether disclosed limitations are engaged with rather than merely mentioned) are directly corroborated by at least one primary reviewer from an unrelated angle — this is convergent, not redundant, evidence. Critically, none of the five CRITICAL findings are "foundation collapse" in the sense of invalidating the paper's core contribution outright (the DA itself notes the feasibility-projection component "is essentially immune to every critique" raised); they instead identify specific places where the manuscript's rhetorical framing outruns what its own evidence supports, or where an experiment the authors clearly have the infrastructure to run (per their own scripts/CSVs) has not yet been run or reported. This is a Major Revision, not a Reject, profile.

### Summary of Key Issues (ranked)
1. **[DA-CRITICAL #5 + R1 + R3]** Efficiency-loss (the paper's own co-primary, DED-relevant metric) is not given the same statistical rigor or rhetorical weight as RMSE, and the headline "wins" claim is RMSE-only.
2. **[DA-CRITICAL #2]** The strongest independent comparator (TCN) has no access to the paper's central mechanism (the attack-type flag) yet statistically ties ACFP-QR — the causal attribution of the gain to attack-type-gating specifically is not yet secured.
3. **[CONSENSUS-3: EIC, R2, R1]** Title/abstract/Contributions overreach relative to validated scope — no network-level (multi-bus, iterative) evaluation of ACFP-QR itself exists in the paper.
4. **[DA-CRITICAL #3, #1 + R1, R3]** The Byzantine-attack result is unstable/reversed across the authors' own development record and under-examined for cause; quantile under-coverage has a more direct explanation than the one given.
5. **[R2, Confidence 5]** Novelty of the feasibility projection is overstated relative to classical DED theory; literature review is thin for the field-level claims made.
6. **[R3 + DA-CRITICAL #4]** No stress-test of the central "attack-type flag is always correct" assumption; an internal contradiction in how the latency overhead is characterized.

---

## Part 2: Revision Roadmap

### Required Revisions (Must Fix)

| # | Revision Item | Source | Priority | Estimated Effort |
|---|---|---|---|---|
| R1 | Run the same paired Wilcoxon + Holm-corrected significance testing on economic efficiency loss (the declared co-primary metric) and report the resulting win/tie/loss table honestly; rebalance abstract/headline framing so it is not RMSE-only. | DA-CRITICAL #5, R1-W1, R3-W3 | P1 | 3–5 days |
| R2 | Address the TCN near-tie's threat to the core causal claim: either attach the attack-type gate to a TCN-strength backbone and show the gain persists, or explicitly temper "attributed to attack-type conditioning" claims. | DA-CRITICAL #2 | P1 | 3–5 days |
| R3 | Resolve the title/scope mismatch: either add a genuine network-level (multi-bus, iterative-DED-loop) evaluation of ACFP-QR itself, or narrow title/abstract/Contributions to a price-recovery-fallback scope and remove/reword the dangling "network-level attack experiments" sentence in Sec. IV-A. | EIC-W1/W2, R2-W1/W2, R1-W5 | P1 | 1–2 weeks (if adding experiment) or 1 day (if rescoping) |
| R4 | Add a gate-misclassification / attack-type-flag-reliability stress test (analogous to the existing attack-strength/contamination-fraction sweeps): flip $g$ to an incorrect branch at a declared rate and report RMSE and efficiency-loss under misclassification. | DA Unexamined Premise, R3-W1 | P1 | 3–5 days |
| R5 | Reconcile and explicitly discuss the Byzantine-attack RMSE reversal between VAL-screening (worse than baseline) and final TEST (better than baseline); propose and, if feasible, test a specific hypothesis (Optuna-objective effect vs. small-sample corrector variance). | DA-CRITICAL #3, R1-W4, R3-W3 | P1 | 2–3 days |
| R6 | Fix the "40–80× slower... small constant factor" internal contradiction between Sec. V.F and the authors' own supplementary characterization; report latency consistently and with unambiguous units everywhere. | DA-CRITICAL #4, DA-MINOR #11 | P1 | <1 day |

### Suggested Revisions (Should Fix)

| # | Revision Item | Source | Priority | Estimated Effort |
|---|---|---|---|---|
| S1 | Fix the Holm-Bonferroni step-down implementation (or justify the simplification); clarify the exact "family" the correction is computed over (11 vs. 14 comparators); replace/supplement the sign-count "effect size" with a true signed-rank rank-biserial correlation and report bootstrap CIs on RMSE mean-differences given the n=10 p-value floor. | R1-W2/W3 | P2 | 2–3 days |
| S2 | Expand literature review: add consensus/ADMM-based DED foundations (Zhang & Chow 2012; Kar & Hug 2012; Boyd et al. 2011) and prior attack-on-ED defense work (Chu et al. 2023); cite the classical origin (Wood, Wollenberg & Sheblé) of the water-filling projection and soften the novelty comparison to Ajeyemi et al.'s harder AC-OPF setting. | R2-W3/W4 (specific references provided in R2's report) | P2 | 2–3 days |
| S3 | Disambiguate "distributed economic dispatch" from the coordinator-based architecture actually studied, at first use in the Introduction. | R2-W5 | P2 | <1 day |
| S4 | Add a robustness test using an attack signature outside the three declared taxonomy shapes ("unknown-shape" contamination), since the gate/correctors were designed around exactly the shapes tested. | DA-MAJOR #6 | P2 | 2–3 days |
| S5 | Investigate whether the Optuna aggregate-RMSE tuning objective is the mechanism behind the Byzantine efficiency-loss regression, as a more specific/testable explanation than the current generic framing. | DA-MAJOR #7 | P2 | 1–2 days |
| S6 | Revise/caveat Contribution #3 ("calibrated uncertainty") given the disclosed 62%-vs-80% coverage gap; consider retraining quantile heads with attack-regime exposure or explicitly scoping the claim to clean-data calibration only. | DA-CRITICAL #1, DA-MAJOR #8 | P2 | 2–4 days |
| S7 | Note the VAL split's own distributional shift (higher mean price than TRAIN or TEST) as a possible contributor to hyperparameter-selection choices and downstream under-coverage. | R1 (Sampling Strategy) | P2 | <1 day |
| S8 | Add an explicit per-branch operational recommendation (e.g., hold back the Byzantine-branch corrector pending a cost-aware objective; enable FDI/replay corrections) given the ablation already shows the branches are mechanically separable at zero extra cost. | R3-W3 | P2 | <1 day |
| S9 | Soften the "best-of-5 stands in for the source paper" comparison given demonstrated cross-split instability in which boosting model wins (LightGBM/source paper vs. XGBoost/this paper's VAL vs. GradientBoosting/this paper's TEST). | DA-MAJOR #9, EIC-Q3 | P2 | <1 day |

### Revision Checklist

#### Priority 1 — Structural Revisions (Estimated total effort: ~2.5–3.5 weeks if the network-level experiment is added; ~1.5 weeks if title/scope is narrowed instead)
- [ ] R1: Efficiency-loss significance testing + rebalanced headline framing
- [ ] R2: TCN-comparator causal-attribution check or claim tempering
- [ ] R3: Network-level experiment OR title/scope narrowing + remove dangling reference
- [ ] R4: Gate-misclassification robustness sweep
- [ ] R5: Byzantine VAL/TEST reversal reconciliation
- [ ] R6: Fix latency-characterization internal contradiction

#### Priority 2 — Content Supplementation (Estimated total effort: ~1.5–2 weeks)
- [ ] S1: Statistical-machinery corrections (Holm step-down, effect size, CIs)
- [ ] S2: Literature review expansion (specific references provided)
- [ ] S3: "Distributed" terminology disambiguation
- [ ] S4: Out-of-taxonomy attack-signature robustness test
- [ ] S5: Investigate Optuna-objective explanation for Byzantine regression
- [ ] S6: Revise/caveat the "calibrated uncertainty" contribution claim
- [ ] S7: Note VAL-split distribution shift
- [ ] S8: Add per-branch operational recommendation
- [ ] S9: Soften source-method-comparator claim

#### Priority 3 — Text and Formatting (Estimated total effort: ~2–3 days)
- [ ] "1 tie" grammar fix in table caption
- [ ] Consistent benchmark-count reporting (≥10 / 11 / 15) across abstract/methods/results
- [ ] Consistent units on all latency figures
- [ ] Surface existing CI95 columns for top competitive models (TCN, MLP, Transformer)
- [ ] Compact table for the abstract's dense per-attack numeric clauses
- [ ] Add gate-misclassification risk and operator-interface/override design to explicit Limitations list
- [ ] Reconsider the "mixture-of-experts"-adjacent framing of $h_g$ in Related Work (it is a hard-gated additive residual, not a learned soft-routed MoE)

### Revision Deadline
Recommended 6–8 weeks (Major Revision).

### Response Letter Template
Authors should structure their response using a Reviewer-comment → Author-response → Change-location (R→A→C) format, addressing every item above individually, including the DA-CRITICAL items (required even where the authors believe the EIC's assessment should be contested — per panel rules, DA-CRITICAL items must be acknowledged in the author response regardless of whether the authors ultimately agree).

---

## Part 3: Reviewer Report Summary (Appendix)

### EIC Report Summary
- Recommendation: Major Revision | Confidence: 4
- Key Point: A well-executed, honestly-reported contribution to the DED recovery sub-problem whose title/framing currently claims more systems-level (network/dispatch) validation than the experiments deliver.

### Reviewer 1 (Methodology) Summary
- Recommendation: Major Revision | Confidence: 4
- Key Point: Leakage-controlled protocol and genuinely validation-only model selection are real strengths, but statistical rigor was applied asymmetrically (RMSE only, not the declared co-primary efficiency-loss metric), and a Byzantine-result discrepancy across the authors' own documents is undisclosed.

### Reviewer 2 (Domain) Summary
- Recommendation: Major Revision | Confidence: 5
- Key Point: The mechanistic critique of the source paper is accurate and source-verified, but the paper's title/abstract oversell a "distributed economic dispatch" contribution that is validated only via a single-zone price benchmark and a topology-free synthetic fleet, and the feasibility-projection novelty is overstated relative to classical ED theory.

### Reviewer 3 (Perspective) Summary
- Recommendation: Major Revision | Confidence: 4
- Key Point: The entire mechanism's trust boundary rests on an unstress-tested assumption that the crypto layer's attack-type flag is always correct, and the disclosed Byzantine cost regression needs an explicit operational recommendation (e.g., disable that branch pending a cost-aware objective) rather than being left as a reported-but-unaddressed number.

### Devil's Advocate Summary
- 5 CRITICAL, 4 MAJOR, 3 MINOR issues, 1 Unexamined Premise identified.
- Key Point: The paper's headline statistical claims (RMSE-only significance, a comparator with no access to the core mechanism nearly matching it, an undisclosed Byzantine-result reversal) are each individually defensible as honest mistakes but collectively suggest the manuscript's final rhetorical packaging is less rigorous than the underlying research process — most of these gaps are already visible in the authors' own supplementary documentation and are addressable without new research, only more complete and consistent reporting.
