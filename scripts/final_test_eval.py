"""
Final, single, frozen-config TEST evaluation (Sec. 15). Run exactly once per reported number;
FINAL_MODEL_CONFIG.yaml is not edited after this point. All models are evaluated on the SAME
per-seed contaminated-TEST realization for a fair, paired comparison (Sec. 22 statistical tests).
Backbone/gate fit on TRAIN+VAL (chronological first-80%/last-20% split of TRAIN+VAL for
Candidate A's gate, mirroring the screening protocol exactly but on the combined fit pool).
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_gefcom import get_dataset
from attack_contamination import contaminate, ATTACK_NAMES
from models_regression import make_classical_models, make_deep_models
from candidates import (CandidateA_ACFP_QR, CandidateB_GBM_ResGRU, CandidateD_QRGBM, make_backbone,
                         water_fill, AttackGate, ENHANCED_GATE_PARAMS)
from economic_impact import _FLEET, implied_demand, efficiency_loss
from metrics import regression_metrics

DETERMINISTIC = {"LinearRegression", "Ridge", "kNN", "SVR"}
SEEDS = list(range(10))
BACKBONE_HP = dict(depth=9, learning_rate=0.022342748011385118, iterations=800)  # frozen, tuned Sec.9 Stage3
GATE_CFG = dict(per_type_params=ENHANCED_GATE_PARAMS, label_noise_rate=0.15)  # revised gate: per-type
# capacity (fixes the Byzantine underfit) + label-noise-robust training (misclassification defense),
# selected on VAL only via scripts/enhance_gate.py and scripts/gate_robustness.py


def feasibility_violation_pct(demand, outputs):
    return float(np.mean(np.abs(demand - outputs.sum(axis=1)) / np.clip(demand, 1e-3, None)) * 100)


def main():
    feat, train, val, test, cols = get_dataset()
    fit_pool = pd.concat([train, val], axis=0)
    n_bb = int(len(fit_pool) * 0.8)
    fit_bb, fit_gate = fit_pool.iloc[:n_bb], fit_pool.iloc[n_bb:]

    scaler = StandardScaler().fit(fit_bb[cols])
    Xbb, ybb = scaler.transform(fit_bb[cols]), fit_bb["price"].values
    y_test = test["price"].values

    rows = []
    for seed in SEEDS:
        rng_gate = np.random.RandomState(1000 + seed)
        gate_df, gate_labels = contaminate(fit_gate, cols, rng_gate, frac=0.20)
        Xgate, ygate = scaler.transform(gate_df[cols]), gate_df["price"].values

        rng_eval = np.random.RandomState(2000 + seed)
        test_df, test_labels = contaminate(test, cols, rng_eval, frac=0.20)
        Xtest_c = scaler.transform(test_df[cols])
        demand_test = implied_demand(y_test)

        # ---- classical / boosting benchmarks ----
        classical = make_classical_models(seed)
        for name, model in classical.items():
            if name in DETERMINISTIC and seed != SEEDS[0]:
                continue
            t0 = time.time()
            model.fit(Xbb, ybb)
            train_t = time.time() - t0
            t1 = time.time()
            pred = model.predict(Xtest_c)
            infer_t = time.time() - t1
            m = regression_metrics(y_test, pred)
            eff = efficiency_loss(y_test, pred)
            rows.append({"model": name, "family": "classical/boosting", "seed": seed,
                         "train_time_s": train_t, "infer_time_s": infer_t, **m, **eff})

        # ---- deep benchmarks ----
        deep = make_deep_models(seed, cols)
        n_val = max(1, int(0.1 * len(Xbb)))
        for name, model in deep.items():
            t0 = time.time()
            model.fit(Xbb[:-n_val], ybb[:-n_val], Xbb[-n_val:], ybb[-n_val:])
            train_t = time.time() - t0
            t1 = time.time()
            pred = model.predict(Xtest_c)
            infer_t = time.time() - t1
            m = regression_metrics(y_test, pred)
            eff = efficiency_loss(y_test, pred)
            n_params = sum(p.numel() for p in model.model.parameters())
            rows.append({"model": name, "family": "deep", "seed": seed,
                         "train_time_s": train_t, "infer_time_s": infer_t, "n_params": n_params, **m, **eff})

        # ---- source-method reproduction marker (best of the Table-III suite, computed post-hoc from rows) ----

        # ---- baseline attack-agnostic backbone (frozen HP, same as Candidate A's backbone) ----
        t0 = time.time()
        base = make_backbone(seed, **BACKBONE_HP)
        base.fit(Xbb, ybb)
        train_t = time.time() - t0
        t1 = time.time()
        pred_base = base.predict(Xtest_c)
        infer_t = time.time() - t1
        m = regression_metrics(y_test, pred_base)
        eff = efficiency_loss(y_test, pred_base)
        out_base_before = np.array([[g.dispatch(p) for g in _FLEET] for p in pred_base])
        rows.append({"model": "Baseline-attack-agnostic-GBM", "family": "ablation", "seed": seed,
                     "train_time_s": train_t, "infer_time_s": infer_t, **m, **eff,
                     "feasibility_violation_before_pct": feasibility_violation_pct(demand_test, out_base_before)})

        # ---- Candidate D (quantile only) ----
        t0 = time.time()
        candD = CandidateD_QRGBM(seed)
        candD.backbone = make_backbone(seed, **BACKBONE_HP)
        candD.fit(Xbb, ybb)
        train_t = time.time() - t0
        predD = candD.predict(Xtest_c)
        m = regression_metrics(y_test, predD)
        eff = efficiency_loss(y_test, predD)
        rows.append({"model": "D_QR-GBM", "family": "ablation", "seed": seed, "train_time_s": train_t,
                     "infer_time_s": np.nan, **m, **eff})

        # ---- Candidate B (GBM + residual GRU) ----
        t0 = time.time()
        candB = CandidateB_GBM_ResGRU(seed, cols)
        candB.backbone = make_backbone(seed, **BACKBONE_HP)
        candB.fit(Xbb, ybb, Xbb[-n_val:], ybb[-n_val:])
        train_t = time.time() - t0
        predB = candB.predict(Xtest_c)
        m = regression_metrics(y_test, predB)
        eff = efficiency_loss(y_test, predB)
        rows.append({"model": "B_GBM+Res-GRU", "family": "ablation", "seed": seed, "train_time_s": train_t,
                     "infer_time_s": np.nan, **m, **eff})

        # ---- Candidate A: FULL final model ----
        t0 = time.time()
        candA = CandidateA_ACFP_QR(seed)
        candA.backbone = make_backbone(seed, **BACKBONE_HP)
        candA.gate = AttackGate(seed, **GATE_CFG)
        candA.backbone.fit(Xbb, ybb)
        candA.quantiles.fit(Xbb, ybb)
        candA.quantiles.calibrate(Xgate, ygate)  # split-conformal correction, held-out contaminated slice
        resid = ygate - candA.backbone.predict(Xgate)
        candA.gate.fit(Xgate, resid, gate_labels)
        train_t = time.time() - t0

        t1 = time.time()
        predA, outA = candA.recover_dispatch(Xtest_c, demand_test, attack_label=test_labels, feasibility=True)
        _, outA_before = candA.recover_dispatch(Xtest_c, demand_test, attack_label=test_labels, feasibility=False)
        infer_t = time.time() - t1
        m = regression_metrics(y_test, predA)
        eff = efficiency_loss(y_test, predA)
        lo, hi = candA.quantiles.predict(Xtest_c)
        coverage = float(np.mean((y_test >= lo) & (y_test <= hi)) * 100)
        rows.append({"model": "ACFP-QR (proposed, FINAL)", "family": "proposed", "seed": seed,
                     "train_time_s": train_t, "infer_time_s": infer_t, **m, **eff,
                     "feasibility_violation_before_pct": feasibility_violation_pct(demand_test, outA_before),
                     "feasibility_violation_after_pct": feasibility_violation_pct(demand_test, outA),
                     "quantile_coverage_10_90_pct": coverage})

        # per-attack-type rows for the final model + baseline (paired diagnostic)
        for t, tname in ATTACK_NAMES.items():
            mask = test_labels == t
            if mask.sum() < 5:
                continue
            mA = regression_metrics(y_test[mask], predA[mask])
            mB = regression_metrics(y_test[mask], pred_base[mask])
            effA = efficiency_loss(y_test[mask], predA[mask])
            effB = efficiency_loss(y_test[mask], pred_base[mask])
            rows.append({"model": f"ACFP-QR__on_{tname}", "family": "diagnostic", "seed": seed, **mA, **effA})
            rows.append({"model": f"Baseline__on_{tname}", "family": "diagnostic", "seed": seed, **mB, **effB})

        print(f"seed {seed} complete ({time.time()-t0:.1f}s for candidate A)")

    df = pd.DataFrame(rows)
    df.to_csv("results/raw/final_test.csv", index=False)
    print(df.groupby("model")["rmse"].agg(["mean", "std", "count"]).sort_values("mean"))


if __name__ == "__main__":
    main()
