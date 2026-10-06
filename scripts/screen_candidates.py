"""
Stage 2/3 validation-only candidate screening (Sec. 9-11). TEST is never touched here.
Backbone is fit on the first 80% of TRAIN (chronological); the last 20% of TRAIN is held out,
contaminated, and used to fit Candidate A's attack-type gate / Candidate B's residual GRU;
evaluation of all candidates + the attack-agnostic baseline happens on a *contaminated* copy of
VAL (3 contamination seeds x the model's own seed).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_gefcom import get_dataset
from attack_contamination import contaminate, ATTACK_NAMES
from candidates import CandidateA_ACFP_QR, CandidateB_GBM_ResGRU, CandidateD_QRGBM, make_backbone, water_fill
from economic_impact import _FLEET, implied_demand
from metrics import regression_metrics


def feasibility_violation_pct(demand, outputs):
    return float(np.mean(np.abs(demand - outputs.sum(axis=1)) / np.clip(demand, 1e-3, None)) * 100)


def run(seeds=(0, 1, 2), contam_frac=0.20, out_path="results/raw/candidate_screening.csv"):
    feat, train, val, test, cols = get_dataset()
    n_bb = int(len(train) * 0.8)
    train_bb, train_gate = train.iloc[:n_bb], train.iloc[n_bb:]

    scaler = StandardScaler().fit(train_bb[cols])
    Xbb = scaler.transform(train_bb[cols])
    ybb = train_bb["price"].values
    Xval_clean = scaler.transform(val[cols])
    yval = val["price"].values
    demand_val = implied_demand(yval)

    rows = []
    for seed in seeds:
        rng = np.random.RandomState(1000 + seed)
        gate_df, gate_labels = contaminate(train_gate, cols, rng, frac=contam_frac)
        Xgate = scaler.transform(gate_df[cols])
        ygate = gate_df["price"].values

        rng_eval = np.random.RandomState(2000 + seed)
        val_df, val_labels = contaminate(val, cols, rng_eval, frac=contam_frac)
        Xval_c = scaler.transform(val_df[cols])

        # --- baseline: attack-agnostic single global backbone (mirrors source method) ---
        base = make_backbone(seed)
        base.fit(Xbb, ybb)
        pred_base = base.predict(Xval_c)
        m = regression_metrics(yval, pred_base)
        out_base = water_fill(_FLEET, demand_val, pred_base)
        rows.append({"candidate": "Baseline-attack-agnostic-GBM", "seed": seed, **m,
                     "feasibility_violation_pct_before_projection":
                         feasibility_violation_pct(demand_val, np.array([[g.dispatch(p) for g in _FLEET] for p in pred_base])),
                     "feasibility_violation_pct_after_projection": feasibility_violation_pct(demand_val, out_base)})

        # --- Candidate D: quantile only, no gate ---
        candD = CandidateD_QRGBM(seed)
        candD.fit(Xbb, ybb)
        predD = candD.predict(Xval_c)
        m = regression_metrics(yval, predD)
        rows.append({"candidate": "D_QR-GBM", "seed": seed, **m,
                     "feasibility_violation_pct_before_projection": None,
                     "feasibility_violation_pct_after_projection": None})

        # --- Candidate B: GBM + residual GRU ---
        candB = CandidateB_GBM_ResGRU(seed, cols)
        candB.fit(Xbb, ybb, Xval_clean[:200], yval[:200])  # small clean slice for early stopping only
        predB = candB.predict(Xval_c)
        m = regression_metrics(yval, predB)
        rows.append({"candidate": "B_GBM+Res-GRU", "seed": seed, **m,
                     "feasibility_violation_pct_before_projection": None,
                     "feasibility_violation_pct_after_projection": None})

        # --- Candidate A: full assembly ---
        candA = CandidateA_ACFP_QR(seed)
        candA.fit(Xbb, ybb, Xgate, ygate, gate_labels)
        predA_noattackinfo = candA.predict(Xval_c, attack_label=None)
        predA, outA = candA.recover_dispatch(Xval_c, demand_val, attack_label=val_labels, feasibility=True)
        _, outA_nofeas = candA.recover_dispatch(Xval_c, demand_val, attack_label=val_labels, feasibility=False)
        m = regression_metrics(yval, predA)
        rows.append({"candidate": "A_ACFP-QR (proposed)", "seed": seed, **m,
                     "feasibility_violation_pct_before_projection": feasibility_violation_pct(demand_val, outA_nofeas),
                     "feasibility_violation_pct_after_projection": feasibility_violation_pct(demand_val, outA)})

        # per-attack-type breakdown for Candidate A (diagnostic)
        for t, tname in ATTACK_NAMES.items():
            mask = val_labels == t
            if mask.sum() < 5:
                continue
            m_t = regression_metrics(yval[mask], predA[mask])
            m_base_t = regression_metrics(yval[mask], pred_base[mask])
            rows.append({"candidate": f"A_ACFP-QR__on_{tname}", "seed": seed, **m_t,
                         "feasibility_violation_pct_before_projection": None,
                         "feasibility_violation_pct_after_projection": None})
            rows.append({"candidate": f"Baseline__on_{tname}", "seed": seed, **m_base_t,
                         "feasibility_violation_pct_before_projection": None,
                         "feasibility_violation_pct_after_projection": None})

        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(df.groupby("candidate")[["rmse", "mae", "r2", "mape"]].mean().sort_values("rmse"))
    return df


if __name__ == "__main__":
    run()
