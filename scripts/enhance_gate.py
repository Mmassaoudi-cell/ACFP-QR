"""
Validation-only diagnosis and fix for the Byzantine-contamination regression flagged in peer
review (Devil's Advocate CRITICAL #3/#5, R1-Methodology Weakness 4, R3-Perspective Weakness 3):
the shallow, shared-hyperparameter gate corrector underfits Byzantine's wider multi-lag
contamination pattern relative to FDI's single-lag / replay's 3-lag signatures. Compares
candidate fixes on TRAIN/VAL only (TEST untouched) and selects the configuration used in the
revised FINAL_MODEL_CONFIG.yaml.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_gefcom import get_dataset
from attack_contamination import contaminate, ATTACK_NAMES
from candidates import CandidateA_ACFP_QR, make_backbone, ENHANCED_GATE_PARAMS
from economic_impact import implied_demand
from metrics import regression_metrics

BACKBONE_HP = dict(depth=9, learning_rate=0.022342748011385118, iterations=800)
SEEDS = (0, 1, 2)

CONFIGS = {
    "baseline": dict(per_type_params=None, label_noise_rate=0.0),
    "byzantine_capacity": dict(per_type_params=ENHANCED_GATE_PARAMS, label_noise_rate=0.0),
    "label_noise_only": dict(per_type_params=None, label_noise_rate=0.15),
    "byzantine_capacity+label_noise": dict(per_type_params=ENHANCED_GATE_PARAMS, label_noise_rate=0.15),
}


def run():
    feat, train, val, test, cols = get_dataset()
    n_bb = int(len(train) * 0.8)
    train_bb, train_gate = train.iloc[:n_bb], train.iloc[n_bb:]
    scaler = StandardScaler().fit(train_bb[cols])
    Xbb, ybb = scaler.transform(train_bb[cols]), train_bb["price"].values
    Xval_raw, yval = val[cols], val["price"].values
    demand_val = implied_demand(yval)

    rows = []
    for cfg_name, cfg in CONFIGS.items():
        for seed in SEEDS:
            backbone = make_backbone(seed, **BACKBONE_HP)
            backbone.fit(Xbb, ybb)

            rng = np.random.RandomState(1000 + seed)
            gate_df, gate_labels = contaminate(train_gate, cols, rng, frac=0.20)
            Xgate = scaler.transform(gate_df[cols])
            ygate = gate_df["price"].values
            resid = ygate - backbone.predict(Xgate)

            from candidates import AttackGate
            gate = AttackGate(seed, **cfg)
            gate.fit(Xgate, resid, gate_labels)

            rng_eval = np.random.RandomState(2000 + seed)
            val_df, val_labels = contaminate(val, cols, rng_eval, frac=0.20)
            Xval_c = scaler.transform(val_df[cols])
            pred = backbone.predict(Xval_c) + gate.predict(Xval_c, val_labels)

            m_overall = regression_metrics(yval, pred)
            rows.append({"config": cfg_name, "seed": seed, "attack": "overall", **m_overall})
            for t, tname in ATTACK_NAMES.items():
                mask = val_labels == t
                if mask.sum() < 5:
                    continue
                m_t = regression_metrics(yval[mask], pred[mask])
                rows.append({"config": cfg_name, "seed": seed, "attack": tname, **m_t})
        print(f"config {cfg_name} done")

    df = pd.DataFrame(rows)
    df.to_csv("results/raw/gate_enhancement_screen.csv", index=False)
    agg = df.groupby(["config", "attack"])[["rmse", "mape"]].mean().unstack()
    pd.set_option("display.width", 160)
    print(agg)
    return df


if __name__ == "__main__":
    run()
