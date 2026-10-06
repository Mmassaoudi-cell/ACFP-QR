"""
Tests whether the attack-type-gated residual-correction MECHANISM generalizes across backbones,
or is only effective because of the specific CatBoost backbone (peer-review Devil's Advocate
CRITICAL #2: the strongest independent comparator, TCN, has no access to the attack-type flag and
nearly ties the full CatBoost-backed ACFP-QR, raising the question of whether the reported gain is
attributable to attack-type conditioning or to backbone capacity alone).

Attaches the identical AttackGate mechanism (enhanced per scripts/enhance_gate.py) to a TCN
backbone and compares TCN-alone vs. TCN+Gate on the same contaminated VALIDATION protocol used
for the CatBoost-backed candidate, to isolate the mechanism's marginal effect independent of
backbone choice. TRAIN/VAL only; TEST untouched.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_gefcom import get_dataset
from attack_contamination import contaminate, ATTACK_NAMES
from candidates import AttackGate, make_backbone, ENHANCED_GATE_PARAMS
from models_regression import TabularTCNRegressor
from metrics import regression_metrics

BACKBONE_HP = dict(depth=9, learning_rate=0.022342748011385118, iterations=800)
GATE_CFG = dict(per_type_params=ENHANCED_GATE_PARAMS, label_noise_rate=0.15)
SEEDS = (0, 1, 2)


def run():
    feat, train, val, test, cols = get_dataset()
    n_bb = int(len(train) * 0.8)
    train_bb, train_gate = train.iloc[:n_bb], train.iloc[n_bb:]
    scaler = StandardScaler().fit(train_bb[cols])
    Xbb, ybb = scaler.transform(train_bb[cols]), train_bb["price"].values
    yval = val["price"].values

    rows = []
    for seed in SEEDS:
        rng = np.random.RandomState(1000 + seed)
        gate_df, gate_labels = contaminate(train_gate, cols, rng, frac=0.20)
        Xgate = scaler.transform(gate_df[cols])
        ygate = gate_df["price"].values

        rng_eval = np.random.RandomState(2000 + seed)
        val_df, val_labels = contaminate(val, cols, rng_eval, frac=0.20)
        Xval_c = scaler.transform(val_df[cols])

        # --- CatBoost backbone (reference) ---
        cat = make_backbone(seed, **BACKBONE_HP)
        cat.fit(Xbb, ybb)
        pred_cat = cat.predict(Xval_c)
        resid_cat = ygate - cat.predict(Xgate)
        gate_cat = AttackGate(seed, **GATE_CFG)
        gate_cat.fit(Xgate, resid_cat, gate_labels)
        pred_cat_gated = pred_cat + gate_cat.predict(Xval_c, val_labels)

        # --- TCN backbone ---
        n_es = max(1, int(0.1 * len(Xbb)))
        tcn = TabularTCNRegressor(cols, seed=seed, epochs=80)
        tcn.fit(Xbb[:-n_es], ybb[:-n_es], Xbb[-n_es:], ybb[-n_es:])
        pred_tcn = tcn.predict(Xval_c)
        resid_tcn = ygate - tcn.predict(Xgate)
        gate_tcn = AttackGate(seed, **GATE_CFG)
        gate_tcn.fit(Xgate, resid_tcn, gate_labels)
        pred_tcn_gated = pred_tcn + gate_tcn.predict(Xval_c, val_labels)

        for name, pred in [("CatBoost-alone", pred_cat), ("CatBoost+Gate", pred_cat_gated),
                            ("TCN-alone", pred_tcn), ("TCN+Gate", pred_tcn_gated)]:
            m = regression_metrics(yval, pred)
            rows.append({"model": name, "seed": seed, "attack": "overall", **m})
            for t, tname in ATTACK_NAMES.items():
                mask = val_labels == t
                if mask.sum() < 5:
                    continue
                m_t = regression_metrics(yval[mask], pred[mask])
                rows.append({"model": name, "seed": seed, "attack": tname, **m_t})
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv("results/raw/backbone_generalization.csv", index=False)
    print(df[df.attack == "overall"].groupby("model")["rmse"].agg(["mean", "std"]))
    return df


if __name__ == "__main__":
    run()
