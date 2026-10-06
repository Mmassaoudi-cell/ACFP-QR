"""
Validation-only test of gate-label-noise-robust training (peer-review Priority-1 item: stress-test
the assumption that the cryptographic layer's attack-type flag g is always correctly classified).
Compares two gate-training regimes under a simulated g-misclassification sweep at evaluation time:
  - "capacity-only": AttackGate(per_type_params=ENHANCED_GATE_PARAMS, label_noise_rate=0.0)
  - "capacity+noise-robust": AttackGate(per_type_params=ENHANCED_GATE_PARAMS, label_noise_rate=0.15)
Selection is made on TRAIN/VAL only; TEST is untouched here.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_gefcom import get_dataset
from attack_contamination import contaminate, ATTACK_NAMES
from candidates import AttackGate, make_backbone, ENHANCED_GATE_PARAMS
from metrics import regression_metrics

BACKBONE_HP = dict(depth=9, learning_rate=0.022342748011385118, iterations=800)
SEEDS = (0, 1, 2)
MISCLASSIFY_RATES = [0.0, 0.10, 0.20, 0.30, 0.40]


def permute_labels(labels, rate, rng, n_types=4):
    labels = labels.copy()
    n = len(labels)
    n_flip = int(rate * n)
    idx = rng.choice(n, size=n_flip, replace=False)
    for i in idx:
        choices = [t for t in range(n_types) if t != labels[i]]
        labels[i] = rng.choice(choices)
    return labels


def run():
    feat, train, val, test, cols = get_dataset()
    n_bb = int(len(train) * 0.8)
    train_bb, train_gate = train.iloc[:n_bb], train.iloc[n_bb:]
    scaler = StandardScaler().fit(train_bb[cols])
    Xbb, ybb = scaler.transform(train_bb[cols]), train_bb["price"].values
    yval = val["price"].values

    rows = []
    for regime_name, label_noise_rate in [("capacity-only", 0.0), ("capacity+noise-robust", 0.15)]:
        for seed in SEEDS:
            backbone = make_backbone(seed, **BACKBONE_HP)
            backbone.fit(Xbb, ybb)

            rng = np.random.RandomState(1000 + seed)
            gate_df, gate_labels = contaminate(train_gate, cols, rng, frac=0.20)
            Xgate = scaler.transform(gate_df[cols])
            ygate = gate_df["price"].values
            resid = ygate - backbone.predict(Xgate)

            gate = AttackGate(seed, per_type_params=ENHANCED_GATE_PARAMS, label_noise_rate=label_noise_rate)
            gate.fit(Xgate, resid, gate_labels)

            rng_eval = np.random.RandomState(2000 + seed)
            val_df, true_labels = contaminate(val, cols, rng_eval, frac=0.20)
            Xval_c = scaler.transform(val_df[cols])

            for rate in MISCLASSIFY_RATES:
                rng_mis = np.random.RandomState(3000 + seed + int(rate * 1000))
                observed_labels = permute_labels(true_labels, rate, rng_mis)
                pred = backbone.predict(Xval_c) + gate.predict(Xval_c, observed_labels)
                m = regression_metrics(yval, pred)
                rows.append({"regime": regime_name, "seed": seed, "misclassify_rate": rate, **m})
        print(f"regime {regime_name} done")

    df = pd.DataFrame(rows)
    df.to_csv("results/raw/gate_robustness_screen.csv", index=False)
    print(df.groupby(["regime", "misclassify_rate"])["rmse"].mean().unstack())
    return df


if __name__ == "__main__":
    run()
