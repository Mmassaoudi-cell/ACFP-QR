"""
Robustness analysis (Sec. 20) of the FROZEN final model (ACFP-QR) vs. the attack-agnostic
baseline on TEST. Architecture/hyperparameters are never changed based on these results (Sec. 11)
— this script only stress-tests the already-frozen model under harder conditions than the
default screening/final-eval setting (contamination_fraction=0.20, strength=1.0).

Sweeps:
  (a) attack strength in {0.5, 1.0, 1.5, 2.0, 3.0} at fixed 20% contamination fraction;
  (b) contamination fraction in {0.10, 0.20, 0.30, 0.40} at fixed strength=1.0;
  (c) feature noise: Gaussian noise added to load-derived features (sigma as a fraction of feature std);
  (d) missing data: random zero-out of a fraction of lag features (simulating dropped telemetry).
5 seeds per condition (lighter than the 10-seed headline run, standard practice for a stress sweep).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_gefcom import get_dataset
from attack_contamination import contaminate, contaminate_unknown_shape
from candidates import CandidateA_ACFP_QR, make_backbone, AttackGate, ENHANCED_GATE_PARAMS
from economic_impact import implied_demand
from metrics import regression_metrics

SEEDS = list(range(5))
BACKBONE_HP = dict(depth=9, learning_rate=0.022342748011385118, iterations=800)
GATE_CFG = dict(per_type_params=ENHANCED_GATE_PARAMS, label_noise_rate=0.15)


def fit_final_model(seed, Xbb, ybb, Xgate, ygate, gate_labels):
    m = CandidateA_ACFP_QR(seed)
    m.backbone = make_backbone(seed, **BACKBONE_HP)
    m.backbone.fit(Xbb, ybb)
    m.quantiles.fit(Xbb, ybb)
    m.gate = AttackGate(seed, **GATE_CFG)
    resid = ygate - m.backbone.predict(Xgate)
    m.gate.fit(Xgate, resid, gate_labels)
    return m


def permute_labels(labels, rate, rng, n_types=4):
    labels = labels.copy()
    n = len(labels)
    n_flip = int(rate * n)
    idx = rng.choice(n, size=n_flip, replace=False)
    for i in idx:
        choices = [t for t in range(n_types) if t != labels[i]]
        labels[i] = rng.choice(choices)
    return labels


def fit_baseline(seed, Xbb, ybb):
    b = make_backbone(seed, **BACKBONE_HP)
    b.fit(Xbb, ybb)
    return b


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
        gate_df, gate_labels = contaminate(fit_gate, cols, rng_gate, frac=0.20, strength=1.0)
        Xgate, ygate = scaler.transform(gate_df[cols]), gate_df["price"].values
        model = fit_final_model(seed, Xbb, ybb, Xgate, ygate, gate_labels)
        baseline = fit_baseline(seed, Xbb, ybb)

        # (a) attack strength sweep
        for strength in [0.5, 1.0, 1.5, 2.0, 3.0]:
            rng = np.random.RandomState(2000 + seed)
            tdf, tlabels = contaminate(test, cols, rng, frac=0.20, strength=strength)
            Xc = scaler.transform(tdf[cols])
            predA = model.predict(Xc, attack_label=tlabels)
            predB = baseline.predict(Xc)
            rows.append({"sweep": "attack_strength", "value": strength, "seed": seed, "model": "ACFP-QR",
                         **regression_metrics(y_test, predA)})
            rows.append({"sweep": "attack_strength", "value": strength, "seed": seed, "model": "Baseline",
                         **regression_metrics(y_test, predB)})

        # (b) contamination fraction sweep
        for frac in [0.10, 0.20, 0.30, 0.40]:
            rng = np.random.RandomState(3000 + seed)
            tdf, tlabels = contaminate(test, cols, rng, frac=frac, strength=1.0)
            Xc = scaler.transform(tdf[cols])
            predA = model.predict(Xc, attack_label=tlabels)
            predB = baseline.predict(Xc)
            rows.append({"sweep": "contamination_fraction", "value": frac, "seed": seed, "model": "ACFP-QR",
                         **regression_metrics(y_test, predA)})
            rows.append({"sweep": "contamination_fraction", "value": frac, "seed": seed, "model": "Baseline",
                         **regression_metrics(y_test, predB)})

        # (c) feature (load) Gaussian noise, no attack contamination
        rng = np.random.RandomState(4000 + seed)
        load_cols = [c for c in cols if "load" in c]
        for sigma in [0.0, 0.1, 0.25, 0.5, 1.0]:
            Xc_df = test.copy()
            for c in load_cols:
                std = test[c].std()
                Xc_df[c] = Xc_df[c] + rng.normal(0, sigma * std, size=len(Xc_df))
            Xc = scaler.transform(Xc_df[cols])
            no_attack_labels = np.zeros(len(test), dtype=int)
            predA = model.predict(Xc, attack_label=no_attack_labels)
            predB = baseline.predict(Xc)
            rows.append({"sweep": "feature_noise_sigma", "value": sigma, "seed": seed, "model": "ACFP-QR",
                         **regression_metrics(y_test, predA)})
            rows.append({"sweep": "feature_noise_sigma", "value": sigma, "seed": seed, "model": "Baseline",
                         **regression_metrics(y_test, predB)})

        # (d) missing data: zero out a fraction of lag columns per-row
        lag_cols = [c for c in cols if "lag" in c]
        rng = np.random.RandomState(5000 + seed)
        for miss_frac in [0.0, 0.1, 0.25, 0.5]:
            Xc_df = test.copy()
            mask = rng.random((len(Xc_df), len(lag_cols))) < miss_frac
            arr = Xc_df[lag_cols].values
            col_means = arr.mean(axis=0)
            arr[mask] = np.take(col_means, np.where(mask)[1])
            Xc_df[lag_cols] = arr
            Xc = scaler.transform(Xc_df[cols])
            no_attack_labels = np.zeros(len(test), dtype=int)
            predA = model.predict(Xc, attack_label=no_attack_labels)
            predB = baseline.predict(Xc)
            rows.append({"sweep": "missing_data_frac", "value": miss_frac, "seed": seed, "model": "ACFP-QR",
                         **regression_metrics(y_test, predA)})
            rows.append({"sweep": "missing_data_frac", "value": miss_frac, "seed": seed, "model": "Baseline",
                         **regression_metrics(y_test, predB)})

        # (e) gate-misclassification sweep: the attack-type flag g is flipped to an incorrect
        # branch at a declared rate, simulating an imperfect (rather than always-correct)
        # cryptographic classifier
        rng_true = np.random.RandomState(2000 + seed)
        tdf_true, true_labels = contaminate(test, cols, rng_true, frac=0.20, strength=1.0)
        Xc_true = scaler.transform(tdf_true[cols])
        for rate in [0.0, 0.10, 0.20, 0.30, 0.40]:
            rng_mis = np.random.RandomState(6000 + seed + int(rate * 1000))
            observed = permute_labels(true_labels, rate, rng_mis)
            predA = model.predict(Xc_true, attack_label=observed)
            predB = baseline.predict(Xc_true)
            rows.append({"sweep": "gate_misclassification_rate", "value": rate, "seed": seed, "model": "ACFP-QR",
                         **regression_metrics(y_test, predA)})
            rows.append({"sweep": "gate_misclassification_rate", "value": rate, "seed": seed, "model": "Baseline",
                         **regression_metrics(y_test, predB)})

        # (f) out-of-taxonomy attack shape: a contamination pattern never seen during gate training
        rng_unk = np.random.RandomState(7000 + seed)
        udf, ulabels = contaminate_unknown_shape(test, cols, rng_unk, frac=0.20, strength=1.0)
        Xc_unk = scaler.transform(udf[cols])
        predA = model.predict(Xc_unk, attack_label=ulabels)
        predB = baseline.predict(Xc_unk)
        rows.append({"sweep": "unknown_attack_shape", "value": 1, "seed": seed, "model": "ACFP-QR",
                     **regression_metrics(y_test, predA)})
        rows.append({"sweep": "unknown_attack_shape", "value": 1, "seed": seed, "model": "Baseline",
                     **regression_metrics(y_test, predB)})

        print(f"robustness seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv("results/raw/robustness.csv", index=False)
    print(df.groupby(["sweep", "value", "model"])["rmse"].mean())


if __name__ == "__main__":
    main()
