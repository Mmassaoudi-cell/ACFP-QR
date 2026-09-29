"""
Stage-3 validation-only hyperparameter tuning for the selected top candidate (ACFP-QR), via
Optuna TPE (Sec. 9, avoids grid search). TEST is never touched. Objective = mean RMSE on
contaminated VAL across 2 contamination seeds, backbone/gate fit on TRAIN only.
"""
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

from data_gefcom import get_dataset
from attack_contamination import contaminate
from candidates import AttackGate
from metrics import regression_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

feat, train, val, test, cols = get_dataset()
n_bb = int(len(train) * 0.8)
train_bb, train_gate = train.iloc[:n_bb], train.iloc[n_bb:]
scaler = StandardScaler().fit(train_bb[cols])
Xbb, ybb = scaler.transform(train_bb[cols]), train_bb["price"].values
Xval, yval = scaler.transform(val[cols]), val["price"].values


def objective(trial):
    depth = trial.suggest_int("depth", 4, 10)
    lr = trial.suggest_float("lr", 0.01, 0.15, log=True)
    iters = trial.suggest_int("iterations", 200, 800, step=100)
    gate_depth = trial.suggest_int("gate_depth", 2, 6)
    gate_iters = trial.suggest_int("gate_iterations", 50, 300, step=50)

    scores = []
    for cseed in (0, 1):
        backbone = CatBoostRegressor(iterations=iters, depth=depth, learning_rate=lr,
                                      random_seed=0, verbose=False)
        backbone.fit(Xbb, ybb)

        rng = np.random.RandomState(1000 + cseed)
        gate_df, gate_labels = contaminate(train_gate, cols, rng, frac=0.20)
        Xgate = scaler.transform(gate_df[cols])
        ygate = gate_df["price"].values
        resid = ygate - backbone.predict(Xgate)
        gate = AttackGate(0)
        for m in gate.correctors.values():
            m.set_params(depth=gate_depth, iterations=gate_iters)
        gate.fit(Xgate, resid, gate_labels)

        rng_eval = np.random.RandomState(2000 + cseed)
        val_df, val_labels = contaminate(val, cols, rng_eval, frac=0.20)
        Xval_c = scaler.transform(val_df[cols])
        pred = backbone.predict(Xval_c) + gate.predict(Xval_c, val_labels)
        scores.append(regression_metrics(yval, pred)["rmse"])
    return float(np.mean(scores))


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)
    import json
    with open("results/raw/tuning_candidateA.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params,
                   "n_trials": len(study.trials)}, f, indent=2)
