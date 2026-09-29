"""
Main benchmark runner for the price-recovery regression task. Two modes:
  --mode screen : train-on-TRAIN, evaluate-on-VAL only (candidate/benchmark screening, Sec. 9).
  --mode test   : train-on-TRAIN+VAL, evaluate-on-TEST (final, run once configs are frozen, Sec. 15).
Multiple seeds for stochastic models (bagging/boosting randomness, NN init); deterministic
models (Linear/Ridge/kNN/SVR) are run once and repeated across the seed column for uniform
downstream aggregation.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_gefcom import get_dataset
from models_regression import make_classical_models, make_deep_models, DEVICE
from metrics import regression_metrics
from economic_impact import efficiency_loss

DETERMINISTIC = {"LinearRegression", "Ridge", "kNN", "SVR"}


def prepare_split(mode: str):
    feat, train, val, test, cols = get_dataset()
    if mode == "screen":
        fit_df, eval_df = train, val
    else:
        fit_df = pd.concat([train, val], axis=0)
        eval_df = test
    scaler = StandardScaler().fit(fit_df[cols])
    X_fit = scaler.transform(fit_df[cols])
    X_eval = scaler.transform(eval_df[cols])
    y_fit = fit_df["price"].values
    y_eval = eval_df["price"].values
    return X_fit, y_fit, X_eval, y_eval, eval_df["system_load"].values, cols


def run(mode: str, seeds: list, out_path: str):
    X_fit, y_fit, X_eval, y_eval, load_eval, cols = prepare_split(mode)
    rows = []
    for seed in seeds:
        classical = make_classical_models(seed)
        deep = make_deep_models(seed, cols)
        all_models = {**classical, **deep}
        for name, model in all_models.items():
            if name in DETERMINISTIC and seed != seeds[0]:
                continue  # deterministic: no point re-running identical fit
            t0 = time.time()
            if hasattr(model, "epochs"):
                n_val = max(1, int(0.1 * len(X_fit)))
                model.fit(X_fit[:-n_val], y_fit[:-n_val], X_fit[-n_val:], y_fit[-n_val:])
            else:
                model.fit(X_fit, y_fit)
            train_time = time.time() - t0
            t1 = time.time()
            pred = model.predict(X_eval)
            infer_time = time.time() - t1
            m = regression_metrics(y_eval, pred)
            eff = efficiency_loss(y_eval, pred, load_eval)
            n_params = sum(p.numel() for p in model.model.parameters()) if hasattr(model, "model") else None
            row = {"model": name, "seed": seed, "mode": mode, "train_time_s": train_time,
                   "infer_time_s": infer_time, "n_params": n_params, **m, **eff}
            rows.append(row)
            print(f"[{mode}] seed={seed} {name:22s} RMSE={m['rmse']:.3f} R2={m['r2']:.4f} "
                  f"MAPE={m['mape']:.2f}% effloss={eff['eff_loss_mean_pct']:.3f}% "
                  f"train={train_time:.2f}s")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["screen", "test"], default="screen")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/raw/benchmark_{args.mode}.csv"
    run(args.mode, args.seeds, out)
