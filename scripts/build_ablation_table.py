"""
Assembles the ablation table (Sec. 19) directly from results/raw/final_test.csv — no extra
retraining needed because the three added components are mechanically separable:
  - the attack-gate is the ONLY component that changes the point prediction (and hence
    RMSE/MAE/R2/MAPE/efficiency-loss);
  - the feasibility (water-filling) projection acts strictly downstream of the price prediction,
    on dispatch MW, so it changes ONLY the feasibility-violation metric, never price accuracy;
  - the quantile heads are separate models that do not feed back into the point prediction at
    all, so they change ONLY interval coverage, never RMSE/MAE/R2/MAPE/efficiency-loss/feasibility.
This is reported explicitly rather than fabricating spurious numeric deltas for components that
are mechanically orthogonal to a given metric.
"""
import pandas as pd

df = pd.read_csv("results/raw/final_test.csv")


def agg(model_name, cols):
    sub = df[df["model"] == model_name]
    return sub[cols].mean()


def main():
    point_cols = ["rmse", "mae", "r2", "mape", "eff_loss_mean_pct"]
    rows = []

    full = agg("ACFP-QR (proposed, FINAL)", point_cols)
    no_gate = agg("D_QR-GBM", point_cols)  # backbone + quantile heads, no gate == "minus gate"
    no_gate_no_quantile = agg("Baseline-attack-agnostic-GBM", point_cols)  # == "minus gate, minus quantile"

    rows.append({"variant": "Full model (ACFP-QR)", **full.to_dict(),
                  "feasibility_violation_after_pct": df[df.model == "ACFP-QR (proposed, FINAL)"]["feasibility_violation_after_pct"].mean(),
                  "quantile_coverage_10_90_pct": df[df.model == "ACFP-QR (proposed, FINAL)"]["quantile_coverage_10_90_pct"].mean()})
    rows.append({"variant": "minus feasibility projection (point metrics unchanged by construction)",
                  **full.to_dict(),
                  "feasibility_violation_after_pct": df[df.model == "ACFP-QR (proposed, FINAL)"]["feasibility_violation_before_pct"].mean(),
                  "quantile_coverage_10_90_pct": df[df.model == "ACFP-QR (proposed, FINAL)"]["quantile_coverage_10_90_pct"].mean()})
    rows.append({"variant": "minus quantile heads (point/feasibility metrics unchanged by construction)",
                  **full.to_dict(),
                  "feasibility_violation_after_pct": df[df.model == "ACFP-QR (proposed, FINAL)"]["feasibility_violation_after_pct"].mean(),
                  "quantile_coverage_10_90_pct": float("nan")})
    rows.append({"variant": "minus attack-gate (= D_QR-GBM)", **no_gate.to_dict(),
                  "feasibility_violation_after_pct": float("nan"), "quantile_coverage_10_90_pct": float("nan")})
    rows.append({"variant": "minus gate, minus quantile, minus feasibility (= parameter-matched backbone only)",
                  **no_gate_no_quantile.to_dict(),
                  "feasibility_violation_after_pct": df[df.model == "Baseline-attack-agnostic-GBM"]["feasibility_violation_before_pct"].mean(),
                  "quantile_coverage_10_90_pct": float("nan")})

    out = pd.DataFrame(rows)
    out.to_csv("results/aggregate/ablation_table.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
