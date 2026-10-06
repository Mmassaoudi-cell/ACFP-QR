"""
BENCHMARK_WTL.csv: win/tie/loss of the proposed model vs. each of the 11 independent benchmarks,
computed identically for both declared co-primary metrics (RMSE and economic efficiency loss),
using the corrected paired-Wilcoxon + step-down-Holm results from stats_tests.py. A "win" requires
BOTH the proposed model ahead AND Holm-significance; ahead-but-not-significant is a "tie"
(practically similar at n=10 seeds); behind (significant or not) is a "loss."
"""
import pandas as pd


def build_one(metric_prefix, lower_is_better=True):
    tests = pd.read_csv(f"results/aggregate/wilcoxon_{metric_prefix}.csv")
    rows = []
    for _, r in tests.iterrows():
        ahead = (r["mean_diff_final_minus_bench"] < 0) if lower_is_better else (r["mean_diff_final_minus_bench"] > 0)
        sig = bool(r["significant_holm_0.05"])
        if ahead and sig:
            outcome = "win"
        elif ahead and not sig:
            outcome = "tie (practically ahead, not significant)"
        elif (not ahead) and sig:
            outcome = "loss"
        else:
            outcome = "tie (practically behind, not significant)"
        rows.append({"metric": metric_prefix, "benchmark": r["benchmark"],
                     "mean_diff_final_minus_bench": r["mean_diff_final_minus_bench"],
                     "diff_ci95_lo": r["diff_ci95_lo"], "diff_ci95_hi": r["diff_ci95_hi"],
                     "p_value": r["p_value"], "effect_size_rank_biserial": r["effect_size_rank_biserial"],
                     "holm_significant": sig, "outcome": outcome,
                     "final_wins_pct_of_seeds": r["final_wins_pct"]})
    return pd.DataFrame(rows).sort_values("mean_diff_final_minus_bench")


def main():
    rmse_wtl = build_one("rmse", lower_is_better=True)
    eff_wtl = build_one("effloss", lower_is_better=True)
    out = pd.concat([rmse_wtl, eff_wtl], axis=0)
    out.to_csv("BENCHMARK_WTL.csv", index=False)

    for name, sub in [("RMSE", rmse_wtl), ("Efficiency loss", eff_wtl)]:
        n_win = (sub["outcome"] == "win").sum()
        n_tie = sub["outcome"].str.startswith("tie").sum()
        n_loss = (sub["outcome"] == "loss").sum()
        print(f"=== {name}: {n_win} wins / {n_tie} ties / {n_loss} losses (of {len(sub)}) ===")
        print(sub.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
