"""
Statistical validation: paired Wilcoxon signed-rank test (ACFP-QR vs. each benchmark, paired by
seed on the SAME contaminated-TEST realization), proper step-down Holm-Bonferroni correction,
matched-pairs rank-biserial effect size computed from signed ranks (not win/loss sign counts),
and a bootstrap 95% CI on the paired RMSE mean-difference (informative at n=10, where the exact
Wilcoxon p-value has a floor around 0.002 and cannot itself discriminate practical magnitude).
Run once for RMSE and once for economic efficiency loss -- both are declared co-primary metrics
(FINAL_MODEL_CONFIG.yaml) and both receive the identical statistical treatment.
"""
import numpy as np
import pandas as pd
from scipy import stats

FINAL_MODEL = "ACFP-QR (proposed, FINAL)"
INDEPENDENT_BENCHMARKS = [
    "RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting", "XGBoost",
    "LightGBM", "CatBoost", "MLP", "GRU", "TCN", "Transformer",
]  # excludes internal ablation controls (D_QR-GBM, B_GBM+Res-GRU, Baseline-attack-agnostic-GBM)
   # and the four deterministic single-seed classical models (insufficient seeds for a paired test)


def ci95(x):
    x = np.asarray(x)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return (x.mean() - h, x.mean() + h)


def bootstrap_ci_diff(a, b, n_boot=10000, seed=0):
    """Bootstrap 95% CI on the paired mean difference a-b (resampling seed-pairs with replacement)."""
    rng = np.random.RandomState(seed)
    diff = a - b
    n = len(diff)
    boots = np.array([diff[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def rank_biserial_signed(a, b):
    """Matched-pairs rank-biserial correlation from Wilcoxon signed ranks (Kerby 2014 formula:
    r = R+ / (R+ + R-) rescaled to [-1,1]), not a sign-count approximation."""
    diff = a - b
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total > 0 else 0.0


def summarize(df, metric="rmse"):
    rows = []
    for model, g in df.groupby("model"):
        vals = g[metric].dropna().values
        lo, hi = ci95(vals)
        rows.append({"model": model, "n": len(vals), "mean": vals.mean(),
                     "std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                     "median": np.median(vals), "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(rows).sort_values("mean")


def holm_bonferroni(pvalues, alpha=0.05):
    """Proper step-down Holm-Bonferroni: sort ascending, compare to alpha/(m-rank+1), and once a
    hypothesis fails to reject, every subsequent (larger-p) hypothesis is also not rejected,
    regardless of its own threshold."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    reject = np.zeros(m, dtype=bool)
    thresholds = np.full(m, np.nan)
    still_rejecting = True
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        thresholds[idx] = thresh
        if still_rejecting and pvalues[idx] < thresh:
            reject[idx] = True
        else:
            still_rejecting = False
            reject[idx] = False
    return reject, thresholds


def paired_tests(df, metric, final_model=FINAL_MODEL, family=INDEPENDENT_BENCHMARKS):
    """Paired Wilcoxon vs. each benchmark in `family`, corrected as one Holm family of exactly
    `len(family)` comparisons (the family actually reported to the reader), with a proper
    step-down procedure, a signed-rank effect size, and a bootstrap CI on the mean difference."""
    final_vals = df[df["model"] == final_model].set_index("seed")[metric]
    results = []
    for model in family:
        other = df[df["model"] == model].set_index("seed")[metric]
        common = final_vals.index.intersection(other.index)
        if len(common) < 3:
            continue
        a, b = final_vals.loc[common].values, other.loc[common].values
        diff = a - b
        if np.allclose(diff, 0):
            p, w = 1.0, 0.0
        else:
            try:
                w, p = stats.wilcoxon(a, b)
            except ValueError:
                w, p = np.nan, 1.0
        ci_lo, ci_hi = bootstrap_ci_diff(a, b)
        results.append({
            "benchmark": model, "n_pairs": len(common), "mean_diff_final_minus_bench": diff.mean(),
            "diff_ci95_lo": ci_lo, "diff_ci95_hi": ci_hi,
            "wilcoxon_stat": w, "p_value": p,
            "effect_size_rank_biserial": rank_biserial_signed(a, b),
            "final_wins_pct": float((diff < 0).mean() * 100),
        })
    res = pd.DataFrame(results)
    reject, thresholds = holm_bonferroni(res["p_value"].values, alpha=0.05)
    res["holm_threshold"] = thresholds
    res["significant_holm_0.05"] = reject
    return res.sort_values("p_value").reset_index(drop=True)


def main():
    df = pd.read_csv("results/raw/final_test.csv")
    df = df[df["family"].isin(["classical/boosting", "deep", "ablation", "proposed"])]

    for metric, out_prefix in [("rmse", "rmse"), ("eff_loss_mean_pct", "effloss")]:
        summary = summarize(df, metric)
        summary.to_csv(f"results/aggregate/{out_prefix}_summary.csv", index=False)
        print(f"=== {metric} summary ===")
        print(summary.to_string(index=False))

        tests = paired_tests(df, metric)
        tests.to_csv(f"results/aggregate/wilcoxon_{out_prefix}.csv", index=False)
        print(f"\n=== Paired Wilcoxon ({metric}, step-down Holm-corrected, family=11 independent benchmarks) ===")
        print(tests.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
