"""
Generates IEEE-ready LaTeX table fragments (Sec. 29/31) directly from results/ CSVs, written to
tables/*.tex for \\input{} inclusion in manuscript/main.tex. No hand-typed numbers.
"""
import pandas as pd
import numpy as np

FINAL = "ACFP-QR (proposed, FINAL)"


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


def main_comparison_table():
    df = pd.read_csv("results/raw/final_test.csv")
    df = df[df["family"].isin(["classical/boosting", "deep", "proposed"])]
    agg = df.groupby("model").agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), r2_mean=("r2", "mean"), mape_mean=("mape", "mean"),
        eff_mean=("eff_loss_mean_pct", "mean"),
        latency_ms=("infer_time_s", lambda x: x.mean() * 1000),
    ).sort_values("rmse_mean")
    lines = [
        r"\begin{table*}[t]",
        r"\centering", r"\footnotesize",
        r"\caption{Test-set price-recovery performance: proposed model vs. all benchmarks (mean over seeds, contaminated TEST).}",
        r"\label{tab:main}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & RMSE & MAE & $R^2$ & MAPE (\%) & Eff.\ loss (\%) & Latency (ms) \\",
        r"\midrule",
    ]
    for model, r in agg.iterrows():
        name = esc(model)
        if model == FINAL:
            name = r"\textbf{" + name + "}"
        lines.append(f"{name} & {r.rmse_mean:.3f} & {r.mae_mean:.3f} & {r.r2_mean:.4f} & "
                     f"{r.mape_mean:.2f} & {r.eff_mean:.3f} & {r.latency_ms:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open("tables/main_comparison.tex", "w") as f:
        f.write("\n".join(lines))


def per_attack_table():
    df = pd.read_csv("results/raw/final_test.csv")
    diag = df[df["family"] == "diagnostic"].copy()
    diag["attack"] = diag["model"].str.extract(r"__on_(\w+)")
    diag["method"] = np.where(diag["model"].str.startswith("ACFP"), "ACFP-QR", "Baseline")
    agg = diag.groupby(["attack", "method"]).agg(rmse=("rmse", "mean"), mape=("mape", "mean")).unstack()
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\caption{Per-attack-type recovery accuracy: attack-agnostic baseline vs.\ proposed model.}",
        r"\label{tab:perattack}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Attack type & Baseline RMSE & ACFP-QR RMSE & Baseline MAPE (\%) & ACFP-QR MAPE (\%) \\",
        r"\midrule",
    ]
    order = [a for a in ["FDI", "replay", "byzantine"] if a in agg.index]
    display_name = {"FDI": "FDI", "replay": "Replay", "byzantine": "Byzantine"}
    for a in order:
        r = agg.loc[a]
        lines.append(f"{esc(display_name[a])} & {r[('rmse','Baseline')]:.3f} & \\textbf{{{r[('rmse','ACFP-QR')]:.3f}}} & "
                     f"{r[('mape','Baseline')]:.2f} & \\textbf{{{r[('mape','ACFP-QR')]:.2f}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open("tables/per_attack.tex", "w") as f:
        f.write("\n".join(lines))


def ablation_table():
    df = pd.read_csv("results/aggregate/ablation_table.csv")
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\caption{Component ablation (Sec.~\ref{sec:results}); components affecting price-domain metrics are mechanically separable from those affecting feasibility/coverage only.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrrrl}", r"\toprule",
        r"Variant & RMSE & MAPE (\%) & Eff.\ loss (\%) & Feasibility viol.\ (\%) \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        fv = r["feasibility_violation_after_pct"]
        fv_s = f"{fv:.4f}" if pd.notna(fv) else "--"
        lines.append(f"{esc(r['variant'])} & {r['rmse']:.3f} & {r['mape']:.2f} & {r['eff_loss_mean_pct']:.3f} & {fv_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open("tables/ablation.tex", "w") as f:
        f.write("\n".join(lines))


def wtl_table():
    df = pd.read_csv("BENCHMARK_WTL.csv")
    rmse_df = df[df["metric"] == "rmse"].sort_values("mean_diff_final_minus_bench")
    eff_df = df[df["metric"] == "effloss"].sort_values("mean_diff_final_minus_bench")

    def counts(sub):
        n_win = (sub["outcome"] == "win").sum()
        n_tie = sub["outcome"].str.startswith("tie").sum()
        n_loss = (sub["outcome"] == "loss").sum()
        return n_win, n_tie, n_loss

    nw_r, nt_r, nl_r = counts(rmse_df)
    nw_e, nt_e, nl_e = counts(eff_df)
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        rf"\caption{{Win/tie/loss of ACFP-QR vs.\ {len(rmse_df)} independent benchmarks on both declared co-primary metrics (paired Wilcoxon, step-down Holm-corrected $\alpha=0.05$, bootstrap 95\% CI on the mean difference). RMSE: {nw_r}W/{nt_r}T/{nl_r}L. Efficiency loss: {nw_e}W/{nt_e}T/{nl_e}L.}}",
        r"\label{tab:wtl}",
        r"\begin{tabular}{lrrrl|rrrl}", r"\toprule",
        r" & \multicolumn{4}{c|}{RMSE} & \multicolumn{4}{c}{Efficiency loss} \\",
        r"Benchmark & $\Delta$ & 95\% CI & $p$ & Outcome & $\Delta$ & 95\% CI & $p$ & Outcome \\",
        r"\midrule",
    ]
    eff_by_bench = eff_df.set_index("benchmark")
    for _, r in rmse_df.iterrows():
        e = eff_by_bench.loc[r["benchmark"]]
        lines.append(
            f"{esc(r['benchmark'])} & {r['mean_diff_final_minus_bench']:.3f} & "
            f"[{r['diff_ci95_lo']:.2f},{r['diff_ci95_hi']:.2f}] & {r['p_value']:.4f} & {esc(r['outcome'])} & "
            f"{e['mean_diff_final_minus_bench']:.3f} & [{e['diff_ci95_lo']:.2f},{e['diff_ci95_hi']:.2f}] & "
            f"{e['p_value']:.4f} & {esc(e['outcome'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open("tables/wtl.tex", "w") as f:
        f.write("\n".join(lines))


def network_validation_table():
    df = pd.read_csv("results/aggregate/network_validation_summary.csv", index_col=0)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{14-bus network-level dispatch cost inflation relative to the true optimum (mean over 300 held-out hours per attack type; real GEFCom2014-derived demand, self-consistent clearing price).}",
        r"\label{tab:network}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Attack & No recovery & Baseline recovery & ACFP-QR recovery \\",
        r" & (\%) & (\%) & (\%) \\",
        r"\midrule",
    ]
    for attack, r in df.iterrows():
        lines.append(f"{esc(attack)} & {r['loss_pct_no_recovery']:.3f} & {r['loss_pct_baseline_recovery']:.5f} & "
                     f"{r['loss_pct_acfpqr_recovery']:.5f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open("tables/network_validation.tex", "w") as f:
        f.write("\n".join(lines))


def backbone_generalization_table():
    df = pd.read_csv("results/raw/backbone_generalization.csv")
    sub = df[df["attack"] == "overall"]
    agg = sub.groupby("model")["rmse"].agg(["mean", "std"])
    order = ["CatBoost-alone", "CatBoost+Gate", "TCN-alone", "TCN+Gate"]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{The attack-type-gated correction mechanism improves both a CatBoost and an unrelated TCN backbone, confirming the gain is attributable to the mechanism, not to backbone choice alone.}",
        r"\label{tab:backbonegen}",
        r"\begin{tabular}{lr}", r"\toprule",
        r"Model & RMSE (mean $\pm$ std) \\", r"\midrule",
    ]
    for name in order:
        r = agg.loc[name]
        lines.append(f"{esc(name)} & {r['mean']:.3f} $\\pm$ {r['std']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open("tables/backbone_generalization.tex", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main_comparison_table()
    per_attack_table()
    ablation_table()
    wtl_table()
    network_validation_table()
    backbone_generalization_table()
    print("tables written to tables/")
