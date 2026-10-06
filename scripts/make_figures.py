"""
Generates Fig. 3-6 (Sec. 31) directly from saved CSV result files — no hand-typed numbers.
Fig. 1 (architecture) and Fig. 2 (conceptual workflow) are TikZ diagrams inside the manuscript
itself (manuscript/main.tex), not matplotlib figures, since they depict structure, not data.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight"})
FINAL = "ACFP-QR (proposed, FINAL)"


def fig3_main_benchmark():
    df = pd.read_csv("results/raw/final_test.csv")
    df = df[df["family"].isin(["classical/boosting", "deep", "proposed"])]
    agg = df.groupby("model")["rmse"].agg(["mean", "std"]).sort_values("mean")
    colors = ["#c0392b" if m == FINAL else "#4a6fa5" for m in agg.index]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.barh(agg.index, agg["mean"], xerr=agg["std"].fillna(0), color=colors, capsize=2)
    ax.set_xlabel("Test RMSE ($/MWh), mean ± std over seeds")
    ax.set_title("Price-recovery RMSE: proposed model vs. all benchmarks (contaminated TEST)")
    ax.invert_yaxis()
    fig.savefig("figures/fig3_main_benchmark.pdf")
    plt.close(fig)


def fig4_per_attack():
    df = pd.read_csv("results/raw/final_test.csv")
    diag = df[df["family"] == "diagnostic"].copy()
    diag["attack"] = diag["model"].str.extract(r"__on_(\w+)")
    diag["method"] = np.where(diag["model"].str.startswith("ACFP"), "ACFP-QR (proposed)", "Baseline (attack-agnostic)")
    agg = diag.groupby(["attack", "method"])["rmse"].mean().unstack()
    order = [a for a in ["FDI", "replay", "byzantine"] if a in agg.index]
    display_name = {"FDI": "FDI", "replay": "Replay", "byzantine": "Byzantine"}
    agg = agg.loc[order]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(agg))
    w = 0.35
    ax.bar(x - w / 2, agg["Baseline (attack-agnostic)"], w, label="Baseline (attack-agnostic)", color="#4a6fa5")
    ax.bar(x + w / 2, agg["ACFP-QR (proposed)"], w, label="ACFP-QR (proposed)", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels([display_name[a] for a in order])
    ax.set_ylabel("Test RMSE ($/MWh)")
    ax.set_title("Per-attack-type recovery accuracy (the mechanism's target experiment)")
    ax.legend()
    fig.savefig("figures/fig4_per_attack.pdf")
    plt.close(fig)


def fig5_robustness():
    df = pd.read_csv("results/raw/robustness.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0))
    sweeps = [("attack_strength", "Attack strength (τ multiplier)"),
              ("contamination_fraction", "Contamination fraction"),
              ("feature_noise_sigma", "Load feature noise σ (× feature std)"),
              ("missing_data_frac", "Missing lag-feature fraction"),
              ("gate_misclassification_rate", "Attack-type flag misclassification rate")]
    for ax, (sweep, xlabel) in zip(axes.flat, sweeps):
        sub = df[df["sweep"] == sweep]
        agg = sub.groupby(["value", "model"])["rmse"].mean().unstack()
        for model, style in [("Baseline", "--o"), ("ACFP-QR", "-s")]:
            if model in agg.columns:
                ax.plot(agg.index, agg[model], style, label=model, markersize=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test RMSE")
        ax.legend(fontsize=7)
    # unknown-attack-shape: single-point bar in the last panel
    ax = axes.flat[5]
    unk = df[df["sweep"] == "unknown_attack_shape"].groupby("model")["rmse"].mean()
    ax.bar(unk.index, unk.values, color=["#4a6fa5", "#c0392b"])
    ax.set_title("Out-of-taxonomy attack shape", fontsize=8)
    ax.set_ylabel("Test RMSE")
    fig.suptitle("Robustness sweeps: frozen final model vs. attack-agnostic baseline")
    fig.tight_layout()
    fig.savefig("figures/fig5_robustness.pdf")
    plt.close(fig)


def fig7_network_validation():
    summary = pd.read_csv("results/aggregate/network_validation_summary.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    x = np.arange(len(summary))
    w = 0.25
    ax.bar(x - w, summary["loss_pct_no_recovery"], w, label="No recovery", color="#7f7f7f")
    ax.bar(x, summary["loss_pct_baseline_recovery"], w, label="Baseline recovery", color="#4a6fa5")
    ax.bar(x + w, summary["loss_pct_acfpqr_recovery"], w, label="ACFP-QR recovery", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index)
    ax.set_ylabel("Converged cost inflation vs. true optimum (%)")
    ax.set_title("14-bus network-level dispatch cost, by attack type")
    ax.legend(fontsize=7)

    ax = axes[1]
    traj = pd.read_csv("results/raw/network_validation_trajectory.csv")
    for col, label, style in [("lambda_no_recovery", "No recovery", "--o"),
                               ("lambda_acfpqr_recovery", "ACFP-QR recovery", "-s"),
                               ("lambda_normal", "No attack (reference)", ":^")]:
        sub = traj.dropna(subset=[col])
        ax.plot(sub["iter"], sub[col], style, label=label, markersize=4)
    ax.set_xlabel("Dual-ascent iteration")
    ax.set_ylabel("Marginal price λ ($/MWh)")
    ax.set_title("Illustrative FDI-attack recovery trajectory")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/fig7_network_validation.pdf")
    plt.close(fig)


def fig8_backbone_generalization():
    df = pd.read_csv("results/raw/backbone_generalization.csv")
    sub = df[df["attack"] == "overall"]
    agg = sub.groupby("model")["rmse"].agg(["mean", "std"])
    order = ["CatBoost-alone", "CatBoost+Gate", "TCN-alone", "TCN+Gate"]
    agg = agg.loc[order]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    colors = ["#4a6fa5", "#c0392b", "#4a6fa5", "#c0392b"]
    hatches = ["", "", "//", "//"]
    bars = ax.bar(order, agg["mean"], yerr=agg["std"], capsize=3, color=colors)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.set_ylabel("Test RMSE ($/MWh)")
    ax.set_title("Attack-type gate generalizes across backbones")
    fig.tight_layout()
    fig.savefig("figures/fig8_backbone_generalization.pdf")
    plt.close(fig)


def fig6_pareto():
    df = pd.read_csv("results/raw/final_test.csv")
    df = df[df["family"].isin(["classical/boosting", "deep", "proposed"])]
    agg = df.groupby("model").agg(rmse=("rmse", "mean"), latency=("infer_time_s", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, r in agg.iterrows():
        is_final = r["model"] == FINAL
        ax.scatter(r["latency"] * 1000, r["rmse"], s=90 if is_final else 40,
                   color="#c0392b" if is_final else "#4a6fa5", zorder=3 if is_final else 2)
        ax.annotate(r["model"], (r["latency"] * 1000, r["rmse"]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency, full TEST set (ms, log scale)")
    ax.set_ylabel("Test RMSE ($/MWh)")
    ax.set_title("Performance-efficiency Pareto: proposed vs. all benchmarks")
    fig.savefig("figures/fig6_pareto.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig3_main_benchmark()
    fig4_per_attack()
    fig5_robustness()
    fig6_pareto()
    fig7_network_validation()
    fig8_backbone_generalization()
    print("figures written to figures/")
