"""
Network-level (multi-generator, iterative dual-ascent) validation of ACFP-QR, added to close the
peer-review gap that the original submission evaluated the recovery model only as a one-shot
regression against held-out rows, never inside an actual DED iterative loop (EIC Weakness 4,
Domain-Reviewer Weakness 2, Methodology-Reviewer Weakness 5).

Builds a historically-driven demand/price series for the reproduced IEEE 14-bus system (Sec.
"14-bus DED reproduction") by (a) rescaling the real GEFCom2014 system-load series to the 14-bus
fleet's capacity range and (b) computing the EXACT market-clearing price for that demand via
bisection (`ded_sim.clearing_price`) against the fleet's own published cost coefficients -- a
self-consistent construction (unlike the source paper's own unpublished PJM-rescaling +
synthetic-commodity-price pipeline, see SOURCE_PAPER_AUDIT.md). A dedicated ACFP-QR instance is
then trained on this 14-bus-specific series and used to drive the online recovery mapping (Eq. 1)
across live dual-ascent iterations under FDI/replay/Byzantine attacks, exactly as it would be
deployed by the secure DED protocol.

Reports (a) an aggregate post-recovery cost-inflation statistic over many held-out hours per
attack type (baseline-recovery vs. ACFP-QR-recovery vs. no-recovery), and (b) one illustrative
iteration-by-iteration cost-convergence trajectory (analogous to the source paper's Fig. 5).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

from ded_sim import Generator, solve_ded, clearing_price, fdi_attack, replay_attack, byzantine_attack
from data_gefcom import load_raw, add_features
from candidates import AttackGate, ENHANCED_GATE_PARAMS
from reproduce_14bus import COEFS, PMAX

# best-fit H2 generator/coefficient assignment from reproduce_14bus.py (matches Table II closely)
H2_ORDER = [(0.01, 32), (0.04, 33), (0.01, 34), (0.05, 31), (0.02, 30)]


def build_14bus_fleet():
    return [Generator(f"G{i+1}", a, b, 0.0, PMAX[i]) for i, (a, b) in enumerate(H2_ORDER)]


def build_14bus_series():
    """Real GEFCom demand shape -> 14-bus capacity range -> exact self-consistent clearing price."""
    raw = load_raw()  # real system_load / zonal_load / price / timestamp, GEFCom2014
    fleet = build_14bus_fleet()
    total_cap = sum(g.pmax for g in fleet)
    lo, hi = np.percentile(raw["system_load"], [2, 98])
    frac = np.clip((raw["system_load"] - lo) / (hi - lo), 0.15, 0.90)
    demand = frac * total_cap * 0.85  # keep headroom below total capacity for feasibility

    df = raw.copy()
    df["system_load"] = demand
    df["zonal_load"] = demand  # single-node 14-bus system: no separate zone
    df["price"] = [clearing_price(fleet, d) for d in demand]
    feat = add_features(df)
    return feat, fleet


def train_14bus_acfpqr(feat, cols, seed=0):
    n = len(feat)
    n_bb = int(n * 0.72)
    n_gate_end = int(n * 0.90)
    fit_bb, fit_gate, holdout = feat.iloc[:n_bb], feat.iloc[n_bb:n_gate_end], feat.iloc[n_gate_end:]

    scaler = StandardScaler().fit(fit_bb[cols])
    Xbb, ybb = scaler.transform(fit_bb[cols]), fit_bb["price"].values

    backbone = CatBoostRegressor(depth=8, learning_rate=0.05, iterations=500, random_seed=seed, verbose=False)
    backbone.fit(Xbb, ybb)

    from attack_contamination import contaminate
    rng = np.random.RandomState(1000 + seed)
    gate_df, gate_labels = contaminate(fit_gate, cols, rng, frac=0.20)
    Xgate = scaler.transform(gate_df[cols])
    ygate = gate_df["price"].values
    resid = ygate - backbone.predict(Xgate)
    gate = AttackGate(seed, per_type_params=ENHANCED_GATE_PARAMS, label_noise_rate=0.15)
    gate.fit(Xgate, resid, gate_labels)

    baseline_backbone = CatBoostRegressor(depth=8, learning_rate=0.05, iterations=500, random_seed=seed, verbose=False)
    baseline_backbone.fit(Xbb, ybb)  # attack-agnostic reference (no gate)

    return backbone, gate, baseline_backbone, scaler, holdout


ATTACK_BUILDERS = {
    "FDI": lambda fleet: fdi_attack(target="G5", tau=6.0),
    "Replay": lambda fleet: replay_attack(targets=["G1", "G5"], freeze_iter=9),
    "Byzantine": lambda fleet: byzantine_attack(cheap=["G5"], expensive=["G3"], tau=6.0),
}
ATTACK_LABEL_CODE = {"FDI": 1, "Replay": 2, "Byzantine": 3}


def recovered_price(model_pack, row, cols, attack_type):
    backbone, gate, _, scaler, _ = model_pack
    x = scaler.transform(row[cols].values.reshape(1, -1))
    label = np.array([ATTACK_LABEL_CODE[attack_type]])
    pred = backbone.predict(x) + gate.predict(x, label)
    return float(pred[0])


def baseline_price(model_pack, row, cols):
    _, _, baseline_backbone, scaler, _ = model_pack
    x = scaler.transform(row[cols].values.reshape(1, -1))
    return float(baseline_backbone.predict(x)[0])


def run_iterative_recovery(fleet, demand, attack_name, recovered_lam, detect_iter=6, max_iter=200):
    """Runs the DED loop: attack active from iter 1; at `detect_iter` the affected supplier(s)
    switch from the corrupted signal to the fixed `recovered_lam` estimate for all remaining
    iterations (mirrors the crypto layer's deterministic detection + ML fallback, Sec. IV-C)."""
    attack_fn = ATTACK_BUILDERS[attack_name](fleet)
    targets = {"FDI": ["G5"], "Replay": ["G1", "G5"], "Byzantine": ["G5", "G3"]}[attack_name]

    def manipulate(k, lam, gens):
        if k < detect_iter:
            return attack_fn(k, lam, gens)
        return {t: recovered_lam for t in targets}

    alpha = 0.5 * 2.0 / sum(1.0 / (2 * g.a) for g in fleet)
    res = solve_ded(fleet, demand, alpha=alpha, lam0=20.0, tol=1e-4, max_iter=max_iter, manipulate=manipulate)
    return res


def normal_cumulative_cost(fleet, demand, n_iters):
    """Reference trajectory with no attack at all, same warm-start dynamics (lam0=20), same
    iteration count -- the correct baseline for transient cost-excess, since the dual-ascent
    warm-up itself (not the attack) accounts for most of the early-iteration cost being below
    the converged optimum's steady-state cost."""
    lam = 20.0
    alpha = 0.5 * 2.0 / sum(1.0 / (2 * g.a) for g in fleet)
    cum_cost = 0.0
    for k in range(1, n_iters + 1):
        outputs = {g.name: g.dispatch(lam) for g in fleet}
        cum_cost += sum(g.cost(outputs[g.name]) for g in fleet)
        lam = lam + alpha * (demand - sum(outputs.values()))
    return cum_cost


def transient_cumulative_cost(fleet, demand, attack_name, recovered_lam, detect_iter=6, window=15):
    """Cumulative dispatch cost over a fixed post-detection recovery window (the economically
    relevant quantity for a system re-dispatched every 5-15 minutes, Sec. I) -- as opposed to the
    final converged cost, which the coordinator's own dual-ascent feedback drives toward the true
    optimum regardless of small recovery-price errors (see the near-zero final-state loss reported
    above), and therefore does not discriminate between recovery models."""
    attack_fn = ATTACK_BUILDERS[attack_name](fleet)
    targets = {"FDI": ["G5"], "Replay": ["G1", "G5"], "Byzantine": ["G5", "G3"]}[attack_name]
    lam = 20.0
    alpha = 0.5 * 2.0 / sum(1.0 / (2 * g.a) for g in fleet)
    cum_cost = 0.0
    for k in range(1, detect_iter + window):
        if recovered_lam is None:
            seen = attack_fn(k, lam, fleet)
        elif k < detect_iter:
            seen = attack_fn(k, lam, fleet)
        else:
            seen = {t: recovered_lam for t in targets}
        outputs = {g.name: g.dispatch(seen[g.name] if (seen and g.name in seen) else lam) for g in fleet}
        cum_cost += sum(g.cost(outputs[g.name]) for g in fleet)
        total_p = sum(outputs.values())
        lam = lam + alpha * (demand - total_p)
    return cum_cost


def main():
    feat, fleet = build_14bus_series()
    exclude = {"timestamp", "price", "hour", "dow", "month"}
    cols = [c for c in feat.columns if c not in exclude]
    model_pack = train_14bus_acfpqr(feat, cols, seed=0)
    holdout = model_pack[-1]

    true_gen = build_14bus_fleet()
    rows = []
    n_hours = min(300, len(holdout))
    sample = holdout.iloc[:n_hours]
    for attack_name in ATTACK_BUILDERS:
        for i in range(n_hours):
            row = sample.iloc[i]
            D = row["system_load"]
            true_price = clearing_price(true_gen, D)
            true_cost = sum(g.cost(g.dispatch(true_price)) for g in true_gen)
            rec_acfp = recovered_price(model_pack, row, cols, attack_name)
            rec_base = baseline_price(model_pack, row, cols)

            res_acfp = run_iterative_recovery(true_gen, D, attack_name, rec_acfp)
            res_base = run_iterative_recovery(true_gen, D, attack_name, rec_base)
            res_noRec = solve_ded(true_gen, D, alpha=0.5 * 2.0 / sum(1 / (2 * g.a) for g in true_gen),
                                   lam0=20.0, tol=1e-4, max_iter=200,
                                   manipulate=ATTACK_BUILDERS[attack_name](true_gen))

            true_cum_cost = normal_cumulative_cost(true_gen, D, 20)  # no-attack reference, same warm-start dynamics
            cum_norec = transient_cumulative_cost(true_gen, D, attack_name, None)
            cum_base = transient_cumulative_cost(true_gen, D, attack_name, rec_base)
            cum_acfp = transient_cumulative_cost(true_gen, D, attack_name, rec_acfp)

            rows.append({"attack": attack_name, "hour": i, "demand": D,
                         "true_cost": true_cost,
                         "cost_no_recovery": res_noRec.total_cost,
                         "cost_baseline_recovery": res_base.total_cost,
                         "cost_acfpqr_recovery": res_acfp.total_cost,
                         "iters_no_recovery": res_noRec.iterations,
                         "iters_acfpqr_recovery": res_acfp.iterations,
                         "true_cum_cost_20iter": true_cum_cost,
                         "cum_cost_no_recovery": cum_norec,
                         "cum_cost_baseline_recovery": cum_base,
                         "cum_cost_acfpqr_recovery": cum_acfp})
        print(f"{attack_name} done")

    df = pd.DataFrame(rows)
    for col in ["cost_no_recovery", "cost_baseline_recovery", "cost_acfpqr_recovery"]:
        df[col.replace("cost", "loss_pct")] = 100 * (df[col] - df["true_cost"]) / df["true_cost"]
    for col in ["cum_cost_no_recovery", "cum_cost_baseline_recovery", "cum_cost_acfpqr_recovery"]:
        df[col.replace("cum_cost", "transient_loss_pct")] = 100 * (df[col] - df["true_cum_cost_20iter"]) / df["true_cum_cost_20iter"]
    df.to_csv("results/raw/network_validation.csv", index=False)

    summary = df.groupby("attack")[[
        "loss_pct_no_recovery", "loss_pct_baseline_recovery", "loss_pct_acfpqr_recovery",
        "transient_loss_pct_no_recovery", "transient_loss_pct_baseline_recovery", "transient_loss_pct_acfpqr_recovery",
    ]].mean()
    summary.to_csv("results/aggregate/network_validation_summary.csv")
    print(summary)

    # illustrative single-hour trajectory for the FDI case (Fig.)
    demo_row = sample.iloc[0]
    D = demo_row["system_load"]
    rec_acfp = recovered_price(model_pack, demo_row, cols, "FDI")
    rec_base = baseline_price(model_pack, demo_row, cols)
    traj_norec = solve_ded(true_gen, D, alpha=0.5 * 2.0 / sum(1 / (2 * g.a) for g in true_gen), lam0=20.0,
                            tol=1e-4, max_iter=60, manipulate=ATTACK_BUILDERS["FDI"](true_gen))
    traj_base = run_iterative_recovery(true_gen, D, "FDI", rec_base, max_iter=60)
    traj_acfp = run_iterative_recovery(true_gen, D, "FDI", rec_acfp, max_iter=60)
    traj_normal = solve_ded(true_gen, D, alpha=0.5 * 2.0 / sum(1 / (2 * g.a) for g in true_gen), lam0=20.0,
                             tol=1e-4, max_iter=60)
    traj_df = pd.DataFrame({
        "iter": range(len(traj_norec.lam_trajectory)),
        "lambda_no_recovery": pd.Series(traj_norec.lam_trajectory),
    })
    traj_df2 = pd.DataFrame({"iter": range(len(traj_base.lam_trajectory)), "lambda_baseline_recovery": pd.Series(traj_base.lam_trajectory)})
    traj_df3 = pd.DataFrame({"iter": range(len(traj_acfp.lam_trajectory)), "lambda_acfpqr_recovery": pd.Series(traj_acfp.lam_trajectory)})
    traj_df4 = pd.DataFrame({"iter": range(len(traj_normal.lam_trajectory)), "lambda_normal": pd.Series(traj_normal.lam_trajectory)})
    merged = traj_df.merge(traj_df2, on="iter", how="outer").merge(traj_df3, on="iter", how="outer").merge(traj_df4, on="iter", how="outer")
    merged.to_csv("results/raw/network_validation_trajectory.csv", index=False)
    print("Demo hour demand:", D, "true price:", clearing_price(true_gen, D))


if __name__ == "__main__":
    main()
