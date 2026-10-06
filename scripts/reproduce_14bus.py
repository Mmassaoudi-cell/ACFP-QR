"""
Reproduce the IEEE 14-bus case study (Sec. VI-A, Table II) of the source paper.

The paper states coefficients "{ (0.01,32), (0.05,31), (0.04,33), (0.01,34), (0.02,30) } for
G1-G5" but does not state the demand D. We test two hypotheses:
  (H1) literal order G1..G5 = as listed;
  (H2) best-fit permutation + demand that minimizes squared error to the reported Normal-case
       outputs (109.25, 14.81, 3.08, 31.85, 100.00) MW and price ($34.19/MWh).
This directly implements the REPRODUCTION_REPORT.md requirement to document assumption-driven
differences rather than silently forcing agreement.
"""
import json
import itertools
import numpy as np
from ded_sim import Generator, solve_ded, fdi_attack, replay_attack, clearing_price

COEFS = [(0.01, 32), (0.05, 31), (0.04, 33), (0.01, 34), (0.02, 30)]
PMAX = [332.4, 140, 100, 100, 100]
REPORTED_NORMAL_P = np.array([109.25, 14.81, 3.08, 31.85, 100.00])
REPORTED_NORMAL_PRICE = 34.19
REPORTED_NORMAL_COST = 8456.25
PUBLISHED = {
    "Normal": {"price": 34.19, "cost": 8456.25},
    "FDI": {"price": None, "cost": 8730.83},
    "Replay": {"price": None, "cost": 8600.35},
}


def make_gens(order, c_terms=None):
    c_terms = c_terms or [0] * 5
    return [Generator(f"G{i+1}", a, b, 0.0, PMAX[i]) for i, (a, b) in enumerate(order)]


def fit_best_permutation():
    best = None
    for perm in itertools.permutations(COEFS):
        gens = make_gens(perm)
        # closed form: for a given lambda, P_i(lambda) as in ded_sim; grid-search lambda and D
        for lam in np.arange(30.0, 40.0, 0.001):
            outs = np.array([g.dispatch(lam) for g in gens])
            err = np.sum((outs - REPORTED_NORMAL_P) ** 2)
            if best is None or err < best[0]:
                best = (err, perm, lam, outs)
    return best


def stable_alpha(gens, safety=0.5):
    slope = sum(1.0 / (2 * g.a) for g in gens)
    return safety * 2.0 / slope


def run_scenario(gens, demand, manipulate=None, alpha=None, lam0=20.0):
    if alpha is None:
        alpha = stable_alpha(gens)
    res = solve_ded(gens, demand, alpha=alpha, lam0=lam0, manipulate=manipulate, max_iter=2000, tol=1e-4)
    return res


def main():
    results = {}

    # --- H1: literal coefficient order, demand inferred from reported outputs sum ---
    gens_h1 = make_gens(COEFS)
    demand = float(REPORTED_NORMAL_P.sum())
    normal_h1 = run_scenario(gens_h1, demand)
    results["H1_literal_order"] = {
        "demand_assumed_MW": demand,
        "price_dual_ascent": normal_h1.price,
        "price_exact_bisection": clearing_price(gens_h1, demand),
        "converged": normal_h1.converged,
        "outputs": normal_h1.outputs,
        "total_cost_marginal_only": normal_h1.total_cost,
        "iterations": normal_h1.iterations,
    }

    # --- H2: best-fit permutation ---
    err, perm, lam_fit, outs_fit = fit_best_permutation()
    gens_h2 = make_gens(perm)
    normal_h2 = run_scenario(gens_h2, demand)
    results["H2_best_fit_permutation"] = {
        "assignment(a,b)_G1..G5": perm,
        "fit_sq_error_MW2": err,
        "demand_assumed_MW": demand,
        "price_dual_ascent": normal_h2.price,
        "price_exact_bisection": clearing_price(gens_h2, demand),
        "converged": normal_h2.converged,
        "outputs": normal_h2.outputs,
        "total_cost_marginal_only": normal_h2.total_cost,
        "iterations": normal_h2.iterations,
    }

    # attacks under H2 (best-fit) assignment, since it matches Table II much more closely
    fdi_res = run_scenario(gens_h2, demand, manipulate=fdi_attack(target="G5", tau=6.0))
    replay_res = run_scenario(gens_h2, demand, manipulate=replay_attack(targets=["G1", "G5"], freeze_iter=9))

    results["FDI_H2"] = {
        "price": fdi_res.price, "converged": fdi_res.converged, "outputs": fdi_res.outputs,
        "total_cost_marginal_only": fdi_res.total_cost, "iterations": fdi_res.iterations,
    }
    results["Replay_H2"] = {
        "price": replay_res.price, "converged": replay_res.converged, "outputs": replay_res.outputs,
        "total_cost_marginal_only": replay_res.total_cost, "iterations": replay_res.iterations,
    }

    with open("results/raw/reproduce_14bus.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
