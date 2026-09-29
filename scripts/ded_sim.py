"""
Distributed Economic Dispatch (DED) simulator reproducing Algorithm 1 / Eqs. (9)-(11)
of the source paper, plus FDI / Replay / Byzantine deception-attack injection per Sec. III.

This is a from-scratch, general-purpose reproduction (works for any generator list), used to:
  (a) reproduce the IEEE 14-bus case studies (Table II, Sec. VI-A);
  (b) build the IEEE 118-bus scenario generator for the ML-recovery economic-impact evaluation
      (Sec. VI-B style experiments), used later by the candidate/benchmark screening pipeline.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Generator:
    name: str
    a: float          # quadratic cost coeff
    b: float          # linear cost coeff
    pmin: float
    pmax: float

    def cost(self, p: float) -> float:
        return self.a * p ** 2 + self.b * p

    def dispatch(self, lam: float) -> float:
        p = (lam - self.b) / (2 * self.a)
        return float(np.clip(p, self.pmin, self.pmax))


@dataclass
class DEDResult:
    iterations: int
    converged: bool
    lam_trajectory: list
    price: float
    outputs: dict
    total_cost: float


def clearing_price(generators: list[Generator], demand: float) -> float:
    """Exact market-clearing lambda via bisection (ground truth, independent of step size)."""
    lo, hi = -1000.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        s = sum(g.dispatch(mid) for g in generators)
        if s < demand:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def solve_ded(
    generators: list[Generator],
    demand: float,
    alpha: float = 0.05,
    lam0: float = 20.0,
    tol: float = 1e-3,
    max_iter: int = 500,
    manipulate=None,
) -> DEDResult:
    """
    Reproduces Algorithm 1 (dual ascent). `manipulate(k, lam, generators) -> dict{name: lam_seen}`
    optionally returns a per-supplier price signal that overrides the broadcast lam for that
    supplier at iteration k (used to inject FDI / Replay / Byzantine deception).
    """
    lam = lam0
    lam_hist = [lam]
    lam_seen_hist = {g.name: [lam] for g in generators}
    for k in range(1, max_iter + 1):
        seen = manipulate(k, lam, generators) if manipulate is not None else None
        outputs = {}
        for g in generators:
            lam_i = seen[g.name] if (seen is not None and g.name in seen) else lam
            outputs[g.name] = g.dispatch(lam_i)
            lam_seen_hist[g.name].append(lam_i)
        total_p = sum(outputs.values())
        mismatch = demand - total_p
        lam_new = lam + alpha * mismatch
        lam_hist.append(lam_new)
        converged = abs(lam_new - lam) < tol and abs(mismatch) < tol
        lam = lam_new
        if converged:
            break
    total_cost = sum(g.cost(outputs[g.name]) for g in generators)
    return DEDResult(
        iterations=k,
        converged=converged,
        lam_trajectory=lam_hist,
        price=lam,
        outputs=outputs,
        total_cost=total_cost,
    )


# ---------------------------------------------------------------------------
# Deception-attack strategies (Sec. III of the source paper)
# ---------------------------------------------------------------------------

def fdi_attack(target: str, tau: float, start_iter: int = 1):
    """Persistent FDI: victim always sees lam - tau (attacker under-states price to a cheap
    competitor to suppress its output), all others see the true lam."""
    def _m(k, lam, gens):
        if k < start_iter:
            return None
        return {target: lam - tau}
    return _m


def replay_attack(targets: list[str], freeze_iter: int = 3):
    """Replay: the attacker records lam at iteration `freeze_iter` and thereafter replays that
    single frozen (stale) value to the targeted suppliers forever, while all other suppliers keep
    receiving the fresh, correctly-updated lam (Eq. 23 of the source paper, S = targets). This
    creates a persistent dispatch bias for the targets rather than a merely-delayed one, matching
    the paper's description of the attack increasing cost while the coordinator's own lam still
    (eventually) converges."""
    state = {"frozen": None}

    def _m(k, lam, gens):
        if k == freeze_iter:
            state["frozen"] = lam
        if state["frozen"] is None:
            return None
        return {t: state["frozen"] for t in targets}
    return _m


def byzantine_attack(cheap: list[str], expensive: list[str], tau: float):
    """Byzantine coordinator: sends lam-tau to cheap units (suppress), lam+tau to expensive
    units (encourage over-production), true lam to mid-cost units."""
    def _m(k, lam, gens):
        seen = {}
        for name in cheap:
            seen[name] = lam - tau
        for name in expensive:
            seen[name] = lam + tau
        return seen
    return _m


def compose_attacks(*attacks):
    def _m(k, lam, gens):
        seen = {}
        for atk in attacks:
            r = atk(k, lam, gens)
            if r:
                seen.update(r)
        return seen or None
    return _m
