"""
Candidate hybrid models (MODEL_CANDIDATES.md). Backbone = CatBoost (data-driven winner of
Stage-1 screening, results/raw/benchmark_screen.csv: lowest mean RMSE 5.02 across 3 seeds).
"""
import numpy as np
from catboost import CatBoostRegressor
import sys
sys.path.insert(0, "scripts") if "scripts" not in sys.path[0] else None
from ded_sim import Generator
from economic_impact import _FLEET


def make_backbone(seed, **kw):
    params = dict(iterations=500, depth=8, learning_rate=0.05, random_seed=seed, verbose=False)
    params.update(kw)
    return CatBoostRegressor(**params)


def water_fill(fleet, demand, prices):
    """Feasibility projection: exact power-balance correction, O(n) per hour, vectorized-ish."""
    outputs = np.array([[g.dispatch(p) for g in fleet] for p in prices])
    pmax = np.array([g.pmax for g in fleet])
    pmin = np.array([g.pmin for g in fleet])
    a = np.array([g.a for g in fleet])
    for row in range(outputs.shape[0]):
        p = outputs[row].copy()
        for _ in range(10):
            mismatch = demand[row] - p.sum()
            if abs(mismatch) < 1e-6:
                break
            free = (p < pmax - 1e-9) if mismatch > 0 else (p > pmin + 1e-9)
            if not free.any():
                break
            weight = 1.0 / (2 * a[free])
            add = mismatch * weight / weight.sum()
            p[free] = np.clip(p[free] + add, pmin[free], pmax[free])
        outputs[row] = p
    return outputs


class QuantileHeads:
    """[q_0.1, q_0.9] boosting quantile heads with a split-conformal calibration step (CQR,
    Romano et al. 2019): the raw quantile-regression interval is systematically under-covered
    whenever the evaluation distribution departs from the clean-only training distribution (e.g.
    under attack contamination); `calibrate()` corrects this using a held-out calibration set
    (never seen during quantile-head training or gate training) so that the reported [10,90]
    interval attains its nominal coverage rate on data resembling deployment conditions."""

    def __init__(self, seed):
        self.lo = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, loss_function="Quantile:alpha=0.1",
                                     random_seed=seed, verbose=False)
        self.hi = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, loss_function="Quantile:alpha=0.9",
                                     random_seed=seed, verbose=False)
        self.conformal_correction = 0.0

    def fit(self, X, y):
        self.lo.fit(X, y)
        self.hi.fit(X, y)
        return self

    def calibrate(self, X_cal, y_cal, alpha=0.2):
        lo, hi = self.lo.predict(X_cal), self.hi.predict(X_cal)
        scores = np.maximum(lo - y_cal, y_cal - hi)
        n = len(scores)
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self.conformal_correction = float(np.quantile(scores, q_level))
        return self

    def predict(self, X):
        return self.lo.predict(X) - self.conformal_correction, self.hi.predict(X) + self.conformal_correction


DEFAULT_GATE_PARAMS = dict(iterations=150, depth=4, learning_rate=0.1)
# Per-type capacity: type 3 (Byzantine) contaminates a wider, harder-to-fit multi-lag window
# (lags 1-24) than FDI (type 1, single-lag) or replay (type 2, 3-lag freeze), so it is given
# more depth/iterations and a Huber loss (robust to the large, low-probability residual outliers
# a gradual multi-lag ramp produces) rather than sharing one shallow configuration across types.
BYZANTINE_TYPE = 3
ENHANCED_GATE_PARAMS = {
    BYZANTINE_TYPE: dict(iterations=250, depth=6, learning_rate=0.08, loss_function="Huber:delta=3.0"),
}


class AttackGate:
    """Per-attack-type residual correctors (Candidate A component 2).

    `per_type_params` overrides DEFAULT_GATE_PARAMS for specific attack types (keyed by the
    ATTACK_NAMES integer code). `label_noise_rate` mixes a controlled fraction of
    other-type-labeled examples into each corrector's training set, so a corrector does not learn
    a correction that is only valid under a perfectly-accurate gate label -- a defense against
    upstream attack-type misclassification (evaluated in scripts/gate_robustness.py)."""

    def __init__(self, seed, n_types=4, per_type_params=None, label_noise_rate=0.0):
        per_type_params = per_type_params or {}
        self.seed = seed
        self.label_noise_rate = label_noise_rate
        self.correctors = {}
        for t in range(1, n_types):  # 0=clean has no corrector
            params = dict(DEFAULT_GATE_PARAMS)
            params.update(per_type_params.get(t, {}))
            params.setdefault("random_seed", seed)
            params.setdefault("verbose", False)
            self.correctors[t] = CatBoostRegressor(**params)

    def fit(self, X, resid, attack_label):
        rng = np.random.RandomState(self.seed)
        for t, model in self.correctors.items():
            mask = attack_label == t
            if self.label_noise_rate > 0:
                mask = mask.copy()
                other_idx = np.where(attack_label != t)[0]
                n_extra = int(self.label_noise_rate * mask.sum())
                if n_extra > 0 and len(other_idx) > 0:
                    extra_idx = rng.choice(other_idx, size=min(n_extra, len(other_idx)), replace=False)
                    mask[extra_idx] = True
            if mask.sum() >= 20:
                model.fit(X[mask], resid[mask])
        return self

    def predict(self, X, attack_label):
        out = np.zeros(len(X))
        for t, model in self.correctors.items():
            mask = attack_label == t
            if mask.any():
                out[mask] = model.predict(X[mask])
        return out


class CandidateA_ACFP_QR:
    """Full assembly: backbone + attack gate + quantile heads + feasibility projection."""
    name = "ACFP-QR (proposed)"

    def __init__(self, seed):
        self.seed = seed
        self.backbone = make_backbone(seed)
        self.gate = AttackGate(seed)
        self.quantiles = QuantileHeads(seed)

    def fit(self, X_clean, y_clean, X_contam=None, y_contam=None, attack_label_contam=None):
        self.backbone.fit(X_clean, y_clean)
        self.quantiles.fit(X_clean, y_clean)
        if X_contam is not None:
            resid = y_contam - self.backbone.predict(X_contam)
            self.gate.fit(X_contam, resid, attack_label_contam)
        return self

    def predict(self, X, attack_label=None):
        base = self.backbone.predict(X)
        if attack_label is not None:
            base = base + self.gate.predict(X, attack_label)
        return base

    def predict_with_quantiles(self, X, attack_label=None):
        pred = self.predict(X, attack_label)
        lo, hi = self.quantiles.predict(X)
        return pred, lo, hi

    def recover_dispatch(self, X, demand, attack_label=None, feasibility=True):
        pred = self.predict(X, attack_label)
        if feasibility:
            outputs = water_fill(_FLEET, demand, pred)
        else:
            outputs = np.array([[g.dispatch(p) for g in _FLEET] for p in pred])
        return pred, outputs


class CandidateB_GBM_ResGRU:
    """Backbone + temporal residual GRU corrector (Candidate B)."""
    name = "GBM+Res-GRU"

    def __init__(self, seed, feature_names):
        from models_regression import TabularGRURegressor
        self.seed = seed
        self.backbone = make_backbone(seed)
        self.residual_gru = TabularGRURegressor(feature_names, seed=seed, epochs=60, hidden=16)

    def fit(self, X, y, X_val=None, y_val=None):
        self.backbone.fit(X, y)
        resid = y - self.backbone.predict(X)
        if X_val is not None:
            resid_val = y_val - self.backbone.predict(X_val)
        else:
            resid_val = None
        self.residual_gru.fit(X, resid, X_val, resid_val)
        return self

    def predict(self, X, attack_label=None):
        return self.backbone.predict(X) + self.residual_gru.predict(X)


class CandidateD_QRGBM:
    """Ablation control: backbone + quantile heads only, no gate, no feasibility projection."""
    name = "QR-GBM (ablation control)"

    def __init__(self, seed):
        self.seed = seed
        self.backbone = make_backbone(seed)
        self.quantiles = QuantileHeads(seed)

    def fit(self, X_clean, y_clean, **kw):
        self.backbone.fit(X_clean, y_clean)
        self.quantiles.fit(X_clean, y_clean)
        return self

    def predict(self, X, attack_label=None):
        return self.backbone.predict(X)
