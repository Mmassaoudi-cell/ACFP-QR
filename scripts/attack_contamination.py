"""
Synthetic contamination of the *recent price-lag history features* used by the ML recovery
model, calibrated to mimic the characteristic signature of each deception attack described in
Sec. III of the source paper (this is an explicit, declared modeling assumption — see
MODEL_CANDIDATES.md Candidate A "expected weakness" — no real deception-attack telemetry exists
for a public price-forecasting dataset). The *target* y (true price) is never touched: this
simulates the recovery model being invoked with a corrupted view of recent history while the
quantity it must predict remains the true, uncorrupted price.

Attack signatures (documented, not tuned to favor any candidate):
  - FDI:       abrupt single-step offset to price_lag1 only (fast to inject, fast to detect —
               Table II shows FDI adds only ~4 extra iterations over Normal).
  - Replay:    price_lag1..lag3 frozen to a single stale repeated value (characteristic
               near-zero short-term variance signature of replaying one old reading).
  - Byzantine: gradual ramp spread across price_lag1..lag24 (deliberately slow/evasive per
               Sec. III-3: "the attacker gradually ramps the manipulation ... to remain stealthy").
ATTACK_TYPES = {0: clean, 1: FDI, 2: replay, 3: byzantine}
"""
import numpy as np

SHORT_LAGS = ["price_lag1", "price_lag2", "price_lag3"]
LONG_LAGS = ["price_lag1", "price_lag2", "price_lag3", "price_lag24", "price_lag48", "price_lag168"]
ATTACK_NAMES = {0: "clean", 1: "FDI", 2: "replay", 3: "byzantine"}


def contaminate(df, cols, rng, frac=0.20, strength=1.0):
    """Returns (df_contaminated, attack_label array aligned to df.index)."""
    df = df.copy()
    n = len(df)
    labels = np.zeros(n, dtype=int)
    n_attacked = int(n * frac)
    attacked_idx = rng.choice(n, size=n_attacked, replace=False)
    thirds = np.array_split(rng.permutation(attacked_idx), 3)
    price_std_local = df["price"].rolling(168, min_periods=24).std().bfill().values

    # FDI: abrupt offset on price_lag1
    idx = thirds[0]
    labels[idx] = 1
    sign = rng.choice([-1, 1], size=len(idx))
    offset = sign * strength * (0.8 + 0.4 * rng.random(len(idx))) * price_std_local[idx]
    df.loc[df.index[idx], "price_lag1"] = df.loc[df.index[idx], "price_lag1"].values + offset

    # Replay: freeze lag1..lag3 to a single stale value (the lag24 reading, i.e. "yesterday")
    idx = thirds[1]
    labels[idx] = 2
    stale = df.loc[df.index[idx], "price_lag24"].values
    for c in SHORT_LAGS:
        df.loc[df.index[idx], c] = stale

    # Byzantine: gradual ramp, larger at lag1, tapering to ~0 by lag24
    idx = thirds[2]
    labels[idx] = 3
    ramp_sign = rng.choice([-1, 1], size=len(idx))
    for lag, weight in zip(["price_lag1", "price_lag2", "price_lag3", "price_lag24"], [1.0, 0.8, 0.6, 0.15]):
        offset = ramp_sign * strength * weight * 0.6 * price_std_local[idx]
        df.loc[df.index[idx], lag] = df.loc[df.index[idx], lag].values + offset

    return df, labels


def contaminate_unknown_shape(df, cols, rng, frac=0.20, strength=1.0):
    """An OUT-OF-TAXONOMY contamination pattern -- an oscillating (alternating-sign) multi-lag
    perturbation whose statistical shape matches none of the three declared signatures -- never
    used to train any corrector. The cryptographic layer's failure-mode classification is a
    protocol-level fact (which check failed: signature, counter, or threshold-aggregate) that is
    independent of the price-manipulation shape an attacker chooses, so this novel-shaped attack
    is still deterministically routed to one of the three known gate branches (assigned uniformly
    at random here, standing in for "whichever protocol check this particular attack happens to
    trip"); what is actually being tested is whether that branch's corrector -- trained on a
    different typical shape for its type -- still helps or instead hurts when the true underlying
    perturbation does not match what it was trained to characterize."""
    df = df.copy()
    n = len(df)
    n_attacked = int(n * frac)
    attacked_idx = rng.choice(n, size=n_attacked, replace=False)
    price_std_local = df["price"].rolling(168, min_periods=24).std().bfill().values
    lags = ["price_lag1", "price_lag2", "price_lag3", "price_lag24"]
    sign = rng.choice([-1, 1], size=len(attacked_idx))
    for j, lag in enumerate(lags):
        osc = (-1) ** j
        offset = sign * osc * strength * 0.5 * price_std_local[attacked_idx]
        df.loc[df.index[attacked_idx], lag] = df.loc[df.index[attacked_idx], lag].values + offset
    labels = np.zeros(n, dtype=int)
    labels[attacked_idx] = rng.choice([1, 2, 3], size=len(attacked_idx))
    return df, labels
