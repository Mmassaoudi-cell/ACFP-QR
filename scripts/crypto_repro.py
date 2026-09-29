"""
Real (not simulated) reproduction of the cryptographic-layer performance claims:
  - ECSM, ECDSA sign, ECDSA verify latency (Sec. VI-C) using secp256k1 via the `ecdsa` library.
  - Pairwise ECDH + HMAC-SHA256 PRF mask derivation (Sec. II-C4) using `cryptography`.
  - Threshold Schnorr partial-signature + aggregation (Eqs. 13-16, 31-34) implemented directly
    over secp256k1 group operations exposed by `ecdsa`.
  - Per-iteration message-count comparison (Fig. 6): proposed 4n vs. coordinator+P2P 2n+n(n-1)
    vs. fully decentralized 2n(n-1).
"""
import json
import time
import hashlib
import hmac as hmac_lib
import statistics as stats

from ecdsa import SigningKey, SECP256k1, ellipticcurve
from ecdsa.util import sigencode_string, sigdecode_string

N_TRIALS = 500
CURVE = SECP256k1
ORDER = CURVE.order
G = CURVE.generator


def time_it(fn, n=N_TRIALS):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)  # ms
    return {"mean_ms": stats.mean(ts), "median_ms": stats.median(ts), "std_ms": stats.pstdev(ts)}


def bench_ecsm():
    sk = SigningKey.generate(curve=CURVE)
    d = sk.privkey.secret_multiplier
    return time_it(lambda: d * G)


def bench_ecdsa_sign():
    sk = SigningKey.generate(curve=CURVE)
    msg = b"DED-price-broadcast-iteration-42"
    return time_it(lambda: sk.sign(msg, sigencode=sigencode_string))


def bench_ecdsa_verify():
    sk = SigningKey.generate(curve=CURVE)
    vk = sk.get_verifying_key()
    msg = b"DED-price-broadcast-iteration-42"
    sig = sk.sign(msg, sigencode=sigencode_string)
    return time_it(lambda: vk.verify(sig, msg, sigdecode=sigdecode_string))


def ecdh_prf_mask(d_i: int, Q_j, label: bytes, modulus: int) -> int:
    """Sec. II-C4: shared secret via ECDH, PRF = HMAC-SHA256 keyed by the shared x-coordinate."""
    shared = d_i * Q_j
    key = shared.x().to_bytes(32, "big")
    digest = hmac_lib.new(key, label, hashlib.sha256).digest()
    return int.from_bytes(digest, "big") % modulus


def bench_pairwise_masking(n_neighbors=3):
    sk = SigningKey.generate(curve=CURVE)
    d_i = sk.privkey.secret_multiplier
    peers = [SigningKey.generate(curve=CURVE).get_verifying_key().pubkey.point for _ in range(n_neighbors)]
    M = 2 ** 32

    def run():
        for j, Qj in enumerate(peers):
            ecdh_prf_mask(d_i, Qj, f"edge-{j}".encode(), M)
    return time_it(run)


def schnorr_partial_and_aggregate(T=3, n=5):
    """Threshold Schnorr partial-signature generation + aggregation (Eqs. 31-34), real group ops."""
    keys = [SigningKey.generate(curve=CURVE).privkey.secret_multiplier for _ in range(T)]
    msg = b"coordinator-message-iteration-42"
    C = int.from_bytes(hashlib.sha256(msg).digest(), "big") % ORDER

    def run():
        nonces = [SigningKey.generate(curve=CURVE).privkey.secret_multiplier for _ in range(T)]
        partials = [(nonces[i] + C * keys[i]) % ORDER for i in range(T)]
        s = sum(partials) % ORDER
        R = sum((nonces[i] * G for i in range(T)), ellipticcurve.INFINITY)
        return s, R
    return time_it(run, n=100)


def message_counts(n_values):
    out = []
    for n in n_values:
        out.append({
            "n": n,
            "proposed_4n": 4 * n,
            "coordinator_p2p_2n_plus_n(n-1)": 2 * n + n * (n - 1),
            "fully_decentralized_2n(n-1)": 2 * n * (n - 1),
        })
    return out


def main():
    results = {
        "ecsm": bench_ecsm(),
        "ecdsa_sign": bench_ecdsa_sign(),
        "ecdsa_verify": bench_ecdsa_verify(),
        "pairwise_masking_3neighbors": bench_pairwise_masking(3),
        "threshold_schnorr_partial_agg_T3": schnorr_partial_and_aggregate(3, 5),
        "published_reference_ms": {"ecsm": 0.49, "ecdsa_sign": 0.50, "ecdsa_verify": 0.45,
                                    "hardware": "AMD Ryzen 7 4700U @ 2.0GHz, 16GB RAM (paper)"},
        "message_counts": message_counts([5, 10, 20, 30, 50, 70, 90, 100]),
    }
    with open("results/raw/crypto_repro.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
