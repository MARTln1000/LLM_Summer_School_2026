"""Local replica of the Matrix Arena instance generator (PDF sections 4, 8).

Y* = w_bil * X W Z^T + w_lat * U V^T + w_nl * sum_l a_l tanh((X p_l)(Z q_l)^T)
     + noise,  each component normalized to unit std before weighting, and the
final Y* normalized to zero mean / unit std.

This is only a faithful-in-structure replica for local testing; the real
grader uses secret seeds and possibly different scalings.
"""

import numpy as np

BUDGETS = {
    "small":  dict(name="small",  n=48,  m=48,  d=12, r=4,  r_nl=2, k=10,
                   solve_timeout=1.0),
    "medium": dict(name="medium", n=96,  m=96,  d=24, r=8,  r_nl=4, k=12,
                   solve_timeout=5.0),
    "large":  dict(name="large",  n=160, m=160, d=32, r=12, r_nl=6, k=16,
                   solve_timeout=15.0),
}

# (name, w_bil, w_lat, w_nl, sigma)
REGIMES = [
    ("bilinear",  0.85, 0.10, 0.05, 0.05),
    ("lowrank",   0.10, 0.85, 0.05, 0.05),
    ("mixed",     0.45, 0.45, 0.10, 0.05),
    ("nonlinear", 0.25, 0.25, 0.50, 0.10),
    ("noisy",     0.60, 0.30, 0.10, 0.20),
    ("spiky",     0.40, 0.40, 0.20, 0.05),
]


def _unit(A):
    return A / (A.std() + 1e-12)


def generate_instance(seed, budget):
    b = BUDGETS[budget] if isinstance(budget, str) else budget
    n, m, d, r, r_nl = b["n"], b["m"], b["d"], b["r"], b["r_nl"]
    rng = np.random.default_rng(seed)

    regime = REGIMES[int(rng.integers(len(REGIMES)))]
    name, w_bil, w_lat, w_nl, sigma = regime

    X = rng.standard_normal((n, d))
    Z = rng.standard_normal((m, d))
    if name == "spiky":
        ri = rng.choice(n, size=max(1, n // 8), replace=False)
        ci = rng.choice(m, size=max(1, m // 8), replace=False)
        X[ri] *= 3.0
        Z[ci] *= 3.0

    W = rng.standard_normal((d, d)) / np.sqrt(d)
    bil = _unit(X @ W @ Z.T)

    U = rng.standard_normal((n, r))
    V = rng.standard_normal((m, r))
    lat = _unit(U @ V.T)

    P = rng.standard_normal((d, r_nl)) / np.sqrt(d)
    Q = rng.standard_normal((d, r_nl)) / np.sqrt(d)
    a = rng.standard_normal(r_nl)
    nl = np.zeros((n, m))
    for l in range(r_nl):
        nl += a[l] * np.tanh(np.outer(X @ P[:, l], Z @ Q[:, l]))
    nl = _unit(nl)

    Y = w_bil * bil + w_lat * lat + w_nl * nl
    Y += sigma * rng.standard_normal((n, m))
    Y = (Y - Y.mean()) / (Y.std() + 1e-12)
    return X, Z, Y, name
