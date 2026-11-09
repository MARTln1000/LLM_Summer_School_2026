"""Compare adversarial mask designs by how much they hurt solvers.

For each design, both my_agent and the hybrid baseline solve the same
instances under that mask; higher NRMSE than under the official random mask
means the design is genuinely adversarial.

Usage: .venv/bin/python attack_eval.py [--seeds 12] [--budgets small,medium]
"""

import argparse
from collections import defaultdict

import numpy as np

from local_arena.baselines import HybridAgent
from local_arena.generator import BUDGETS, generate_instance
from local_arena.grading import derived_seed, load_agent, solve_instance
from local_arena.masks import is_valid_mask, official_mask


def band_mask(scores_r, scores_c, k, anti):
    n, m = scores_r.size, scores_c.size
    order_r = np.argsort(scores_r, kind="stable")
    order_c = np.argsort(scores_c, kind="stable")
    mask = np.zeros((n, m), dtype=bool)
    p = np.arange(n)
    for t in range(k):
        q = (m - 1 - p + t) % m if anti else (p + t) % m
        mask[order_r[p], order_c[q]] = True
    return mask


def leverage(A):
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    keep = s > s[0] * 1e-10
    return (U[:, keep] ** 2).sum(axis=1)


def pc1(A):
    U, s, _ = np.linalg.svd(A - A.mean(axis=0), full_matrices=False)
    return U[:, 0] * s[0]


def block_mask(scores_r, scores_c, k):
    """Feature-clustered blocks, circulant inside each block, chained by
    degree-preserving edge swaps so the graph stays (barely) connected."""
    n, m = scores_r.size, scores_c.size
    assert n == m
    order_r = np.argsort(scores_r, kind="stable")
    order_c = np.argsort(scores_c, kind="stable")
    nb = max(1, n // (k + 1))
    bounds = [round(b * n / nb) for b in range(nb + 1)]
    S = np.zeros((n, m), dtype=bool)
    for b in range(nb):
        s, e = bounds[b], bounds[b + 1]
        L = e - s
        p = np.arange(L)
        for t in range(k):
            S[s + p, s + (p + t) % L] = True
    for b in range(nb - 1):
        i = bounds[b + 1] - 1  # last row of block b; (i, i) is its t=0 edge
        j = bounds[b + 1]      # first row of block b+1
        S[i, i] = S[j, j] = False
        S[i, j] = S[j, i] = True
    M = np.zeros((n, m), dtype=bool)
    pr, pc = np.nonzero(S)
    M[order_r[pr], order_c[pc]] = True
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--budgets", default="small,medium")
    args = ap.parse_args()

    me = load_agent("my_agent.py")
    solvers = {"my_agent": me, "hybrid": HybridAgent()}
    losses = defaultdict(list)

    for bname in args.budgets.split(","):
        b = BUDGETS[bname]
        k = b["k"]
        for s in range(args.seeds):
            X, Z, Y, regime = generate_instance(s, b)
            designs = {
                "official": official_mask(
                    b["n"], b["m"], k,
                    np.random.default_rng(derived_seed(s, "off"))),
                "anti_lev": band_mask(leverage(X), leverage(Z), k, True),
                "diag_lev": band_mask(leverage(X), leverage(Z), k, False),
                "anti_pc1": band_mask(pc1(X), pc1(Z), k, True),
                "diag_pc1": band_mask(pc1(X), pc1(Z), k, False),
                "block_pc1": block_mask(pc1(X), pc1(Z), k),
                "block_lev": block_mask(leverage(X), leverage(Z), k),
                "block_rand": block_mask(
                    np.random.default_rng(derived_seed(s, "br")).permutation(
                        float(1) * np.arange(b["n"])),
                    np.random.default_rng(derived_seed(s, "bc")).permutation(
                        float(1) * np.arange(b["m"])), k),
            }
            for dname, Mk in designs.items():
                ok, why = is_valid_mask(Mk, k)
                assert ok, f"{dname}: {why}"
                for sname, agent in solvers.items():
                    loss, _, status = solve_instance(
                        agent, X, Z, Y, Mk, b, derived_seed(s, sname))
                    losses[(dname, sname)].append(min(loss, 10.0))

    print(f"\n{'design':<10} {'vs my_agent':>12} {'vs hybrid':>12}"
          f"   (mean NRMSE inflicted; higher = stronger attack)")
    print("-" * 60)
    for dname in ["official", "anti_lev", "diag_lev", "anti_pc1", "diag_pc1",
                  "block_pc1", "block_lev", "block_rand"]:
        a = losses[(dname, "my_agent")]
        h = losses[(dname, "hybrid")]
        print(f"{dname:<10} {sum(a)/len(a):>12.3f} {sum(h)/len(h):>12.3f}")


if __name__ == "__main__":
    main()
