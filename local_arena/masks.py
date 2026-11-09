"""Mask utilities: official random k-regular connected mask + validity check."""

import numpy as np


def is_connected(M):
    n, m = M.shape
    if not M.any():
        return False
    seen_r = np.zeros(n, bool)
    seen_c = np.zeros(m, bool)
    stack = [("r", int(np.flatnonzero(M.any(axis=1))[0]))]
    seen_r[stack[0][1]] = True
    while stack:
        kind, idx = stack.pop()
        if kind == "r":
            for j in np.flatnonzero(M[idx]):
                if not seen_c[j]:
                    seen_c[j] = True
                    stack.append(("c", int(j)))
        else:
            for i in np.flatnonzero(M[:, idx]):
                if not seen_r[i]:
                    seen_r[i] = True
                    stack.append(("r", int(i)))
    rows_used = M.any(axis=1)
    cols_used = M.any(axis=0)
    return bool((seen_r >= rows_used).all() and (seen_c >= cols_used).all())


def is_valid_mask(M, k):
    M = np.asarray(M)
    if M.dtype != bool:
        return False, "dtype is not bool"
    n, m = M.shape
    if not (M.sum(axis=1) == k).all():
        return False, "row sums != k"
    if not (M.sum(axis=0) == k).all():
        return False, "col sums != k"
    if not is_connected(M):
        return False, "bipartite graph not connected"
    return True, "ok"


def official_mask(n, m, k, rng):
    """Random k-regular connected mask: union of k disjoint random matchings."""
    assert n == m, "official mask assumes square instances"
    for _ in range(50):
        M = np.zeros((n, m), dtype=bool)
        ok = True
        for _t in range(k):
            placed = False
            for _try in range(200):
                perm = rng.permutation(n)
                if not M[np.arange(n), perm].any():
                    M[np.arange(n), perm] = True
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok and is_connected(M):
            return M
    # deterministic fallback: circulant on random orderings (always valid)
    orow = rng.permutation(n)
    ocol = rng.permutation(m)
    M = np.zeros((n, m), dtype=bool)
    p = np.arange(n)
    for t in range(k):
        M[orow[p], ocol[(p + t) % m]] = True
    return M
