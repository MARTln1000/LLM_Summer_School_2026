"""Reference baselines from the PDF (section 13), reimplemented locally."""

import numpy as np

from .masks import official_mask


def _prep(Y_obs, mask):
    rows, cols = np.nonzero(mask)
    y = Y_obs[rows, cols]
    mu = float(y.mean()) if y.size else 0.0
    return rows, cols, y, mu


def _random_attack(X, Z, k, seed):
    rng = np.random.default_rng(seed)
    return official_mask(X.shape[0], Z.shape[0], k, rng)


class MeanAgent:
    def solve(self, X, Z, Y_obs, mask, budget, seed):
        _, _, _, mu = _prep(Y_obs, mask)
        return np.full(Y_obs.shape, mu)

    def attack(self, X, Z, k, budget, seed):
        return _random_attack(X, Z, k, seed)


class RidgeAgent:
    """Ridge regression on [x_i, z_j, x_i * z_j]."""

    def solve(self, X, Z, Y_obs, mask, budget, seed):
        rows, cols, y, mu = _prep(Y_obs, mask)
        sd = float(y.std()) + 1e-9
        yc = (y - mu) / sd
        F = np.hstack([X[rows], Z[cols], X[rows] * Z[cols],
                       np.ones((y.size, 1))])
        A = F.T @ F
        A[np.diag_indices_from(A)] += 1e-2 * y.size
        w = np.linalg.solve(A, F.T @ yc)
        n, m = Y_obs.shape
        pred = np.empty((n, m))
        for i in range(n):
            Fi = np.hstack([np.repeat(X[i][None, :], m, axis=0), Z,
                            X[i][None, :] * Z, np.ones((m, 1))])
            pred[i] = Fi @ w
        return mu + sd * pred

    def attack(self, X, Z, k, budget, seed):
        return _random_attack(X, Z, k, seed)


def _als(R, M, rank, lam, iters, rng):
    n, m = R.shape
    P = 0.1 * rng.standard_normal((n, rank))
    Q = 0.1 * rng.standard_normal((m, rank))
    obs_i = [np.flatnonzero(M[i]) for i in range(n)]
    obs_j = [np.flatnonzero(M[:, j]) for j in range(m)]
    I = np.eye(rank)
    for _ in range(iters):
        for i in range(n):
            j = obs_i[i]
            if j.size:
                Qj = Q[j]
                P[i] = np.linalg.solve(Qj.T @ Qj + lam * j.size * I,
                                       Qj.T @ R[i, j])
        for jj in range(m):
            i = obs_j[jj]
            if i.size:
                Pi = P[i]
                Q[jj] = np.linalg.solve(Pi.T @ Pi + lam * i.size * I,
                                        Pi.T @ R[i, jj])
    return P, Q


class LowRankAgent:
    """ALS matrix factorization Y ~ P Q^T."""

    def solve(self, X, Z, Y_obs, mask, budget, seed):
        rows, cols, y, mu = _prep(Y_obs, mask)
        sd = float(y.std()) + 1e-9
        Yc = np.zeros(Y_obs.shape)
        Yc[rows, cols] = (y - mu) / sd
        r = budget.get("r", 8) if isinstance(budget, dict) else 8
        rng = np.random.default_rng(seed)
        P, Q = _als(Yc, mask.astype(bool), r, 0.1, 20, rng)
        return mu + sd * (P @ Q.T)

    def attack(self, X, Z, k, budget, seed):
        return _random_attack(X, Z, k, seed)


class HybridAgent:
    """Bilinear term X W Z^T plus low-rank ALS on the residual."""

    def solve(self, X, Z, Y_obs, mask, budget, seed):
        rows, cols, y, mu = _prep(Y_obs, mask)
        sd = float(y.std()) + 1e-9
        yc = (y - mu) / sd
        n, m = Y_obs.shape
        Xa = np.hstack([X, np.ones((n, 1))])
        Za = np.hstack([Z, np.ones((m, 1))])
        F = (Xa[rows][:, :, None] * Za[cols][:, None, :]).reshape(y.size, -1)
        A = F.T @ F
        A[np.diag_indices_from(A)] += 3e-3 * y.size
        w = np.linalg.solve(A, F.T @ yc)
        W = w.reshape(Xa.shape[1], Za.shape[1])
        pred = (Xa @ W) @ Za.T

        Yc = np.zeros((n, m))
        Yc[rows, cols] = yc
        Mb = mask.astype(bool)
        R = np.where(Mb, Yc - pred, 0.0)
        r = budget.get("r", 8) if isinstance(budget, dict) else 8
        rng = np.random.default_rng(seed)
        P, Q = _als(R, Mb, r, 0.1, 20, rng)
        return mu + sd * (pred + P @ Q.T)

    def attack(self, X, Z, k, budget, seed):
        return _random_attack(X, Z, k, seed)


BASELINES = {
    "mean_agent": MeanAgent,
    "ridge_agent": RidgeAgent,
    "low_rank_agent": LowRankAgent,
    "hybrid_agent": HybridAgent,
}
