"""Matrix Arena submission.

The solver fits a handful of models on 85% of the observed cells, picks
whichever does best on the held-out 15%, and refits that one on all
observations. The candidates cover the generator's three mechanisms:
a bilinear ridge on (tanh-augmented) features, masked ALS for the latent
low-rank part, a bilinear+ALS hybrid with a backfitting round, soft-impute,
and mean / row+column-effect fallbacks.

The attack sorts rows and columns by their first PC score, splits them into
~(k+1)-sized groups of look-alikes, and lets each group observe only itself
(a circulant inside each block). Degree-preserving edge swaps chain the
blocks so the mask stays exactly k-regular and connected, but it is about
the worst expander you can build under those constraints: the opponent's
observations are highly redundant and the latent factors barely propagate
between blocks.

Only numpy. All randomness goes through one default_rng(seed), so the agent
is deterministic. No file I/O, no state kept between calls.
"""

import time

import numpy as np


class Agent:
    # settled by local ablations (see ablation.py in the dev folder)
    N_BACKFIT = 1
    USE_SOFT_IMPUTE = True

    def solve(self, X, Z, Y_obs, mask, budget, seed):
        """X (n,d), Z (m,d), Y_obs (n,m) with zeros where hidden,
        mask (n,m) bool. Returns Y_hat (n,m)."""
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        Y_obs = np.asarray(Y_obs, dtype=np.float64)
        M = np.asarray(mask, dtype=bool)
        n, m = Y_obs.shape

        # stay well inside the budget; the deadline is only an emergency
        # brake, the workload below is sized to finish long before it
        deadline = t0 + 0.80 * self._time_limit(budget, n)

        rows, cols = np.nonzero(M)
        y = Y_obs[rows, cols]
        nobs = y.size
        mu = float(y.mean()) if nobs else 0.0
        sd = float(y.std()) + 1e-9
        if nobs < 50:
            return np.full((n, m), mu)

        # work in centered/scaled units throughout
        yc = (y - mu) / sd
        Yc = np.zeros((n, m))
        Yc[rows, cols] = yc

        rng = np.random.default_rng(seed)
        cfg = self._config(budget, n, m)

        try:
            pred = self._reconstruct(X, Z, Yc, M, yc, rows, cols, rng, cfg,
                                     deadline)
        except Exception:
            # should never happen; better a weak prediction than a crash
            pred = self._rowcol(Yc, M)

        pred = np.where(np.isfinite(pred), pred, 0.0)
        np.clip(pred, -25.0, 25.0, out=pred)
        out = mu + sd * pred
        out[rows, cols] = y
        np.clip(out, -1e5, 1e5, out=out)
        return out

    def _reconstruct(self, X, Z, Yc, M, yc, rows, cols, rng, cfg, deadline):
        n, m = Yc.shape
        nobs = yc.size

        # hold out 15% of the observed cells for model selection
        perm = rng.permutation(nobs)
        n_val = max(1, int(round(0.15 * nobs)))
        va, tr = perm[:n_val], perm[n_val:]
        y_va = yc[va]
        r_tr, c_tr, y_tr = rows[tr], cols[tr], yc[tr]
        Mt = M.copy()
        Mt[rows[va], cols[va]] = False
        Yct = Yc.copy()
        Yct[rows[va], cols[va]] = 0.0

        def val_err(pred):
            d = pred[rows[va], cols[va]] - y_va
            return float(np.sqrt(np.mean(d * d)))

        cands = []  # (holdout rmse, name, train-fit prediction, params)

        def consider(name, pred, params=None):
            cands.append((val_err(pred), name, pred, params))

        consider("mean", np.zeros((n, m)))
        consider("rowcol", self._rowcol(Yct, Mt))

        # feature blocks; the tanh random features give the bilinear model
        # a shot at the generator's weak nonlinear term
        Xs = X / (X.std(axis=0) + 1e-9)
        Zs = Z / (Z.std(axis=0) + 1e-9)
        d = X.shape[1]
        Rp = rng.standard_normal((d, cfg["nf"])) * (1.5 / np.sqrt(d))
        Rq = rng.standard_normal((d, cfg["nf"])) * (1.5 / np.sqrt(d))
        feats = {
            "plain": (np.hstack([Xs, np.ones((n, 1))]),
                      np.hstack([Zs, np.ones((m, 1))])),
            "aug": (np.hstack([Xs, np.tanh(Xs @ Rp), np.ones((n, 1))]),
                    np.hstack([Zs, np.tanh(Zs @ Rq), np.ones((m, 1))])),
        }

        # bilinear ridge Y ~ Xa W Za^T; one eigendecomposition per feature
        # set pays for the whole lambda grid
        factors = {}
        best_bil = None  # (rmse, mode, lam, pred)
        for mode, (Xa, Za) in feats.items():
            factors[mode] = self._bilinear_factor(Xa, Za, r_tr, c_tr)
            for lam in cfg["lam_grid"]:
                pred = self._bilinear_solve(Xa, Za, factors[mode], y_tr, lam)
                err = val_err(pred)
                if best_bil is None or err < best_bil[0]:
                    best_bil = (err, mode, lam, pred)
        _, mode_b, lam_b, bil_tr = best_bil
        consider("bil", bil_tr, (mode_b, lam_b))

        # masked ALS on the bilinear residual
        Rmat = np.where(Mt, Yc - bil_tr, 0.0)
        best_hyb = None  # (rmse, rank, lam, pred)
        for rank in cfg["ranks"]:
            for lam in cfg["als_lams"]:
                P, Q = self._als(Rmat, Mt, rank, lam, cfg["iters"], rng,
                                 deadline)
                pred = bil_tr + P @ Q.T
                err = val_err(pred)
                if best_hyb is None or err < best_hyb[0]:
                    best_hyb = (err, rank, lam, pred)
        _, rank_h, lam_h, hyb_tr = best_hyb
        consider("hyb", hyb_tr, (rank_h, lam_h))

        # a round of backfitting between the two components sometimes helps
        # when neither dominates (the "mixed" regime)
        if self.N_BACKFIT > 0:
            Xa, Za = feats[mode_b]
            bil_bf = bil_tr
            low_bf = hyb_tr - bil_tr
            for _ in range(self.N_BACKFIT):
                bil_bf = self._bilinear_solve(Xa, Za, factors[mode_b],
                                              y_tr - low_bf[r_tr, c_tr],
                                              lam_b)
                Rm = np.where(Mt, Yc - bil_bf, 0.0)
                P, Q = self._als(Rm, Mt, rank_h, lam_h, cfg["iters"], rng,
                                 deadline)
                low_bf = P @ Q.T
            consider("hyb_bf", bil_bf + low_bf, (rank_h, lam_h))

        # plain ALS, for when the low-rank term dominates
        best_als = None
        for rank in cfg["ranks"]:
            for lam in cfg["als_lams"]:
                P, Q = self._als(Yct, Mt, rank, lam, cfg["iters"], rng,
                                 deadline)
                pred = P @ Q.T
                err = val_err(pred)
                if best_als is None or err < best_als[0]:
                    best_als = (err, (rank, lam), pred)
        consider("als", best_als[2], best_als[1])

        # soft-impute path (nuclear-norm flavored; behaves differently from
        # ALS when observations are scarce)
        if self.USE_SOFT_IMPUTE:
            best_si = None
            for thr, pred in self._soft_impute(Yct, Mt, cfg["si_iters"],
                                               deadline):
                err = val_err(pred)
                if best_si is None or err < best_si[0]:
                    best_si = (err, thr, pred)
            consider("si", best_si[2], best_si[1])

        # pick on the holdout and refit the winner on all observed cells.
        # (I also tried ridge-stacking the candidates here; it was
        # consistently a bit worse than plain selection, so: selection.)
        _, name, train_pred, params = min(cands, key=lambda c: c[0])

        if time.perf_counter() > deadline:
            return train_pred  # out of time, keep the train-split fit

        if name == "mean":
            return np.zeros((n, m))
        if name == "rowcol":
            return self._rowcol(Yc, M)

        if name in ("bil", "hyb", "hyb_bf"):
            Xa, Za = feats[mode_b]
            factor = self._bilinear_factor(Xa, Za, rows, cols)
            bil = self._bilinear_solve(Xa, Za, factor, yc, lam_b)
            if name == "bil":
                return bil
            Rf = np.where(M, Yc - bil, 0.0)
            P, Q = self._als(Rf, M, rank_h, lam_h, cfg["iters"], rng,
                             deadline)
            low = P @ Q.T
            if name == "hyb":
                return bil + low
            for _ in range(self.N_BACKFIT):
                bil = self._bilinear_solve(Xa, Za, factor, yc - low[rows, cols],
                                           lam_b)
                Rf = np.where(M, Yc - bil, 0.0)
                P, Q = self._als(Rf, M, rank_h, lam_h, cfg["iters"], rng,
                                 deadline)
                low = P @ Q.T
            return bil + low

        if name == "als":
            rank, lam = params
            P, Q = self._als(Yc, M, rank, lam, cfg["iters"], rng, deadline)
            return P @ Q.T

        # "si": walk the warm-started path again down to the chosen threshold
        pred = None
        for _, sp in self._soft_impute(Yc, M, cfg["si_iters"], deadline,
                                       stop_at=params):
            pred = sp
        return pred

    def attack(self, X, Z, k, budget, seed):
        """Returns a k-regular connected (n,m) bool mask for the opponent."""
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        n, m = X.shape[0], Z.shape[0]
        k = int(k)

        if n != m or k < 2 or k > m:
            return self._circulant_mask(n, m, k)

        try:
            order_r = np.argsort(self._pc1(X), kind="stable")
            order_c = np.argsort(self._pc1(Z), kind="stable")
        except Exception:
            order_r = np.arange(n)
            order_c = np.arange(m)

        # blocks of look-alike rows/columns, each observing only itself
        # through an in-block circulant
        nb = max(1, n // (k + 1))
        bounds = [round(b * n / nb) for b in range(nb + 1)]
        S = np.zeros((n, m), dtype=bool)
        idx = np.arange(n)
        for b in range(nb):
            s, e = bounds[b], bounds[b + 1]
            L = e - s
            p = idx[:L]
            for t in range(k):
                S[s + p, s + (p + t) % L] = True

        # swap two edges between neighboring blocks: degrees are unchanged
        # and the graph becomes connected (barely -- which is the point)
        for b in range(nb - 1):
            i = bounds[b + 1] - 1  # last row of block b; (i,i) is its t=0 edge
            j = bounds[b + 1]      # first row of block b+1
            S[i, i] = S[j, j] = False
            S[i, j] = S[j, i] = True

        mask = np.zeros((n, m), dtype=bool)
        pr, pc = np.nonzero(S)
        mask[order_r[pr], order_c[pc]] = True

        # never hand the grader an invalid mask (0.5 penalty): if anything
        # above went wrong, fall back to a circulant band, which is always
        # k-regular and connected
        if not self._mask_valid(mask, k):
            mask = np.zeros((n, m), dtype=bool)
            for t in range(k):
                mask[order_r[idx], order_c[(m - 1 - idx + t) % m]] = True
        return mask

    # --- helpers ---

    @staticmethod
    def _time_limit(budget, n):
        if isinstance(budget, dict):
            for key in ("solve_timeout", "timeout", "time_limit"):
                v = budget.get(key)
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
        return 1.0 if n <= 64 else (5.0 if n <= 128 else 15.0)

    @staticmethod
    def _config(budget, n, m):
        r = None
        if isinstance(budget, dict):
            v = budget.get("r")
            if isinstance(v, (int, float)) and v > 0:
                r = int(v)
        if r is None:  # budget dict didn't say; guess from the size
            r = 4 if n <= 64 else (8 if n <= 128 else 12)
        rmax = max(2, min(n, m) // 3)
        ranks = sorted({min(r, rmax), min(r + 6, rmax)})
        if n <= 64:
            iters, nf = 15, 6
        elif n <= 128:
            iters, nf = 20, 8
        else:
            iters, nf = 25, 8
        return {
            "ranks": ranks,
            "als_lams": (0.05, 0.3),
            "iters": iters,
            "nf": nf,
            "lam_grid": (3e-5, 3e-4, 3e-3, 3e-2, 3e-1),
            "si_iters": 10,
        }

    @staticmethod
    def _rowcol(Yc, M, reg=6.0):
        """Shrunk row effects + column effects."""
        a = Yc.sum(axis=1) / (M.sum(axis=1) + reg)
        resid = np.where(M, Yc - a[:, None], 0.0)
        b = resid.sum(axis=0) / (M.sum(axis=0) + reg)
        return a[:, None] + b[None, :]

    @staticmethod
    def _bilinear_factor(Xa, Za, r_idx, c_idx):
        """Normal equations for Y ~ Xa W Za^T restricted to the given cells,
        eigendecomposed once so every lambda (and every new target vector in
        backfitting) is just a cheap re-solve."""
        nobs = r_idx.size
        F = (Xa[r_idx][:, :, None] * Za[c_idx][:, None, :]).reshape(nobs, -1)
        A = F.T @ F
        evals, evecs = np.linalg.eigh(A)
        return F, evals, evecs

    @staticmethod
    def _bilinear_solve(Xa, Za, factor, yvals, lam):
        F, evals, evecs = factor
        b = evecs.T @ (F.T @ yvals)
        w = evecs @ (b / (evals + lam * yvals.size))
        W = w.reshape(Xa.shape[1], Za.shape[1])
        return (Xa @ W) @ Za.T

    @staticmethod
    def _als(R, M, rank, lam, iters, rng, deadline):
        """Masked ALS, R[M] ~ (P Q^T)[M]. Warm-started from the SVD of the
        zero-filled matrix rescaled by 1/density (unbiased estimate of the
        full matrix), which converges much faster than a random init."""
        n, m = R.shape
        dens = max(float(M.mean()), 1e-6)
        U, s, Vt = np.linalg.svd(R / dens, full_matrices=False)
        rank = min(rank, s.size)
        root = np.sqrt(s[:rank])
        P = U[:, :rank] * root + 0.01 * rng.standard_normal((n, rank))
        Q = Vt[:rank].T * root + 0.01 * rng.standard_normal((m, rank))

        obs_i = [np.flatnonzero(M[i]) for i in range(n)]
        obs_j = [np.flatnonzero(M[:, j]) for j in range(m)]
        I = np.eye(rank)
        for _ in range(iters):
            for i in range(n):
                j = obs_i[i]
                if j.size == 0:
                    P[i] = 0.0
                    continue
                Qj = Q[j]
                P[i] = np.linalg.solve(Qj.T @ Qj + lam * j.size * I,
                                       Qj.T @ R[i, j])
            for jj in range(m):
                i = obs_j[jj]
                if i.size == 0:
                    Q[jj] = 0.0
                    continue
                Pi = P[i]
                Q[jj] = np.linalg.solve(Pi.T @ Pi + lam * i.size * I,
                                        Pi.T @ R[i, jj])
            if time.perf_counter() > deadline:  # emergency brake
                break
        return P, Q

    @staticmethod
    def _soft_impute(Yc, M, iters, deadline, stop_at=None):
        """Soft-impute: alternate fill-in with SVD soft-thresholding, warm
        starting each threshold from the previous (larger) one. Returns the
        list of (threshold fraction, prediction) along the path; pass
        stop_at to cut the path short when refitting a chosen threshold."""
        s0 = np.linalg.svd(Yc, compute_uv=False)[0]
        Yhat = np.zeros(Yc.shape)
        out = []
        for frac in (0.5, 0.3, 0.2, 0.1, 0.05):
            thr = frac * s0
            for _ in range(iters):
                Yfill = np.where(M, Yc, Yhat)
                U, s, Vt = np.linalg.svd(Yfill, full_matrices=False)
                Ynew = (U * np.maximum(s - thr, 0.0)) @ Vt
                converged = np.linalg.norm(Ynew - Yhat) <= 1e-4 * (
                    np.linalg.norm(Yhat) + 1e-9)
                Yhat = Ynew
                if converged:
                    break
            out.append((frac, Yhat.copy()))
            if stop_at is not None and frac == stop_at:
                break
            if time.perf_counter() > deadline:  # emergency brake
                break
        return out

    @staticmethod
    def _pc1(A):
        Ac = A - A.mean(axis=0)
        U, s, _ = np.linalg.svd(Ac, full_matrices=False)
        if s.size == 0:
            return np.zeros(A.shape[0])
        return U[:, 0] * s[0]

    @staticmethod
    def _mask_valid(M, k):
        n, m = M.shape
        if not ((M.sum(axis=1) == k).all() and (M.sum(axis=0) == k).all()):
            return False
        # BFS over the bipartite graph
        seen_r = np.zeros(n, dtype=bool)
        seen_c = np.zeros(m, dtype=bool)
        stack = [(True, 0)]
        seen_r[0] = True
        while stack:
            is_row, idx = stack.pop()
            if is_row:
                for j in np.flatnonzero(M[idx]):
                    if not seen_c[j]:
                        seen_c[j] = True
                        stack.append((False, int(j)))
            else:
                for i in np.flatnonzero(M[:, idx]):
                    if not seen_r[i]:
                        seen_r[i] = True
                        stack.append((True, int(i)))
        return bool(seen_r.all() and seen_c.all())

    @staticmethod
    def _circulant_mask(n, m, k):
        """Last-resort mask for shapes the block construction can't handle
        (non-square, tiny k). Valid whenever a valid mask exists at all."""
        k = max(1, min(int(k), m))
        mask = np.zeros((n, m), dtype=bool)
        p = np.arange(n)
        for t in range(k):
            mask[p, (p + t) % m] = True
        return mask
