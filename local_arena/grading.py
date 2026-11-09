"""Local grading: NRMSE, solve score (PDF eqs. 6-7), solve-only eval, duels."""

import hashlib
import importlib.util
import sys
import time

import numpy as np

from .generator import BUDGETS, generate_instance
from .masks import is_valid_mask, official_mask


def derived_seed(gen_seed, tag=""):
    """One-way derived agent seed, mimicking the grader's decoupling."""
    h = hashlib.sha256(f"{gen_seed}:{tag}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def nrmse(Y_hat, Y_true, mask):
    hidden = ~mask
    diff = (Y_hat - Y_true)[hidden]
    denom = Y_true[hidden].std() + 1e-8
    return float(np.sqrt(np.mean(diff * diff)) / denom)


def prediction_valid(Y_hat, shape):
    if not isinstance(Y_hat, np.ndarray) or Y_hat.shape != shape:
        return False
    if not np.isfinite(Y_hat).all():
        return False
    return bool((np.abs(Y_hat) <= 1e5).all())


def load_agent(path):
    spec = importlib.util.spec_from_file_location("submitted_agent", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submitted_agent"] = mod
    spec.loader.exec_module(mod)
    return mod.Agent()


def solve_instance(agent, X, Z, Y, mask, budget, agent_seed):
    Y_obs = np.where(mask, Y, 0.0)
    t0 = time.perf_counter()
    try:
        Y_hat = agent.solve(X, Z, Y_obs, mask.copy(), dict(budget), agent_seed)
        elapsed = time.perf_counter() - t0
        if not prediction_valid(np.asarray(Y_hat, dtype=float), Y.shape):
            return 1e6, elapsed, "invalid"
        if elapsed > budget["solve_timeout"]:
            return 1e6, elapsed, "timeout"
        return nrmse(np.asarray(Y_hat, dtype=float), Y, mask), elapsed, "ok"
    except Exception as e:  # noqa: BLE001 - grader treats any crash as invalid
        return 1e6, time.perf_counter() - t0, f"crash: {e}"


def run_solve_eval(agents, seeds, budgets):
    """agents: {name: agent}. Returns rows of per-instance results."""
    results = []
    for bname in budgets:
        budget = BUDGETS[bname]
        for s in seeds:
            X, Z, Y, regime = generate_instance(s, budget)
            mrng = np.random.default_rng(derived_seed(s, "official-mask"))
            mask = official_mask(budget["n"], budget["m"], budget["k"], mrng)
            mu = Y[mask].mean()
            base = nrmse(np.full(Y.shape, mu), Y, mask)
            for name, agent in agents.items():
                loss, elapsed, status = solve_instance(
                    agent, X, Z, Y, mask, budget, derived_seed(s, name))
                score = float(np.log(base / max(loss, 1e-12)))
                results.append(dict(budget=bname, seed=s, regime=regime,
                                    agent=name, nrmse=loss, base=base,
                                    score=score, time=elapsed, status=status))
    return results


def get_attack_mask(agent, X, Z, k, budget, agent_seed):
    """Returns (mask, penalty). Invalid masks -> official mask + delta=0.5."""
    try:
        M = agent.attack(X, Z, k, dict(budget), agent_seed)
        M = np.asarray(M)
        ok, _why = is_valid_mask(M, k)
        if ok:
            return M.astype(bool), 0.0
    except Exception:  # noqa: BLE001
        pass
    rng = np.random.default_rng(derived_seed(agent_seed, "fallback"))
    return official_mask(X.shape[0], Z.shape[0], k, rng), 0.5


def run_duel(agent_a, agent_b, seed, budget):
    """One duel per PDF section 6. Returns dict with losses and winner."""
    b = BUDGETS[budget] if isinstance(budget, str) else budget
    X, Z, Y, regime = generate_instance(seed, b)
    sa, sb = derived_seed(seed, "duel-A"), derived_seed(seed, "duel-B")
    mask_b, pen_a = get_attack_mask(agent_a, X, Z, b["k"], b, sa)
    mask_a, pen_b = get_attack_mask(agent_b, X, Z, b["k"], b, sb)
    la, _, _ = solve_instance(agent_a, X, Z, Y, mask_a, b, sa)
    lb, _, _ = solve_instance(agent_b, X, Z, Y, mask_b, b, sb)
    La, Lb = la + pen_a, lb + pen_b
    eps = 1e-4
    if La < Lb * (1 - eps):
        winner = "A"
    elif Lb < La * (1 - eps):
        winner = "B"
    else:
        winner = "draw"
    return dict(seed=seed, regime=regime, L_A=La, L_B=Lb, winner=winner,
                pen_A=pen_a, pen_B=pen_b)
