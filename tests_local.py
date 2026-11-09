"""Validity checks for my_agent.py: masks, determinism, prediction validity,
timing. Run: .venv/bin/python tests_local.py
"""

import time

import numpy as np

from local_arena.generator import BUDGETS, generate_instance
from local_arena.grading import derived_seed, load_agent
from local_arena.masks import is_valid_mask, official_mask


def main():
    agent = load_agent("my_agent.py")
    failures = []

    for bname, budget in BUDGETS.items():
        for s in (0, 1, 2):
            X, Z, Y, regime = generate_instance(s, budget)
            aseed = derived_seed(s, "test")

            # --- attack: validity + determinism + speed -----------------
            t0 = time.perf_counter()
            M1 = agent.attack(X, Z, budget["k"], dict(budget), aseed)
            t_att = time.perf_counter() - t0
            ok, why = is_valid_mask(np.asarray(M1), budget["k"])
            if not ok:
                failures.append(f"[{bname} s{s}] attack mask invalid: {why}")
            M2 = agent.attack(X, Z, budget["k"], dict(budget), aseed)
            if not np.array_equal(np.asarray(M1), np.asarray(M2)):
                failures.append(f"[{bname} s{s}] attack not deterministic")

            # --- solve: validity + determinism + timing ------------------
            mrng = np.random.default_rng(derived_seed(s, "official-mask"))
            mask = official_mask(budget["n"], budget["m"], budget["k"], mrng)
            Y_obs = np.where(mask, Y, 0.0)
            t0 = time.perf_counter()
            Y1 = agent.solve(X, Z, Y_obs.copy(), mask.copy(), dict(budget),
                             aseed)
            t_solve = time.perf_counter() - t0
            Y1 = np.asarray(Y1)
            if Y1.shape != Y.shape:
                failures.append(f"[{bname} s{s}] bad prediction shape")
            if not np.isfinite(Y1).all() or (np.abs(Y1) > 1e5).any():
                failures.append(f"[{bname} s{s}] non-finite/huge prediction")
            Y2 = np.asarray(agent.solve(X, Z, Y_obs.copy(), mask.copy(),
                                        dict(budget), aseed))
            if not np.array_equal(Y1, Y2):
                failures.append(f"[{bname} s{s}] solve not deterministic")
            if t_solve > budget["solve_timeout"]:
                failures.append(
                    f"[{bname} s{s}] solve too slow: {t_solve:.2f}s "
                    f"> {budget['solve_timeout']}s")

            print(f"[{bname} s{s} {regime:<9}] solve {t_solve:5.2f}s "
                  f"(limit {budget['solve_timeout']:4.1f}s)  "
                  f"attack {t_att*1000:6.1f}ms  mask={'ok' if ok else why}")

    # --- solve with a missing-budget dict (defensive parsing) ----------
    budget = BUDGETS["small"]
    X, Z, Y, _ = generate_instance(7, budget)
    mask = official_mask(48, 48, 10, np.random.default_rng(7))
    Yh = agent.solve(X, Z, np.where(mask, Y, 0.0), mask, {}, 123)
    if np.asarray(Yh).shape != (48, 48):
        failures.append("solve failed with empty budget dict")
    Mh = agent.attack(X, Z, 10, {}, 123)
    ok, why = is_valid_mask(np.asarray(Mh), 10)
    if not ok:
        failures.append(f"attack failed with empty budget dict: {why}")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
