"""Ablation over the agent's STACK / SVD_INIT / BACKFIT flags.

Usage: .venv/bin/python ablation.py [--seeds 20] [--budgets small,medium]
"""

import argparse
from collections import defaultdict

from local_arena.grading import load_agent, run_solve_eval


def make_variant(base_cls, **flags):
    return type("Agent", (base_cls,), flags)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budgets", default="small,medium")
    args = ap.parse_args()

    base = load_agent("my_agent.py").__class__
    variants = {
        "base":      make_variant(base, N_BACKFIT=0, USE_SOFT_IMPUTE=False),
        "bf":        make_variant(base, N_BACKFIT=1, USE_SOFT_IMPUTE=False),
        "si":        make_variant(base, N_BACKFIT=0, USE_SOFT_IMPUTE=True),
        "bf_si":     make_variant(base, N_BACKFIT=1, USE_SOFT_IMPUTE=True),
    }
    budgets = args.budgets.split(",")
    results = run_solve_eval(variants, range(args.seeds), budgets)

    by = defaultdict(list)
    reg = defaultdict(list)
    for r in results:
        by[(r["agent"], r["budget"])].append(r)
        reg[(r["agent"], r["regime"])].append(r["score"])

    print(f"\n{'variant':<14} " + " ".join(f"{b:>8}" for b in budgets)
          + f" {'overall':>8} {'max_t':>6} {'bad':>4}")
    print("-" * (16 + 9 * (len(budgets) + 1) + 12))
    for name in variants:
        per_b = [sum(r['score'] for r in by[(name, b)]) / len(by[(name, b)])
                 for b in budgets]
        allr = [r for b in budgets for r in by[(name, b)]]
        overall = sum(r["score"] for r in allr) / len(allr)
        tmax = max(r["time"] for r in allr)
        bad = sum(r["status"] != "ok" for r in allr)
        print(f"{name:<14} " + " ".join(f"{s:>8.3f}" for s in per_b)
              + f" {overall:>8.3f} {tmax:>5.2f}s {bad:>4}")

    regimes = sorted({r for (_, r) in reg})
    print(f"\n{'variant':<14} " + " ".join(f"{r[:9]:>9}" for r in regimes))
    for name in variants:
        vals = []
        for r in regimes:
            s = reg[(name, r)]
            vals.append(sum(s) / len(s) if s else float("nan"))
        print(f"{name:<14} " + " ".join(f"{v:>9.3f}" for v in vals))


if __name__ == "__main__":
    main()
