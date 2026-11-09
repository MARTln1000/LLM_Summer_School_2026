"""Solve-only phase, locally: my_agent vs the PDF baselines.

Usage:
    .venv/bin/python run_local_grader.py [--seeds 20] [--budgets small,medium]
"""

import argparse
from collections import defaultdict

from local_arena.baselines import BASELINES
from local_arena.grading import load_agent, run_solve_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="my_agent.py")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budgets", default="small,medium")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    agents = {"my_agent": load_agent(args.agent)}
    if not args.skip_baselines:
        for name, cls in BASELINES.items():
            agents[name] = cls()

    budgets = args.budgets.split(",")
    results = run_solve_eval(agents, range(args.seeds), budgets)

    by = defaultdict(list)
    for r in results:
        by[(r["agent"], r["budget"])].append(r)

    print(f"\n{'agent':<16} {'budget':<8} {'score':>7} {'nrmse':>7} "
          f"{'max_t':>7} {'bad':>4}")
    print("-" * 55)
    for agent in agents:
        for b in budgets:
            rows = by[(agent, b)]
            score = sum(r["score"] for r in rows) / len(rows)
            loss = sum(min(r["nrmse"], 10) for r in rows) / len(rows)
            tmax = max(r["time"] for r in rows)
            bad = sum(r["status"] != "ok" for r in rows)
            print(f"{agent:<16} {b:<8} {score:>7.3f} {loss:>7.3f} "
                  f"{tmax:>6.2f}s {bad:>4}")

    print("\nmy_agent per regime (mean solve score):")
    reg = defaultdict(list)
    for r in results:
        if r["agent"] == "my_agent":
            reg[(r["regime"], r["budget"])].append(r["score"])
    for (regime, b), scores in sorted(reg.items()):
        print(f"  {regime:<10} {b:<8} {sum(scores)/len(scores):>7.3f} "
              f"(n={len(scores)})")


if __name__ == "__main__":
    main()
