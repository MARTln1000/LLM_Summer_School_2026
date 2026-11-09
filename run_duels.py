"""Arena phase, locally: duels of my_agent vs each baseline.

Usage:
    .venv/bin/python run_duels.py [--seeds 10] [--budgets small,medium]
"""

import argparse
from collections import Counter

from local_arena.baselines import BASELINES
from local_arena.grading import load_agent, run_duel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="my_agent.py")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budgets", default="small,medium")
    args = ap.parse_args()

    me = load_agent(args.agent)
    budgets = args.budgets.split(",")

    for name, cls in BASELINES.items():
        opp = cls()
        tally = Counter()
        losses_me, losses_opp = [], []
        for b in budgets:
            for s in range(args.seeds):
                d = run_duel(me, opp, 10_000 + s, b)
                tally[d["winner"]] += 1
                losses_me.append(d["L_A"])
                losses_opp.append(d["L_B"])
        n = sum(tally.values())
        pts = (tally["A"] + 0.5 * tally["draw"]) / n
        print(f"my_agent vs {name:<16} W{tally['A']:>3} D{tally['draw']:>3} "
              f"L{tally['B']:>3}  arena_pts={pts:.3f}  "
              f"my_nrmse={sum(losses_me)/n:.3f} "
              f"opp_nrmse={sum(losses_opp)/n:.3f}")


if __name__ == "__main__":
    main()
