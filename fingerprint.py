"""Hash the agent's outputs on a few fixed instances — used to confirm a
refactor didn't change behavior."""

import hashlib

import numpy as np

from local_arena.generator import BUDGETS, generate_instance
from local_arena.grading import derived_seed, load_agent
from local_arena.masks import official_mask

agent = load_agent("my_agent.py")
h = hashlib.sha256()
for bname in ("small", "medium"):
    b = BUDGETS[bname]
    for s in range(3):
        X, Z, Y, _ = generate_instance(s, b)
        mask = official_mask(b["n"], b["m"], b["k"],
                             np.random.default_rng(derived_seed(s, "official-mask")))
        Yh = agent.solve(X, Z, np.where(mask, Y, 0.0), mask, dict(b),
                         derived_seed(s, "fp"))
        h.update(np.asarray(Yh).tobytes())
        Mk = agent.attack(X, Z, b["k"], dict(b), derived_seed(s, "fpa"))
        h.update(np.asarray(Mk).tobytes())
print(h.hexdigest())
