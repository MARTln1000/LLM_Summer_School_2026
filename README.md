# Matrix Arena — LLM Summer School 2026 entrance competition

Everything for the competition described in `LLM_Summer_school.pdf`.
The file that actually gets submitted is **`my_agent.py`** — one class,
two methods, numpy only. The rest of this folder is tooling I built to
test it, since the official scaffold isn't distributed with the PDF.

## The agent in short

**solve** fits several models on 85% of the observed cells and keeps
whichever one predicts the held-out 15% best, refit on all observations:

- global mean and shrunk row/column effects (safety nets),
- bilinear ridge `Y ≈ Xa W Za^T`, where Xa optionally includes random
  tanh features to soak up the generator's weak nonlinear term. One
  eigendecomposition of the normal equations covers the whole lambda
  grid,
- the bilinear fit plus masked ALS on its residual, with one backfitting
  round between the two components,
- plain masked ALS (for low-rank-dominant instances),
- soft-impute, i.e. repeated SVD soft-thresholding.

ALS is warm-started from an SVD of the zero-filled matrix divided by the
observation density. Everything is deterministic given the seed, and the
workload is sized to finish at roughly a fifth of each budget's timeout
(worst measured: 0.3s / 1s small, 1.6s / 5s medium, 4.5s / 15s large).

**attack** sorts rows and columns by first-PC score, groups look-alikes
into blocks of about k+1, and has each block observe only itself through
an in-block circulant. Edge swaps between neighboring blocks keep the
mask exactly k-regular and connected while making it a terrible
expander — the opponent's observations are redundant and the latent
low-rank factors barely propagate between blocks. A validity self-check
falls back to a circulant band, so the 0.5 invalid-mask penalty can
never trigger.

Things I tried that didn't make the cut: ridge-stacking the candidate
models instead of picking one (consistently slightly worse), and
band-shaped leverage masks for the attack (weaker than blocks — see
`attack_eval.py`).

## Local tooling

`local_arena/` recreates the grader from the PDF spec: the instance
generator with all six regimes (`generator.py`), the official random
k-regular mask and validity check (`masks.py`), NRMSE / solve score /
duel protocol (`grading.py`), and the four documented baselines
(`baselines.py`). Sanity check: my low-rank baseline lands at
NRMSE ≈ 1.12 on average, the same number shown in the PDF's Figure 3.

Scripts, all run with `.venv/bin/python`:

- `tests_local.py` — validity, determinism and timing checks
- `run_local_grader.py` — solve-only phase against the baselines
- `run_duels.py` — arena duels against each baseline
- `attack_eval.py` — compares mask designs by damage inflicted
- `ablation.py` — solver flag ablations
- `fingerprint.py` — hashes agent outputs to catch accidental behavior
  changes when refactoring

Last full run: solve scores 0.53 / 0.43 / 0.26 (small / medium / large)
vs 0.42 / 0.21 / 0.11 for the best baseline, duels 54W–3L–0D, no
crashes, timeouts or invalid outputs.

## Environment

Python 3.14 venv in `.venv/` with numpy (pypdf/pdfplumber were only used
to read the PDF). The agent itself imports numpy and time.
