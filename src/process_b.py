"""Process B pipelines (simulated penicillin fermentation; 40 cal batches, only 5 rejected; fixed length 5001).

Challenge: strong dynamics may hide deviations; very few rejected batches -> lean on the unsupervised
monitor and low-capacity models.  B is already at the ceiling on the leaderboard, so it is used as a
REGRESSION GUARD (see benchmark.py) plus robustness check, not as a discriminating benchmark.

    python src/benchmark.py --proc B --configs all
"""
from pipeline import Config, standard_grid

CONFIGS = standard_grid()

_b = CONFIGS["baseline"]
for _c in [
    _b.variant("B_dynamics", clf_features=("summary", "phase", "corr"), mon_features=("traj", "corr")),
]:
    CONFIGS[_c.name] = _c
