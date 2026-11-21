"""Process A pipelines (industrial; 161 cal batches, 40 rejected; variable length 196-2932; sensor outliers).

Challenge: class imbalance + sharp discrimination near the acceptance border (A_101..A_104).
Add A-specific candidates below; they are benchmarked exactly like everything else:

    python src/benchmark.py --proc A --configs all
"""
from pipeline import Config, standard_grid

CONFIGS = standard_grid()

_b = CONFIGS["baseline"]
for _c in [
    # winsorised signals + extra dynamic features: outlier-robust variant for the heavy A spikes
    _b.variant("A_winsor_phase", preprocess="winsor", clf_features=("summary", "phase")),
    _b.variant("A_winsor_phase_hgb", preprocess="winsor", clf_features=("summary", "phase"),
               models=(("lr", 1.0), ("et", 1.0), ("hgb", 1.0))),
]:
    CONFIGS[_c.name] = _c
