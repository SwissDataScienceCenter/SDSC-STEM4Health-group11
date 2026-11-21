# Process A: sensor-wise model with learned safe ranges

Branch: `ramona-sensorwise` · Code: [`src/sensorwise.py`](src/sensorwise.py)

## TL;DR

A new model for **process A** that replaces the global-PCA approach in `src/brpmm.py` for that process. Process B is unchanged and still handled by `brpmm.py`.

| Model (process A) | F1, accepted = positive (grading metric) | AUC | Bad batches caught (CV) |
|---|---|---|---|
| Global PCA + classifier (`brpmm.py`) | 0.891 CV | – | 27 / 40 |
| Sensor-wise + safe ranges (previous version) | 0.911 (fresh seeds) | 0.927 | 28 / 40 |
| **Sensor-wise + safe ranges + subtle-trained trees (final, `SubtleModel`)** | **0.938 (fresh seeds)** | **0.956** | **35 / 40** |

"Fresh seeds" = F1 at the chosen flag rate, re-measured on 5 new random CV splits (the threshold was not tuned on them), so it is the fairest estimate. Run-to-run noise is about ±0.005.

## Grading metric

- **score = mean of the per-project F1 scores**
- F1 treats **`accepted` as the positive class**

Because F1 is on the majority class, it's better to flag only the clearly suspicious batches as rejected and call borderline ones accepted. That's why the decision threshold is tuned for F1(accepted), not for balanced accuracy.

## What we found in the data

Plotting every raw batch, with accepted and rejected in different colours, showed:

1. **Process A is steady-state.** Every sensor holds a flat level for the whole batch, so trajectory shape carries little information. That's why the global PCA on time-binned trajectories plateaued.
2. **Faults are mostly ONE sensor shifted to a wrong level, and a different sensor each time.** This is the brief's point about "low correlation, several independent variables". A single multivariate model dilutes a fault on one sensor.
3. **Shifts come in fixed, designed sizes.** Small shifts are still accepted and large shifts are rejected:

   | Sensor | Accepted shifts | Rejected shifts |
   |---|---|---|
   | temperature_3 | ±1, −2 | ±3, ±4 |
   | temperature_4 | ±1 | ±3 |
   | flow_rate_1 | −3, −0.9, +2.1 | −10, −3.9, +5.9, +6.5 |
   | flow_rate_2 | down to −48 | −102, −199 |
   | pressure_1 | none seen | ±4.4, −10 |

   This is the brief's "acceptable vs. unacceptable process variation", in concrete form.
4. **There are two operating modes.** Batches are about 2,370 or about 2,725 rows long. The short mode runs about 1.5° hotter on temperature_2 and about 3.4 higher on pressure_3, and it has twice the rejection rate (37% vs 19%).
5. **There's a hidden third recipe.** Among the long batches, a group of accepted batches has temperature_2 ≈ 91.5 (instead of ≈ 86), pressure_2 ≈ 8.6, pressure_3 ≈ 58 and pressure_4 ≈ 5, all shifted together. It's 9 batches in calibration and 18 in test. Treating it as its own mode tightens the safe ranges (temperature_2 ±8.2 → ±5.6, pressure_3 ±21.6 → ±16.7) and improves ranking (CV AUC 0.926 → 0.940). F1 gains only +0.003, which is noise, because the classifier had already learned the recipe.
6. **The "hard" rejected batches leave a weak, combined trace.** About 20 rejected batches have no out-of-range level shift, and the previous model missed about 11 of them. One statistic at a time they look normal (nothing survives a multiple-testing correction, and we also checked batch order, missing values, decimals, start/end transients, time lags between sensors and mass-balance ratios). But a model trained only to separate those 11 from accepted batches reaches AUC 0.83, while shuffled labels peak at 0.70. So there is a signal, spread over several small effects: temperature_1 tends to start high and dip mid-batch with a wider spread, pressure_4 swings harder, and density is slightly low in the normal long recipe.
7. **Training the trees only on the subtle cases recovers most of them.** When the obvious (out-of-range) rejected batches are left out of the classifier's training (the safe ranges already handle them), the trees focus on the fine boundary. Same features, same settings: F1 0.911 → 0.938 on fresh seeds, rejected caught 28 → 35 / 40, false alarms 5 → 6 / 121. About 5 rejected batches remain indistinguishable; those likely need information the sensors don't record (lab assays), so borderline batches should still go to a review zone.

## The model

```
raw batch ─► per-sensor statistics ─► (a) learned safe ranges ──► flagged? → rejected
                                      (b) ExtraTrees, trained on the  ─► rank the rest
                                          subtle cases only               └► flag top ~25% overall
```

1. **Per-sensor statistics** (`batch_stats`), computed on the middle 90% of each batch: level (median), fast noise, spread, slow drift ("wander"), oscillation, spike rate, tail heaviness, and start/end transients. That's 9 statistics × 11 sensors, plus batch length.
2. **Robust z-scores** of every statistic against the accepted training batches (`ModeScaler`).
3. **Learned safe ranges** (`Envelope`): for each sensor, the shift is its level minus the accepted-batch median of the same operating mode. A batch is flagged if any sensor is shifted more than **1.5× the largest shift ever seen in an accepted batch** (and more than 8 robust standard deviations, so normal noise never triggers it).
   - On its own this catches about 16/40 bad batches with about 3/121 false alarms.
   - It's fully explainable, e.g. "temperature_4 shifted 3.0°, the safe limit is 1.5°".
4. **ExtraTrees** on robust z-scores computed per operating mode (`SubtleModel`). It is trained only on accepted batches vs the rejected batches the safe ranges do **not** flag, so it specialises in the subtle faults instead of relearning "big shift = bad". (The previous version, `SensorwiseModel`, trained on all batches with pooled z-scores plus worst-deviation features; it is kept in the code for comparison.)
5. **Final ranking**: batches outside a safe range come first, then the rest ordered by ExtraTrees probability. The fraction to flag as rejected is chosen to maximise out-of-fold F1(accepted) over 10× repeated 5-fold CV. It comes out at **about 25%** (close to the 25% rejection rate in calibration), which is 41 of the 161 test batches. None of the 18 hot-recipe test batches are flagged.

### How it maps to the challenge requirements

| Requirement | How it's addressed |
|---|---|
| Limited calibration set | Compact per-batch statistics instead of full trajectories. Robust (median/MAD) scaling. Repeated CV over 5 seeds for every decision. |
| Low correlation / independent variables | Each sensor is monitored separately (per-sensor z-scores and safe ranges), with no single global projection. |
| Sensitivity to unseen variability | Safe ranges use a 1.5× margin over the largest acceptable shift, and a sensor-agnostic "worst deviation" feature generalises to faults on sensors never seen faulty in training. |
| Acceptable vs unacceptable variation | Safe ranges learn the boundary between acceptable and rejected shift sizes per sensor. |
| Recalibration / self-supervised learning | Everything is refit from data with no hand-set thresholds. Recalibration means refitting on the enlarged calibration set. Confidently-accepted new batches can be added to the reference set (as `brpmm.py` already does). |

## Things we tried that did NOT help (all ±0.005 F1, i.e. noise)

Evaluated on the same CV splits across 5 seeds:

- **Block-wise PCA** (one PCA each for temperatures, pressures and flows) and per-sensor PCA in place of the global PCA
- **"Shakiness"** (per-bin step size) features
- **Oversampling or undersampling** the rejected class. This only trades caught bad batches for false alarms, and the classifier already uses class weighting, which is equivalent to oversampling.
- **Mode-aware scaling.** The classifier learns the mode offsets itself from the short-mode flag. Mode-awareness is kept only inside the safe ranges.
- **Event, oscillation-amplitude and step-jump features**
- **Smaller feature sets** and ExtraTrees hyperparameter changes
- **Blending with the `brpmm.py` score.** This was worse than the new model alone.
- **Hundreds of extra features** (tsfresh-style stats, time segments, multi-scale, cross-correlations, sensor-relation residuals) and other classifiers (RandomForest, gradient boosting, L1 logistic regression, SVM, kNN, ensembles). Adding the extra features to the final model also didn't help.
- **A fixed "≥ 2.5 units" rule.** This dropped F1 to 0.76 because noisy sensors such as temperature_2 and pressure_2 naturally vary that much. The learned per-sensor ranges solve this.

## How to run

From the repo root, with the data at `../input-data/Project 2 - BRPMM/` (the default layout in Renku):

```bash
python src/sensorwise.py            # cross-validated evaluation for process A
python src/sensorwise.py --predict  # + writes predictions_A_sensorwise.csv (test set, process A)
```

The first run computes per-batch statistics (about 1–2 min on the slow mount) and caches them in `../cache/`.

To build a full submission, use this model for A and `brpmm.py` for B:

```bash
python src/brpmm.py --out submission.csv   # writes submission_scores.csv with A+B labels
python src/sensorwise.py --predict
python - <<'EOF'
import pandas as pd
t = pd.read_csv("../input-data/Project 2 - BRPMM/BRPMM_template.csv")
b = pd.read_csv("submission_scores.csv").set_index("batch_id")["label"]
a = pd.read_csv("predictions_A_sensorwise.csv").set_index("batch_id")["label"]
t["label"] = [a[i] if p == "A" else b[i] for i, p in zip(t.batch_id, t.project)]
t.to_csv("submission_sensorwise.csv", index=False)
EOF
```

## Caveats

- All numbers are cross-validation estimates on the 161 calibration batches (40 rejected). With so few positives, differences below about 0.01 F1 are noise.
- The final model's idea came from studying the batches we were missing, and we tried about 30 variants on the same 161 batches. Its true gain is probably somewhat smaller than the CV numbers suggest (our guess: +0.01 to +0.02 F1), but it is consistently positive.
- The flag fraction (about 25%) is tuned on out-of-fold scores, so the CV F1 is slightly optimistic. The same bias applies to the baseline comparison.
- Safe ranges assume test shifts resemble calibration shifts. A larger but still acceptable shift on a sensor would be a false alarm, and that's exactly the "unseen variability" trade-off the brief mentions.
