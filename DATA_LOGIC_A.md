# Process A – what the data logic looks like, and what did not move the score

Measured on the real calibration data (161 batches, 40 rejected), same CV as `SENSORWISE.md`
(5 seeds x 5-fold, F1 accepted @ 25 % flag rate). Reference `SubtleModel` = **0.946**, noise about ±0.01.

## Structure found
1. **Recipe offsets.** Every sensor is a flat level; the level depends on the recipe (short ~2370 rows,
   long ~2725 rows, hot recipe temperature_2 ~91.5). Within a recipe, accepted batches still scatter
   (temperature_1 13.2-15.5), so a level alone cannot decide.
2. **temperature_1** = batch offset + one common shape (step near row 1850-2000; a step down near
   2400-2500 in long batches). Short-mode rejected batches sit at high offsets (+1.0 to +1.9) and often
   carry noise bursts.
3. **Pressures share one disturbance.** Detrended residuals of pressure_2/3/4 correlate about 0.9 inside a
   batch (0.68-0.98 across batches), flow_rate_2 about 0.3 with them, flow_rate_1-pressure_1 0.19. Noise
   amplitudes of the three pressures correlate 0.97-0.98 across batches. Pressure noise scale (AUC 0.78-0.80)
   and kurtosis (0.78 inverse) separate the subtle subset.
4. Obvious faults are designed shifts (temperature_3 +-3/4, temperature_4 +-3, flow_rate_1 +-6/10,
   flow_rate_2 -20..-200, pressure_1 +-4/10). The ~24 subtle rejected batches have no single out-of-range
   level (e.g. A_100, A_104, A_289: every sensor within a few tenths of nominal).

## Tried, no gain (F1 @25 %, paired with the reference on identical folds)
| idea | F1 |
|---|---|
| phase-aware stats (phase 1 only / both phases) | 0.941 / 0.923 |
| windowed local-activity features | 0.939 |
| Mahalanobis / whitened level, wander, spread blocks (Ledoit-Wolf) | 0.938-0.943 |
| raw std/MAD/kurtosis/skew, pressure common-factor structure, 55 pair correlations | 0.944-0.949 |
| GP classifier / shrinkage LDA / ARD | 0.943 / 0.924 / 0.929 |
| ExtraTrees + GP blend (`bayes_models.py`) | 0.951 (+0.005, 3/5 seeds) |
| label cleaning (drop training batches contradicting an inner OOF prediction) | 0.891-0.939, worse |
| up-weighting boundary batches | 0.946-0.948 |
| self-training / pseudo-labels (`selftrain.py`) | 0.936-0.944 |

Unsupervised PCA (SPE/T2) on autoscaled mean+std per sensor already reaches AUC ~0.87 (0.80-0.83 on the
subtle subset) with no labels, but supervised trees on the same information are not improved by adding it.

## Reading
Misclassified batches are on the boundary (OOF rank 0.72-0.80) and are genuine, not label noise: removing
the "hard" accepted batches makes things clearly worse, they are what defines the boundary. About 5 rejected
batches (A_289, A_008, A_097, A_163, A_244) show no deviation in any statistic we computed.
