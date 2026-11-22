# Process A model with 93 features

Process A uses one class-weighted Extra Trees classifier for both operating modes. The classifier has 160 trees and a minimum of one sample per leaf. Its inputs describe sensor levels, changes during the batch, and variability at different scales.

## Classifier inputs

| Group | Definition | Count |
|---|---|---:|
| Sensor levels | 5th percentile, median, and 95th percentile for each of the 11 sensors | 33 |
| Whole-batch variability | Median absolute deviation (MAD) of each sensor's readings | 11 |
| Overall trend | Least-squares slope for each sensor over sample positions scaled from −1 to +1 | 11 |
| Short-term fluctuation | Median absolute difference between consecutive readings for each sensor | 11 |
| Relative peak timing | Position of the highest median among 30 consecutive windows, scaled from 0 to 1, for each sensor | 11 |
| Local instability | Maximum rolling MAD for each sensor, using overlapping windows roughly 1/30 of a batch long | 11 |
| Batch context | Natural log of the row count and the largest mode-adjusted sensor deviation | 2 |
| Targeted features | Mid-run temperature dip, slow pressure variation, and mode-relative density | 3 |
| **Total** | | **93** |

The targeted inputs are:

- **Mid-run `temperature_1` dip.** Subtract the median of the first five window medians from the minimum of windows 8–23.
- **Slow `pressure_4` variation.** Take the 97.5th-to-2.5th percentile spread of a centered rolling-median trace over the middle 90% of the batch.
- **Mode-relative density.** Subtract the typical accepted density for the operating mode from the batch median density.

Window positions and slopes use row order because the calibration CSV contains no timestamps.

## Operating modes and decision rule

Batch length serves as a proxy for the short and long operating modes, which typically contain about 2,370 and 2,725 rows. The mode-adjusted deviation feature uses a 2,548-row split. It compares each sensor median with its accepted reference for that mode and takes the largest relative deviation across sensors. The density feature and sensor-limit rule use a 2,550-row split.

The sensor-limit rule adjusts the rejection score after the classifier runs. For each sensor, it learns a baseline from accepted training-batch medians in each mode. The allowable shift is 1.5 times the largest absolute shift across accepted training batches relative to their respective baselines, with a floor based on typical variation. If any sensor exceeds its limit in either direction, the rule adds 1 to the classifier's rejection score. The batch is labeled `rejected` when the resulting score meets or exceeds the cutoff.

## Validation and saved model

The following results pool held-out predictions from four repeats of five-fold cross-validation on 161 labeled calibration batches. In each outer fold, sensor references were learned from training data and the cutoff was chosen to maximize accepted-class F1 on out-of-fold scores within that training data.

| Accuracy | Balanced accuracy | Accepted F1 | Rejected F1 | Rejected predictions caught | False alarms |
|---:|---:|---:|---:|---:|---:|
| 90.53% | 84.70% | 93.86% | 79.32% | 117/160 | 18 |

The final model was fitted on all 161 calibration batches. The artifact includes its sensor order, feature configuration, and rejection cutoff of 0.43125. It labeled 130 Process A test batches as accepted and 31 as rejected.

[Implementation](src/brpmm.py) · [Saved model](models/process_a_93.joblib)

The Process A test labels are in `predictions_A_93_features.csv` in the shared team workspace.
