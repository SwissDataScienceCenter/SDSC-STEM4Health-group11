# Process A model with 95 features

Process A uses the 93 engineered Extra Trees inputs plus two temperature forecast-error features. The classifier still uses 160 class-weighted trees and the binary sensor-limit rule. The new cutoff maximizes **rejected-class F1** on out-of-fold calibration scores.

## Engineered features

| Inputs | Count |
|---|---:|
| 5th percentile, median, and 95th percentile per sensor | 33 |
| Whole-batch MAD, overall slope, and median absolute consecutive change per sensor | 33 |
| Relative peak timing and maximum rolling MAD per sensor | 22 |
| Log batch length and maximum mode-adjusted sensor deviation | 2 |
| Mid-run temperature 1 dip, slow pressure 4 variation, and mode-relative density | 3 |

Peak timing uses 30 equal windows by sample position. Rolling MAD uses overlapping windows about one-thirtieth of a batch long. Batch length distinguishes the two typical operating modes. A sensor-limit breach adds one to the Extra Trees rejection score before applying the cutoff.

## Temperature features

One GRU takes a history of 16 readings from all four temperature sensors and directly predicts the next eight readings for each sensor. It learns from accepted batches only. Its inputs are centered separately for short and long operating modes using a 2,550-row split. Each accepted training batch supplies up to 24 evenly spaced forecast windows.

At scoring time, each batch supplies up to 96 evenly spaced forecast windows. For each of `temperature_1` and `temperature_2`, the feature is the 95th percentile of absolute forecast errors pooled across the forecast windows and eight horizons. The other two temperatures are GRU inputs and outputs but their errors are not classifier inputs. Classifier-training forecast features are cross-fitted by whole batch.

## Validation

Both models used rejected-F1 cutoff selection inside the training folds. Each comparison used two repeats of five-fold cross-validation on the same 161 labeled batches.

| Split set | 93-feature rejected F1 | 95-feature rejected F1 | Rejects within 25 false alarms, 93 / 95 features |
|---|---:|---:|---:|
| Original feature-selection splits | 80.00% | 80.00% | 67 / 69 |
| Fresh split assignments | 73.75% | 75.64% | 64 / 67 |

On the fresh splits, the 95-feature model had 88.20% accuracy, 77.63% rejected precision, and 73.75% rejected recall. Both versions caught 59 of 80 held-out reject predictions at the training-selected cutoff. False alarms fell from 21 to 17.

## Training and inference

Run these commands from the repository root. Install NumPy, pandas, scikit-learn, joblib, and CPU PyTorch in the training environment.

```sh
python -m src.process_a.model --model /tmp/process_a_95.joblib train --data "/path/to/Project 2 - BRPMM"
python -m src.process_a.model --model /tmp/process_a_95.joblib predict --input "/path/to/Project 2 - BRPMM/process A/test data/test_data.csv" --out /tmp/process_a_predictions.csv
```

The data root contains `process A/calibration data/calibration_data.csv` and `process A/calibration data/calibration_labels.csv`. Test input is a long-format CSV with `batch_id` and the same 11 sensor columns. Keep readings in acquisition order within each batch. Each batch needs at least 24 readings for the GRU. The prediction CSV contains `batch_id`, `score`, and `label`.

Training writes the classifier, GRU, sensor order, and cutoff to the chosen local path. The repository includes the source and documentation. Use the same Python environment to train and reload the local artifact. If you trained an artifact before the move into `src/`, rerun the training command to save it with the new module path.

[Model source](model.py)
