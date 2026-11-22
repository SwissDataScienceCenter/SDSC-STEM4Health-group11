# Process A model with 95 features

Process A uses the [original 93 Extra Trees inputs](93_feat.md) plus two temperature forecast-error features. The classifier still uses 160 class-weighted trees and the binary sensor-limit rule. The new cutoff maximizes **rejected-class F1** on out-of-fold calibration scores.

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

Run `python -m src.process_a_95 train` from the repository root to fit and save [the model](models/process_a_95.joblib). Run `python -m src.process_a_95 predict --input path/to/test_data.csv --out predictions.csv` to apply it. The saved artifact stores the fitted classifier, temperature GRU, sensor order, and rejection cutoff. The combined submission CLI in `src/brpmm.py` trains the 95-feature pipeline for Process A.

[Temperature features and model](src/process_a_95.py)
