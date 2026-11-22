# SDSC-STEM4Health

Welcome to the SDSC STEM4Health Hackathon! This git repository is your team's shared codebase: use it to keep your code in sync and under version control.

## Project 2 · Batch process monitoring

The BRPMM challenge uses sensor readings to decide whether to accept or reject each manufacturing batch. For both processes, the data includes labeled calibration batches and unlabeled test batches.

- **Process A** records 11 sensors across industrial batches of varying lengths. The two typical run lengths have different normal sensor levels, so the same reading can have a different meaning in each operating mode.
- **Process B** records 20 sensors during fermentation batches. Rejects are rare in the calibration data. Changes in aeration, fermentation volume, and feed temperature are among the signals the team investigated.

## Our solution · Team 11

Each process has its own feature extraction and classifier.

| Process | Selected model | Model inputs |
|---|---|---|
| A | Class-weighted Extra Trees with temperature GRU features and a sensor-limit rule | 95 batch features from 11 sensors |
| B | Standardized logistic regression | Six batch features from three sensors |

### Process A

The [Process A model](src/process_a/model.py) starts with 93 engineered inputs. The final two features capture temperature behavior that is hard to predict from accepted runs.

| Feature group | Selection | Count |
|---|---|---:|
| Sensor levels | 5th percentile, median, and 95th percentile per sensor | 33 |
| Trend and variability | Whole-batch median absolute deviation (MAD), overall slope, and median absolute consecutive change per sensor | 33 |
| Timing and local instability | Relative peak timing across 30 windows and maximum rolling MAD per sensor | 22 |
| Batch context | Log batch length and maximum mode-adjusted sensor deviation | 2 |
| Targeted signals | Mid-run temperature 1 dip, slow pressure 4 variation, and mode-relative density | 3 |
| Temperature forecast errors | 95th-percentile absolute errors for temperature 1 and temperature 2 | 2 |

One GRU trains on accepted batches and uses 16 readings from all four temperature sensors to predict the next eight readings for each. Its inputs are centered by operating mode. The error features pool across forecast positions and horizons, with cross-fitting by batch for classifier training.

The 160-tree classifier scores the 95 features. The sensor-limit rule adds one to the rejection score if any sensor median exceeds its learned limit relative to the accepted baseline for its operating mode. The final cutoff is tuned for rejected-class F1 on out-of-fold calibration scores. [Process A details](src/process_a/README.md)

### Process B

The submitted [Process B classifier](src/process_b_model.py) uses the **mean** and **range** (maximum minus minimum) of each of these sensors, giving six inputs per batch:

- `aeration_rate`
- `fermentation_volume`
- `feed_temperature`

The team compared mean and range summaries from six candidate sensors with the smaller three-sensor set. The selected classifier standardizes the six inputs using calibration data, fits logistic regression on the labeled batches, and rejects a batch if its predicted rejection probability exceeds 0.5. The Process B modeling notebook also explores a PCA monitor and self-training. The submission command uses the six-feature classifier.

## Reproduce the combined submission

Run from the repository root in the Renku environment. The commands use NumPy, pandas, SciPy, scikit-learn, joblib, and CPU PyTorch. Set `DATA` to the directory containing `process A`, `process B`, and `BRPMM_template.csv`.

```sh
DATA="../input-data/Project 2 - BRPMM"
python -m src.process_a.model --model /tmp/process_a_95.joblib train --data "$DATA"
python -m src.process_a.model --model /tmp/process_a_95.joblib predict --input "$DATA/process A/test data/test_data.csv" --out /tmp/predictions_a.csv
python -m src.process_b_model --data "$DATA" --out /tmp/predictions_b.csv
python -m src.make_submission --data "$DATA" --a /tmp/predictions_a.csv --b /tmp/predictions_b.csv --out submission.csv
```

The combined command checks that both prediction files cover their assigned batches exactly once and writes `batch_id,project,label` in template order. Keep calibration data, fitted weights, and generated submissions local or in the shared team workspace.
