
Project 2 – BRPMM (SDSC STEM4Health hackathon, group 11).

## Task
Batch-level binary classification: predict `accepted` / `rejected` for each manufacturing batch from its sensor time series. Two independent processes (A and B) — model them separately.

## Data location
`../input-data/Project 2 - BRPMM/` (relative to repo root; slow mount — full CSV loads take ~1–2 min, cache to parquet if iterating).

```
BRPMM_template.csv                    # submission template
Leaderboard_template_project2.ipynb   # leaderboard submit helper
process A/calibration data/{calibration_data.csv, calibration_labels.csv, variable_names.xlsx}
process A/test data/{test_data.csv, variable_names.xlsx}
process B/calibration data/{calibration_data.csv, calibration_labels.csv, variable_names.xlsx}
process B/test data/{test_data.csv, variable_names.xlsx}
```

## Format
- Long format: one row per timestep, first column `batch_id` (e.g. `A_003`, `B_001`). No time column — row order within a batch is the time order.
- Calibration and test batch IDs are interleaved (disjoint sets).
- `calibration_labels.csv`: `batch_id,label`; IDs exactly match calibration data.
- `variable_names.xlsx`: just the list of column names (no descriptions/units).
- No missing values anywhere.

| | Process A | Process B |
|---|---|---|
| Sensors | 11 | 20 |
| Calibration batches | 161 (121 accepted / 40 rejected) | 40 (35 accepted / 5 rejected) |
| Test batches | 161 | 40 |
| Rows per batch | variable: 196–2932 (median ~2720) | fixed: 5001 |

### Process A columns
`flow_rate_1, density, flow_rate_2, pressure_1, temperature_1, temperature_2, temperature_3, temperature_4, pressure_2, pressure_3, pressure_4`

- Some batches are very short (~200 rows vs ~2700) — possibly aborted runs; batch length may be predictive.
- Large outliers in all sensors (e.g. `flow_rate_1` ~131 typical, min −19; `flow_rate_2` ~1845 typical, min 347) — startup/shutdown or genuine faults.

### Process B columns (penicillin fermentation)
`substrate_concentration, dissolved_oxygen, biomass_concentration, penicillin_concentration, fermentation_volume, dissolved_co2, ph, reactor_temperature, reaction_heat, feed_rate, aeration_rate, agitator_power, feed_temperature, water_flow_rate, hot_cold_switch, base_flow, acid_flow, feed_substate_concentration, cooling_water_temperature, heating_water_temperature`

- Constant across all data → drop: `hot_cold_switch` (0), `feed_substate_concentration` (600), `cooling_water_temperature` (290), `heating_water_temperature` (323), `reactor_temperature` (~298).
- Near-constant: `ph`, `agitator_power`, `feed_temperature`.
- Informative signals are likely trajectories of substrate, biomass, penicillin, dissolved O₂/CO₂, feed rate, aeration, water flow.
- Only 5 rejected batches — prefer simple per-batch features + cross-validation over deep models.

## Submission
- `BRPMM_template.csv`: columns `batch_id, project, label`; 201 rows (161 A + 40 B), all pre-filled `rejected`. Overwrite `label` with predictions.
- Submit via the notebook's `submit(df, user="Team XX", token=...)`, which POSTs to the SDSC leaderboard API and returns metrics; `get_leaderboard()` shows standings. Scoring metric not documented — check the response of the first submission.

## Suggested approach
Aggregate each batch into one feature row (length, mean/std/min/max, slopes, start/end values per sensor), then train one classifier per process with class-imbalance handling.
