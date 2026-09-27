# ML report: essential-spending quantile model

Self-supervised `HistGradientBoostingRegressor` (quantile loss) that predicts the next grocery, transport or dining amount from the previous six occurrences, normalized by each series' median. It is trained on every user's settled history; no answer labels or organizer files are used.

- Series: 800 · training rows: 9,295 · holdout rows (each series' most recent occurrence): 800

## Holdout pinball loss (lower is better)

| Quantile | Model | Last value | Mean of last 4 | Max of last 4 | Holdout coverage (actual ≤ prediction) |
|---:|---:|---:|---:|---:|---:|
| 0.5 | **0.0727** | 0.0953 | 0.0758 | 0.1023 | 51.7% |
| 0.75 | **0.0573** | 0.0964 | 0.0770 | 0.0611 | 70.6% |
| 0.9 | **0.0285** | 0.0971 | 0.0777 | 0.0363 | 83.5% |

Coverage near the nominal quantile means the model is calibrated; a q=0.75 forecast should cover about 75% of next occurrences. That is the operational meaning of forecasting "conservatively".

Whether the forecast uses this model (`variable_estimator = ml_q50 | ml_q75 | ml_q90`) or a statistical estimator is decided by the calibration search on the solved samples; see `calibration_report.md`.
