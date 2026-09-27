# Calibration report

Forecast rule identified by coordinate search against the 25 solved
samples in `dataset/sample_requests.csv`. The rule below is applied
unchanged to every evaluation request.

**Result:** field hits 110/150, amount exact 3/25, earliest exact 21/25, mean rel err 0.2560

## Selected rule

| Parameter | Value |
|---|---|
| `variable_estimator` | `'mean'` |
| `variable_lookback` | `4` |
| `variable_placement` | `'replay'` |
| `variable_cadence` | `'mode'` |
| `variable_occurrences` | `4` |
| `fixed_estimator` | `'mean'` |
| `fixed_lookback` | `2` |
| `income_require_stable` | `True` |
| `income_stability` | `'last2eq'` |
| `income_estimator` | `'last'` |
| `min_occurrences` | `2` |
| `live_within_months` | `1` |
| `income_min_occurrences` | `2` |
| `variable_income` | `'median'` |
| `variable_income_occurrences` | `6` |

## Per-sample results

| Sample | Expected safe | Got | Match | Expected earliest | Got | Match |
|---|---:|---:|:---:|---|---|:---:|
| request_01 | 25256 | 25256.00 | ✓ | 2024-03-03 | 2024-03-03 | ✓ |
| request_02 | 17229139.2 | 18451118.46 | ✗ | 2025-09-15 | 2025-09-15 | ✓ |
| request_03 | 873000 | 908407.05 | ✗ | 2019-11-15 | 2019-11-15 | ✓ |
| request_04 | 8401800 | 7499947.84 | ✗ | 2024-06-15 | 2024-06-15 | ✓ |
| request_05 | 737 | 0 | ✗ | — | — | ✓ |
| request_06 | 603.3 | 536.09 | ✗ | 2026-01-15 | 2026-01-15 | ✓ |
| request_07 | 87170.56 | 90699.19 | ✗ | 2024-10-23 | 2024-10-23 | ✓ |
| request_08 | 284.57 | 0 | ✗ | 2025-04-15 | — | ✗ |
| request_09 | 166.61 | 166.61 | ✓ | 2026-07-04 | 2026-07-04 | ✓ |
| request_10 | 12700 | 0 | ✗ | — | — | ✓ |
| request_11 | 12510645 | 12288424.06 | ✗ | 2025-07-15 | 2025-07-15 | ✓ |
| request_12 | 65164 | 58513.31 | ✗ | 2026-04-05 | — | ✗ |
| request_13 | 433.4 | 478.59 | ✗ | 2024-05-15 | — | ✗ |
| request_14 | 597.74 | 472.64 | ✗ | — | — | ✓ |
| request_15 | 83.05 | 94.96 | ✗ | — | — | ✓ |
| request_16 | 122500 | 122500.00 | ✓ | 2023-08-12 | 2023-08-12 | ✓ |
| request_17 | 243849.58 | 236490.08 | ✗ | 2026-03-15 | 2026-04-15 | ✗ |
| request_18 | 462 | 572.29 | ✗ | 2026-09-15 | 2026-09-15 | ✓ |
| request_19 | 28820 | 26521.37 | ✗ | 2024-09-15 | 2024-09-15 | ✓ |
| request_20 | 5400 | 9682.89 | ✗ | — | — | ✓ |
| request_21 | 1543.35 | 1573.80 | ✗ | 2026-04-15 | 2026-04-15 | ✓ |
| request_22 | 475.46 | 451.79 | ✗ | 2025-01-15 | 2025-01-15 | ✓ |
| request_23 | 9152 | 10071.26 | ✗ | 2025-07-15 | 2025-07-15 | ✓ |
| request_24 | 13420 | 11519.98 | ✗ | — | — | ✓ |
| request_25 | 1425000 | 0 | ✗ | — | — | ✓ |
