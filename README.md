# BiLSTM Weekly Drought Classification (Kansas - Nebraska 20 Counties)

Project description: This repository implements a Bi-directional LSTM (BiLSTM) sequence classifier to predict weekly drought categories for Kansas counties. It uses the previous 52 weeks of county-level weather and drought history to output one of six drought classes (None, D0..D4). The repo includes data integration scripts, feature engineering, model training/evaluation (`BiLSTM_Weekly_Kansas_Clean.py`), scenario experiments, and output analysis.

## Statement of the Problem

Accurate weekly monitoring and short-term prediction of drought severity at the county level is essential for agricultural planning and resource allocation. Existing operational drought indicators are often descriptive and lag actual hydrometeorological conditions; there is a need for a data-driven model that leverages historical weather and drought persistence signals to predict the categorical drought state one week ahead. This work addresses the problem of mapping 52 weeks of historical county-level weather and USDM-derived drought signals to a single weekly drought class.

## Objectives

- **Primary:** Build and validate a BiLSTM sequence classifier that predicts the weekly drought class (None, D0..D4) for Kansas counties using the previous 52 weeks of features.
- **Secondary:** Engineer robust temporal features and drought-memory signals, compare balancing and loss strategies (e.g., ROS, focal loss), tune model/configuration by validation macro-F1, and produce reproducible training and evaluation artifacts for an undergraduate final-year project.

## Scope and Limitations

- **Geographic scope:** Model and experiments are limited to a fixed set of 20 Kansas counties (see `Integrated_weekly_KAN_20counties.csv`).
- **Temporal scope:** Uses weekly-aggregated NASA POWER weather and USDM weekly drought coverage covering ~2010–2025 as available in the repository.
- **Inputs & outputs:** Inputs are historical weather and cumulative USDM coverage features; output is a single categorical drought class per county-week. The model does not perform causal attribution or hydrological simulation.
- **Limitations:** Labels derive from USDM coverage and include aggregation/measurement noise; class imbalance and temporal non-stationarity can affect generalization; spatial transferability beyond the selected counties is not evaluated; model predictions are not real-time operational products and require additional validation for deployment.
- **Resource assumptions:** Training experiments assume access to a moderately capable workstation (GPU recommended for faster runs) and the Python dependencies listed in `requirements.txt`.

## Why Kansas and Nebraska?

This repository contains experiments for two nearby U.S. states (Kansas and Nebraska) to demonstrate the model's spatial generalization potential. The Kansas dataset (20 counties) is the primary development and evaluation corpus used for the undergraduate final project, while the Nebraska scripts and example outputs are provided to:

- show how the same pipeline and model architecture can be applied to a different but climatically related region,
- provide a simple transfer / generalization check (retrain or fine-tune on Nebraska data to compare performance), and
- surface limitations due to differing label distributions, county selections, and local climate regimes.

Note: Nebraska experiments are intended as illustrative transfer tests rather than fully tuned, production-ready models. Successful generalization requires careful retraining, calibration, and evaluation on region-specific data.

## What This Model Does
This project predicts **weekly drought category** for Kansas counties using weather history and drought persistence signals.

- Input: weekly county-level weather + drought history
- Output: one of 6 drought classes per county-week
  - `0: None`
  - `1: D0`
  - `2: D1`
  - `3: D2`
  - `4: D3`
  - `5: D4`

The model is a **sequence classifier**: it uses the previous **52 weeks** to classify drought state for the target week.

---

## Main Training Script
- [`BiLSTM_Weekly_Kansas_Clean.py`](./BiLSTM_Weekly_Kansas_Clean.py)

Default dataset in script:
- [`Integrated_weekly_KAN_20counties.csv`](./Integrated_weekly_KAN_20counties.csv)

Outputs are written to:
- `output_weekly_kansas_20counties/`

---

## Datasets Explained

### Raw source datasets
- [`NASA_POWER_daily_2010_2025_KAN.csv`](./NASA_POWER_daily_2010_2025_KAN.csv)
  - Daily NASA POWER weather for Kansas
  - File rows in this repo: `116,880`
- [`KAN_dm_export_20100101_20251231.csv`](./KAN_dm_export_20100101_20251231.csv)
  - Weekly USDM drought coverage by county
  - File rows in this repo: `87,780`

### Integrated datasets
- [`Integrated_weekly_KAN.csv`](./Integrated_weekly_KAN.csv)
  - Full Kansas integrated weekly table
  - File rows in this repo: `77,748` across `93` counties
- [`Integrated_weekly_KAN_20counties.csv`](./Integrated_weekly_KAN_20counties.csv)
  - Reduced 20-county Kansas integrated weekly table used by training script
  - File rows in this repo: `16,720` across `20` counties
  - Date range: `2009-12-29` to `2025-12-30`

### How integration is built
Integration logic is implemented in:
- [`integrate_weekly_nasa_usdm_20_counties.py`](./integrate_weekly_nasa_usdm_20_counties.py)

Steps:
1. NASA daily weather is aggregated to weekly means using USDM-aligned week start.
2. USDM weekly table is cleaned and filtered by state.
3. A fixed county set (20 counties) is selected deterministically.
4. NASA weekly features and USDM weekly drought fields are merged on `week_start`.

### Integrated dataset schema
Columns in `Integrated_weekly_KAN_20counties.csv`:
- Time/index: `week_start, ValidEnd, Year, Month, YearWeek`
- Location: `FIPS, County, State`
- Weather: `ALLSKY_SFC_SW_DWN, PRECTOTCORR, PS, RH2M, T2M, WS2M`
- Drought coverage: `None, D0, D1, D2, D3, D4`

Meaning of drought coverage columns:
- They are cumulative area percentages by severity threshold.
- Example: `D2` means area in at least D2 drought severity.

---

## Data and Labeling Pipeline

### 1) Load and sort
- Reads integrated weekly dataset
- Converts `week_start` and `ValidEnd` to datetime
- Sorts by county (`FIPS`) and time (`week_start`)

### 2) Convert cumulative drought fields into class labels
The source has cumulative drought coverage columns (`None, D0, D1, D2, D3, D4`).

The script decumulates these into PMF-like components:
- `PMF_D4 = D4`
- `PMF_D3 = D3 - D4`
- `PMF_D2 = D2 - D3`
- `PMF_D1 = D1 - D2`
- `PMF_D0 = D0 - D1`
- `PMF_None = None`

Then label is assigned by `argmax` over `PMF_None..PMF_D4`.

### 3) Feature list and calculations

Total model input features after engineering: **41**

Base weather features (direct):
- `ALLSKY_SFC_SW_DWN`
- `PRECTOTCORR`
- `PS`
- `RH2M`
- `T2M`
- `WS2M`

Lag features by county (`groupby(FIPS).shift(k)`):
- `PREC_lag1, PREC_lag2, PREC_lag4, PREC_lag8` from `PRECTOTCORR`
- `T2M_lag1, T2M_lag2, T2M_lag4, T2M_lag8` from `T2M`
- `RH2M_lag1, RH2M_lag2, RH2M_lag4, RH2M_lag8` from `RH2M`

Rolling features by county (all shifted by 1 week before rolling):
- `PREC_roll4_mean = mean(PRECTOTCORR[t-1 ... t-4])`
- `PREC_roll4_std = std(PRECTOTCORR[t-1 ... t-4])`
- `PREC_roll12_mean = mean(PRECTOTCORR[t-1 ... t-12])`
- `PREC_roll12_std = std(PRECTOTCORR[t-1 ... t-12])`
- `T2M_roll4_mean = mean(T2M[t-1 ... t-4])`
- `T2M_roll12_mean = mean(T2M[t-1 ... t-12])`

Seasonality encoding:
- `week_sin = sin(2*pi*iso_week/52)`
- `week_cos = cos(2*pi*iso_week/52)`

Drought memory features:
- Lag-1 and lag-2 for each cumulative drought series:
  - `None_lag1, D0_lag1, D1_lag1, D2_lag1, D3_lag1, D4_lag1`
  - `None_lag2, D0_lag2, D1_lag2, D2_lag2, D3_lag2, D4_lag2`
- Composite carryover signals:
  - `drought_carryover_lag1 = D0_lag1 + D1_lag1 + 0.5*D2_lag1`
  - `severe_carryover_lag1 = D3_lag1 + D4_lag1`

Interaction feature:
- `heat_dry_stress = T2M * (1 - RH2M/100)`

Complete flat feature list used by the model (`feature_cols`):
1. `ALLSKY_SFC_SW_DWN`
2. `PRECTOTCORR`
3. `PS`
4. `RH2M`
5. `T2M`
6. `WS2M`
7. `PREC_lag1`
8. `PREC_lag2`
9. `PREC_lag4`
10. `PREC_lag8`
11. `T2M_lag1`
12. `T2M_lag2`
13. `T2M_lag4`
14. `T2M_lag8`
15. `RH2M_lag1`
16. `RH2M_lag2`
17. `RH2M_lag4`
18. `RH2M_lag8`
19. `PREC_roll4_mean`
20. `PREC_roll4_std`
21. `PREC_roll12_mean`
22. `PREC_roll12_std`
23. `T2M_roll4_mean`
24. `T2M_roll12_mean`
25. `week_sin`
26. `week_cos`
27. `None_lag1`
28. `D0_lag1`
29. `D1_lag1`
30. `D2_lag1`
31. `D3_lag1`
32. `D4_lag1`
33. `None_lag2`
34. `D0_lag2`
35. `D1_lag2`
36. `D2_lag2`
37. `D3_lag2`
38. `D4_lag2`
39. `drought_carryover_lag1`
40. `severe_carryover_lag1`
41. `heat_dry_stress`

Rows with missing values in required features are dropped.

---

## Temporal Split Strategy
Time-based split (no random split):
- Train: `week_start <= 2019-12-31`
- Validation: `2020-01-01` to `2021-12-31`
- Test: `week_start >= 2022-01-01`

Why this matters:
- Mimics real forecasting deployment
- Avoids leakage from future into training

---

## Scaling and Sequence Construction

### Scaling
- `MinMaxScaler` is fit on **train only**
- Same scaler applied to full timeline (train/val/test rows)

### Sequences
- Sequence length = `52`
- Built per county (`FIPS`), sorted by time
- Target label is class at the final timestep in each sequence
- Validation and test targets are date-filtered, while still allowing historical context from earlier weeks

---

## Model Architecture
The model is a configurable BiLSTM classifier:

1. Input: `(52, n_features)`
2. `Bidirectional(LSTM(u1, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))`
3. `BatchNormalization`
4. `Dropout`
5. `Bidirectional(LSTM(u2, dropout=0.2, recurrent_dropout=0.1))`
6. `BatchNormalization`
7. `Dropout`
8. `Dense(dense_units, activation='relu')`
9. `Dropout`
10. `Dense(6, activation='softmax')`

Loss:
- **Categorical focal loss** (configurable `gamma`, optional class-wise `alpha`)

Optimizer:
- Adam with configurable learning rate

Metric for model selection:
- Validation **macro-F1** via custom callback

---

## Balancing Strategy
The script runs a small trial set (`TRIAL_CONFIGS`) and picks best by validation macro-F1.

Trial dimensions include:
- Balancer: `ROS` or `NONE` (framework also supports `SMOTE`, `BSMOTE`, `ADASYN`, `SMOTETOMEK`, `SMOTEENN`)
- Class weights on/off
- Focal loss params
- Architecture size/dropout
- Learning rate and patience

### Current best observed trial
From latest summary:
- Trial: `ros_focal_no_cw_96x48`
- Balancing: `ROS`
- Class weights: off
- Focal gamma: `1.5`
- LSTM units: `(96, 48)`

---

## Softmax and Final Decision Rule

### Raw prediction
The network outputs class probabilities via softmax:
- `y_pred_raw = argmax(softmax_probs)`

### Optional post-softmax calibration
Script also tunes **class multipliers** on validation data only:
- `y_pred_tuned = argmax(softmax_probs * class_multipliers)`

This is a decision-layer calibration aimed at improving macro-F1. It is not temperature scaling.

Important: in your latest run, tuned validation improved, but tuned test was slightly lower than raw test macro-F1.

---

## Training Outputs
In `output_weekly_kansas_20counties/`:
- `best_model.keras` (best trial model)
- `best_model_<trial_name>.keras` (per-trial checkpoints)
- `training_history.png`
- `confusion_matrix.png`
- `per_class_f1.png`
- `results_summary.txt`

`results_summary.txt` includes:
- Best trial config
- Validation macro-F1 and tuned validation macro-F1
- Raw vs tuned test metrics
- Trial leaderboard
- Per-class F1 and classification report

---

## How to Run
Recommended environment:
- Conda env: `bilstm-gpu`

Run:
```bash
conda run -n bilstm-gpu python BiLSTM_Weekly_Kansas_Clean.py
```

---

## Interpreting Success for This Project
Your stated target is macro-F1 >= 0.80.

From your latest result file:
- **Raw test macro-F1 = 0.8137** (target achieved)
- Tuned test macro-F1 = 0.8039 (still above 0.80)

If deployment priority is best generalization on held-out test, raw softmax argmax is currently the stronger choice.
