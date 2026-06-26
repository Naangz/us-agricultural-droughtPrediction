# Kansas Grand Summary of Scenario Results

This document compiles and summarizes all prediction results across various experimental scenarios and baseline configurations. The goal is to collect all results in one file for comparison, including class multipliers, best trials, selected features, test performance (raw and tuned), per-class F1, and raw/tuned classification reports.

## 📊 Scenario Comparison Table

The table below summarizes the key test and validation metrics for each scenario run on the dataset:

| Scenario / File | Best Trial | Features | Val F1 | Test Acc (Raw) | Test F1 (Raw) | Test Acc (Tuned) | Test F1 (Tuned) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (20 counties)** | ros_focal_no_cw_96x48 | All | 0.8480 / 0.8536 | 0.8177 | 0.8166 | 0.8014 | 0.7921 |
| **Scenario 2 (Correlation Top-15)** | ros_focal_inv_no_cw_128x64 | 15 | 0.6040 / 0.7630 | 0.6038 | 0.4755 | 0.5952 | 0.4595 |
| **Scenario 2A (Correlation Top-20)** | ros_focal_inv_no_cw_128x64 | 20 | 0.6340 / 0.6450 | 0.6222 | 0.4628 | 0.6282 | 0.4643 |
| **Scenario 2B (Correlation Top-25)** | ros_focal_no_cw_96x48 | 25 | 0.8444 / 0.8546 | 0.8184 | 0.8245 | 0.8182 | 0.8258 |
| **Scenario 2C (Correlation Top-30)** | ros_focal_no_cw_96x48 | 25 | 0.8444 / 0.8546 | 0.8184 | 0.8245 | 0.8182 | 0.8258 |
| **Scenario 3 (F1 Permutation Selection)** | ros_focal_inv_no_cw_128x64 | 25 | 0.8470 / 0.8511 | 0.8060 | 0.8062 | 0.8062 | 0.8008 |
| **Scenario 4 (Weather Only)** | ros_focal_no_cw_96x48 | 6 | 0.1746 / 0.2060 | 0.1031 | 0.0965 | 0.1438 | 0.1185 |
| **Scenario 5 (Weather + Lag Only)** | none_focal_inv_cw_64x32 | 20 | 0.2068 / 0.2498 | 0.1718 | 0.1380 | 0.1715 | 0.1152 |
| **Scenario 6 (No Drought History)** | none_focal_inv_cw_64x32 | 26 | 0.2510 / 0.2570 | 0.2031 | 0.1900 | 0.2134 | 0.1991 |
| **Scenario 7 (Drought History Only)** | ros_focal_inv_no_cw_128x64 | 12 | 0.8446 / 0.8481 | 0.8177 | 0.8202 | 0.8246 | 0.8289 |

---

## 🔍 Detailed Scenario Breakdown

### 📌 Baseline (20 counties)
- **Summary File:** `output_weekly_kansas_20counties/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features:** None / Full Feature Set (41 features)
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.800000`, `1.125549`, `0.805861`, `0.811980`, `0.753668`, `1.324415`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8480 | 0.8536 |
| **Test Accuracy** | 0.8177 | 0.8014 |
| **Test Macro F1** | 0.8166 | 0.7921 |
| **Test Weighted F1** | 0.8186 | 0.7990 |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9022 | 0.8969 |
| **D0** | 0.7773 | 0.7468 |
| **D1** | 0.7804 | 0.7571 |
| **D2** | 0.7721 | 0.7742 |
| **D3** | 0.7653 | 0.7054 |
| **D4** | 0.9026 | 0.8720 |

#### Classification Report (Raw)
```
======================================================================
              precision    recall  f1-score   support

        None     0.9323    0.8739    0.9022      1150
          D0     0.7495    0.8072    0.7773       975
          D1     0.7989    0.7627    0.7804       948
          D2     0.7640    0.7804    0.7721       560
          D3     0.7500    0.7812    0.7653       288
          D4     0.8764    0.9305    0.9026       259

    accuracy                         0.8177      4180
   macro avg     0.8118    0.8226    0.8166      4180
weighted avg     0.8208    0.8177    0.8186      4180
```

#### Classification Report (Tuned)
```
======================================================================
              precision    recall  f1-score   support

        None     0.8688    0.9270    0.8969      1150
          D0     0.7390    0.7549    0.7468       975
          D1     0.8238    0.7004    0.7571       948
          D2     0.7564    0.7929    0.7742       560
          D3     0.7673    0.6528    0.7054       288
          D4     0.7900    0.9730    0.8720       259

    accuracy                         0.8014      4180
   macro avg     0.7909    0.8001    0.7921      4180
weighted avg     0.8014    0.8014    0.7990      4180
```

---

### 📌 Scenario 2 (Correlation Top-15)
- **Summary File:** `output_weekly_kansas_scenario2/results_summary.txt`
- **Best Trial:** `ros_focal_inv_no_cw_128x64`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_inv_no_cw_128x64', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (128, 64), 'dropout': 0.28, 'dense_units': 96, 'lr': 0.0006, 'patience': 14}
  ```
- **Selected Features (15 total):**
  `D1_lag1`, `None_lag1`, `RH2M`, `PREC_roll12_std`, `T2M`, `T2M_lag4`, `RH2M_lag4`, `PREC_roll4_mean`, `RH2M_lag2`, `PRECTOTCORR`, `RH2M_lag8`, `PREC_lag2`, `RH2M_lag1`, `PREC_roll4_std`, `PREC_roll12_mean`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `0.756665`, `0.715555`, `0.846751`, `0.835950`, `1.229635`, `0.743149`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.6040 | 0.7630 |
| **Test Accuracy** | 0.6038 | 0.5952 |
| **Test Macro F1** | 0.4755 | 0.4595 |
| **Test Weighted F1** | 0.5947 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.8853 | 0.8884 |
| **D0** | 0.7333 | 0.7311 |
| **D1** | 0.5142 | 0.5159 |
| **D2** | 0.3870 | 0.3054 |
| **D3** | 0.2910 | 0.3165 |
| **D4** | 0.0423 | 0.0000 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.8915    0.8791    0.8853      1150
          D0     0.7292    0.7374    0.7333       975
          D1     0.6028    0.4483    0.5142       948
          D2     0.3346    0.4589    0.3870       560
          D3     0.2452    0.3576    0.2910       288
          D4     0.0539    0.0347    0.0423       259

    accuracy                         0.6038      4180
   macro avg     0.4762    0.4860    0.4755      4180
weighted avg     0.6172    0.6038    0.6057      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.8873    0.8896    0.8884      1150
          D0     0.7492    0.7138    0.7311       975
          D1     0.5981    0.4536    0.5159       948
          D2     0.3191    0.2929    0.3054       560
          D3     0.2139    0.6076    0.3165       288
          D4     0.0000    0.0000    0.0000       259

    accuracy                         0.5952      4180
   macro avg     0.4612    0.4929    0.4595      4180
weighted avg     0.6120    0.5952    0.5947      4180
```

---

### 📌 Scenario 2A (Correlation Top-20)
- **Summary File:** `output_weekly_kansas_scenario2A/results_summary.txt`
- **Best Trial:** `ros_focal_inv_no_cw_128x64`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_inv_no_cw_128x64', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (128, 64), 'dropout': 0.28, 'dense_units': 96, 'lr': 0.0006, 'patience': 14}
  ```
- **Selected Features (20 total):**
  `D1_lag1`, `None_lag1`, `RH2M`, `PREC_roll12_std`, `T2M`, `T2M_lag4`, `RH2M_lag4`, `PREC_roll4_mean`, `RH2M_lag2`, `PRECTOTCORR`, `RH2M_lag8`, `PREC_lag2`, `RH2M_lag1`, `PREC_roll4_std`, `PREC_roll12_mean`, `ALLSKY_SFC_SW_DWN`, `PREC_lag8`, `PREC_lag4`, `WS2M`, `PREC_lag1`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `0.918367`, `1.062982`, `1.185233`, `0.884054`, `1.002350`, `0.998703`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.6340 | 0.6450 |
| **Test Accuracy** | 0.6222 | 0.6282 |
| **Test Macro F1** | 0.4628 | 0.4643 |
| **Test Weighted F1** | 0.6171 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.8930 | 0.8893 |
| **D0** | 0.7623 | 0.7644 |
| **D1** | 0.5514 | 0.5847 |
| **D2** | 0.3824 | 0.3666 |
| **D3** | 0.1876 | 0.1808 |
| **D4** | 0.0000 | 0.0000 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.8973    0.8887    0.8930      1150
          D0     0.7385    0.7877    0.7623       975
          D1     0.5727    0.5316    0.5514       948
          D2     0.3278    0.4589    0.3824       560
          D3     0.2041    0.1736    0.1876       288
          D4     0.0000    0.0000    0.0000       259

    accuracy                         0.6222      4180
   macro avg     0.4567    0.4734    0.4628      4180
weighted avg     0.6070    0.6222    0.6127      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9099    0.8696    0.8893      1150
          D0     0.7371    0.7938    0.7644       975
          D1     0.5538    0.6192    0.5847       948
          D2     0.3507    0.3839    0.3666       560
          D3     0.1887    0.1736    0.1808       288
          D4     0.0000    0.0000    0.0000       259

    accuracy                         0.6282      4180
   macro avg     0.4567    0.4734    0.4643      4180
weighted avg     0.6079    0.6282    0.6171      4180
```

---

### 📌 Scenario 2B (Correlation Top-25)
- **Summary File:** `output_weekly_kansas_scenario2B/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (25 total):**
  `D1_lag1`, `None_lag1`, `RH2M`, `PREC_roll12_std`, `T2M`, `T2M_lag4`, `RH2M_lag4`, `PREC_roll4_mean`, `RH2M_lag2`, `PRECTOTCORR`, `RH2M_lag8`, `PREC_lag2`, `RH2M_lag1`, `PREC_roll4_std`, `PREC_roll12_mean`, `ALLSKY_SFC_SW_DWN`, `PREC_lag8`, `PREC_lag4`, `WS2M`, `PREC_lag1`, `T2M_lag8`, `PS`, `D2_lag1`, `D3_lag1`, `D4_lag1`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.202489`, `1.457245`, `1.096861`, `0.874849`, `0.550000`, `1.021403`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8444 | 0.8546 |
| **Test Accuracy** | 0.8184 | 0.8182 |
| **Test Macro F1** | 0.8245 | 0.8258 |
| **Test Weighted F1** | 0.8184 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9015 | 0.8948 |
| **D0** | 0.7720 | 0.7677 |
| **D1** | 0.7628 | 0.7555 |
| **D2** | 0.7771 | 0.8192 |
| **D3** | 0.8006 | 0.8045 |
| **D4** | 0.9328 | 0.9134 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9119    0.8913    0.9015      1150
          D0     0.7338    0.8144    0.7720       975
          D1     0.8373    0.7004    0.7628       948
          D2     0.7670    0.7875    0.7771       560
          D3     0.7508    0.8576    0.8006       288
          D4     0.9025    0.9653    0.9328       259

    accuracy                         0.8184      4180
   macro avg     0.8172    0.8361    0.8245      4180
weighted avg     0.8224    0.8184    0.8182      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9264    0.8652    0.8948      1150
          D0     0.6983    0.8523    0.7677       975
          D1     0.8429    0.6846    0.7555       948
          D2     0.7875    0.8536    0.8192       560
          D3     0.8770    0.7431    0.8045       288
          D4     0.8576    0.9768    0.9134       259

    accuracy                         0.8182      4180
   macro avg     0.8316    0.8293    0.8258      4180
weighted avg     0.8280    0.8182    0.8184      4180
```

---

### 📌 Scenario 2C (Correlation Top-30)
- **Summary File:** `output_weekly_kansas_scenario2C/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (25 total):**
  `D1_lag1`, `None_lag1`, `RH2M`, `PREC_roll12_std`, `T2M`, `T2M_lag4`, `RH2M_lag4`, `PREC_roll4_mean`, `RH2M_lag2`, `PRECTOTCORR`, `RH2M_lag8`, `PREC_lag2`, `RH2M_lag1`, `PREC_roll4_std`, `PREC_roll12_mean`, `ALLSKY_SFC_SW_DWN`, `PREC_lag8`, `PREC_lag4`, `WS2M`, `PREC_lag1`, `T2M_lag8`, `PS`, `D2_lag1`, `D3_lag1`, `D4_lag1`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.202489`, `1.457245`, `1.096861`, `0.874849`, `0.550000`, `1.021403`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8444 | 0.8546 |
| **Test Accuracy** | 0.8184 | 0.8182 |
| **Test Macro F1** | 0.8245 | 0.8258 |
| **Test Weighted F1** | 0.8184 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9015 | 0.8948 |
| **D0** | 0.7720 | 0.7677 |
| **D1** | 0.7628 | 0.7555 |
| **D2** | 0.7771 | 0.8192 |
| **D3** | 0.8006 | 0.8045 |
| **D4** | 0.9328 | 0.9134 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9119    0.8913    0.9015      1150
          D0     0.7338    0.8144    0.7720       975
          D1     0.8373    0.7004    0.7628       948
          D2     0.7670    0.7875    0.7771       560
          D3     0.7508    0.8576    0.8006       288
          D4     0.9025    0.9653    0.9328       259

    accuracy                         0.8184      4180
   macro avg     0.8172    0.8361    0.8245      4180
weighted avg     0.8224    0.8184    0.8182      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9264    0.8652    0.8948      1150
          D0     0.6983    0.8523    0.7677       975
          D1     0.8429    0.6846    0.7555       948
          D2     0.7875    0.8536    0.8192       560
          D3     0.8770    0.7431    0.8045       288
          D4     0.8576    0.9768    0.9134       259

    accuracy                         0.8182      4180
   macro avg     0.8316    0.8293    0.8258      4180
weighted avg     0.8280    0.8182    0.8184      4180
```

---

### 📌 Scenario 3 (F1 Permutation Selection)
- **Summary File:** `output_weekly_kansas_scenario3/results_summary.txt`
- **Best Trial:** `ros_focal_inv_no_cw_128x64`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_inv_no_cw_128x64', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (128, 64), 'dropout': 0.28, 'dense_units': 96, 'lr': 0.0006, 'patience': 14}
  ```
- **Selected Features (25 total):**
  `D3_lag1`, `D1_lag1`, `D2_lag1`, `None_lag1`, `D0_lag1`, `D2_lag2`, `D3_lag2`, `PREC_lag2`, `week_cos`, `None_lag2`, `T2M_roll4_mean`, `PREC_roll4_mean`, `D4_lag1`, `D4_lag2`, `T2M_roll12_mean`, `T2M_lag2`, `T2M_lag1`, `PREC_lag8`, `PRECTOTCORR`, `RH2M_lag1`, `D0_lag2`, `RH2M_lag2`, `T2M_lag4`, `T2M_lag8`, `PREC_lag1`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.202415`, `1.208490`, `1.154968`, `1.130114`, `0.908317`, `1.266484`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8470 | 0.8511 |
| **Test Accuracy** | 0.8060 | 0.8062 |
| **Test Macro F1** | 0.8062 | 0.8008 |
| **Test Weighted F1** | 0.8054 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.8987 | 0.8983 |
| **D0** | 0.7466 | 0.7519 |
| **D1** | 0.7698 | 0.7742 |
| **D2** | 0.7547 | 0.7616 |
| **D3** | 0.7529 | 0.7210 |
| **D4** | 0.9142 | 0.8980 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.8983    0.8991    0.8987      1150
          D0     0.7593    0.7344    0.7466       975
          D1     0.7760    0.7637    0.7698       948
          D2     0.7595    0.7500    0.7547       560
          D3     0.7121    0.7986    0.7529       288
          D4     0.8845    0.9459    0.9142       259

    accuracy                         0.8060      4180
   macro avg     0.7983    0.8153    0.8062      4180
weighted avg     0.8059    0.8060    0.8056      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.8983    0.8983    0.8983      1150
          D0     0.7594    0.7446    0.7519       975
          D1     0.7838    0.7648    0.7742       948
          D2     0.7453    0.7786    0.7616       560
          D3     0.7538    0.6910    0.7210       288
          D4     0.8367    0.9691    0.8980       259

    accuracy                         0.8062      4180
   macro avg     0.7962    0.8077    0.8008      4180
weighted avg     0.8056    0.8062    0.8054      4180
```

---

### 📌 Scenario 4 (Weather Only)
- **Summary File:** `output_weekly_kansas_scenario4/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': True, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (6 total):**
  `ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.382284`, `1.002400`, `1.268524`, `0.863332`, `0.580552`, `1.083837`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.1746 | 0.2060 |
| **Test Accuracy** | 0.1031 | 0.1438 |
| **Test Macro F1** | 0.0965 | 0.1185 |
| **Test Weighted F1** | 0.1201 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.0437 | 0.1256 |
| **D0** | 0.1070 | 0.0758 |
| **D1** | 0.0121 | 0.1301 |
| **D2** | 0.1897 | 0.2062 |
| **D3** | 0.0880 | 0.0000 |
| **D4** | 0.1385 | 0.1734 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.6500    0.0226    0.0437      1150
          D0     0.2708    0.0667    0.1070       975
          D1     0.1500    0.0063    0.0121       948
          D2     0.1833    0.1964    0.1897       560
          D3     0.0539    0.2396    0.0880       288
          D4     0.0783    0.5985    0.1385       259

    accuracy                         0.1031      4180
   macro avg     0.2311    0.1883    0.0965      4180
weighted avg     0.3091    0.1031    0.0798      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.5786    0.0704    0.1256      1150
          D0     0.2687    0.0441    0.0758       975
          D1     0.1645    0.1076    0.1301       948
          D2     0.1811    0.2393    0.2062       560
          D3     0.0000    0.0000    0.0000       288
          D4     0.0956    0.9305    0.1734       259

    accuracy                         0.1438      4180
   macro avg     0.2148    0.2320    0.1185      4180
weighted avg     0.2894    0.1438    0.1201      4180
```

---

### 📌 Scenario 5 (Weather + Lag Only)
- **Summary File:** `output_weekly_kansas_scenario5/results_summary.txt`
- **Best Trial:** `none_focal_inv_cw_64x32`
- **Best Trial Config:**
  ```python
  {'name': 'none_focal_inv_cw_64x32', 'balancer': 'NONE', 'use_class_weight': True, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (64, 32), 'dropout': 0.35, 'dense_units': 64, 'lr': 0.001, 'patience': 16}
  ```
- **Selected Features (20 total):**
  `ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`, `PREC_lag1`, `PREC_lag2`, `PREC_lag4`, `PREC_lag8`, `T2M_lag1`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `RH2M_lag1`, `RH2M_lag2`, `RH2M_lag4`, `RH2M_lag8`, `week_sin`, `week_cos`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.800000`, `0.655210`, `0.839104`, `0.859227`, `1.800000`, `0.672980`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.2068 | 0.2498 |
| **Test Accuracy** | 0.1718 | 0.1715 |
| **Test Macro F1** | 0.1380 | 0.1152 |
| **Test Weighted F1** | 0.1532 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.1603 | 0.4184 |
| **D0** | 0.1939 | 0.0000 |
| **D1** | 0.2634 | 0.1209 |
| **D2** | 0.0000 | 0.0034 |
| **D3** | 0.0643 | 0.1485 |
| **D4** | 0.1458 | 0.0000 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.6562    0.0913    0.1603      1150
          D0     0.2860    0.1467    0.1939       975
          D1     0.2108    0.3513    0.2634       948
          D2     0.0000    0.0000    0.0000       560
          D3     0.0437    0.1215    0.0643       288
          D4     0.0895    0.3938    0.1458       259

    accuracy                         0.1718      4180
   macro avg     0.2144    0.1841    0.1380      4180
weighted avg     0.3036    0.1718    0.1625      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.5529    0.3365    0.4184      1150
          D0     0.0000    0.0000    0.0000       975
          D1     0.2808    0.0770    0.1209       948
          D2     0.0500    0.0018    0.0034       560
          D3     0.0810    0.8889    0.1485       288
          D4     0.0000    0.0000    0.0000       259

    accuracy                         0.1715      4180
   macro avg     0.1608    0.2174    0.1152      4180
weighted avg     0.2281    0.1715    0.1532      4180
```

---

### 📌 Scenario 6 (No Drought History)
- **Summary File:** `output_weekly_kansas_scenario6/results_summary.txt`
- **Best Trial:** `none_focal_inv_cw_64x32`
- **Best Trial Config:**
  ```python
  {'name': 'none_focal_inv_cw_64x32', 'balancer': 'NONE', 'use_class_weight': True, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (64, 32), 'dropout': 0.35, 'dense_units': 64, 'lr': 0.001, 'patience': 16}
  ```
- **Selected Features (26 total):**
  `ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`, `PREC_lag1`, `PREC_lag2`, `PREC_lag4`, `PREC_lag8`, `T2M_lag1`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `RH2M_lag1`, `RH2M_lag2`, `RH2M_lag4`, `RH2M_lag8`, `PREC_roll4_mean`, `PREC_roll4_std`, `PREC_roll12_mean`, `PREC_roll12_std`, `T2M_roll4_mean`, `T2M_roll12_mean`, `week_sin`, `week_cos`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.687043`, `1.043530`, `1.012851`, `1.031347`, `0.936718`, `0.877563`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.2510 | 0.2570 |
| **Test Accuracy** | 0.2031 | 0.2134 |
| **Test Macro F1** | 0.1900 | 0.1991 |
| **Test Weighted F1** | 0.2232 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.1741 | 0.2366 |
| **D0** | 0.2771 | 0.2610 |
| **D1** | 0.2509 | 0.2584 |
| **D2** | 0.1419 | 0.1511 |
| **D3** | 0.0892 | 0.0836 |
| **D4** | 0.2070 | 0.2041 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.5042    0.1052    0.1741      1150
          D0     0.2920    0.2636    0.2771       975
          D1     0.2142    0.3027    0.2509       948
          D2     0.2033    0.1089    0.1419       560
          D3     0.0565    0.2118    0.0892       288
          D4     0.1824    0.2394    0.2070       259

    accuracy                         0.2031      4180
   macro avg     0.2421    0.2053    0.1900      4180
weighted avg     0.2978    0.2031    0.2074      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.4763    0.1574    0.2366      1150
          D0     0.2936    0.2349    0.2610       975
          D1     0.2155    0.3228    0.2584       948
          D2     0.2000    0.1214    0.1511       560
          D3     0.0541    0.1840    0.0836       288
          D4     0.1964    0.2124    0.2041       259

    accuracy                         0.2134      4180
   macro avg     0.2393    0.2055    0.1991      4180
weighted avg     0.2911    0.2134    0.2232      4180
```

---

### 📌 Scenario 7 (Drought History Only)
- **Summary File:** `output_weekly_kansas_scenario7/results_summary.txt`
- **Best Trial:** `ros_focal_inv_no_cw_128x64`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_inv_no_cw_128x64', 'balancer': 'ROS', 'use_class_weight': True, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (128, 64), 'dropout': 0.28, 'dense_units': 96, 'lr': 0.0006, 'patience': 14}
  ```
- **Selected Features (12 total):**
  `None_lag1`, `D0_lag1`, `D1_lag1`, `D2_lag1`, `D3_lag1`, `D4_lag1`, `None_lag2`, `D0_lag2`, `D1_lag2`, `D2_lag2`, `D3_lag2`, `D4_lag2`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.498996`, `1.214531`, `1.228827`, `1.199457`, `0.660953`, `0.768868`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8446 | 0.8481 |
| **Test Accuracy** | 0.8177 | 0.8246 |
| **Test Macro F1** | 0.8202 | 0.8289 |
| **Test Weighted F1** | 0.8240 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9043 | 0.9045 |
| **D0** | 0.7743 | 0.7661 |
| **D1** | 0.7767 | 0.7784 |
| **D2** | 0.7549 | 0.8062 |
| **D3** | 0.7695 | 0.7891 |
| **D4** | 0.9416 | 0.9294 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9215    0.8878    0.9043      1150
          D0     0.7364    0.8164    0.7743       975
          D1     0.8272    0.7321    0.7767       948
          D2     0.7895    0.7232    0.7549       560
          D3     0.6866    0.8750    0.7695       288
          D4     0.9191    0.9653    0.9416       259

    accuracy                         0.8177      4180
   macro avg     0.8134    0.8333    0.8202      4180
weighted avg     0.8229    0.8177    0.8181      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.8994    0.9096    0.9045      1150
          D0     0.7485    0.7846    0.7661       975
          D1     0.8243    0.7373    0.7784       948
          D2     0.7756    0.8393    0.8062       560
          D3     0.8282    0.7535    0.7891       288
          D4     0.8961    0.9653    0.9294       259

    accuracy                         0.8246      4180
   macro avg     0.8287    0.8316    0.8289      4180
weighted avg     0.8255    0.8246    0.8240      4180
```

---