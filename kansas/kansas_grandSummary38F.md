# Kansas Grand Summary of Scenario Results

This document compiles and summarizes all prediction results across various experimental scenarios and baseline configurations. The goal is to collect all results in one file for comparison, including class multipliers, best trials, selected features, test performance (raw and tuned), per-class F1, and raw/tuned classification reports.

## 📊 Scenario Comparison Table

The table below summarizes the key test and validation metrics for each scenario run on the dataset:

| Scenario / File | Best Trial | Features | Val F1 | Test Acc (Raw) | Test F1 (Raw) | Test Acc (Tuned) | Test F1 (Tuned) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | - | All | - | 0.6968 | 0.7327 | - | - |
| **Clean Baseline ExpB** | - | All | - | 0.6774 | 0.7303 | - | - |
| **Baseline (20 counties)** (results_summary - KAN.txt) | ros_focal_no_cw_96x48 | All | 0.8514 / 0.8553 | 0.8089 | 0.8137 | 0.8067 | 0.8039 |
| **Baseline (20 counties)** | ros_focal_no_cw_96x48 | All | 0.8529 / 0.8553 | 0.8093 | 0.8025 | 0.8081 | 0.7970 |
| **Scenario 2 (Correlation Top-15)** | ros_focal_inv_no_cw_128x64 | 15 | 0.6573 / 0.7108 | 0.5751 | 0.4637 | 0.5931 | 0.4635 |
| **Scenario 2A (Correlation Top-20)** | ros_focal_no_cw_96x48 | 20 | 0.7709 / 0.8049 | 0.5971 | 0.4755 | 0.5856 | 0.4560 |
| **Scenario 2B (Correlation Top-25)** | ros_focal_no_cw_96x48 | 25 | 0.8453 / 0.8522 | 0.8117 | 0.8081 | 0.7969 | 0.7896 |
| **Scenario 2C (Correlation Top-30)** | ros_focal_no_cw_96x48 | 25 | 0.8453 / 0.8522 | 0.8117 | 0.8081 | 0.7969 | 0.7896 |
| **Scenario 3 (F1 Permutation Selection)** | ros_focal_no_cw_96x48 | 20 | 0.8486 / 0.8516 | 0.8132 | 0.8057 | 0.8129 | 0.7940 |
| **Scenario 4 (Weather Only)** | ros_focal_no_cw_96x48 | 6 | 0.1746 / 0.2060 | 0.1031 | 0.0965 | 0.1438 | 0.1185 |
| **Scenario 5 (Weather + Lag Only)** | none_focal_inv_cw_64x32 | 20 | 0.2068 / 0.2498 | 0.1718 | 0.1380 | 0.1715 | 0.1152 |
| **Scenario 6 (No Drought History)** | ros_focal_inv_no_cw_128x64 | 26 | 0.2319 / 0.2457 | 0.2505 | 0.1776 | 0.2404 | 0.1715 |
| **Scenario 7 (Drought History Only)** | ros_focal_inv_no_cw_128x64 | 12 | 0.8446 / 0.8481 | 0.8177 | 0.8202 | 0.8246 | 0.8289 |

---

## 🔍 Detailed Scenario Breakdown

### 📌 Clean Baseline
- **Summary File:** `output_weekly_kansas_clean/results_summary.txt`
- **Best Trial:** `N/A`
- **Selected Features:** None / Full Feature Set (41 features)

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | - | - |
| **Test Accuracy** | 0.6968 | - |
| **Test Macro F1** | 0.7327 | - |
| **Test Weighted F1** | 0.6970 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.7215 | - |
| **D0** | 0.5522 | - |
| **D1** | 0.7586 | - |
| **D2** | 0.6768 | - |
| **D3** | 0.7715 | - |
| **D4** | 0.9154 | - |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.7086    0.7349    0.7215      4361
          D0     0.5348    0.5707    0.5522      3673
          D1     0.7659    0.7515    0.7586      3444
          D2     0.8434    0.5652    0.6768      1534
          D3     0.7303    0.8177    0.7715       861
          D4     0.8842    0.9488    0.9154       821

    accuracy                         0.6968     14694
   macro avg     0.7445    0.7315    0.7327     14694
weighted avg     0.7037    0.6968    0.6970     14694
```

---

### 📌 Clean Baseline ExpB
- **Summary File:** `output_weekly_kansas_clean_expB/results_summary.txt`
- **Best Trial:** `N/A`
- **Selected Features:** None / Full Feature Set (41 features)

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | - | - |
| **Test Accuracy** | 0.6774 | - |
| **Test Macro F1** | 0.7303 | - |
| **Test Weighted F1** | 0.6818 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.6352 | - |
| **D0** | 0.5721 | - |
| **D1** | 0.7602 | - |
| **D2** | 0.7223 | - |
| **D3** | 0.7775 | - |
| **D4** | 0.9147 | - |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.7439    0.5542    0.6352      4361
          D0     0.4868    0.6937    0.5721      3673
          D1     0.7785    0.7427    0.7602      3444
          D2     0.8375    0.6349    0.7223      1534
          D3     0.7779    0.7770    0.7775       861
          D4     0.8736    0.9598    0.9147       821

    accuracy                         0.6774     14694
   macro avg     0.7497    0.7271    0.7303     14694
weighted avg     0.7068    0.6774    0.6818     14694
```

---

### 📌 Baseline (20 counties) (results_summary - KAN.txt)
- **Summary File:** `output_weekly_kansas_20counties/results_summary - KAN.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features:** None / Full Feature Set (41 features)
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.800000`, `1.800000`, `1.132381`, `0.743147`, `0.550000`, `1.083627`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8514 | 0.8553 |
| **Test Accuracy** | 0.8089 | 0.8067 |
| **Test Macro F1** | 0.8137 | 0.8039 |
| **Test Weighted F1** | 0.8095 | 0.8061 |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.8977 | - |
| **D0** | 0.7614 | - |
| **D1** | 0.7435 | - |
| **D2** | 0.8028 | - |
| **D3** | 0.7348 | - |
| **D4** | 0.8835 | - |

#### Classification Report (Raw)
```
======================================================================
              precision    recall  f1-score   support

        None     0.9229    0.8739    0.8977      1150
          D0     0.6921    0.8462    0.7614       975
          D1     0.8288    0.6741    0.7435       948
          D2     0.7817    0.8250    0.8028       560
          D3     0.8462    0.6493    0.7348       288
          D4     0.8038    0.9807    0.8835       259

    accuracy                         0.8067      4180
   macro avg     0.8126    0.8082    0.8039      4180
weighted avg     0.8161    0.8067    0.8061      4180
```

---

### 📌 Baseline (20 counties)
- **Summary File:** `output_weekly_kansas_20counties/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features:** None / Full Feature Set (41 features)
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.077242`, `1.023779`, `0.952111`, `0.914989`, `0.879959`, `1.072059`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8529 | 0.8553 |
| **Test Accuracy** | 0.8093 | 0.8081 |
| **Test Macro F1** | 0.8025 | 0.7970 |
| **Test Weighted F1** | 0.8102 | 0.8087 |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.8999 | 0.9002 |
| **D0** | 0.7817 | 0.7836 |
| **D1** | 0.7726 | 0.7725 |
| **D2** | 0.7480 | 0.7520 |
| **D3** | 0.7188 | 0.6954 |
| **D4** | 0.8938 | 0.8781 |

#### Classification Report (Raw)
```
======================================================================
              precision    recall  f1-score   support

        None     0.9345    0.8678    0.8999      1150
          D0     0.7431    0.8246    0.7817       975
          D1     0.8126    0.7363    0.7726       948
          D2     0.7408    0.7554    0.7480       560
          D3     0.6901    0.7500    0.7188       288
          D4     0.8502    0.9421    0.8938       259

    accuracy                         0.8093      4180
   macro avg     0.7952    0.8127    0.8025      4180
weighted avg     0.8142    0.8093    0.8102      4180
```

#### Classification Report (Tuned)
```
======================================================================
              precision    recall  f1-score   support

        None     0.9320    0.8704    0.9002      1150
          D0     0.7391    0.8338    0.7836       975
          D1     0.8216    0.7289    0.7725       948
          D2     0.7435    0.7607    0.7520       560
          D3     0.6894    0.7014    0.6954       288
          D4     0.8194    0.9459    0.8781       259

    accuracy                         0.8081      4180
   macro avg     0.7908    0.8069    0.7970      4180
weighted avg     0.8130    0.8081    0.8087      4180
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
  `D1_lag1`, `None_lag1`, `T2M_lag2`, `PREC_roll12_std`, `RH2M`, `RH2M_lag2`, `RH2M_lag8`, `PREC_roll4_std`, `T2M_roll12_mean`, `PRECTOTCORR`, `RH2M_lag4`, `ALLSKY_SFC_SW_DWN`, `PREC_lag2`, `RH2M_lag1`, `PREC_lag1`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `0.846579`, `0.965049`, `0.986662`, `0.876644`, `1.102061`, `0.580173`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.6573 | 0.7108 |
| **Test Accuracy** | 0.5751 | 0.5931 |
| **Test Macro F1** | 0.4637 | 0.4635 |
| **Test Weighted F1** | 0.5971 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.8900 | 0.8874 |
| **D0** | 0.7127 | 0.7158 |
| **D1** | 0.5000 | 0.5318 |
| **D2** | 0.3330 | 0.3224 |
| **D3** | 0.2315 | 0.3118 |
| **D4** | 0.1148 | 0.0116 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.8862    0.8939    0.8900      1150
          D0     0.7526    0.6769    0.7127       975
          D1     0.6091    0.4241    0.5000       948
          D2     0.3455    0.3214    0.3330       560
          D3     0.1794    0.3264    0.2315       288
          D4     0.0913    0.1544    0.1148       259

    accuracy                         0.5751      4180
   macro avg     0.4773    0.4662    0.4637      4180
weighted avg     0.6218    0.5751    0.5922      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9013    0.8739    0.8874      1150
          D0     0.7351    0.6974    0.7158       975
          D1     0.6029    0.4757    0.5318       948
          D2     0.3933    0.2732    0.3224       560
          D3     0.2048    0.6528    0.3118       288
          D4     0.0235    0.0077    0.0116       259

    accuracy                         0.5931      4180
   macro avg     0.4768    0.4968    0.4635      4180
weighted avg     0.6245    0.5931    0.5971      4180
```

---

### 📌 Scenario 2A (Correlation Top-20)
- **Summary File:** `output_weekly_kansas_scenario2A/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (20 total):**
  `D1_lag1`, `None_lag1`, `T2M_lag2`, `PREC_roll12_std`, `RH2M`, `RH2M_lag2`, `RH2M_lag8`, `PREC_roll4_std`, `T2M_roll12_mean`, `PRECTOTCORR`, `RH2M_lag4`, `ALLSKY_SFC_SW_DWN`, `PREC_lag2`, `RH2M_lag1`, `PREC_lag1`, `PREC_roll4_mean`, `PREC_roll12_mean`, `WS2M`, `PREC_lag4`, `PREC_lag8`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.265012`, `1.530874`, `1.200340`, `1.051582`, `1.450273`, `0.973100`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.7709 | 0.8049 |
| **Test Accuracy** | 0.5971 | 0.5856 |
| **Test Macro F1** | 0.4755 | 0.4560 |
| **Test Weighted F1** | 0.5932 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.8824 | 0.8731 |
| **D0** | 0.7303 | 0.7440 |
| **D1** | 0.5320 | 0.5310 |
| **D2** | 0.4095 | 0.2900 |
| **D3** | 0.2115 | 0.2419 |
| **D4** | 0.0876 | 0.0562 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.8975    0.8678    0.8824      1150
          D0     0.7344    0.7262    0.7303       975
          D1     0.6288    0.4610    0.5320       948
          D2     0.3782    0.4464    0.4095       560
          D3     0.1721    0.2743    0.2115       288
          D4     0.0830    0.0927    0.0876       259

    accuracy                         0.5971      4180
   macro avg     0.4823    0.4781    0.4755      4180
weighted avg     0.6285    0.5971    0.6086      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9226    0.8287    0.8731      1150
          D0     0.6985    0.7959    0.7440       975
          D1     0.6093    0.4705    0.5310       948
          D2     0.4019    0.2268    0.2900       560
          D3     0.1634    0.4653    0.2419       288
          D4     0.0714    0.0463    0.0562       259

    accuracy                         0.5856      4180
   macro avg     0.4778    0.4722    0.4560      4180
weighted avg     0.6244    0.5856    0.5932      4180
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
  `D1_lag1`, `None_lag1`, `T2M_lag2`, `PREC_roll12_std`, `RH2M`, `RH2M_lag2`, `RH2M_lag8`, `PREC_roll4_std`, `T2M_roll12_mean`, `PRECTOTCORR`, `RH2M_lag4`, `ALLSKY_SFC_SW_DWN`, `PREC_lag2`, `RH2M_lag1`, `PREC_lag1`, `PREC_roll4_mean`, `PREC_roll12_mean`, `WS2M`, `PREC_lag4`, `PREC_lag8`, `PS`, `D2_lag1`, `D3_lag1`, `D4_lag2`, `week_sin`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.158620`, `1.433461`, `1.090687`, `0.898787`, `1.106003`, `0.550000`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8453 | 0.8522 |
| **Test Accuracy** | 0.8117 | 0.7969 |
| **Test Macro F1** | 0.8081 | 0.7896 |
| **Test Weighted F1** | 0.7979 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.8999 | 0.8970 |
| **D0** | 0.7571 | 0.7563 |
| **D1** | 0.7772 | 0.7561 |
| **D2** | 0.7768 | 0.7483 |
| **D3** | 0.7556 | 0.7332 |
| **D4** | 0.8821 | 0.8468 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9007    0.8991    0.8999      1150
          D0     0.7533    0.7610    0.7571       975
          D1     0.7913    0.7637    0.7772       948
          D2     0.7897    0.7643    0.7768       560
          D3     0.7543    0.7569    0.7556       288
          D4     0.8206    0.9537    0.8821       259

    accuracy                         0.8117      4180
   macro avg     0.8016    0.8165    0.8081      4180
weighted avg     0.8116    0.8117    0.8113      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9119    0.8826    0.8970      1150
          D0     0.7092    0.8103    0.7563       975
          D1     0.7967    0.7194    0.7561       948
          D2     0.8134    0.6929    0.7483       560
          D3     0.6423    0.8542    0.7332       288
          D4     0.8861    0.8108    0.8468       259

    accuracy                         0.7969      4180
   macro avg     0.7933    0.7950    0.7896      4180
weighted avg     0.8051    0.7969    0.7979      4180
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
  `D1_lag1`, `None_lag1`, `T2M_lag2`, `PREC_roll12_std`, `RH2M`, `RH2M_lag2`, `RH2M_lag8`, `PREC_roll4_std`, `T2M_roll12_mean`, `PRECTOTCORR`, `RH2M_lag4`, `ALLSKY_SFC_SW_DWN`, `PREC_lag2`, `RH2M_lag1`, `PREC_lag1`, `PREC_roll4_mean`, `PREC_roll12_mean`, `WS2M`, `PREC_lag4`, `PREC_lag8`, `PS`, `D2_lag1`, `D3_lag1`, `D4_lag2`, `week_sin`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.158620`, `1.433461`, `1.090687`, `0.898787`, `1.106003`, `0.550000`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8453 | 0.8522 |
| **Test Accuracy** | 0.8117 | 0.7969 |
| **Test Macro F1** | 0.8081 | 0.7896 |
| **Test Weighted F1** | 0.7979 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.8999 | 0.8970 |
| **D0** | 0.7571 | 0.7563 |
| **D1** | 0.7772 | 0.7561 |
| **D2** | 0.7768 | 0.7483 |
| **D3** | 0.7556 | 0.7332 |
| **D4** | 0.8821 | 0.8468 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9007    0.8991    0.8999      1150
          D0     0.7533    0.7610    0.7571       975
          D1     0.7913    0.7637    0.7772       948
          D2     0.7897    0.7643    0.7768       560
          D3     0.7543    0.7569    0.7556       288
          D4     0.8206    0.9537    0.8821       259

    accuracy                         0.8117      4180
   macro avg     0.8016    0.8165    0.8081      4180
weighted avg     0.8116    0.8117    0.8113      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9119    0.8826    0.8970      1150
          D0     0.7092    0.8103    0.7563       975
          D1     0.7967    0.7194    0.7561       948
          D2     0.8134    0.6929    0.7483       560
          D3     0.6423    0.8542    0.7332       288
          D4     0.8861    0.8108    0.8468       259

    accuracy                         0.7969      4180
   macro avg     0.7933    0.7950    0.7896      4180
weighted avg     0.8051    0.7969    0.7979      4180
```

---

### 📌 Scenario 3 (F1 Permutation Selection)
- **Summary File:** `output_weekly_kansas_scenario3/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (20 total):**
  `D3_lag1`, `D2_lag1`, `D1_lag1`, `None_lag1`, `D3_lag2`, `PREC_lag1`, `D0_lag1`, `None_lag2`, `WS2M`, `week_cos`, `T2M_lag4`, `T2M_roll12_mean`, `T2M_lag1`, `week_sin`, `T2M_roll4_mean`, `PREC_lag4`, `PREC_lag8`, `D4_lag1`, `D4_lag2`, `T2M_lag2`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.077213`, `1.142240`, `1.138655`, `0.745990`, `0.610306`, `1.081898`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8486 | 0.8516 |
| **Test Accuracy** | 0.8132 | 0.8129 |
| **Test Macro F1** | 0.8057 | 0.7940 |
| **Test Weighted F1** | 0.8117 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9105 | 0.9119 |
| **D0** | 0.7796 | 0.7832 |
| **D1** | 0.7786 | 0.7872 |
| **D2** | 0.7261 | 0.7434 |
| **D3** | 0.7209 | 0.6727 |
| **D4** | 0.9185 | 0.8655 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9054    0.9157    0.9105      1150
          D0     0.7737    0.7856    0.7796       975
          D1     0.8145    0.7458    0.7786       948
          D2     0.7403    0.7125    0.7261       560
          D3     0.6667    0.7847    0.7209       288
          D4     0.8826    0.9575    0.9185       259

    accuracy                         0.8132      4180
   macro avg     0.7972    0.8170    0.8057      4180
weighted avg     0.8141    0.8132    0.8128      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9100    0.9139    0.9119      1150
          D0     0.7738    0.7928    0.7832       975
          D1     0.7940    0.7806    0.7872       948
          D2     0.7815    0.7089    0.7434       560
          D3     0.7019    0.6458    0.6727       288
          D4     0.7819    0.9691    0.8655       259

    accuracy                         0.8129      4180
   macro avg     0.7905    0.8019    0.7940      4180
weighted avg     0.8124    0.8129    0.8117      4180
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
- **Best Trial:** `ros_focal_inv_no_cw_128x64`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_inv_no_cw_128x64', 'balancer': 'ROS', 'use_class_weight': True, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (128, 64), 'dropout': 0.28, 'dense_units': 96, 'lr': 0.0006, 'patience': 14}
  ```
- **Selected Features (26 total):**
  `ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`, `PREC_lag1`, `PREC_lag2`, `PREC_lag4`, `PREC_lag8`, `T2M_lag1`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `RH2M_lag1`, `RH2M_lag2`, `RH2M_lag4`, `RH2M_lag8`, `PREC_roll4_mean`, `PREC_roll4_std`, `PREC_roll12_mean`, `PREC_roll12_std`, `T2M_roll4_mean`, `T2M_roll12_mean`, `week_sin`, `week_cos`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `0.564677`, `0.642517`, `0.838090`, `1.405612`, `0.791669`, `0.782838`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.2319 | 0.2457 |
| **Test Accuracy** | 0.2505 | 0.2404 |
| **Test Macro F1** | 0.1776 | 0.1715 |
| **Test Weighted F1** | 0.2183 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.4464 | 0.3124 |
| **D0** | 0.1752 | 0.1169 |
| **D1** | 0.2802 | 0.2812 |
| **D2** | 0.0686 | 0.2992 |
| **D3** | 0.0759 | 0.0000 |
| **D4** | 0.0195 | 0.0193 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.5022    0.4017    0.4464      1150
          D0     0.2300    0.1415    0.1752       975
          D1     0.2100    0.4209    0.2802       948
          D2     0.1714    0.0429    0.0686       560
          D3     0.1062    0.0590    0.0759       288
          D4     0.0152    0.0270    0.0195       259

    accuracy                         0.2505      4180
   macro avg     0.2058    0.1822    0.1776      4180
weighted avg     0.2707    0.2505    0.2429      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.4889    0.2296    0.3124      1150
          D0     0.2167    0.0800    0.1169       975
          D1     0.2517    0.3186    0.2812       948
          D2     0.1956    0.6357    0.2992       560
          D3     0.0000    0.0000    0.0000       288
          D4     0.0192    0.0193    0.0193       259

    accuracy                         0.2404      4180
   macro avg     0.1953    0.2139    0.1715      4180
weighted avg     0.2695    0.2404    0.2183      4180
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