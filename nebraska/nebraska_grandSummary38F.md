# Nebraska Grand Summary of Scenario Results

This document compiles and summarizes all prediction results across various experimental scenarios and baseline configurations. The goal is to collect all results in one file for comparison, including class multipliers, best trials, selected features, test performance (raw and tuned), per-class F1, and raw/tuned classification reports.

## 📊 Scenario Comparison Table

The table below summarizes the key test and validation metrics for each scenario run on the dataset:

| Scenario / File | Best Trial | Features | Val F1 | Test Acc (Raw) | Test F1 (Raw) | Test Acc (Tuned) | Test F1 (Tuned) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (20 counties)** (NE_results_summary.txt) | ros_focal_no_cw_96x48 | All | 0.8578 / 0.8687 | 0.8136 | 0.8104 | 0.8201 | 0.8173 |
| **Baseline (20 counties)** (results_summary - NEB.txt) | ros_focal_no_cw_96x48 | All | 0.8578 / 0.8687 | 0.8136 | 0.8104 | 0.8201 | 0.8173 |
| **Baseline (20 counties)** | ros_focal_no_cw_96x48 | All | 0.8443 / 0.8670 | 0.8074 | 0.8053 | 0.8122 | 0.8099 |
| **Scenario 2 (Correlation Top-15)** | ros_focal_inv_no_cw_128x64 | 15 | 0.5203 / 0.5343 | 0.4098 | 0.3711 | 0.4471 | 0.3813 |
| **Scenario 2A (Correlation Top-20)** | ros_focal_no_cw_96x48 | 20 | 0.4761 / 0.5021 | 0.4031 | 0.3456 | 0.4557 | 0.3975 |
| **Scenario 2B (Correlation Top-25)** | ros_focal_no_cw_96x48 | 25 | 0.6525 / 0.6997 | 0.7589 | 0.6803 | 0.7672 | 0.6895 |
| **Scenario 2C (Correlation Top-30)** | ros_focal_no_cw_96x48 | 26 | 0.8567 / 0.8678 | 0.8072 | 0.7811 | 0.8079 | 0.7799 |
| **Scenario 3 (F1 Permutation Selection)** | ros_focal_no_cw_96x48 | 10 | 0.6386 / 0.6719 | 0.6749 | 0.6575 | 0.6988 | 0.6761 |
| **Scenario 4 (Weather Only)** | none_focal_inv_cw_64x32 | 6 | 0.1520 / 0.2774 | 0.1689 | 0.1063 | 0.1990 | 0.1516 |
| **Scenario 5 (Weather + Lag Only)** | ros_focal_no_cw_96x48 | 20 | 0.3156 / 0.3183 | 0.3079 | 0.2680 | 0.3383 | 0.2709 |
| **Scenario 6 (No Drought History)** | ros_focal_no_cw_96x48 | 26 | 0.2985 / 0.3056 | 0.2914 | 0.2607 | 0.2868 | 0.2565 |
| **Scenario 7 (Drought History Only)** | ros_focal_no_cw_96x48 | 12 | 0.8623 / 0.8711 | 0.8254 | 0.8136 | 0.8254 | 0.8161 |

---

## 🔍 Detailed Scenario Breakdown

### 📌 Baseline (20 counties) (NE_results_summary.txt)
- **Summary File:** `output_weekly_nebraska_20counties/NE_results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features:** None / Full Feature Set (41 features)
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.636890`, `1.461624`, `1.007362`, `0.915742`, `0.961145`, `1.070525`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8578 | 0.8687 |
| **Test Accuracy** | 0.8136 | 0.8201 |
| **Test Macro F1** | 0.8104 | 0.8173 |
| **Test Weighted F1** | 0.8219 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9158 | - |
| **D0** | 0.7402 | - |
| **D1** | 0.8058 | - |
| **D2** | 0.8056 | - |
| **D3** | 0.7821 | - |
| **D4** | 0.8544 | - |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9443    0.8890    0.9158       973
          D0     0.6910    0.7971    0.7402       547
          D1     0.8515    0.7647    0.8058       952
          D2     0.8395    0.7743    0.8056      1081
          D3     0.6898    0.9027    0.7821       473
          D4     0.8333    0.8766    0.8544       154

    accuracy                         0.8201      4180
   macro avg     0.8082    0.8341    0.8173      4180
weighted avg     0.8300    0.8201    0.8219      4180
```

---

### 📌 Baseline (20 counties) (results_summary - NEB.txt)
- **Summary File:** `output_weekly_nebraska_20counties/results_summary - NEB.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features:** None / Full Feature Set (41 features)
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.636890`, `1.461624`, `1.007362`, `0.915742`, `0.961145`, `1.070525`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8578 | 0.8687 |
| **Test Accuracy** | 0.8136 | 0.8201 |
| **Test Macro F1** | 0.8104 | 0.8173 |
| **Test Weighted F1** | 0.8156 | 0.8219 |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9158 | - |
| **D0** | 0.7402 | - |
| **D1** | 0.8058 | - |
| **D2** | 0.8056 | - |
| **D3** | 0.7821 | - |
| **D4** | 0.8544 | - |

#### Classification Report (Raw)
```
======================================================================
              precision    recall  f1-score   support

        None     0.9443    0.8890    0.9158       973
          D0     0.6910    0.7971    0.7402       547
          D1     0.8515    0.7647    0.8058       952
          D2     0.8395    0.7743    0.8056      1081
          D3     0.6898    0.9027    0.7821       473
          D4     0.8333    0.8766    0.8544       154

    accuracy                         0.8201      4180
   macro avg     0.8082    0.8341    0.8173      4180
weighted avg     0.8300    0.8201    0.8219      4180
```

---

### 📌 Baseline (20 counties)
- **Summary File:** `output_weekly_nebraska_20counties/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features:** None / Full Feature Set (41 features)
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.687080`, `1.218358`, `0.795265`, `0.692086`, `0.856665`, `0.909710`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8443 | 0.8670 |
| **Test Accuracy** | 0.8074 | 0.8122 |
| **Test Macro F1** | 0.8053 | 0.8099 |
| **Test Weighted F1** | 0.8087 | 0.8132 |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9086 | 0.9175 |
| **D0** | 0.7004 | 0.7247 |
| **D1** | 0.7809 | 0.7917 |
| **D2** | 0.8017 | 0.7943 |
| **D3** | 0.7851 | 0.7732 |
| **D4** | 0.8553 | 0.8580 |

#### Classification Report (Raw)
```
======================================================================
              precision    recall  f1-score   support

        None     0.9526    0.8684    0.9086       973
          D0     0.6916    0.7093    0.7004       547
          D1     0.8158    0.7489    0.7809       952
          D2     0.7976    0.8057    0.8017      1081
          D3     0.7010    0.8922    0.7851       473
          D4     0.8293    0.8831    0.8553       154

    accuracy                         0.8074      4180
   macro avg     0.7980    0.8180    0.8053      4180
weighted avg     0.8142    0.8074    0.8087      4180
```

#### Classification Report (Tuned)
```
======================================================================
              precision    recall  f1-score   support

        None     0.9330    0.9024    0.9175       973
          D0     0.6983    0.7532    0.7247       547
          D1     0.8504    0.7405    0.7917       952
          D2     0.8124    0.7771    0.7943      1081
          D3     0.6834    0.8901    0.7732       473
          D4     0.8176    0.9026    0.8580       154

    accuracy                         0.8122      4180
   macro avg     0.7992    0.8276    0.8099      4180
weighted avg     0.8198    0.8122    0.8132      4180
```

---

### 📌 Scenario 2 (Correlation Top-15)
- **Summary File:** `output_weekly_nebraska_scenario2/results_summary.txt`
- **Best Trial:** `ros_focal_inv_no_cw_128x64`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_inv_no_cw_128x64', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (128, 64), 'dropout': 0.28, 'dense_units': 96, 'lr': 0.0006, 'patience': 14}
  ```
- **Selected Features (15 total):**
  `D0_lag1`, `PREC_roll12_std`, `PS`, `RH2M_lag8`, `PREC_roll12_mean`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `RH2M_lag4`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `PREC_roll4_std`, `T2M`, `RH2M_lag2`, `PREC_roll4_mean`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.188089`, `1.121013`, `0.951797`, `1.800000`, `1.272076`, `0.798624`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.5203 | 0.5343 |
| **Test Accuracy** | 0.4098 | 0.4471 |
| **Test Macro F1** | 0.3711 | 0.3813 |
| **Test Weighted F1** | 0.4421 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9100 | 0.9131 |
| **D0** | 0.4088 | 0.3992 |
| **D1** | 0.3378 | 0.1796 |
| **D2** | 0.2516 | 0.3471 |
| **D3** | 0.1905 | 0.3939 |
| **D4** | 0.1279 | 0.0552 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9569    0.8674    0.9100       973
          D0     0.4523    0.3729    0.4088       547
          D1     0.4245    0.2805    0.3378       952
          D2     0.2787    0.2294    0.2516      1081
          D3     0.2338    0.1607    0.1905       473
          D4     0.0738    0.4805    0.1279       154

    accuracy                         0.4098      4180
   macro avg     0.4033    0.3986    0.3711      4180
weighted avg     0.4799    0.4098    0.4336      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9551    0.8746    0.9131       973
          D0     0.4319    0.3711    0.3992       547
          D1     0.4303    0.1134    0.1796       952
          D2     0.3193    0.3802    0.3471      1081
          D3     0.2936    0.5983    0.3939       473
          D4     0.0410    0.0844    0.0552       154

    accuracy                         0.4471      4180
   macro avg     0.4119    0.4037    0.3813      4180
weighted avg     0.4942    0.4471    0.4421      4180
```

---

### 📌 Scenario 2A (Correlation Top-20)
- **Summary File:** `output_weekly_nebraska_scenario2A/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (20 total):**
  `D0_lag1`, `PREC_roll12_std`, `PS`, `RH2M_lag8`, `PREC_roll12_mean`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `RH2M_lag4`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `PREC_roll4_std`, `T2M`, `RH2M_lag2`, `PREC_roll4_mean`, `RH2M_lag1`, `WS2M`, `PREC_lag2`, `PREC_lag8`, `PREC_lag4`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.109224`, `0.889099`, `0.910978`, `1.588428`, `1.096336`, `0.550000`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.4761 | 0.5021 |
| **Test Accuracy** | 0.4031 | 0.4557 |
| **Test Macro F1** | 0.3456 | 0.3975 |
| **Test Weighted F1** | 0.4577 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9175 | 0.9252 |
| **D0** | 0.3725 | 0.3644 |
| **D1** | 0.4021 | 0.2653 |
| **D2** | 0.1729 | 0.3280 |
| **D3** | 0.0717 | 0.4056 |
| **D4** | 0.1370 | 0.0967 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9514    0.8859    0.9175       973
          D0     0.3310    0.4260    0.3725       547
          D1     0.4636    0.3550    0.4021       952
          D2     0.2462    0.1332    0.1729      1081
          D3     0.1361    0.0486    0.0717       473
          D4     0.0782    0.5519    0.1370       154

    accuracy                         0.4031      4180
   macro avg     0.3677    0.4001    0.3456      4180
weighted avg     0.4523    0.4031    0.4118      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9424    0.9085    0.9252       973
          D0     0.3344    0.4004    0.3644       547
          D1     0.5651    0.1733    0.2653       952
          D2     0.3223    0.3340    0.3280      1081
          D3     0.3307    0.5243    0.4056       473
          D4     0.0659    0.1818    0.0967       154

    accuracy                         0.4557      4180
   macro avg     0.4268    0.4204    0.3975      4180
weighted avg     0.5150    0.4557    0.4577      4180
```

---

### 📌 Scenario 2B (Correlation Top-25)
- **Summary File:** `output_weekly_nebraska_scenario2B/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (25 total):**
  `D0_lag1`, `PREC_roll12_std`, `PS`, `RH2M_lag8`, `PREC_roll12_mean`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `RH2M_lag4`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `PREC_roll4_std`, `T2M`, `RH2M_lag2`, `PREC_roll4_mean`, `RH2M_lag1`, `WS2M`, `PREC_lag2`, `PREC_lag8`, `PREC_lag4`, `PREC_lag1`, `PRECTOTCORR`, `D1_lag1`, `D2_lag1`, `D3_lag1`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.575870`, `1.400141`, `0.838991`, `0.657664`, `1.147917`, `0.550000`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.6525 | 0.6997 |
| **Test Accuracy** | 0.7589 | 0.7672 |
| **Test Macro F1** | 0.6803 | 0.6895 |
| **Test Weighted F1** | 0.7750 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9073 | 0.9121 |
| **D0** | 0.7123 | 0.7357 |
| **D1** | 0.8013 | 0.8109 |
| **D2** | 0.7863 | 0.7700 |
| **D3** | 0.6178 | 0.6440 |
| **D4** | 0.2571 | 0.2644 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9599    0.8602    0.9073       973
          D0     0.6926    0.7331    0.7123       547
          D1     0.8316    0.7731    0.8013       952
          D2     0.8118    0.7623    0.7863      1081
          D3     0.5684    0.6765    0.6178       473
          D4     0.2030    0.3506    0.2571       154

    accuracy                         0.7589      4180
   macro avg     0.6779    0.6926    0.6803      4180
weighted avg     0.7852    0.7589    0.7696      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9530    0.8746    0.9121       973
          D0     0.6727    0.8117    0.7357       547
          D1     0.8644    0.7637    0.8109       952
          D2     0.8379    0.7123    0.7700      1081
          D3     0.5483    0.7801    0.6440       473
          D4     0.2371    0.2987    0.2644       154

    accuracy                         0.7672      4180
   macro avg     0.6856    0.7069    0.6895      4180
weighted avg     0.7942    0.7672    0.7750      4180
```

---

### 📌 Scenario 2C (Correlation Top-30)
- **Summary File:** `output_weekly_nebraska_scenario2C/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (26 total):**
  `D0_lag1`, `PREC_roll12_std`, `PS`, `RH2M_lag8`, `PREC_roll12_mean`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `RH2M_lag4`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `PREC_roll4_std`, `T2M`, `RH2M_lag2`, `PREC_roll4_mean`, `RH2M_lag1`, `WS2M`, `PREC_lag2`, `PREC_lag8`, `PREC_lag4`, `PREC_lag1`, `PRECTOTCORR`, `D1_lag1`, `D2_lag1`, `D3_lag1`, `D4_lag2`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.383968`, `1.195750`, `0.882496`, `0.753483`, `1.024591`, `1.077168`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8567 | 0.8678 |
| **Test Accuracy** | 0.8072 | 0.8079 |
| **Test Macro F1** | 0.7811 | 0.7799 |
| **Test Weighted F1** | 0.8109 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9031 | 0.9140 |
| **D0** | 0.7204 | 0.7281 |
| **D1** | 0.8102 | 0.8169 |
| **D2** | 0.8137 | 0.8074 |
| **D3** | 0.7496 | 0.7330 |
| **D4** | 0.6897 | 0.6803 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9542    0.8571    0.9031       973
          D0     0.6981    0.7441    0.7204       547
          D1     0.8374    0.7847    0.8102       952
          D2     0.8257    0.8020    0.8137      1081
          D3     0.6496    0.8858    0.7496       473
          D4     0.7353    0.6494    0.6897       154

    accuracy                         0.8072      4180
   macro avg     0.7834    0.7872    0.7811      4180
weighted avg     0.8183    0.8072    0.8097      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9451    0.8849    0.9140       973
          D0     0.6937    0.7660    0.7281       547
          D1     0.8570    0.7805    0.8169       952
          D2     0.8467    0.7715    0.8074      1081
          D3     0.6241    0.8879    0.7330       473
          D4     0.7143    0.6494    0.6803       154

    accuracy                         0.8079      4180
   macro avg     0.7801    0.7900    0.7799      4180
weighted avg     0.8219    0.8079    0.8109      4180
```

---

### 📌 Scenario 3 (F1 Permutation Selection)
- **Summary File:** `output_weekly_nebraska_scenario3/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (10 total):**
  `D2_lag1`, `None_lag1`, `D1_lag1`, `D0_lag1`, `D3_lag1`, `D1_lag2`, `None_lag2`, `D0_lag2`, `PREC_roll12_std`, `D4_lag1`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.800000`, `1.171256`, `1.041891`, `1.087185`, `0.650898`, `0.787839`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.6386 | 0.6719 |
| **Test Accuracy** | 0.6749 | 0.6988 |
| **Test Macro F1** | 0.6575 | 0.6761 |
| **Test Weighted F1** | 0.7005 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9081 | 0.9198 |
| **D0** | 0.6098 | 0.6331 |
| **D1** | 0.6216 | 0.6247 |
| **D2** | 0.6126 | 0.6601 |
| **D3** | 0.5812 | 0.5976 |
| **D4** | 0.6113 | 0.6212 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9455    0.8736    0.9081       973
          D0     0.6326    0.5887    0.6098       547
          D1     0.6830    0.5704    0.6216       952
          D2     0.6726    0.5624    0.6126      1081
          D3     0.4335    0.8816    0.5812       473
          D4     0.7297    0.5260    0.6113       154

    accuracy                         0.6749      4180
   macro avg     0.6828    0.6671    0.6575      4180
weighted avg     0.7083    0.6749    0.6795      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9260    0.9137    0.9198       973
          D0     0.6728    0.5978    0.6331       547
          D1     0.7099    0.5578    0.6247       952
          D2     0.6682    0.6522    0.6601      1081
          D3     0.4773    0.7992    0.5976       473
          D4     0.6547    0.5909    0.6212       154

    accuracy                         0.6988      4180
   macro avg     0.6848    0.6852    0.6761      4180
weighted avg     0.7162    0.6988    0.7005      4180
```

---

### 📌 Scenario 4 (Weather Only)
- **Summary File:** `output_weekly_nebraska_scenario4/results_summary.txt`
- **Best Trial:** `none_focal_inv_cw_64x32`
- **Best Trial Config:**
  ```python
  {'name': 'none_focal_inv_cw_64x32', 'balancer': 'NONE', 'use_class_weight': True, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (64, 32), 'dropout': 0.35, 'dense_units': 64, 'lr': 0.001, 'patience': 16}
  ```
- **Selected Features (6 total):**
  `ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.800000`, `0.844216`, `0.762617`, `0.734996`, `0.938679`, `0.596986`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.1520 | 0.2774 |
| **Test Accuracy** | 0.1689 | 0.1990 |
| **Test Macro F1** | 0.1063 | 0.1516 |
| **Test Weighted F1** | 0.1455 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.0000 | 0.0000 |
| **D0** | 0.0000 | 0.0000 |
| **D1** | 0.1331 | 0.1749 |
| **D2** | 0.2756 | 0.2589 |
| **D3** | 0.0980 | 0.2780 |
| **D4** | 0.1308 | 0.1979 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.0000    0.0000    0.0000       973
          D0     0.0000    0.0000    0.0000       547
          D1     0.2929    0.0861    0.1331       952
          D2     0.1947    0.4718    0.2756      1081
          D3     0.1778    0.0677    0.0980       473
          D4     0.0745    0.5325    0.1308       154

    accuracy                         0.1689      4180
   macro avg     0.1233    0.1930    0.1063      4180
weighted avg     0.1399    0.1689    0.1175      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.0000    0.0000    0.0000       973
          D0     0.0000    0.0000    0.0000       547
          D1     0.2857    0.1261    0.1749       952
          D2     0.1882    0.4144    0.2589      1081
          D3     0.1957    0.4799    0.2780       473
          D4     0.1682    0.2403    0.1979       154

    accuracy                         0.1990      4180
   macro avg     0.1396    0.2101    0.1516      4180
weighted avg     0.1421    0.1990    0.1455      4180
```

---

### 📌 Scenario 5 (Weather + Lag Only)
- **Summary File:** `output_weekly_nebraska_scenario5/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': True, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (20 total):**
  `ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`, `PREC_lag1`, `PREC_lag2`, `PREC_lag4`, `PREC_lag8`, `T2M_lag1`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `RH2M_lag1`, `RH2M_lag2`, `RH2M_lag4`, `RH2M_lag8`, `week_sin`, `week_cos`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `0.998747`, `0.865971`, `0.890396`, `0.835963`, `1.640680`, `0.801996`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.3156 | 0.3183 |
| **Test Accuracy** | 0.3079 | 0.3383 |
| **Test Macro F1** | 0.2680 | 0.2709 |
| **Test Weighted F1** | 0.3013 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.5725 | 0.5704 |
| **D0** | 0.2664 | 0.2612 |
| **D1** | 0.2895 | 0.3129 |
| **D2** | 0.1012 | 0.0594 |
| **D3** | 0.1697 | 0.4215 |
| **D4** | 0.2091 | 0.0000 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.5183    0.6393    0.5725       973
          D0     0.1949    0.4205    0.2664       547
          D1     0.3361    0.2542    0.2895       952
          D2     0.1558    0.0749    0.1012      1081
          D3     0.3714    0.1099    0.1697       473
          D4     0.1429    0.3896    0.2091       154

    accuracy                         0.3079      4180
   macro avg     0.2866    0.3147    0.2680      4180
weighted avg     0.3103    0.3079    0.2871      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.4955    0.6721    0.5704       973
          D0     0.2020    0.3693    0.2612       547
          D1     0.3474    0.2847    0.3129       952
          D2     0.1100    0.0407    0.0594      1081
          D3     0.3574    0.5137    0.4215       473
          D4     0.0000    0.0000    0.0000       154

    accuracy                         0.3383      4180
   macro avg     0.2520    0.3134    0.2709      4180
weighted avg     0.2898    0.3383    0.3013      4180
```

---

### 📌 Scenario 6 (No Drought History)
- **Summary File:** `output_weekly_nebraska_scenario6/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': True, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (26 total):**
  `ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`, `PREC_lag1`, `PREC_lag2`, `PREC_lag4`, `PREC_lag8`, `T2M_lag1`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `RH2M_lag1`, `RH2M_lag2`, `RH2M_lag4`, `RH2M_lag8`, `PREC_roll4_mean`, `PREC_roll4_std`, `PREC_roll12_mean`, `PREC_roll12_std`, `T2M_roll4_mean`, `T2M_roll12_mean`, `week_sin`, `week_cos`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `0.637500`, `0.735192`, `0.851000`, `1.445603`, `0.825430`, `1.211748`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.2985 | 0.3056 |
| **Test Accuracy** | 0.2914 | 0.2868 |
| **Test Macro F1** | 0.2607 | 0.2565 |
| **Test Weighted F1** | 0.2992 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.4796 | 0.4164 |
| **D0** | 0.2949 | 0.2774 |
| **D1** | 0.3696 | 0.3416 |
| **D2** | 0.1653 | 0.2951 |
| **D3** | 0.0914 | 0.0542 |
| **D4** | 0.1637 | 0.1544 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.6161    0.3926    0.4796       973
          D0     0.2250    0.4278    0.2949       547
          D1     0.3643    0.3750    0.3696       952
          D2     0.2481    0.1240    0.1653      1081
          D3     0.2000    0.0592    0.0914       473
          D4     0.0965    0.5390    0.1637       154

    accuracy                         0.2914      4180
   macro avg     0.2917    0.3196    0.2607      4180
weighted avg     0.3462    0.2914    0.2935      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.6905    0.2980    0.4164       973
          D0     0.2177    0.3821    0.2774       547
          D1     0.3905    0.3036    0.3416       952
          D2     0.3039    0.2868    0.2951      1081
          D3     0.1875    0.0317    0.0542       473
          D4     0.0896    0.5584    0.1544       154

    accuracy                         0.2868      4180
   macro avg     0.3133    0.3101    0.2565      4180
weighted avg     0.3813    0.2868    0.2992      4180
```

---

### 📌 Scenario 7 (Drought History Only)
- **Summary File:** `output_weekly_nebraska_scenario7/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': True, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (12 total):**
  `None_lag1`, `D0_lag1`, `D1_lag1`, `D2_lag1`, `D3_lag1`, `D4_lag1`, `None_lag2`, `D0_lag2`, `D1_lag2`, `D2_lag2`, `D3_lag2`, `D4_lag2`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.800000`, `1.576066`, `0.912133`, `0.836971`, `0.906954`, `1.165289`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8623 | 0.8711 |
| **Test Accuracy** | 0.8254 | 0.8254 |
| **Test Macro F1** | 0.8136 | 0.8161 |
| **Test Weighted F1** | 0.8267 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9254 | 0.9274 |
| **D0** | 0.7278 | 0.7509 |
| **D1** | 0.8104 | 0.8047 |
| **D2** | 0.8150 | 0.8128 |
| **D3** | 0.7971 | 0.7882 |
| **D4** | 0.8059 | 0.8127 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9463    0.9054    0.9254       973
          D0     0.7534    0.7038    0.7278       547
          D1     0.7891    0.8330    0.8104       952
          D2     0.8579    0.7761    0.8150      1081
          D3     0.6950    0.9345    0.7971       473
          D4     0.9244    0.7143    0.8059       154

    accuracy                         0.8254      4180
   macro avg     0.8277    0.8112    0.8136      4180
weighted avg     0.8331    0.8254    0.8259      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9427    0.9126    0.9274       973
          D0     0.7083    0.7989    0.7509       547
          D1     0.8271    0.7836    0.8047       952
          D2     0.8693    0.7632    0.8128      1081
          D3     0.6849    0.9281    0.7882       473
          D4     0.8915    0.7468    0.8127       154

    accuracy                         0.8254      4180
   macro avg     0.8206    0.8222    0.8161      4180
weighted avg     0.8356    0.8254    0.8267      4180
```

---