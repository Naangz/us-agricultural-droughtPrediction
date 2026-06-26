# Nebraska Grand Summary of Scenario Results

This document compiles and summarizes all prediction results across various experimental scenarios and baseline configurations. The goal is to collect all results in one file for comparison, including class multipliers, best trials, selected features, test performance (raw and tuned), per-class F1, and raw/tuned classification reports.

## 📊 Scenario Comparison Table

The table below summarizes the key test and validation metrics for each scenario run on the dataset:

| Scenario / File | Best Trial | Features | Val F1 | Test Acc (Raw) | Test F1 (Raw) | Test Acc (Tuned) | Test F1 (Tuned) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (20 counties)** | ros_focal_no_cw_96x48 | All | 0.8514 / 0.8641 | 0.8077 | 0.7958 | 0.7969 | 0.7718 |
| **Scenario 2 (Correlation Top-15)** | ros_focal_no_cw_96x48 | 15 | 0.4707 / 0.4932 | 0.4340 | 0.3892 | 0.4124 | 0.3428 |
| **Scenario 2A (Correlation Top-20)** | ros_focal_no_cw_96x48 | All | 0.4832 / 0.5086 | 0.4529 | 0.3962 | 0.4438 | 0.3802 |
| **Scenario 2B (Correlation Top-25)** | ros_focal_no_cw_96x48 | All | 0.6525 / 0.6889 | 0.7536 | 0.6695 | 0.7589 | 0.6695 |
| **Scenario 2C (Correlation Top-30)** | ros_focal_no_cw_96x48 | All | 0.8598 / 0.8653 | 0.8086 | 0.8010 | 0.8029 | 0.7867 |
| **Scenario 3 (F1 Permutation Selection)** | ros_focal_no_cw_96x48 | 10 | 0.6064 / 0.7612 | 0.6433 | 0.5875 | 0.6567 | 0.5973 |
| **Scenario 4 (Weather Only)** | none_focal_inv_cw_64x32 | 6 | 0.1520 / 0.2774 | 0.1689 | 0.1063 | 0.1990 | 0.1516 |
| **Scenario 5 (Weather + Lag Only)** | ros_focal_no_cw_96x48 | 20 | 0.3156 / 0.3183 | 0.3079 | 0.2680 | 0.3383 | 0.2709 |
| **Scenario 6 (No Drought History)** | ros_focal_no_cw_96x48 | 26 | 0.3240 / 0.3306 | 0.3347 | 0.2865 | 0.3344 | 0.2725 |
| **Scenario 7 (Drought History Only)** | ros_focal_no_cw_96x48 | 12 | 0.8623 / 0.8711 | 0.8254 | 0.8136 | 0.8254 | 0.8161 |

---

## 🔍 Detailed Scenario Breakdown

### 📌 Baseline (20 counties)
- **Summary File:** `output_weekly_nebraska_20counties/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features:** None / Full Feature Set (41 features)
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.353896`, `1.090890`, `0.795373`, `0.780979`, `1.800000`, `0.738624`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8514 | 0.8641 |
| **Test Accuracy** | 0.8077 | 0.7969 |
| **Test Macro F1** | 0.7958 | 0.7718 |
| **Test Weighted F1** | 0.8086 | 0.7980 |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9134 | 0.9187 |
| **D0** | 0.6852 | 0.7263 |
| **D1** | 0.7893 | 0.7972 |
| **D2** | 0.8061 | 0.7686 |
| **D3** | 0.7836 | 0.7401 |
| **D4** | 0.7974 | 0.6800 |

#### Classification Report (Raw)
```
======================================================================
              precision    recall  f1-score   support

        None     0.9451    0.8839    0.9134       973
          D0     0.6962    0.6746    0.6852       547
          D1     0.7877    0.7910    0.7893       952
          D2     0.8248    0.7882    0.8061      1081
          D3     0.7012    0.8879    0.7836       473
          D4     0.8026    0.7922    0.7974       154

    accuracy                         0.8077      4180
   macro avg     0.7929    0.8030    0.7958      4180
weighted avg     0.8127    0.8077    0.8086      4180
```

#### Classification Report (Tuned)
```
======================================================================
              precision    recall  f1-score   support

        None     0.9323    0.9054    0.9187       973
          D0     0.7110    0.7422    0.7263       547
          D1     0.8264    0.7700    0.7972       952
          D2     0.8235    0.7206    0.7686      1081
          D3     0.6082    0.9450    0.7401       473
          D4     0.8854    0.5519    0.6800       154

    accuracy                         0.7969      4180
   macro avg     0.7978    0.7725    0.7718      4180
weighted avg     0.8127    0.7969    0.7980      4180
```

---

### 📌 Scenario 2 (Correlation Top-15)
- **Summary File:** `output_weekly_nebraska_scenario2/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features (15 total):**
  `None_lag1`, `RH2M_lag8`, `T2M_lag8`, `ALLSKY_SFC_SW_DWN`, `PREC_roll12_std`, `T2M_lag1`, `RH2M_lag4`, `T2M_lag4`, `PS`, `PREC_roll4_mean`, `heat_dry_stress`, `RH2M_lag1`, `RH2M_lag2`, `PREC_roll12_mean`, `PREC_lag2`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.156250`, `0.826559`, `0.550000`, `0.603301`, `1.594338`, `0.734359`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.4707 | 0.4932 |
| **Test Accuracy** | 0.4340 | 0.4124 |
| **Test Macro F1** | 0.3892 | 0.3428 |
| **Test Weighted F1** | 0.3803 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9088 | 0.9246 |
| **D0** | 0.4061 | 0.3827 |
| **D1** | 0.3472 | 0.2237 |
| **D2** | 0.2029 | 0.0788 |
| **D3** | 0.3793 | 0.3562 |
| **D4** | 0.0911 | 0.0908 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9273    0.8911    0.9088       973
          D0     0.3993    0.4132    0.4061       547
          D1     0.3931    0.3109    0.3472       952
          D2     0.2772    0.1600    0.2029      1081
          D3     0.3243    0.4567    0.3793       473
          D4     0.0566    0.2338    0.0911       154

    accuracy                         0.4340      4180
   macro avg     0.3963    0.4109    0.3892      4180
weighted avg     0.4681    0.4340    0.4425      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9104    0.9394    0.9246       973
          D0     0.3352    0.4461    0.3827       547
          D1     0.4777    0.1460    0.2237       952
          D2     0.2394    0.0472    0.0788      1081
          D3     0.2350    0.7357    0.3562       473
          D4     0.0605    0.1818    0.0908       154

    accuracy                         0.4124      4180
   macro avg     0.3763    0.4160    0.3428      4180
weighted avg     0.4553    0.4124    0.3803      4180
```

---

### 📌 Scenario 2A (Correlation Top-20)
- **Summary File:** `output_weekly_nebraska_scenario2A/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features:** None / Full Feature Set (41 features)
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.090156`, `0.749149`, `0.652116`, `0.661279`, `1.490751`, `0.744676`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.4832 | 0.5086 |
| **Test Accuracy** | 0.4529 | 0.4438 |
| **Test Macro F1** | 0.3962 | 0.3802 |
| **Test Weighted F1** | 0.4164 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9100 | 0.9337 |
| **D0** | 0.3559 | 0.3395 |
| **D1** | 0.4573 | 0.3598 |
| **D2** | 0.2238 | 0.0982 |
| **D3** | 0.2190 | 0.3545 |
| **D4** | 0.2111 | 0.1957 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9388    0.8828    0.9100       973
          D0     0.3126    0.4132    0.3559       547
          D1     0.4383    0.4779    0.4573       952
          D2     0.2997    0.1785    0.2238      1081
          D3     0.2857    0.1776    0.2190       473
          D4     0.1343    0.4935    0.2111       154

    accuracy                         0.4529      4180
   macro avg     0.4016    0.4373    0.3962      4180
weighted avg     0.4740    0.4529    0.4530      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9193    0.9486    0.9337       973
          D0     0.2847    0.4205    0.3395       547
          D1     0.4119    0.3193    0.3598       952
          D2     0.3119    0.0583    0.0982      1081
          D3     0.2463    0.6321    0.3545       473
          D4     0.1682    0.2338    0.1957       154

    accuracy                         0.4438      4180
   macro avg     0.3904    0.4354    0.3802      4180
weighted avg     0.4598    0.4438    0.4164      4180
```

---

### 📌 Scenario 2B (Correlation Top-25)
- **Summary File:** `output_weekly_nebraska_scenario2B/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features:** None / Full Feature Set (41 features)
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.447933`, `1.148992`, `0.699877`, `0.854198`, `1.786296`, `0.653757`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.6525 | 0.6889 |
| **Test Accuracy** | 0.7536 | 0.7589 |
| **Test Macro F1** | 0.6695 | 0.6695 |
| **Test Weighted F1** | 0.7602 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9135 | 0.9200 |
| **D0** | 0.6929 | 0.7182 |
| **D1** | 0.7972 | 0.7721 |
| **D2** | 0.7931 | 0.7607 |
| **D3** | 0.5684 | 0.6334 |
| **D4** | 0.2519 | 0.2128 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9441    0.8849    0.9135       973
          D0     0.6949    0.6910    0.6929       547
          D1     0.8216    0.7742    0.7972       952
          D2     0.8265    0.7623    0.7931      1081
          D3     0.5164    0.6321    0.5684       473
          D4     0.2032    0.3312    0.2519       154

    accuracy                         0.7536      4180
   macro avg     0.6678    0.6793    0.6695      4180
weighted avg     0.7775    0.7536    0.7636      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9297    0.9106    0.9200       973
          D0     0.6775    0.7642    0.7182       547
          D1     0.8847    0.6849    0.7721       952
          D2     0.7986    0.7262    0.7607      1081
          D3     0.5019    0.8584    0.6334       473
          D4     0.3086    0.1623    0.2128       154

    accuracy                         0.7589      4180
   macro avg     0.6835    0.6844    0.6695      4180
weighted avg     0.7812    0.7589    0.7602      4180
```

---

### 📌 Scenario 2C (Correlation Top-30)
- **Summary File:** `output_weekly_nebraska_scenario2C/results_summary.txt`
- **Best Trial:** `ros_focal_no_cw_96x48`
- **Best Trial Config:**
  ```python
  {'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}
  ```
- **Selected Features:** None / Full Feature Set (41 features)
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.039749`, `0.979278`, `0.670292`, `0.643043`, `1.184767`, `0.550000`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.8598 | 0.8653 |
| **Test Accuracy** | 0.8086 | 0.8029 |
| **Test Macro F1** | 0.8010 | 0.7867 |
| **Test Weighted F1** | 0.8058 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9093 | 0.9132 |
| **D0** | 0.7183 | 0.7343 |
| **D1** | 0.8103 | 0.8103 |
| **D2** | 0.7937 | 0.7809 |
| **D3** | 0.7522 | 0.7348 |
| **D4** | 0.8224 | 0.7464 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9456    0.8756    0.9093       973
          D0     0.6927    0.7459    0.7183       547
          D1     0.8248    0.7962    0.8103       952
          D2     0.8461    0.7475    0.7937      1081
          D3     0.6502    0.8922    0.7522       473
          D4     0.7904    0.8571    0.8224       154

    accuracy                         0.8086      4180
   macro avg     0.7916    0.8191    0.8010      4180
weighted avg     0.8201    0.8086    0.8109      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9411    0.8869    0.9132       973
          D0     0.6835    0.7934    0.7343       547
          D1     0.8501    0.7742    0.8103       952
          D2     0.8548    0.7188    0.7809      1081
          D3     0.6055    0.9345    0.7348       473
          D4     0.8443    0.6688    0.7464       154

    accuracy                         0.8029      4180
   macro avg     0.7965    0.7961    0.7867      4180
weighted avg     0.8228    0.8029    0.8058      4180
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
  `D3_lag1`, `D1_lag1`, `D2_lag1`, `None_lag1`, `D1_lag2`, `D0_lag1`, `D0_lag2`, `D2_lag2`, `D3_lag2`, `PREC_roll4_mean`
- **Class Multipliers (None, D0, D1, D2, D3, D4):**
  `1.760622`, `1.351568`, `0.904283`, `0.915277`, `1.113876`, `0.712563`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.6064 | 0.7612 |
| **Test Accuracy** | 0.6433 | 0.6567 |
| **Test Macro F1** | 0.5875 | 0.5973 |
| **Test Weighted F1** | 0.6662 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.9153 | 0.9245 |
| **D0** | 0.6391 | 0.6849 |
| **D1** | 0.7099 | 0.7069 |
| **D2** | 0.5534 | 0.5304 |
| **D3** | 0.4269 | 0.4729 |
| **D4** | 0.2802 | 0.2642 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.9443    0.8880    0.9153       973
          D0     0.6294    0.6490    0.6391       547
          D1     0.7506    0.6733    0.7099       952
          D2     0.7500    0.4385    0.5534      1081
          D3     0.3325    0.5962    0.4269       473
          D4     0.1989    0.4740    0.2802       154

    accuracy                         0.6433      4180
   macro avg     0.6010    0.6198    0.5875      4180
weighted avg     0.7120    0.6433    0.6601      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.9303    0.9188    0.9245       973
          D0     0.6426    0.7331    0.6849       547
          D1     0.8081    0.6282    0.7069       952
          D2     0.7454    0.4117    0.5304      1081
          D3     0.3439    0.7569    0.4729       473
          D4     0.2258    0.3182    0.2642       154

    accuracy                         0.6567      4180
   macro avg     0.6160    0.6278    0.5973      4180
weighted avg     0.7247    0.6567    0.6662      4180
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
  `1.015671`, `0.829377`, `0.853891`, `0.713567`, `1.461893`, `1.376722`

#### Performance Metrics
| Metric | Raw / Baseline | Tuned (Tuned Multipliers) |
| :--- | :---: | :---: |
| **Validation Macro F1** | 0.3240 | 0.3306 |
| **Test Accuracy** | 0.3347 | 0.3344 |
| **Test Macro F1** | 0.2865 | 0.2725 |
| **Test Weighted F1** | 0.2966 | - |

#### Per-Class F1 Score
| Class | Raw F1 | Tuned F1 |
| :--- | :---: | :---: |
| **None** | 0.5981 | 0.6176 |
| **D0** | 0.2927 | 0.2092 |
| **D1** | 0.3953 | 0.3813 |
| **D2** | 0.1499 | 0.0254 |
| **D3** | 0.1079 | 0.2261 |
| **D4** | 0.1751 | 0.1751 |

#### Classification Report (Raw)
```
precision    recall  f1-score   support

        None     0.5900    0.6064    0.5981       973
          D0     0.2417    0.3711    0.2927       547
          D1     0.3979    0.3929    0.3953       952
          D2     0.2308    0.1110    0.1499      1081
          D3     0.2667    0.0677    0.1079       473
          D4     0.1053    0.5195    0.1751       154

    accuracy                         0.3347      4180
   macro avg     0.3054    0.3447    0.2865      4180
weighted avg     0.3533    0.3347    0.3250      4180
```

#### Classification Report (Tuned)
```
precision    recall  f1-score   support

        None     0.5204    0.7595    0.6176       973
          D0     0.2000    0.2194    0.2092       547
          D1     0.4175    0.3508    0.3813       952
          D2     0.1500    0.0139    0.0254      1081
          D3     0.2200    0.2326    0.2261       473
          D4     0.1053    0.5195    0.1751       154

    accuracy                         0.3344      4180
   macro avg     0.2689    0.3493    0.2725      4180
weighted avg     0.3100    0.3344    0.2966      4180
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