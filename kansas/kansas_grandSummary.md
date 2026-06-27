# Kansas Grand Summary

Gabungan dan perapihan seluruh `results_summary.txt` yang ada di folder `kansas`.

## Ringkasan Utama

| Folder | Scenario | Best Trial | Features | Val Macro F1 Raw | Val Macro F1 Tuned | Test Acc Raw | Test Acc Tuned | Test Macro F1 Raw | Test Macro F1 Tuned |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `output_weekly_kansas_20counties` | Scenario 1: Baseline | `ros_focal_no_cw_96x48` | 39 | 0.8480 | 0.8536 | 0.8177 | 0.8014 | 0.8166 | 0.7921 |
| `output_weekly_kansas_scenario2` | Scenario 2: Correlation-aware Feature Selection | `ros_focal_inv_no_cw_128x64` | 15 | 0.6338 | 0.6847 | 0.6100 | 0.6081 | 0.4795 | 0.4754 |
| `output_weekly_kansas_scenario2A` | Scenario 2A: Correlation-aware Feature Selection | `ros_focal_inv_no_cw_128x64` | 20 | 0.6340 | 0.6450 | 0.6222 | 0.6282 | 0.4628 | 0.4643 |
| `output_weekly_kansas_scenario2B` | Scenario 2B: Correlation-aware Feature Selection | `ros_focal_no_cw_96x48` | 25 | 0.8444 | 0.8546 | 0.8184 | 0.8182 | 0.8245 | 0.8258 |
| `output_weekly_kansas_scenario2C` | Scenario 2C: Correlation-aware Feature Selection | `ros_focal_no_cw_96x48` | 25 | 0.8444 | 0.8546 | 0.8184 | 0.8182 | 0.8245 | 0.8258 |
| `output_weekly_kansas_scenario3` | Scenario 3: Macro F1-driven Feature Selection | `ros_focal_inv_no_cw_128x64` | 25 | 0.1990 | 0.2093 | 0.3364 | 0.3333 | 0.2483 | 0.2391 |
| `output_weekly_kansas_scenario4` | Scenario 4: Weather Only | `ros_focal_no_cw_96x48` | 6 | 0.1746 | 0.2060 | 0.1031 | 0.1438 | 0.0965 | 0.1185 |
| `output_weekly_kansas_scenario5` | Scenario 5: Weather + Lag Only | `none_focal_inv_cw_64x32` | 20 | 0.2068 | 0.2498 | 0.1718 | 0.1715 | 0.1380 | 0.1152 |
| `output_weekly_kansas_scenario6` | Scenario 6: No Drought History | `none_focal_inv_cw_64x32` | 27 | 0.2510 | 0.2570 | 0.2031 | 0.2134 | 0.1900 | 0.1991 |
| `output_weekly_kansas_scenario7` | Scenario 7: Drought History Only | `ros_focal_inv_no_cw_128x64` | 12 | 0.8446 | 0.8481 | 0.8177 | 0.8246 | 0.8202 | 0.8289 |

---

## `output_weekly_kansas_20counties`

### Scenario 1: Baseline

- Best trial: `ros_focal_no_cw_96x48`
- Best model path: `F:\Projectan\TA_new\enhprota\kansas\output_weekly_kansas_20counties\best_model.keras`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq length: `52`
- Feature count: `39`
- Val Macro F1 (raw/tuned): `0.8480 / 0.8536`
- Class multipliers: `[1.7999999523162842, 1.1255494356155396, 0.8058606386184692, 0.8119803667068481, 0.7536675333976746, 1.3244149684906006]`

#### Selected Features

`ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`, `PREC_lag1`, `PREC_lag2`, `PREC_lag4`, `PREC_lag8`, `T2M_lag1`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `RH2M_lag1`, `RH2M_lag2`, `RH2M_lag4`, `RH2M_lag8`, `PREC_roll4_mean`, `PREC_roll4_std`, `PREC_roll12_mean`, `PREC_roll12_std`, `T2M_roll4_mean`, `T2M_roll12_mean`, `week_sin`, `week_cos`, `None_lag1`, `D0_lag1`, `D1_lag1`, `D2_lag1`, `D3_lag1`, `D4_lag1`, `None_lag2`, `D0_lag2`, `D1_lag2`, `D2_lag2`, `D3_lag2`, `D4_lag2`, `heat_dry_stress`

#### Results

| Metric | Raw | Tuned |
| --- | ---: | ---: |
| Accuracy | 0.8177 | 0.8014 |
| Macro F1 | 0.8166 | 0.7921 |
| Weighted F1 | 0.8186 | 0.7990 |

#### Per-class F1

| Class | Raw | Tuned |
| --- | ---: | ---: |
| None | 0.9022 | 0.8969 |
| D0 | 0.7773 | 0.7468 |
| D1 | 0.7804 | 0.7571 |
| D2 | 0.7721 | 0.7742 |
| D3 | 0.7653 | 0.7054 |
| D4 | 0.9026 | 0.8720 |

#### Classification Report (Raw)

```text
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

```text
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

## `output_weekly_kansas_scenario2`

### Scenario 2: Correlation-aware Feature Selection

- Best trial: `ros_focal_inv_no_cw_128x64`
- Best model path: `F:\Projectan\TA_new\enhprota\kansas\output_weekly_kansas_scenario2\best_model.keras`
- Best trial config: `{'name': 'ros_focal_inv_no_cw_128x64', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (128, 64), 'dropout': 0.28, 'dense_units': 96, 'lr': 0.0006, 'patience': 14}`
- Seq length: `52`
- Feature count: `15`
- Val Macro F1 (raw/tuned): `0.6338 / 0.6847`
- Class multipliers: `[0.687532901763916, 0.8795230388641357, 1.0119997262954712, 0.8031442761421204, 1.0250754356384277, 1.0201678276062012]`

#### Selected Features

`D1_lag1`, `None_lag1`, `RH2M`, `PREC_roll12_std`, `T2M`, `T2M_lag4`, `RH2M_lag4`, `PREC_roll4_mean`, `RH2M_lag2`, `PRECTOTCORR`, `RH2M_lag8`, `PREC_lag2`, `RH2M_lag1`, `PREC_roll4_std`, `PREC_roll12_mean`

#### Results

| Metric | Raw | Tuned |
| --- | ---: | ---: |
| Accuracy | 0.6100 | 0.6081 |
| Macro F1 | 0.4795 | 0.4754 |
| Weighted F1 | 0.6099 | 0.6101 |

#### Per-class F1

| Class | Raw | Tuned |
| --- | ---: | ---: |
| None | 0.8906 | 0.8895 |
| D0 | 0.7384 | 0.7445 |
| D1 | 0.5221 | 0.5522 |
| D2 | 0.3765 | 0.3214 |
| D3 | 0.3100 | 0.3052 |
| D4 | 0.0394 | 0.0395 |

#### Classification Report (Raw)

```text
              precision    recall  f1-score   support

        None     0.8756    0.9061    0.8906      1150
          D0     0.7521    0.7251    0.7384       975
          D1     0.6250    0.4483    0.5221       948
          D2     0.3118    0.4750    0.3765       560
          D3     0.2757    0.3542    0.3100       288
          D4     0.0544    0.0309    0.0394       259

    accuracy                         0.6100      4180
   macro avg     0.4824    0.4899    0.4795      4180
weighted avg     0.6222    0.6100    0.6099      4180
```

#### Classification Report (Tuned)

```text
              precision    recall  f1-score   support

        None     0.9084    0.8713    0.8895      1150
          D0     0.7374    0.7518    0.7445       975
          D1     0.5899    0.5190    0.5522       948
          D2     0.3145    0.3286    0.3214       560
          D3     0.2375    0.4271    0.3052       288
          D4     0.0548    0.0309    0.0395       259

    accuracy                         0.6081      4180
   macro avg     0.4738    0.4881    0.4754      4180
weighted avg     0.6176    0.6081    0.6101      4180
```

---

## `output_weekly_kansas_scenario2A`

### Scenario 2A: Correlation-aware Feature Selection

- Best trial: `ros_focal_inv_no_cw_128x64`
- Best model path: `F:\Projectan\TA_new\enhprota\kansas\output_weekly_kansas_scenario2A\best_model.keras`
- Best trial config: `{'name': 'ros_focal_inv_no_cw_128x64', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (128, 64), 'dropout': 0.28, 'dense_units': 96, 'lr': 0.0006, 'patience': 14}`
- Seq length: `52`
- Feature count: `20`
- Val Macro F1 (raw/tuned): `0.6340 / 0.6450`
- Class multipliers: `[0.9183673858642578, 1.0629817247390747, 1.1852333545684814, 0.8840540051460266, 1.0023502111434937, 0.9987034201622009]`

#### Selected Features

`D1_lag1`, `None_lag1`, `RH2M`, `PREC_roll12_std`, `T2M`, `T2M_lag4`, `RH2M_lag4`, `PREC_roll4_mean`, `RH2M_lag2`, `PRECTOTCORR`, `RH2M_lag8`, `PREC_lag2`, `RH2M_lag1`, `PREC_roll4_std`, `PREC_roll12_mean`, `ALLSKY_SFC_SW_DWN`, `PREC_lag8`, `PREC_lag4`, `WS2M`, `PREC_lag1`

#### Results

| Metric | Raw | Tuned |
| --- | ---: | ---: |
| Accuracy | 0.6222 | 0.6282 |
| Macro F1 | 0.4628 | 0.4643 |
| Weighted F1 | 0.6127 | 0.6171 |

#### Per-class F1

| Class | Raw | Tuned |
| --- | ---: | ---: |
| None | 0.8930 | 0.8893 |
| D0 | 0.7623 | 0.7644 |
| D1 | 0.5514 | 0.5847 |
| D2 | 0.3824 | 0.3666 |
| D3 | 0.1876 | 0.1808 |
| D4 | 0.0000 | 0.0000 |

#### Classification Report (Raw)

```text
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

```text
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

## `output_weekly_kansas_scenario2B`

### Scenario 2B: Correlation-aware Feature Selection

- Best trial: `ros_focal_no_cw_96x48`
- Best model path: `F:\Projectan\TA_new\enhprota\kansas\output_weekly_kansas_scenario2B\best_model.keras`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq length: `52`
- Feature count: `25`
- Val Macro F1 (raw/tuned): `0.8444 / 0.8546`
- Class multipliers: `[1.2024885416030884, 1.4572447538375854, 1.0968611240386963, 0.8748490810394287, 0.550000011920929, 1.0214027166366577]`

#### Selected Features

`D1_lag1`, `None_lag1`, `RH2M`, `PREC_roll12_std`, `T2M`, `T2M_lag4`, `RH2M_lag4`, `PREC_roll4_mean`, `RH2M_lag2`, `PRECTOTCORR`, `RH2M_lag8`, `PREC_lag2`, `RH2M_lag1`, `PREC_roll4_std`, `PREC_roll12_mean`, `ALLSKY_SFC_SW_DWN`, `PREC_lag8`, `PREC_lag4`, `WS2M`, `PREC_lag1`, `T2M_lag8`, `PS`, `D2_lag1`, `D3_lag1`, `D4_lag1`

#### Results

| Metric | Raw | Tuned |
| --- | ---: | ---: |
| Accuracy | 0.8184 | 0.8182 |
| Macro F1 | 0.8245 | 0.8258 |
| Weighted F1 | 0.8182 | 0.8184 |

#### Per-class F1

| Class | Raw | Tuned |
| --- | ---: | ---: |
| None | 0.9015 | 0.8948 |
| D0 | 0.7720 | 0.7677 |
| D1 | 0.7628 | 0.7555 |
| D2 | 0.7771 | 0.8192 |
| D3 | 0.8006 | 0.8045 |
| D4 | 0.9328 | 0.9134 |

#### Classification Report (Raw)

```text
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

```text
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

## `output_weekly_kansas_scenario2C`

### Scenario 2C: Correlation-aware Feature Selection

- Best trial: `ros_focal_no_cw_96x48`
- Best model path: `F:\Projectan\TA_new\enhprota\kansas\output_weekly_kansas_scenario2C\best_model.keras`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq length: `52`
- Feature count: `25`
- Val Macro F1 (raw/tuned): `0.8444 / 0.8546`
- Class multipliers: `[1.2024885416030884, 1.4572447538375854, 1.0968611240386963, 0.8748490810394287, 0.550000011920929, 1.0214027166366577]`

#### Selected Features

`D1_lag1`, `None_lag1`, `RH2M`, `PREC_roll12_std`, `T2M`, `T2M_lag4`, `RH2M_lag4`, `PREC_roll4_mean`, `RH2M_lag2`, `PRECTOTCORR`, `RH2M_lag8`, `PREC_lag2`, `RH2M_lag1`, `PREC_roll4_std`, `PREC_roll12_mean`, `ALLSKY_SFC_SW_DWN`, `PREC_lag8`, `PREC_lag4`, `WS2M`, `PREC_lag1`, `T2M_lag8`, `PS`, `D2_lag1`, `D3_lag1`, `D4_lag1`

#### Results

| Metric | Raw | Tuned |
| --- | ---: | ---: |
| Accuracy | 0.8184 | 0.8182 |
| Macro F1 | 0.8245 | 0.8258 |
| Weighted F1 | 0.8182 | 0.8184 |

#### Per-class F1

| Class | Raw | Tuned |
| --- | ---: | ---: |
| None | 0.9015 | 0.8948 |
| D0 | 0.7720 | 0.7677 |
| D1 | 0.7628 | 0.7555 |
| D2 | 0.7771 | 0.8192 |
| D3 | 0.8006 | 0.8045 |
| D4 | 0.9328 | 0.9134 |

#### Classification Report (Raw)

```text
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

```text
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

## `output_weekly_kansas_scenario3`

### Scenario 3: Macro F1-driven Feature Selection

- Best trial: `ros_focal_inv_no_cw_128x64`
- Best model path: `F:\Projectan\TA_new\enhprota\kansas\output_weekly_kansas_scenario3\best_model.keras`
- Best trial config: `{'name': 'ros_focal_inv_no_cw_128x64', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (128, 64), 'dropout': 0.28, 'dense_units': 96, 'lr': 0.0006, 'patience': 14}`
- Seq length: `52`
- Feature count: `25`
- Val Macro F1 (raw/tuned): `0.1990 / 0.2093`
- Class multipliers: `[0.6714562177658081, 1.7999999523162842, 1.3767106533050537, 0.858252227306366, 0.7342633605003357, 1.16351318359375]`

#### Selected Features

`D3_lag1`, `D1_lag1`, `D2_lag1`, `None_lag1`, `D0_lag1`, `D2_lag2`, `D3_lag2`, `PREC_lag2`, `week_cos`, `None_lag2`, `T2M_roll4_mean`, `PREC_roll4_mean`, `D4_lag1`, `D4_lag2`, `T2M_roll12_mean`, `T2M_lag2`, `T2M_lag1`, `PREC_lag8`, `PRECTOTCORR`, `RH2M_lag1`, `D0_lag2`, `RH2M_lag2`, `T2M_lag4`, `T2M_lag8`, `PREC_lag1`

#### Results

| Metric | Raw | Tuned |
| --- | ---: | ---: |
| Accuracy | 0.3364 | 0.3333 |
| Macro F1 | 0.2483 | 0.2391 |
| Weighted F1 | 0.3048 | 0.3054 |

#### Per-class F1

| Class | Raw | Tuned |
| --- | ---: | ---: |
| None | 0.8985 | 0.8916 |
| D0 | 0.1150 | 0.1479 |
| D1 | 0.0053 | 0.0052 |
| D2 | 0.0033 | 0.0033 |
| D3 | 0.0195 | 0.0120 |
| D4 | 0.4481 | 0.3748 |

#### Classification Report (Raw)

```text
              precision    recall  f1-score   support

        None     0.8710    0.9278    0.8985      1150
          D0     0.7093    0.0626    0.1150       975
          D1     0.0161    0.0032    0.0053       948
          D2     0.0031    0.0036    0.0033       560
          D3     0.0122    0.0486    0.0195       288
          D4     0.2887    1.0000    0.4481       259

    accuracy                         0.3364      4180
   macro avg     0.3168    0.3410    0.2483      4180
weighted avg     0.4279    0.3364    0.3048      4180
```

#### Classification Report (Tuned)

```text
              precision    recall  f1-score   support

        None     0.8791    0.9043    0.8916      1150
          D0     0.6119    0.0841    0.1479       975
          D1     0.0143    0.0032    0.0052       948
          D2     0.0031    0.0036    0.0033       560
          D3     0.0079    0.0243    0.0120       288
          D4     0.2306    1.0000    0.3748       259

    accuracy                         0.3333      4180
   macro avg     0.2912    0.3366    0.2391      4180
weighted avg     0.4031    0.3333    0.3054      4180
```

---

## `output_weekly_kansas_scenario4`

### Scenario 4: Weather Only

- Best trial: `ros_focal_no_cw_96x48`
- Best model path: `F:\Projectan\TA_new\enhprota\kansas\output_weekly_kansas_scenario4\best_model.keras`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': True, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq length: `52`
- Feature count: `6`
- Val Macro F1 (raw/tuned): `0.1746 / 0.2060`
- Class multipliers: `[1.3822839260101318, 1.0023996829986572, 1.268524169921875, 0.8633322715759277, 0.5805516242980957, 1.0838371515274048]`

#### Selected Features

`ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`

#### Results

| Metric | Raw | Tuned |
| --- | ---: | ---: |
| Accuracy | 0.1031 | 0.1438 |
| Macro F1 | 0.0965 | 0.1185 |
| Weighted F1 | 0.0798 | 0.1201 |

#### Per-class F1

| Class | Raw | Tuned |
| --- | ---: | ---: |
| None | 0.0437 | 0.1256 |
| D0 | 0.1070 | 0.0758 |
| D1 | 0.0121 | 0.1301 |
| D2 | 0.1897 | 0.2062 |
| D3 | 0.0880 | 0.0000 |
| D4 | 0.1385 | 0.1734 |

#### Classification Report (Raw)

```text
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

```text
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

## `output_weekly_kansas_scenario5`

### Scenario 5: Weather + Lag Only

- Best trial: `none_focal_inv_cw_64x32`
- Best model path: `F:\Projectan\TA_new\enhprota\kansas\output_weekly_kansas_scenario5\best_model.keras`
- Best trial config: `{'name': 'none_focal_inv_cw_64x32', 'balancer': 'NONE', 'use_class_weight': True, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (64, 32), 'dropout': 0.35, 'dense_units': 64, 'lr': 0.001, 'patience': 16}`
- Seq length: `52`
- Feature count: `20`
- Val Macro F1 (raw/tuned): `0.2068 / 0.2498`
- Class multipliers: `[1.7999999523162842, 0.6552096605300903, 0.8391038179397583, 0.8592265844345093, 1.7999999523162842, 0.6729795336723328]`

#### Selected Features

`ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`, `PREC_lag1`, `PREC_lag2`, `PREC_lag4`, `PREC_lag8`, `T2M_lag1`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `RH2M_lag1`, `RH2M_lag2`, `RH2M_lag4`, `RH2M_lag8`, `week_sin`, `week_cos`

#### Results

| Metric | Raw | Tuned |
| --- | ---: | ---: |
| Accuracy | 0.1718 | 0.1715 |
| Macro F1 | 0.1380 | 0.1152 |
| Weighted F1 | 0.1625 | 0.1532 |

#### Per-class F1

| Class | Raw | Tuned |
| --- | ---: | ---: |
| None | 0.1603 | 0.4184 |
| D0 | 0.1939 | 0.0000 |
| D1 | 0.2634 | 0.1209 |
| D2 | 0.0000 | 0.0034 |
| D3 | 0.0643 | 0.1485 |
| D4 | 0.1458 | 0.0000 |

#### Classification Report (Raw)

```text
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

```text
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

## `output_weekly_kansas_scenario6`

### Scenario 6: No Drought History

- Best trial: `none_focal_inv_cw_64x32`
- Best model path: `F:\Projectan\TA_new\enhprota\kansas\output_weekly_kansas_scenario6\best_model.keras`
- Best trial config: `{'name': 'none_focal_inv_cw_64x32', 'balancer': 'NONE', 'use_class_weight': True, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (64, 32), 'dropout': 0.35, 'dense_units': 64, 'lr': 0.001, 'patience': 16}`
- Seq length: `52`
- Feature count: `27`
- Val Macro F1 (raw/tuned): `0.2510 / 0.2570`
- Class multipliers: `[1.6870434284210205, 1.0435301065444946, 1.0128508806228638, 1.0313466787338257, 0.9367184042930603, 0.8775627613067627]`

#### Selected Features

`ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`, `PREC_lag1`, `PREC_lag2`, `PREC_lag4`, `PREC_lag8`, `T2M_lag1`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`, `RH2M_lag1`, `RH2M_lag2`, `RH2M_lag4`, `RH2M_lag8`, `PREC_roll4_mean`, `PREC_roll4_std`, `PREC_roll12_mean`, `PREC_roll12_std`, `T2M_roll4_mean`, `T2M_roll12_mean`, `week_sin`, `week_cos`, `heat_dry_stress`

#### Results

| Metric | Raw | Tuned |
| --- | ---: | ---: |
| Accuracy | 0.2031 | 0.2134 |
| Macro F1 | 0.1900 | 0.1991 |
| Weighted F1 | 0.2074 | 0.2232 |

#### Per-class F1

| Class | Raw | Tuned |
| --- | ---: | ---: |
| None | 0.1741 | 0.2366 |
| D0 | 0.2771 | 0.2610 |
| D1 | 0.2509 | 0.2584 |
| D2 | 0.1419 | 0.1511 |
| D3 | 0.0892 | 0.0836 |
| D4 | 0.2070 | 0.2041 |

#### Classification Report (Raw)

```text
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

```text
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

## `output_weekly_kansas_scenario7`

### Scenario 7: Drought History Only

- Best trial: `ros_focal_inv_no_cw_128x64`
- Best model path: `F:\Projectan\TA_new\enhprota\kansas\output_weekly_kansas_scenario7\best_model.keras`
- Best trial config: `{'name': 'ros_focal_inv_no_cw_128x64', 'balancer': 'ROS', 'use_class_weight': True, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (128, 64), 'dropout': 0.28, 'dense_units': 96, 'lr': 0.0006, 'patience': 14}`
- Seq length: `52`
- Feature count: `12`
- Val Macro F1 (raw/tuned): `0.8446 / 0.8481`
- Class multipliers: `[1.4989957809448242, 1.2145313024520874, 1.2288269996643066, 1.1994571685791016, 0.6609528660774231, 0.7688682675361633]`

#### Selected Features

`None_lag1`, `D0_lag1`, `D1_lag1`, `D2_lag1`, `D3_lag1`, `D4_lag1`, `None_lag2`, `D0_lag2`, `D1_lag2`, `D2_lag2`, `D3_lag2`, `D4_lag2`

#### Results

| Metric | Raw | Tuned |
| --- | ---: | ---: |
| Accuracy | 0.8177 | 0.8246 |
| Macro F1 | 0.8202 | 0.8289 |
| Weighted F1 | 0.8181 | 0.8240 |

#### Per-class F1

| Class | Raw | Tuned |
| --- | ---: | ---: |
| None | 0.9043 | 0.9045 |
| D0 | 0.7743 | 0.7661 |
| D1 | 0.7767 | 0.7784 |
| D2 | 0.7549 | 0.8062 |
| D3 | 0.7695 | 0.7891 |
| D4 | 0.9416 | 0.9294 |

#### Classification Report (Raw)

```text
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

```text
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

