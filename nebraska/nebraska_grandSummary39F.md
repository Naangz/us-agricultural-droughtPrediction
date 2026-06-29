# Nebraska Grand Summary

Dokumen ini adalah kompilasi seluruh `results_summary.txt` untuk scenario Nebraska.

## Scenario 1 - Nebraska Baseline

### Best Configuration

- Best trial: `ros_focal_no_cw_96x48`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq Length: `52`
- Val Macro F1 (best trial): `0.8514`
- Val Macro F1 (raw): `0.8514`
- Val Macro F1 (tuned): `0.8641`
- Class multipliers: `[1.3538963794708252, 1.0908902883529663, 0.7953732013702393, 0.7809790968894958, 1.7999999523162842, 0.7386244535446167]`

### Features Used

- `ALLSKY_SFC_SW_DWN`
- `PRECTOTCORR`
- `PS`
- `RH2M`
- `T2M`
- `WS2M`
- `PREC_lag1`
- `PREC_lag2`
- `PREC_lag4`
- `PREC_lag8`
- `T2M_lag1`
- `T2M_lag2`
- `T2M_lag4`
- `T2M_lag8`
- `RH2M_lag1`
- `RH2M_lag2`
- `RH2M_lag4`
- `RH2M_lag8`
- `PREC_roll4_mean`
- `PREC_roll4_std`
- `PREC_roll12_mean`
- `PREC_roll12_std`
- `T2M_roll4_mean`
- `T2M_roll12_mean`
- `week_sin`
- `week_cos`
- `None_lag1`
- `D0_lag1`
- `D1_lag1`
- `D2_lag1`
- `D3_lag1`
- `D4_lag1`
- `None_lag2`
- `D0_lag2`
- `D1_lag2`
- `D2_lag2`
- `D3_lag2`
- `D4_lag2`
- `heat_dry_stress`

### Raw Results

- Accuracy: `0.8077`
- Macro F1: `0.7958`
- Weighted F1: `0.8086`

### Per-Class F1 Raw

- `None`: `0.9134`
- `D0`: `0.6852`
- `D1`: `0.7893`
- `D2`: `0.8061`
- `D3`: `0.7836`
- `D4`: `0.7974`

### Classification Report Raw

```text
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

### Tuned Results

- Accuracy: `0.7969`
- Macro F1: `0.7718`
- Weighted F1: `0.7980`

### Per-Class F1 Tuned

- `None`: `0.9187`
- `D0`: `0.7263`
- `D1`: `0.7972`
- `D2`: `0.7686`
- `D3`: `0.7401`
- `D4`: `0.6800`

### Classification Report Tuned

```text
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

## Scenario 2 - Correlation-aware Feature Selection

### Best Configuration

- Best trial: `ros_focal_no_cw_96x48`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq Length: `52`
- Val Macro F1 (best trial): `0.4707`
- Val Macro F1 (raw): `0.4707`
- Val Macro F1 (tuned): `0.4932`
- Class multipliers: `[1.1562503576278687, 0.8265590667724609, 0.550000011920929, 0.6033008694648743, 1.5943379402160645, 0.7343587875366211]`

### Features Used

- `None_lag1`
- `RH2M_lag8`
- `T2M_lag8`
- `ALLSKY_SFC_SW_DWN`
- `PREC_roll12_std`
- `T2M_lag1`
- `RH2M_lag4`
- `T2M_lag4`
- `PS`
- `PREC_roll4_mean`
- `heat_dry_stress`
- `RH2M_lag1`
- `RH2M_lag2`
- `PREC_roll12_mean`
- `PREC_lag2`

### Raw Results

- Accuracy: `0.4340`
- Macro F1: `0.3892`
- Weighted F1: `0.4425`

### Per-Class F1 Raw

- `None`: `0.9088`
- `D0`: `0.4061`
- `D1`: `0.3472`
- `D2`: `0.2029`
- `D3`: `0.3793`
- `D4`: `0.0911`

### Classification Report Raw

```text
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

### Tuned Results

- Accuracy: `0.4124`
- Macro F1: `0.3428`
- Weighted F1: `0.3803`

### Per-Class F1 Tuned

- `None`: `0.9246`
- `D0`: `0.3827`
- `D1`: `0.2237`
- `D2`: `0.0788`
- `D3`: `0.3562`
- `D4`: `0.0908`

### Classification Report Tuned

```text
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

## Scenario 2A - Correlation-aware Feature Selection (Top-20)

### Best Configuration

- Best trial: `ros_focal_no_cw_96x48`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq Length: `52`
- Val Macro F1 (best trial): `0.4832`
- Val Macro F1 (raw): `0.4832`
- Val Macro F1 (tuned): `0.5086`
- Class multipliers: `[1.090155839920044, 0.7491489052772522, 0.6521157622337341, 0.6612790822982788, 1.4907511472702026, 0.7446761727333069]`

### Features Used

- `None_lag1`
- `RH2M_lag8`
- `T2M_lag8`
- `ALLSKY_SFC_SW_DWN`
- `PREC_roll12_std`
- `T2M_lag1`
- `RH2M_lag4`
- `T2M_lag4`
- `PS`
- `PREC_roll4_mean`
- `heat_dry_stress`
- `RH2M_lag1`
- `RH2M_lag2`
- `PREC_roll12_mean`
- `PREC_lag2`
- `PREC_roll4_std`
- `RH2M`
- `PREC_lag8`
- `WS2M`
- `PRECTOTCORR`

### Raw Results

- Accuracy: `0.4529`
- Macro F1: `0.3962`
- Weighted F1: `0.4530`

### Per-Class F1 Raw

- `None`: `0.9100`
- `D0`: `0.3559`
- `D1`: `0.4573`
- `D2`: `0.2238`
- `D3`: `0.2190`
- `D4`: `0.2111`

### Classification Report Raw

```text
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

### Tuned Results

- Accuracy: `0.4438`
- Macro F1: `0.3802`
- Weighted F1: `0.4164`

### Per-Class F1 Tuned

- `None`: `0.9337`
- `D0`: `0.3395`
- `D1`: `0.3598`
- `D2`: `0.0982`
- `D3`: `0.3545`
- `D4`: `0.1957`

### Classification Report Tuned

```text
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

## Scenario 2B - Correlation-aware Feature Selection (Top-25)

### Best Configuration

- Best trial: `ros_focal_no_cw_96x48`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq Length: `52`
- Val Macro F1 (best trial): `0.6525`
- Val Macro F1 (raw): `0.6525`
- Val Macro F1 (tuned): `0.6889`
- Class multipliers: `[1.4479331970214844, 1.148991584777832, 0.6998769044876099, 0.854198157787323, 1.7862962484359741, 0.6537574529647827]`

### Features Used

- `None_lag1`
- `RH2M_lag8`
- `T2M_lag8`
- `ALLSKY_SFC_SW_DWN`
- `PREC_roll12_std`
- `T2M_lag1`
- `RH2M_lag4`
- `T2M_lag4`
- `PS`
- `PREC_roll4_mean`
- `heat_dry_stress`
- `RH2M_lag1`
- `RH2M_lag2`
- `PREC_roll12_mean`
- `PREC_lag2`
- `PREC_roll4_std`
- `RH2M`
- `PREC_lag8`
- `WS2M`
- `PRECTOTCORR`
- `PREC_lag1`
- `PREC_lag4`
- `D1_lag1`
- `D2_lag1`
- `D3_lag1`

### Raw Results

- Accuracy: `0.7536`
- Macro F1: `0.6695`
- Weighted F1: `0.7636`

### Per-Class F1 Raw

- `None`: `0.9135`
- `D0`: `0.6929`
- `D1`: `0.7972`
- `D2`: `0.7931`
- `D3`: `0.5684`
- `D4`: `0.2519`

### Classification Report Raw

```text
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

### Tuned Results

- Accuracy: `0.7589`
- Macro F1: `0.6695`
- Weighted F1: `0.7602`

### Per-Class F1 Tuned

- `None`: `0.9200`
- `D0`: `0.7182`
- `D1`: `0.7721`
- `D2`: `0.7607`
- `D3`: `0.6334`
- `D4`: `0.2128`

### Classification Report Tuned

```text
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

## Scenario 2C - Correlation-aware Feature Selection (Top-30)

### Best Configuration

- Best trial: `ros_focal_no_cw_96x48`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq Length: `52`
- Val Macro F1 (best trial): `0.8598`
- Val Macro F1 (raw): `0.8598`
- Val Macro F1 (tuned): `0.8653`
- Class multipliers: `[1.0397485494613647, 0.9792780876159668, 0.6702917218208313, 0.6430429816246033, 1.1847668886184692, 0.550000011920929]`

### Features Used

- `None_lag1`
- `RH2M_lag8`
- `T2M_lag8`
- `ALLSKY_SFC_SW_DWN`
- `PREC_roll12_std`
- `T2M_lag1`
- `RH2M_lag4`
- `T2M_lag4`
- `PS`
- `PREC_roll4_mean`
- `heat_dry_stress`
- `RH2M_lag1`
- `RH2M_lag2`
- `PREC_roll12_mean`
- `PREC_lag2`
- `PREC_roll4_std`
- `RH2M`
- `PREC_lag8`
- `WS2M`
- `PRECTOTCORR`
- `PREC_lag1`
- `PREC_lag4`
- `D1_lag1`
- `D2_lag1`
- `D3_lag1`
- `D4_lag1`

### Raw Results

- Accuracy: `0.8086`
- Macro F1: `0.8010`
- Weighted F1: `0.8109`

### Per-Class F1 Raw

- `None`: `0.9093`
- `D0`: `0.7183`
- `D1`: `0.8103`
- `D2`: `0.7937`
- `D3`: `0.7522`
- `D4`: `0.8224`

### Classification Report Raw

```text
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

### Tuned Results

- Accuracy: `0.8029`
- Macro F1: `0.7867`
- Weighted F1: `0.8058`

### Per-Class F1 Tuned

- `None`: `0.9132`
- `D0`: `0.7343`
- `D1`: `0.8103`
- `D2`: `0.7809`
- `D3`: `0.7348`
- `D4`: `0.7464`

### Classification Report Tuned

```text
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

## Scenario 3 - Macro F1-driven Feature Selection

### Best Configuration

- Best trial: `ros_focal_inv_no_cw_128x64`
- Best trial config: `{'name': 'ros_focal_inv_no_cw_128x64', 'balancer': 'ROS', 'use_class_weight': False, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (128, 64), 'dropout': 0.28, 'dense_units': 96, 'lr': 0.0006, 'patience': 14}`
- Seq Length: `52`
- Val Macro F1 (best trial): `0.8241`
- Val Macro F1 (raw): `0.8241`
- Val Macro F1 (tuned): `0.8246`
- Class multipliers: `[0.9220073223114014, 0.9669707417488098, 0.9690529704093933, 0.9767623543739319, 0.9451612234115601, 1.078676462173462]`

### Features Used

- `None_lag1`
- `D2_lag1`
- `D1_lag1`
- `D3_lag1`
- `D1_lag2`
- `D0_lag1`
- `None_lag2`
- `D3_lag2`
- `D0_lag2`
- `D2_lag2`
- `T2M`
- `WS2M`
- `D4_lag1`
- `D4_lag2`
- `PS`

### Raw Results

- Accuracy: `0.7589`
- Macro F1: `0.7440`
- Weighted F1: `0.7590`

### Per-Class F1 Raw

- `None`: `0.9126`
- `D0`: `0.6871`
- `D1`: `0.7411`
- `D2`: `0.6978`
- `D3`: `0.7189`
- `D4`: `0.7067`

### Classification Report Raw

```text
              precision    recall  f1-score   support

        None     0.9353    0.8911    0.9126       973
          D0     0.6599    0.7166    0.6871       547
          D1     0.6972    0.7910    0.7411       952
          D2     0.8312    0.6013    0.6978      1081
          D3     0.6206    0.8541    0.7189       473
          D4     0.7260    0.6883    0.7067       154

    accuracy                         0.7589      4180
   macro avg     0.7450    0.7571    0.7440      4180
weighted avg     0.7748    0.7589    0.7590      4180
```

### Tuned Results

- Accuracy: `0.7593`
- Macro F1: `0.7465`
- Weighted F1: `0.7599`

### Per-Class F1 Tuned

- `None`: `0.9107`
- `D0`: `0.6864`
- `D1`: `0.7414`
- `D2`: `0.7019`
- `D3`: `0.7167`
- `D4`: `0.7220`

### Classification Report Tuned

```text
              precision    recall  f1-score   support

        None     0.9370    0.8859    0.9107       973
          D0     0.6556    0.7203    0.6864       547
          D1     0.6993    0.7889    0.7414       952
          D2     0.8287    0.6087    0.7019      1081
          D3     0.6266    0.8372    0.7167       473
          D4     0.7107    0.7338    0.7220       154

    accuracy                         0.7593      4180
   macro avg     0.7430    0.7625    0.7465      4180
weighted avg     0.7745    0.7593    0.7599      4180
```

## Scenario 4 - Weather Only

### Best Configuration

- Best trial: `none_focal_inv_cw_64x32`
- Best trial config: `{'name': 'none_focal_inv_cw_64x32', 'balancer': 'NONE', 'use_class_weight': True, 'focal_gamma': 2.0, 'focal_alpha_mode': 'inverse_train_seq', 'manual_focal_alpha': None, 'lstm_units': (64, 32), 'dropout': 0.35, 'dense_units': 64, 'lr': 0.001, 'patience': 16}`
- Seq Length: `52`
- Val Macro F1 (best trial): `0.1520`
- Val Macro F1 (raw): `0.1520`
- Val Macro F1 (tuned): `0.2774`
- Class multipliers: `[1.7999999523162842, 0.8442162275314331, 0.762617290019989, 0.7349960207939148, 0.938678503036499, 0.5969864726066589]`

### Features Used

- `ALLSKY_SFC_SW_DWN`
- `PRECTOTCORR`
- `PS`
- `RH2M`
- `T2M`
- `WS2M`

### Raw Results

- Accuracy: `0.1689`
- Macro F1: `0.1063`
- Weighted F1: `0.1175`

### Per-Class F1 Raw

- `None`: `0.0000`
- `D0`: `0.0000`
- `D1`: `0.1331`
- `D2`: `0.2756`
- `D3`: `0.0980`
- `D4`: `0.1308`

### Classification Report Raw

```text
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

### Tuned Results

- Accuracy: `0.1990`
- Macro F1: `0.1516`
- Weighted F1: `0.1455`

### Per-Class F1 Tuned

- `None`: `0.0000`
- `D0`: `0.0000`
- `D1`: `0.1749`
- `D2`: `0.2589`
- `D3`: `0.2780`
- `D4`: `0.1979`

### Classification Report Tuned

```text
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

## Scenario 5 - Weather + Lag Only

### Best Configuration

- Best trial: `ros_focal_no_cw_96x48`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': True, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq Length: `52`
- Val Macro F1 (best trial): `0.3156`
- Val Macro F1 (raw): `0.3156`
- Val Macro F1 (tuned): `0.3183`
- Class multipliers: `[0.9987471699714661, 0.8659706115722656, 0.8903961777687073, 0.83596271276474, 1.640679955482483, 0.8019959330558777]`

### Features Used

- `ALLSKY_SFC_SW_DWN`
- `PRECTOTCORR`
- `PS`
- `RH2M`
- `T2M`
- `WS2M`
- `PREC_lag1`
- `PREC_lag2`
- `PREC_lag4`
- `PREC_lag8`
- `T2M_lag1`
- `T2M_lag2`
- `T2M_lag4`
- `T2M_lag8`
- `RH2M_lag1`
- `RH2M_lag2`
- `RH2M_lag4`
- `RH2M_lag8`
- `week_sin`
- `week_cos`

### Raw Results

- Accuracy: `0.3079`
- Macro F1: `0.2680`
- Weighted F1: `0.2871`

### Per-Class F1 Raw

- `None`: `0.5725`
- `D0`: `0.2664`
- `D1`: `0.2895`
- `D2`: `0.1012`
- `D3`: `0.1697`
- `D4`: `0.2091`

### Classification Report Raw

```text
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

### Tuned Results

- Accuracy: `0.3383`
- Macro F1: `0.2709`
- Weighted F1: `0.3013`

### Per-Class F1 Tuned

- `None`: `0.5704`
- `D0`: `0.2612`
- `D1`: `0.3129`
- `D2`: `0.0594`
- `D3`: `0.4215`
- `D4`: `0.0000`

### Classification Report Tuned

```text
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

## Scenario 6 - No Drought History

### Best Configuration

- Best trial: `ros_focal_no_cw_96x48`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': True, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq Length: `52`
- Val Macro F1 (best trial): `0.3240`
- Val Macro F1 (raw): `0.3240`
- Val Macro F1 (tuned): `0.3306`
- Class multipliers: `[1.0156711339950562, 0.829376757144928, 0.8538907766342163, 0.7135670781135559, 1.4618934392929077, 1.3767223358154297]`

### Features Used

- `ALLSKY_SFC_SW_DWN`
- `PRECTOTCORR`
- `PS`
- `RH2M`
- `T2M`
- `WS2M`
- `PREC_lag1`
- `PREC_lag2`
- `PREC_lag4`
- `PREC_lag8`
- `T2M_lag1`
- `T2M_lag2`
- `T2M_lag4`
- `T2M_lag8`
- `RH2M_lag1`
- `RH2M_lag2`
- `RH2M_lag4`
- `RH2M_lag8`
- `PREC_roll4_mean`
- `PREC_roll4_std`
- `PREC_roll12_mean`
- `PREC_roll12_std`
- `T2M_roll4_mean`
- `T2M_roll12_mean`
- `week_sin`
- `week_cos`
- `heat_dry_stress`

### Raw Results

- Accuracy: `0.3347`
- Macro F1: `0.2865`
- Weighted F1: `0.3250`

### Per-Class F1 Raw

- `None`: `0.5981`
- `D0`: `0.2927`
- `D1`: `0.3953`
- `D2`: `0.1499`
- `D3`: `0.1079`
- `D4`: `0.1751`

### Classification Report Raw

```text
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

### Tuned Results

- Accuracy: `0.3344`
- Macro F1: `0.2725`
- Weighted F1: `0.2966`

### Per-Class F1 Tuned

- `None`: `0.6176`
- `D0`: `0.2092`
- `D1`: `0.3813`
- `D2`: `0.0254`
- `D3`: `0.2261`
- `D4`: `0.1751`

### Classification Report Tuned

```text
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

## Scenario 7 - Drought History Only

### Best Configuration

- Best trial: `ros_focal_no_cw_96x48`
- Best trial config: `{'name': 'ros_focal_no_cw_96x48', 'balancer': 'ROS', 'use_class_weight': True, 'focal_gamma': 1.5, 'focal_alpha_mode': 'none', 'manual_focal_alpha': None, 'lstm_units': (96, 48), 'dropout': 0.3, 'dense_units': 64, 'lr': 0.0008, 'patience': 14}`
- Seq Length: `52`
- Val Macro F1 (best trial): `0.8623`
- Val Macro F1 (raw): `0.8623`
- Val Macro F1 (tuned): `0.8711`
- Class multipliers: `[1.7999999523162842, 1.576066255569458, 0.9121327996253967, 0.8369708061218262, 0.9069540500640869, 1.1652894020080566]`

### Features Used

- `None_lag1`
- `D0_lag1`
- `D1_lag1`
- `D2_lag1`
- `D3_lag1`
- `D4_lag1`
- `None_lag2`
- `D0_lag2`
- `D1_lag2`
- `D2_lag2`
- `D3_lag2`
- `D4_lag2`

### Raw Results

- Accuracy: `0.8254`
- Macro F1: `0.8136`
- Weighted F1: `0.8259`

### Per-Class F1 Raw

- `None`: `0.9254`
- `D0`: `0.7278`
- `D1`: `0.8104`
- `D2`: `0.8150`
- `D3`: `0.7971`
- `D4`: `0.8059`

### Classification Report Raw

```text
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

### Tuned Results

- Accuracy: `0.8254`
- Macro F1: `0.8161`
- Weighted F1: `0.8267`

### Per-Class F1 Tuned

- `None`: `0.9274`
- `D0`: `0.7509`
- `D1`: `0.8047`
- `D2`: `0.8128`
- `D3`: `0.7882`
- `D4`: `0.8127`

### Classification Report Tuned

```text
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
