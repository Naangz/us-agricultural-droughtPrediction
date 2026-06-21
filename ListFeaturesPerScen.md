# List Features Per Scenario

Dokumen ini berisi daftar fitur yang dipakai pada semua skenario untuk semua wilayah yang ada di repo ini, yaitu:

- `kansas`
- `nebraska`

## Ringkasan Cepat

| Scenario | Kansas | Nebraska | Catatan |
| --- | --- | --- | --- |
| Scenario 1 | 39 fitur | 39 fitur | Sama |
| Scenario 2 | 15 fitur terpilih | 15 fitur terpilih | Berbeda |
| Scenario 2A | 20 fitur terpilih | 20 fitur terpilih | Berbeda |
| Scenario 2B | 25 fitur terpilih | 25 fitur terpilih | Berbeda |
| Scenario 2C | 25 fitur terpilih | 26 fitur terpilih | Berbeda |
| Scenario 3 | 10 fitur terpilih | 15 fitur terpilih | Berbeda |
| Scenario 4 | 6 fitur | 6 fitur | Sama |
| Scenario 5 | 20 fitur | 20 fitur | Sama |
| Scenario 6 | 27 fitur | 27 fitur | Sama |
| Scenario 7 | 12 fitur | 12 fitur | Sama |

## Feature Group Legend

- **Weather dasar**
  - `ALLSKY_SFC_SW_DWN`
  - `PRECTOTCORR`
  - `PS`
  - `RH2M`
  - `T2M`
  - `WS2M`
- **Lag cuaca**
  - `PREC_lag1`, `PREC_lag2`, `PREC_lag4`, `PREC_lag8`
  - `T2M_lag1`, `T2M_lag2`, `T2M_lag4`, `T2M_lag8`
  - `RH2M_lag1`, `RH2M_lag2`, `RH2M_lag4`, `RH2M_lag8`
- **Rolling stats**
  - `PREC_roll4_mean`, `PREC_roll4_std`
  - `PREC_roll12_mean`, `PREC_roll12_std`
  - `T2M_roll4_mean`, `T2M_roll12_mean`
- **Seasonality**
  - `week_sin`
  - `week_cos`
- **Drought history**
  - `None_lag1`, `D0_lag1`, `D1_lag1`, `D2_lag1`, `D3_lag1`, `D4_lag1`
  - `None_lag2`, `D0_lag2`, `D1_lag2`, `D2_lag2`, `D3_lag2`, `D4_lag2`
- **Derived drought features**
  - `drought_carryover_lag1`
  - `severe_carryover_lag1`
- **Interaction**
  - `heat_dry_stress`

## Kansas

### Scenario 1

Jumlah fitur: `39`

```text
ALLSKY_SFC_SW_DWN
PRECTOTCORR
PS
RH2M
T2M
WS2M
PREC_lag1
PREC_lag2
PREC_lag4
PREC_lag8
T2M_lag1
T2M_lag2
T2M_lag4
T2M_lag8
RH2M_lag1
RH2M_lag2
RH2M_lag4
RH2M_lag8
PREC_roll4_mean
PREC_roll4_std
PREC_roll12_mean
PREC_roll12_std
T2M_roll4_mean
T2M_roll12_mean
week_sin
week_cos
None_lag1
D0_lag1
D1_lag1
D2_lag1
D3_lag1
D4_lag1
None_lag2
D0_lag2
D1_lag2
D2_lag2
D3_lag2
D4_lag2
heat_dry_stress
```

### Scenario 2

Jumlah fitur: `15`

```text
drought_carryover_lag1
D0_lag1
RH2M_lag8
T2M
RH2M
RH2M_lag4
PREC_roll12_std
PREC_roll4_mean
RH2M_lag2
T2M_roll12_mean
ALLSKY_SFC_SW_DWN
RH2M_lag1
PREC_roll12_mean
PRECTOTCORR
PREC_lag2
```

### Scenario 2A

Jumlah fitur: `20`

```text
drought_carryover_lag1
D0_lag1
RH2M_lag8
T2M
RH2M
RH2M_lag4
PREC_roll12_std
PREC_roll4_mean
RH2M_lag2
T2M_roll12_mean
ALLSKY_SFC_SW_DWN
RH2M_lag1
PREC_roll12_mean
PRECTOTCORR
PREC_lag2
PREC_lag1
PREC_lag4
PREC_roll4_std
WS2M
PS
```

### Scenario 2B

Jumlah fitur: `25`

```text
drought_carryover_lag1
D0_lag1
RH2M_lag8
T2M
RH2M
RH2M_lag4
PREC_roll12_std
PREC_roll4_mean
RH2M_lag2
T2M_roll12_mean
ALLSKY_SFC_SW_DWN
RH2M_lag1
PREC_roll12_mean
PRECTOTCORR
PREC_lag2
PREC_lag1
PREC_lag4
PREC_roll4_std
WS2M
PS
PREC_lag8
D2_lag1
severe_carryover_lag1
D4_lag1
week_sin
```

### Scenario 2C

Jumlah fitur: `25`

Catatan: pada Kansas, hasil akhir Scenario 2C identik dengan Scenario 2B.

```text
drought_carryover_lag1
D0_lag1
RH2M_lag8
T2M
RH2M
RH2M_lag4
PREC_roll12_std
PREC_roll4_mean
RH2M_lag2
T2M_roll12_mean
ALLSKY_SFC_SW_DWN
RH2M_lag1
PREC_roll12_mean
PRECTOTCORR
PREC_lag2
PREC_lag1
PREC_lag4
PREC_roll4_std
WS2M
PS
PREC_lag8
D2_lag1
severe_carryover_lag1
D4_lag1
week_sin
```

### Scenario 3

Jumlah fitur: `10`

```text
D3_lag1
D2_lag1
None_lag1
D1_lag1
D0_lag1
severe_carryover_lag1
D2_lag2
PREC_lag1
drought_carryover_lag1
PREC_roll12_mean
```

### Scenario 4

Jumlah fitur: `6`

```text
ALLSKY_SFC_SW_DWN
PRECTOTCORR
PS
RH2M
T2M
WS2M
```

### Scenario 5

Jumlah fitur: `20`

```text
ALLSKY_SFC_SW_DWN
PRECTOTCORR
PS
RH2M
T2M
WS2M
PREC_lag1
PREC_lag2
PREC_lag4
PREC_lag8
T2M_lag1
T2M_lag2
T2M_lag4
T2M_lag8
RH2M_lag1
RH2M_lag2
RH2M_lag4
RH2M_lag8
week_sin
week_cos
```

### Scenario 6

Jumlah fitur: `27`

```text
ALLSKY_SFC_SW_DWN
PRECTOTCORR
PS
RH2M
T2M
WS2M
PREC_lag1
PREC_lag2
PREC_lag4
PREC_lag8
T2M_lag1
T2M_lag2
T2M_lag4
T2M_lag8
RH2M_lag1
RH2M_lag2
RH2M_lag4
RH2M_lag8
PREC_roll4_mean
PREC_roll4_std
PREC_roll12_mean
PREC_roll12_std
T2M_roll4_mean
T2M_roll12_mean
week_sin
week_cos
heat_dry_stress
```

### Scenario 7

Jumlah fitur: `12`

```text
None_lag1
D0_lag1
D1_lag1
D2_lag1
D3_lag1
D4_lag1
None_lag2
D0_lag2
D1_lag2
D2_lag2
D3_lag2
D4_lag2
```

## Nebraska

### Scenario 1

Jumlah fitur: `39`

```text
ALLSKY_SFC_SW_DWN
PRECTOTCORR
PS
RH2M
T2M
WS2M
PREC_lag1
PREC_lag2
PREC_lag4
PREC_lag8
T2M_lag1
T2M_lag2
T2M_lag4
T2M_lag8
RH2M_lag1
RH2M_lag2
RH2M_lag4
RH2M_lag8
PREC_roll4_mean
PREC_roll4_std
PREC_roll12_mean
PREC_roll12_std
T2M_roll4_mean
T2M_roll12_mean
week_sin
week_cos
None_lag1
D0_lag1
D1_lag1
D2_lag1
D3_lag1
D4_lag1
None_lag2
D0_lag2
D1_lag2
D2_lag2
D3_lag2
D4_lag2
heat_dry_stress
```

### Scenario 2

Jumlah fitur: `15`

```text
drought_carryover_lag1
PREC_roll12_std
PREC_roll4_mean
RH2M_lag1
T2M_roll12_mean
RH2M_lag4
ALLSKY_SFC_SW_DWN
PREC_roll4_std
RH2M_lag8
PREC_roll12_mean
PS
RH2M
RH2M_lag2
T2M_roll4_mean
WS2M
```

### Scenario 2A

Jumlah fitur: `20`

```text
drought_carryover_lag1
PREC_roll12_std
PREC_roll4_mean
RH2M_lag1
T2M_roll12_mean
RH2M_lag4
ALLSKY_SFC_SW_DWN
PREC_roll4_std
RH2M_lag8
PREC_roll12_mean
PS
RH2M
RH2M_lag2
T2M_roll4_mean
WS2M
heat_dry_stress
PREC_lag2
PREC_lag1
PREC_lag8
PRECTOTCORR
```

### Scenario 2B

Jumlah fitur: `25`

```text
drought_carryover_lag1
PREC_roll12_std
PREC_roll4_mean
RH2M_lag1
T2M_roll12_mean
RH2M_lag4
ALLSKY_SFC_SW_DWN
PREC_roll4_std
RH2M_lag8
PREC_roll12_mean
PS
RH2M
RH2M_lag2
T2M_roll4_mean
WS2M
heat_dry_stress
PREC_lag2
PREC_lag1
PREC_lag8
PRECTOTCORR
PREC_lag4
D0_lag2
D2_lag1
severe_carryover_lag1
D4_lag2
```

### Scenario 2C

Jumlah fitur: `26`

```text
drought_carryover_lag1
PREC_roll12_std
PREC_roll4_mean
RH2M_lag1
T2M_roll12_mean
RH2M_lag4
ALLSKY_SFC_SW_DWN
PREC_roll4_std
RH2M_lag8
PREC_roll12_mean
PS
RH2M
RH2M_lag2
T2M_roll4_mean
WS2M
heat_dry_stress
PREC_lag2
PREC_lag1
PREC_lag8
PRECTOTCORR
PREC_lag4
D0_lag2
D2_lag1
severe_carryover_lag1
D4_lag2
week_sin
```

### Scenario 3

Jumlah fitur: `15`

```text
D2_lag1
None_lag1
D1_lag1
D0_lag1
D3_lag1
D1_lag2
None_lag2
drought_carryover_lag1
week_sin
PREC_lag2
PREC_roll4_std
severe_carryover_lag1
RH2M_lag8
PREC_roll4_mean
D4_lag1
```

### Scenario 4

Jumlah fitur: `6`

```text
ALLSKY_SFC_SW_DWN
PRECTOTCORR
PS
RH2M
T2M
WS2M
```

### Scenario 5

Jumlah fitur: `20`

```text
ALLSKY_SFC_SW_DWN
PRECTOTCORR
PS
RH2M
T2M
WS2M
PREC_lag1
PREC_lag2
PREC_lag4
PREC_lag8
T2M_lag1
T2M_lag2
T2M_lag4
T2M_lag8
RH2M_lag1
RH2M_lag2
RH2M_lag4
RH2M_lag8
week_sin
week_cos
```

### Scenario 6

Jumlah fitur: `27`

```text
ALLSKY_SFC_SW_DWN
PRECTOTCORR
PS
RH2M
T2M
WS2M
PREC_lag1
PREC_lag2
PREC_lag4
PREC_lag8
T2M_lag1
T2M_lag2
T2M_lag4
T2M_lag8
RH2M_lag1
RH2M_lag2
RH2M_lag4
RH2M_lag8
PREC_roll4_mean
PREC_roll4_std
PREC_roll12_mean
PREC_roll12_std
T2M_roll4_mean
T2M_roll12_mean
week_sin
week_cos
heat_dry_stress
```

### Scenario 7

Jumlah fitur: `12`

```text
None_lag1
D0_lag1
D1_lag1
D2_lag1
D3_lag1
D4_lag1
None_lag2
D0_lag2
D1_lag2
D2_lag2
D3_lag2
D4_lag2
```

## Catatan Penting

- **Scenario 1, 4, 5, 6, dan 7** memakai daftar fitur yang sama pada Kansas dan Nebraska.
- **Scenario 2, 2A, 2B, 2C, dan 3** berbeda antar wilayah karena fiturnya adalah hasil proses seleksi fitur.
- Pada **Scenario 2 dan 3**, daftar fitur final diambil dari `results_summary.txt`, bukan hanya dari `feature_cols` awal di script, karena fitur akhirnya merupakan hasil seleksi.

## Referensi File

- Kansas scripts: [`kansas/`](./kansas/)
- Nebraska scripts: [`nebraska/`](./nebraska/)
- Kansas Scenario 2 summary: [`kansas/output_weekly_kansas_scenario2/results_summary.txt`](./kansas/output_weekly_kansas_scenario2/results_summary.txt)
- Nebraska Scenario 2 summary: [`nebraska/output_weekly_nebraska_scenario2/results_summary.txt`](./nebraska/output_weekly_nebraska_scenario2/results_summary.txt)
