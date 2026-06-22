# Scenario Comparison: Kansas dan Nebraska

Dokumen ini merangkum seluruh skenario pada folder `kansas/` dan `nebraska/`, sekaligus menjelaskan model, trial hyperparameter, arti nama konfigurasi seperti `ros_focal_no_cw_96x48`, dan hasil akhir tiap skenario.

## Gambaran Umum Pipeline Model

Semua skenario menggunakan kerangka model yang sama. Perbedaan utama antar skenario ada pada **subset fitur** yang dipakai.

### Arsitektur model

- Model inti: **BiLSTM classifier** untuk prediksi kelas drought mingguan
- Input sequence: `52` time steps
- Output class: `6` kelas, yaitu `None`, `D0`, `D1`, `D2`, `D3`, `D4`
- Optimizer: `Adam`
- Loss utama: **categorical focal loss**
- Monitoring training: `val_macro_f1`

### Bentuk arsitektur

Secara umum arsitektur modelnya:

1. `Input(shape=(52, n_features))`
2. `Bidirectional(LSTM(u1, return_sequences=True))`
3. `BatchNormalization`
4. `Dropout`
5. `Bidirectional(LSTM(u2))`
6. `BatchNormalization`
7. `Dropout`
8. `Dense(dense_units, activation='relu')`
9. `Dropout`
10. `Dense(6, activation='softmax')`

Parameter `u1`, `u2`, `dropout`, `dense_units`, dan `lr` berubah sesuai trial hyperparameter.

## Arti Nama Trial Hyperparameter

Nama trial di repo ini disusun secara padat. Format umumnya:

`{balancer}_focal_{alpha_mode}_{cw_flag}_{lstm_units}`

Contoh:

- `ros_focal_no_cw_96x48`
- `none_focal_inv_cw_64x32`
- `ros_focal_inv_no_cw_128x64`

### Arti tiap bagian

| Komponen nama | Arti |
| --- | --- |
| `ros` | `RandomOverSampler` dipakai untuk menyeimbangkan data train |
| `none` | Tidak memakai resampling |
| `focal` | Loss function yang dipakai adalah focal loss |
| `no` | `focal_alpha_mode = none`, artinya focal loss tanpa alpha vector tambahan |
| `inv` | `focal_alpha_mode = inverse_train_seq`, alpha dihitung dari inverse distribusi kelas train sequence |
| `cw` | Menggunakan `class_weight` saat training |
| `no_cw` | Tidak menggunakan `class_weight` |
| `96x48` | `lstm_units=(96, 48)` |
| `64x32` | `lstm_units=(64, 32)` |
| `128x64` | `lstm_units=(128, 64)` |

## Trial Hyperparameter yang Diuji

Secara umum, skenario-skenario ini membandingkan tiga trial utama.

### Trial A: `ros_focal_no_cw_96x48`

| Hyperparameter | Nilai |
| --- | --- |
| Balancer | `ROS` |
| Class weight | `False` |
| Focal gamma | `1.5` |
| Focal alpha mode | `none` |
| LSTM units | `(96, 48)` |
| Dropout | `0.30` |
| Dense units | `64` |
| Learning rate | `8e-4` |
| Patience | `14` |

Interpretasi:

- Data train diseimbangkan dulu dengan `RandomOverSampler`
- Loss memakai focal loss, tapi tanpa alpha vector tambahan
- Tidak memakai `class_weight`
- Model menengah, tidak terlalu kecil dan tidak terlalu besar

### Trial B: `none_focal_inv_cw_64x32`

| Hyperparameter | Nilai |
| --- | --- |
| Balancer | `NONE` |
| Class weight | `True` |
| Focal gamma | `2.0` |
| Focal alpha mode | `inverse_train_seq` |
| LSTM units | `(64, 32)` |
| Dropout | `0.35` |
| Dense units | `64` |
| Learning rate | `1e-3` |
| Patience | `16` |

Interpretasi:

- Tidak ada oversampling
- Ketidakseimbangan kelas ditangani lewat kombinasi **focal alpha inverse** dan `class_weight`
- Arsitekturnya paling kecil dari tiga trial
- Dropout lebih besar, jadi regularisasi lebih kuat

### Trial C: `ros_focal_inv_no_cw_128x64`

| Hyperparameter | Nilai |
| --- | --- |
| Balancer | `ROS` |
| Class weight | `False` |
| Focal gamma | `2.0` |
| Focal alpha mode | `inverse_train_seq` |
| LSTM units | `(128, 64)` |
| Dropout | `0.28` |
| Dense units | `96` |
| Learning rate | `6e-4` |
| Patience | `14` |

Interpretasi:

- Data train diseimbangkan dengan `RandomOverSampler`
- Focal loss memakai alpha inverse dari distribusi kelas
- Tidak memakai `class_weight`
- Ini trial dengan kapasitas model terbesar

## Mekanisme Tuning dan Training

### 1. Balancing data train

Pipeline training bisa memakai:

- `ROS` = `RandomOverSampler`
- `NONE` = tanpa resampling

Oversampling dilakukan setelah sequence training dibentuk, lalu sequence di-flatten untuk resampling, kemudian dibentuk lagi ke tensor sequence.

### 2. Focal loss

Focal loss dipakai untuk memberi perhatian lebih besar ke contoh yang sulit atau minoritas.

- `gamma=1.5` atau `2.0`
- `alpha=None` jika mode `none`
- `alpha=inverse_train_seq` jika mode `inv`

Mode `inverse_train_seq` menghitung bobot alpha dari inverse frekuensi kelas pada train sequence.

### 3. Class weight

Beberapa trial memakai `class_weight`, beberapa tidak.

- Jika `cw=True`, bobot kelas dihitung dengan `compute_class_weight(class_weight='balanced')`
- Jika `cw=False`, model hanya mengandalkan resampling dan focal loss

### 4. Callback training

Training memakai callback:

- `MacroF1Callback` untuk menghitung `val_macro_f1`
- `EarlyStopping` berdasarkan `val_macro_f1`
- `ModelCheckpoint`
- `ReduceLROnPlateau`

Jadi model terbaik dipilih dari skor **validation macro-F1**, bukan validation accuracy.

### 5. Post-training class multiplier calibration

Setelah trial terbaik dipilih, repo ini masih melakukan tuning tambahan pada output probabilitas:

- Probabilitas kelas dikalikan dengan `class_multipliers`
- Multiplier dicari dengan random search pada validation set
- Tujuan utamanya menaikkan **macro-F1**

Karena itu di `results_summary.txt` selalu ada dua versi:

- `raw`: prediksi langsung dari model
- final: prediksi setelah class multiplier calibration

## Ringkasan Jenis Skenario

| Skenario | Deskripsi | Pola fitur utama |
| --- | --- | --- |
| Scenario 1 | Baseline | Weather dasar + lag weather + rolling stats + seasonality + drought history lag +  |
| Scenario 2 | Correlation-aware FS Top-15 | Seleksi fitur statistik dari 41 kandidat |
| Scenario 2A | Correlation-aware FS Top-20 | Sama seperti Scenario 2, target 20 fitur |
| Scenario 2B | Correlation-aware FS Top-25 | Sama seperti Scenario 2, target 25 fitur |
| Scenario 2C | Correlation-aware FS Top-30 | Sama seperti Scenario 2, target 30 fitur |
| Scenario 3 | Macro-F1-driven FS | Seleksi fitur berbasis kontribusi ke validation macro-F1 |
| Scenario 4 | Weather Only | Hanya 6 fitur weather dasar |
| Scenario 5 | Weather + Lag Only | Weather dasar + lag weather + seasonality |
| Scenario 6 | No Drought History | Semua weather engineering tanpa drought history |
| Scenario 7 | Drought History Only | Hanya drought history lag-1 dan lag-2 |

## Definisi Fitur per Skenario

### Scenario 1

- Jumlah fitur akhir: `39`
- Isi fitur:
  - weather dasar
  - lag `PREC`, `T2M`, `RH2M`
  - rolling precipitation dan temperature
  - `week_sin`, `week_cos`
  - drought history `lag1` dan `lag2`
  - 

### Scenario 2, 2A, 2B, 2C

- Pool kandidat awal: `39`
- Tambahan dibanding baseline 39 fitur:
  - `drought_carryover_lag1`
  - `severe_carryover_lag1`
- Metode seleksi:
  - mutual information ranking
  - correlation pruning dengan threshold `0.9`

### Scenario 3

- Pool kandidat awal: `39`
- Seleksi fitur berdasarkan permutation importance terhadap validation macro-F1
- Subset terbaik ditentukan dengan evaluasi beberapa ukuran subset

### Scenario 4

- Jumlah fitur: `6`
- Hanya weather dasar

### Scenario 5

- Jumlah fitur: `20`
- Weather dasar + lag cuaca + seasonality

### Scenario 6

- Jumlah fitur: `27`
- Weather dasar + lag cuaca + rolling stats + seasonality + 
- Tidak ada drought history

### Scenario 7

- Jumlah fitur: `12`
- Hanya `None/D0/D1/D2/D3/D4` untuk `lag1` dan `lag2`

## Kansas

### Ringkasan Hasil

| Skenario | Fitur akhir | Best trial | Arti singkat trial terbaik | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Scenario 1 | 39 | `ros_focal_inv_no_cw_128x64` | ROS + focal alpha inverse + tanpa class weight + BiLSTM besar | 0.8487 | 0.8098 | 0.8041 | 0.8100 |
| Scenario 2 | 15 | `none_focal_inv_cw_64x32` | tanpa ROS + focal alpha inverse + class weight + BiLSTM kecil | 0.6280 | 0.5184 | 0.4460 | 0.5294 |
| Scenario 2A | 20 | `none_focal_inv_cw_64x32` | tanpa ROS + focal alpha inverse + class weight + BiLSTM kecil | 0.6663 | 0.5313 | 0.4376 | 0.5536 |
| Scenario 2B | 25 | `ros_focal_inv_no_cw_128x64` | ROS + focal alpha inverse + tanpa class weight + BiLSTM besar | 0.8429 | 0.7940 | 0.8014 | 0.7932 |
| Scenario 2C | 25 | `ros_focal_inv_no_cw_128x64` | ROS + focal alpha inverse + tanpa class weight + BiLSTM besar | 0.8429 | 0.7940 | 0.8014 | 0.7932 |
| Scenario 3 | 10 | `ros_focal_no_cw_96x48` | ROS + focal tanpa alpha + tanpa class weight + BiLSTM medium | 0.8320 | 0.7536 | 0.6710 | 0.7491 |
| Scenario 4 | 6 | `ros_focal_no_cw_96x48` | ROS + focal tanpa alpha + tanpa class weight + BiLSTM medium | 0.1746 | 0.1438 | 0.1185 | 0.1201 |
| Scenario 5 | 20 | `none_focal_inv_cw_64x32` | tanpa ROS + focal alpha inverse + class weight + BiLSTM kecil | 0.2068 | 0.1715 | 0.1152 | 0.1532 |
| Scenario 6 | 27 | `none_focal_inv_cw_64x32` | tanpa ROS + focal alpha inverse + class weight + BiLSTM kecil | 0.2510 | 0.2134 | 0.1991 | 0.2232 |
| Scenario 7 | 12 | `ros_focal_inv_no_cw_128x64` | ROS + focal alpha inverse + tanpa class weight + BiLSTM besar | 0.8446 | 0.8246 | 0.8289 | 0.8240 |

### Pola pemenang trial di Kansas

- Trial terbaik paling sering adalah `ros_focal_inv_no_cw_128x64`
- Ini menunjukkan Kansas cenderung cocok dengan:
  - oversampling
  - focal alpha inverse
  - model berkapasitas besar
- Scenario 2 dan 2A justru dimenangkan oleh `none_focal_inv_cw_64x32`, tetapi performa akhirnya rendah karena reduksi fitur terlalu agresif

### Fitur seleksi penting di Kansas

| Skenario | Hasil seleksi fitur |
| --- | --- |
| Scenario 2 | Top-15, dominan `drought_carryover_lag1`, `D0_lag1`, `RH2M_lag8`, `T2M`, `RH2M`, `PREC_roll12_std` |
| Scenario 2A | Top-20, menambah `PREC_lag1`, `PREC_lag4`, `PREC_roll4_std`, `WS2M`, `PS` |
| Scenario 2B | Top-25, menambah `PREC_lag8`, `D2_lag1`, `severe_carryover_lag1`, `D4_lag1`, `week_sin` |
| Scenario 2C | Identik dengan 2B, final tetap 25 fitur |
| Scenario 3 | Top-10: `D3_lag1`, `D2_lag1`, `None_lag1`, `D1_lag1`, `D0_lag1`, `severe_carryover_lag1`, `D2_lag2`, `PREC_lag1`, `drought_carryover_lag1`, `PREC_roll12_mean` |

### Kesimpulan Kansas

- Best overall: **Scenario 7**
- Best compact selected-features scenario: **Scenario 2B/2C**
- Weather-only dan weather-without-drought-history sangat lemah
- Drought history adalah sinyal paling dominan di Kansas

## Nebraska

### Ringkasan Hasil

| Skenario | Fitur akhir | Best trial | Arti singkat trial terbaik | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Scenario 1 | 39 | `ros_focal_no_cw_96x48` | ROS + focal tanpa alpha + tanpa class weight + BiLSTM medium | 0.8514 | 0.7969 | 0.7718 | 0.7980 |
| Scenario 2 | 15 | `ros_focal_no_cw_96x48` | ROS + focal tanpa alpha + tanpa class weight + BiLSTM medium | 0.6092 | 0.6007 | 0.5262 | 0.6124 |
| Scenario 2A | 20 | `none_focal_inv_cw_64x32` | tanpa ROS + focal alpha inverse + class weight + BiLSTM kecil | 0.6040 | 0.5878 | 0.4942 | 0.5830 |
| Scenario 2B | 25 | `ros_focal_no_cw_96x48` | ROS + focal tanpa alpha + tanpa class weight + BiLSTM medium | 0.8510 | 0.7689 | 0.7517 | 0.7752 |
| Scenario 2C | 26 | `ros_focal_no_cw_96x48` | ROS + focal tanpa alpha + tanpa class weight + BiLSTM medium | 0.8558 | 0.7782 | 0.7603 | 0.7825 |
| Scenario 3 | 15 | `ros_focal_no_cw_96x48` | ROS + focal tanpa alpha + tanpa class weight + BiLSTM medium | 0.7835 | 0.7627 | 0.7476 | 0.7636 |
| Scenario 4 | 6 | `none_focal_inv_cw_64x32` | tanpa ROS + focal alpha inverse + class weight + BiLSTM kecil | 0.1520 | 0.1990 | 0.1516 | 0.1455 |
| Scenario 5 | 20 | `ros_focal_no_cw_96x48` | ROS + focal tanpa alpha + tanpa class weight + BiLSTM medium | 0.3156 | 0.3383 | 0.2709 | 0.3013 |
| Scenario 6 | 27 | `ros_focal_no_cw_96x48` | ROS + focal tanpa alpha + tanpa class weight + BiLSTM medium | 0.3240 | 0.3344 | 0.2725 | 0.2966 |
| Scenario 7 | 12 | `ros_focal_no_cw_96x48` | ROS + focal tanpa alpha + tanpa class weight + BiLSTM medium | 0.8623 | 0.8254 | 0.8161 | 0.8267 |

### Pola pemenang trial di Nebraska

- Trial terbaik paling sering adalah `ros_focal_no_cw_96x48`
- Ini menunjukkan Nebraska cenderung cocok dengan:
  - oversampling
  - focal loss standar tanpa alpha inverse tambahan
  - model ukuran medium
- Hanya beberapa skenario yang lebih cocok dengan `none_focal_inv_cw_64x32`

### Fitur seleksi penting di Nebraska

| Skenario | Hasil seleksi fitur |
| --- | --- |
| Scenario 2 | Top-15, dominan weather dan rolling stats: `drought_carryover_lag1`, `PREC_roll12_std`, `PREC_roll4_mean`, `RH2M_lag1`, `T2M_roll12_mean`, `RH2M_lag4` |
| Scenario 2A | Top-20, menambah , `PREC_lag2`, `PREC_lag1`, `PREC_lag8`, `PRECTOTCORR` |
| Scenario 2B | Top-25, menambah `PREC_lag4`, `D0_lag2`, `D2_lag1`, `severe_carryover_lag1`, `D4_lag2` |
| Scenario 2C | Top-26, menambah `week_sin` dibanding 2B |
| Scenario 3 | Top-15: `D2_lag1`, `None_lag1`, `D1_lag1`, `D0_lag1`, `D3_lag1`, `D1_lag2`, `None_lag2`, `drought_carryover_lag1`, `week_sin`, `PREC_lag2`, `PREC_roll4_std`, `severe_carryover_lag1`, `RH2M_lag8`, `PREC_roll4_mean`, `D4_lag1` |

### Kesimpulan Nebraska

- Best overall: **Scenario 7**
- Best selected-features scenario: **Scenario 2C**
- Baseline tetap kuat
- Weather-only naik sedikit dibanding Kansas, tetapi tetap jauh di bawah skenario yang memakai drought history

## Kansas vs Nebraska

### Perbandingan Test Macro F1

| Skenario | Kansas | Nebraska | Lebih baik |
| --- | ---: | ---: | --- |
| Scenario 1 | 0.8041 | 0.7718 | Kansas |
| Scenario 2 | 0.4460 | 0.5262 | Nebraska |
| Scenario 2A | 0.4376 | 0.4942 | Nebraska |
| Scenario 2B | 0.8014 | 0.7517 | Kansas |
| Scenario 2C | 0.8014 | 0.7603 | Kansas |
| Scenario 3 | 0.6710 | 0.7476 | Nebraska |
| Scenario 4 | 0.1185 | 0.1516 | Nebraska |
| Scenario 5 | 0.1152 | 0.2709 | Nebraska |
| Scenario 6 | 0.1991 | 0.2725 | Nebraska |
| Scenario 7 | 0.8289 | 0.8161 | Kansas |

### Perbandingan pola tuning

| Aspek | Kansas | Nebraska |
| --- | --- | --- |
| Trial pemenang dominan | `ros_focal_inv_no_cw_128x64` | `ros_focal_no_cw_96x48` |
| Preferensi kapasitas model | Lebih besar | Medium |
| Alpha inverse | Sering membantu | Tidak selalu diperlukan |
| Drought history | Sangat dominan | Sangat dominan |

## Kesimpulan Utama

1. Pada dua state, skenario terbaik adalah **Scenario 7: Drought History Only**.
2. **Scenario 1 baseline** tetap sangat kuat dan menjadi pembanding yang masuk akal.
3. Reduksi fitur yang terlalu agresif pada **Scenario 2** dan **2A** cenderung menurunkan performa.
4. Untuk kompromi antara jumlah fitur dan performa:
   - Kansas: **Scenario 2B** atau **2C**
   - Nebraska: **Scenario 2C**
5. Secara konsisten, informasi **riwayat drought** lebih penting daripada **weather saja**.

## Referensi File

- Kansas baseline model: [`kansas/BiLSTM_Scenario1_Kansas_Baseline.py`](./kansas/BiLSTM_Scenario1_Kansas_Baseline.py)
- Nebraska baseline model: [`nebraska/BiLSTM_Scenario1_Nebraska_Baseline.py`](./nebraska/BiLSTM_Scenario1_Nebraska_Baseline.py)
- Kansas scenario 7: [`kansas/BiLSTM_Scenario7_DroughtHistoryOnly.py`](./kansas/BiLSTM_Scenario7_DroughtHistoryOnly.py)
- Nebraska scenario 7: [`nebraska/BiLSTM_Scenario7_DroughtHistoryOnly.py`](./nebraska/BiLSTM_Scenario7_DroughtHistoryOnly.py)
- Kansas summary: [`kansas/output_weekly_kansas_20counties/results_summary.txt`](./kansas/output_weekly_kansas_20counties/results_summary.txt)
- Nebraska summary: [`nebraska/output_weekly_nebraska_20counties/results_summary.txt`](./nebraska/output_weekly_nebraska_20counties/results_summary.txt)
