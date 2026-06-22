# Perbandingan Skenario Kansas dan Nebraska

Dokumen ini merangkum seluruh skenario eksperimen pada folder `kansas/` dan `nebraska/`, mencakup perbedaan fitur, jumlah fitur, serta metrik utama dari `results_summary.txt`.

## Ringkasan Jenis Skenario

| Skenario | Deskripsi | Pola fitur utama |
| --- | --- | --- |
| Scenario 1 | Baseline | Weather dasar + lag weather + rolling stats + seasonal features + drought history lag +  |
| Scenario 2 | Correlation-aware FS Top-15 | Seleksi fitur berbasis mutual information dan pruning korelasi dari 41 kandidat |
| Scenario 2A | Correlation-aware FS Top-20 | Sama seperti Scenario 2, target 20 fitur |
| Scenario 2B | Correlation-aware FS Top-25 | Sama seperti Scenario 2, target 25 fitur |
| Scenario 2C | Correlation-aware FS Top-30 | Sama seperti Scenario 2, target 30 fitur, tetapi hasil akhir bisa kurang dari 30 |
| Scenario 3 | Macro-F1-driven FS | Seleksi fitur berdasarkan dampak ke validation macro-F1 |
| Scenario 4 | Weather Only | Hanya 6 weather dasar |
| Scenario 5 | Weather + Lag Only | Weather dasar + lag weather + seasonality |
| Scenario 6 | No Drought History | Semua weather engineering tanpa drought history |
| Scenario 7 | Drought History Only | Hanya riwayat kelas drought lag-1 dan lag-2 |

## Definisi Fitur per Skenario

### Scenario 1 Baseline

- Jumlah fitur: `39`
- Komponen:
  - Weather dasar: `ALLSKY_SFC_SW_DWN`, `PRECTOTCORR`, `PS`, `RH2M`, `T2M`, `WS2M`
  - Lag cuaca: `PREC/T2M/RH2M` untuk lag `1, 2, 4, 8`
  - Rolling stats: `PREC_roll4_mean`, `PREC_roll4_std`, `PREC_roll12_mean`, `PREC_roll12_std`, `T2M_roll4_mean`, `T2M_roll12_mean`
  - Musiman: `week_sin`, `week_cos`
  - Drought history: `None/D0/D1/D2/D3/D4` untuk `lag1` dan `lag2`
  - Interaksi: 

### Scenario 2, 2A, 2B, 2C

- Pool awal fitur: `41`
- Tambahan dibanding baseline:
  - `drought_carryover_lag1`
  - `severe_carryover_lag1`
- Metode:
  - ranking dengan mutual information
  - pruning antar fitur dengan `correlation threshold = 0.9`

### Scenario 3

- Pool awal fitur: `41`
- Seleksi berdasarkan permutation importance terhadap validation macro-F1
- Subset fitur terbaik berbeda antar state

### Scenario 4

- Jumlah fitur: `6`
- Fitur: weather dasar saja

### Scenario 5

- Jumlah fitur: `20`
- Fitur: weather dasar + lag `PREC/T2M/RH2M` + `week_sin`, `week_cos`

### Scenario 6

- Jumlah fitur: `27`
- Fitur: weather dasar + lag cuaca + rolling stats + seasonality + 
- Tidak memakai drought history

### Scenario 7

- Jumlah fitur: `12`
- Fitur: `None/D0/D1/D2/D3/D4` untuk `lag1` dan `lag2`

## Kansas

### Tabel Ringkas

| Skenario | Jumlah fitur akhir | Best trial | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 | Catatan |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Scenario 1 | 39 | `ros_focal_inv_no_cw_128x64` | 0.8487 | 0.8098 | 0.8041 | 0.8100 | Baseline 20 counties |
| Scenario 2 | 15 | `none_focal_inv_cw_64x32` | 0.6280 | 0.5184 | 0.4460 | 0.5294 | Seleksi fitur terlalu agresif |
| Scenario 2A | 20 | `none_focal_inv_cw_64x32` | 0.6663 | 0.5313 | 0.4376 | 0.5536 | Masih lemah |
| Scenario 2B | 25 | `ros_focal_inv_no_cw_128x64` | 0.8429 | 0.7940 | 0.8014 | 0.7932 | Hampir setara baseline |
| Scenario 2C | 25 | `ros_focal_inv_no_cw_128x64` | 0.8429 | 0.7940 | 0.8014 | 0.7932 | Identik dengan 2B |
| Scenario 3 | 10 | `ros_focal_no_cw_96x48` | 0.8320 | 0.7536 | 0.6710 | 0.7491 | Top-10 fitur terbaik |
| Scenario 4 | 6 | `ros_focal_no_cw_96x48` | 0.1746 | 0.1438 | 0.1185 | 0.1201 | Weather only sangat lemah |
| Scenario 5 | 20 | `none_focal_inv_cw_64x32` | 0.2068 | 0.1715 | 0.1152 | 0.1532 | Weather + lag saja belum cukup |
| Scenario 6 | 27 | `none_focal_inv_cw_64x32` | 0.2510 | 0.2134 | 0.1991 | 0.2232 | Weather engineering tanpa drought history masih lemah |
| Scenario 7 | 12 | `ros_focal_inv_no_cw_128x64` | 0.8446 | 0.8246 | 0.8289 | 0.8240 | Performa terbaik |

### Fitur hasil seleksi penting

| Skenario | Hasil seleksi fitur |
| --- | --- |
| Scenario 2 | Top-15, dominan `drought_carryover_lag1`, `D0_lag1`, `RH2M_lag8`, `T2M`, `RH2M`, `PREC_roll12_std` |
| Scenario 2A | Top-20, menambah `PREC_lag1`, `PREC_lag4`, `PREC_roll4_std`, `WS2M`, `PS` |
| Scenario 2B | Top-25, menambah `PREC_lag8`, `D2_lag1`, `severe_carryover_lag1`, `D4_lag1`, `week_sin` |
| Scenario 2C | Sama persis dengan 2B, final tetap 25 fitur |
| Scenario 3 | Top-10: `D3_lag1`, `D2_lag1`, `None_lag1`, `D1_lag1`, `D0_lag1`, `severe_carryover_lag1`, `D2_lag2`, `PREC_lag1`, `drought_carryover_lag1`, `PREC_roll12_mean` |

### Temuan utama Kansas

- Skenario terbaik adalah **Scenario 7** dengan `Macro F1 = 0.8289`.
- **Scenario 1 baseline** tetap sangat kuat dengan `Macro F1 = 0.8041`.
- **Scenario 2B dan 2C** menunjukkan bahwa seleksi fitur bisa mempertahankan performa mendekati baseline.
- **Scenario 4, 5, 6** menunjukkan weather tanpa drought history sangat tidak cukup.
- Pada Kansas, sinyal paling dominan berasal dari **riwayat drought**.

## Nebraska

### Tabel Ringkas

| Skenario | Jumlah fitur akhir | Best trial | Val Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 | Catatan |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Scenario 1 | 39 | `ros_focal_no_cw_96x48` | 0.8514 | 0.7969 | 0.7718 | 0.7980 | Baseline 20 counties |
| Scenario 2 | 15 | `ros_focal_no_cw_96x48` | 0.6092 | 0.6007 | 0.5262 | 0.6124 | Top-15 terlalu sempit |
| Scenario 2A | 20 | `none_focal_inv_cw_64x32` | 0.6040 | 0.5878 | 0.4942 | 0.5830 | Belum stabil |
| Scenario 2B | 25 | `ros_focal_no_cw_96x48` | 0.8510 | 0.7689 | 0.7517 | 0.7752 | Mendekati baseline |
| Scenario 2C | 26 | `ros_focal_no_cw_96x48` | 0.8558 | 0.7782 | 0.7603 | 0.7825 | Varian FS terbaik Nebraska |
| Scenario 3 | 15 | `ros_focal_no_cw_96x48` | 0.7835 | 0.7627 | 0.7476 | 0.7636 | Top-15 fitur terbaik |
| Scenario 4 | 6 | `none_focal_inv_cw_64x32` | 0.1520 | 0.1990 | 0.1516 | 0.1455 | Weather only sangat lemah |
| Scenario 5 | 20 | `ros_focal_no_cw_96x48` | 0.3156 | 0.3383 | 0.2709 | 0.3013 | Ada kenaikan dari weather only |
| Scenario 6 | 27 | `ros_focal_no_cw_96x48` | 0.3240 | 0.3344 | 0.2725 | 0.2966 | Masih jauh dari baseline |
| Scenario 7 | 12 | `ros_focal_no_cw_96x48` | 0.8623 | 0.8254 | 0.8161 | 0.8267 | Performa terbaik |

### Fitur hasil seleksi penting

| Skenario | Hasil seleksi fitur |
| --- | --- |
| Scenario 2 | Top-15, dominan weather dan rolling stats: `drought_carryover_lag1`, `PREC_roll12_std`, `PREC_roll4_mean`, `RH2M_lag1`, `T2M_roll12_mean`, `RH2M_lag4` |
| Scenario 2A | Top-20, menambah , `PREC_lag2`, `PREC_lag1`, `PREC_lag8`, `PRECTOTCORR` |
| Scenario 2B | Top-25, menambah `PREC_lag4`, `D0_lag2`, `D2_lag1`, `severe_carryover_lag1`, `D4_lag2` |
| Scenario 2C | Top-26, menambah `week_sin` dibanding 2B |
| Scenario 3 | Top-15: `D2_lag1`, `None_lag1`, `D1_lag1`, `D0_lag1`, `D3_lag1`, `D1_lag2`, `None_lag2`, `drought_carryover_lag1`, `week_sin`, `PREC_lag2`, `PREC_roll4_std`, `severe_carryover_lag1`, `RH2M_lag8`, `PREC_roll4_mean`, `D4_lag1` |

### Temuan utama Nebraska

- Skenario terbaik adalah **Scenario 7** dengan `Macro F1 = 0.8161`.
- **Scenario 1 baseline** juga kuat dengan `Macro F1 = 0.7718`.
- **Scenario 2C** adalah varian feature selection terbaik di Nebraska dan cukup dekat dengan baseline.
- **Scenario 4, 5, 6** kembali menunjukkan bahwa weather saja belum cukup untuk menandingi fitur riwayat drought.
- Nebraska sedikit lebih baik daripada Kansas pada skenario weather-only, tetapi gap terhadap skenario berbasis drought history tetap besar.

## Perbandingan Langsung Kansas vs Nebraska

### Berdasarkan Test Macro F1

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

### Pola Umum

- Pada **dua state**, skenario terbaik adalah **Scenario 7: Drought History Only**.
- **Scenario 1 baseline** konsisten kuat pada dua state.
- **Feature selection yang terlalu sedikit fitur** seperti Scenario 2 dan 2A cenderung menurunkan performa.
- **Scenario 2B/2C** adalah kompromi terbaik antara reduksi fitur dan performa.
- **Scenario 4 sampai 6** menunjukkan bahwa informasi cuaca tanpa drought history tidak cukup kuat untuk prediksi kelas drought mingguan.

## Kesimpulan Singkat

- Jika tujuan utama adalah **akurasi prediksi**, maka **Scenario 7** adalah pilihan terbaik untuk Kansas dan Nebraska.
- Jika tujuan utama adalah **menjaga performa sambil mereduksi fitur**, maka:
  - Kansas: **Scenario 2B** atau **2C**
  - Nebraska: **Scenario 2C**
- Jika butuh **baseline komprehensif**, gunakan **Scenario 1**.
- Secara keseluruhan, repo ini menunjukkan bahwa **riwayat drought sebelumnya lebih informatif daripada weather saja** untuk prediksi drought mingguan.

## Referensi File

- Kansas baseline: [`kansas/BiLSTM_Scenario1_Kansas_Baseline.py`](./kansas/BiLSTM_Scenario1_Kansas_Baseline.py)
- Nebraska baseline: [`nebraska/BiLSTM_Scenario1_Nebraska_Baseline.py`](./nebraska/BiLSTM_Scenario1_Nebraska_Baseline.py)
- Kansas results: [`kansas/output_weekly_kansas_20counties/results_summary.txt`](./kansas/output_weekly_kansas_20counties/results_summary.txt)
- Nebraska results: [`nebraska/output_weekly_nebraska_20counties/results_summary.txt`](./nebraska/output_weekly_nebraska_20counties/results_summary.txt)
