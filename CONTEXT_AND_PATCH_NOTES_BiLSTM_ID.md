Analisis lintas-skenario — ringkasan & catatan patch


**1) Catatan penting tentang kesetaraan perbandingan**
- Clean runs menggunakan ukuran test dan dukungan kelas yang berbeda (14694 total) dibandingkan dengan 20-county/skenario runs Anda (4180 total).
- Jadi perbandingan clean vs 20-county harus diartikan sebagai arahan saja, bukan perbandingan yang tepat secara langsung.
- Dalam keluarga 20-county (Baseline + Scenario 2/3/4/5/6/7), perbandingan adalah valid.


**2) Peringkat kinerja pada benchmark 20-county bersama (Test Macro F1)**
| Peringkat | Experiment | Val Macro F1 | Test Macro F1 | Test accuracy | Catatan |
|---|---:|---:|---:|---:|---|
| 1 | Scenario 7 (Drought history only) | 0.8436 | 0.8085 | 0.8167 | Best test macro among scenarios |
| 2 | Baseline 20 counties | 0.8514 | 0.8039 | 0.8067 | Strongest validation, near-best test |
| 3 | Scenario 2B | 0.8429 | 0.8014 | 0.7940 | Best reduced-feature model |
| 3 | Scenario 2C | 0.8429 | 0.8014 | 0.7940 | Exactly same as 2B |
| 5 | Scenario 3 (Top-10 by permutation) | 0.8320 | 0.6710 | 0.7536 | High accuracy, much lower macro |
| 6 | Scenario 2A | 0.6663 | 0.4376 | 0.5313 | Moderate collapse |
| 7 | Scenario 2 | 0.5638 | 0.4165 | 0.5433 | More collapse |
| 8 | Scenario 6 (No drought history) | 0.2510 | 0.1991 | 0.2134 | Severe degradation |
| 9 | Scenario 4 (Weather only) | 0.1746 | 0.1185 | 0.1438 | Very poor |
| 10 | Scenario 5 (Weather + lag only) | 0.2068 | 0.1152 | 0.1715 | Very poor |


**3) Kesimpulan inti pemodelan**
- Drought-memory features adalah sinyal paling dominan di setup Anda.
- Bukti:
  - Scenario 7 (hanya drought history) adalah yang terbaik pada Test Macro F1.
  - Menghapus drought history (Scenario 6) menurunkan Macro F1 dari ~0.80 menjadi ~0.20.
  - Weather-only dan weather+lag-only hampir runtuh.
- Ini menunjukkan model terutama mempelajari dinamika persistensi drought, sedangkan weather berperan sebagai penguat sekunder.


**4) Scenario feature-selection (2/2A/2B/2C/3)**
- Scenario 2B adalah tradeoff feature-selection yang paling kuat:
  - Mempertahankan banyak kualitas Baseline dengan jumlah fitur lebih sedikit.
  - Keseimbangan antar-class lebih baik dibandingkan reduced sets lain.
- Scenario 2C tampak duplikat dari 2B:
  - Jumlah fitur terpilih sama (25), daftar fitur sama, dan metrik identik di results_summary.txt.
  - Kemungkinan target top-30 dibatasi oleh correlation pruning sehingga berakhir identik dengan top-25.
- Scenario 3 (Top-10) terlalu terkompresi:
  - Validasi tampak kuat, tetapi Test Macro drop signifikan.
  - Menunjukkan underrepresentation struktur kelas minoritas setelah pruning agresif.


**5) Celah generalisasi (Val best vs Test Macro F1)**
Semakin kecil gap biasanya berarti transfer dari validation ke test lebih baik:
- Scenario 7: sekitar -0.035 (stabilitas terbaik)
- Baseline: sekitar -0.048
- Scenario 2B/2C: sekitar -0.042
- Scenario 6: sekitar -0.052 (tetap skor absolut buruk)
- Scenario 4: sekitar -0.056
- Scenario 5: sekitar -0.092
- Scenario 2: sekitar -0.147
- Scenario 3: sekitar -0.161
- Scenario 2A: sekitar -0.229 (terburuk)

Interpretasi:
- Model dengan nilai absolut terbaik juga umumnya generalize lebih baik (baseline, scenario 7, scenario 2B).
- Reduced-feature set berkinerja sedang/rendah menunjukkan mismatch validation-to-test yang jauh lebih besar.


**6) Tren perilaku per-class**
- Performa class paling seimbang:
  - Baseline dan Scenario 7.
  - Keduanya mempertahankan D4 kuat dan D3 yang wajar.
- Scenario 2B/2C:
  - Ternyata tangguh di berbagai kelas; D4 sangat kuat.
- Scenario 3:
  - D4 runtuh parah (F1 sekitar 0.315) meskipun akurasi keseluruhan lumayan.
- Scenario 2 dan 2A:
  - D2/D3/D4 tidak stabil; kelas minoritas menjadi lemah.
- Scenario 4 dan 5:
  - Perilaku kelas degenerate.
  - Scenario 4 memprediksi D4 berat (recall D4 tinggi, D3 hampir nol).
  - Scenario 5 memiliki D0 dan D4 F1 = 0, dengan recall D3 sangat tinggi tetapi precision sangat rendah.
- Scenario 6:
  - Semua kelas lemah; konfirmasi bahwa menghapus drought-history sangat merusak.


**7) Tentang clean baselines**
- results_summary.txt: Macro F1 0.7327
- results_summary.txt: Macro F1 0.7303
- Ini konsisten dan stabil, tetapi karena dukungan sampel berbeda jauh dari 20-county runs, perlakukan sebagai track benchmark terpisah.


**8) Kesimpulan akhir**
- Jika tujuan Anda adalah Macro F1 kelas drought terbaik di benchmark 20-county ini, urutan saat ini:
  1. Scenario 7
  2. Baseline 20 counties
  3. Scenario 2B/2C
- Untuk reduksi fitur dengan kehilangan minimal, Scenario 2B adalah pemenang praktis.
- Untuk interpretabilitas fisika terkait causalitas weather-only, hasil saat ini menunjukkan sinyal weather-only tidak cukup tanpa variabel drought-memory.


**1. Baseline (20 counties)**
Script: BiLSTM_Weekly_Kansas_Clean.py  
Output: results_summary.txt


Yang digunakan:
- Full engineered feature set (41 features).
- Tidak ada langkah feature-selection.
- Konfigurasi trial asli (dua ROS tanpa class weight, satu NONE dengan class weight).


Arti:
- Ini adalah model referensi full-information untuk eksperimen 20-county.


---


**2. Scenario 2 (Correlation-aware feature selection, Top-15 target)**
Script: BiLSTM_Scenario2_Correlation.py  
Output: results_summary.txt


Perubahan:
- Menambahkan blok feature selection antara split dan scaling.
- Metode: mutual information ranking + correlation pruning.
- Correlation threshold: 0.9.
- Target jumlah fitur: 15.


Fitur terpilih:
- drought_carryover_lag1
- D0_lag1
- RH2M_lag8
- T2M
- RH2M
- RH2M_lag4
- PREC_roll12_std
- PREC_roll4_mean
- RH2M_lag2
- T2M_roll12_mean
- ALLSKY_SFC_SW_DWN
- RH2M_lag1
- PREC_roll12_mean
- PRECTOTCORR
- PREC_lag2


---


**3. Scenario 2A (Correlation-aware feature selection, Top-20 target)**
Script: BiLSTM_Scenario2A_Correlation.py  
Output: results_summary.txt


Perubahan:
- Metode sama seperti Scenario 2.
- Correlation threshold: 0.9.
- Target jumlah fitur: 20.


Fitur terpilih:
- drought_carryover_lag1
- D0_lag1
- RH2M_lag8
- T2M
- RH2M
- RH2M_lag4
- PREC_roll12_std
- PREC_roll4_mean
- RH2M_lag2
- T2M_roll12_mean
- ALLSKY_SFC_SW_DWN
- RH2M_lag1
- PREC_roll12_mean
- PRECTOTCORR
- PREC_lag2
- PREC_lag1
- PREC_lag4
- PREC_roll4_std
- WS2M
- PS


---


**4. Scenario 2B (Correlation-aware feature selection, Top-25 target)**
Script: BiLSTM_Scenario2B_Correlation.py  
Output: results_summary.txt


Perubahan:
- Metode sama seperti Scenario 2/2A.
- Correlation threshold: 0.9.
- Target jumlah fitur: 25.


Fitur terpilih:
- drought_carryover_lag1
- D0_lag1
- RH2M_lag8
- T2M
- RH2M
- RH2M_lag4
- PREC_roll12_std
- PREC_roll4_mean
- RH2M_lag2
- T2M_roll12_mean
- ALLSKY_SFC_SW_DWN
- RH2M_lag1
- PREC_roll12_mean
- PRECTOTCORR
- PREC_lag2
- PREC_lag1
- PREC_lag4
- PREC_roll4_std
- WS2M
- PS
- PREC_lag8
- D2_lag1
- severe_carryover_lag1
- D4_lag1
- week_sin


---


**5. Scenario 2C (Correlation-aware feature selection, Top-30 target)**
Script: BiLSTM_Scenario2C_Correlation.py  
Output: results_summary.txt


Perubahan:
- Target jumlah fitur yang dimaksud: 30.
- Metode dan threshold sama seperti 2/2A/2B.


Hasil penting:
- Jumlah fitur akhir berhenti di 25 (bukan 30), daftar fitur sama seperti Scenario 2B.
- Hasil pada output sama dengan Scenario 2B.


---


**6. Scenario 3 (Macro-F1 driven feature selection)**
Script: BiLSTM_Scenario3_F1Selection.py  
Output: results_summary.txt


Perubahan:
- Strategi feature selection diganti total.
- Metode: permutation importance berdasarkan validation Macro F1.
- Ukuran subset diuji: 30, 25, 20, 15, 10.
- Subset terbaik: Top-10 fitur.


Top-10 fitur terpilih:
- D3_lag1
- D2_lag1
- None_lag1
- D1_lag1
- D0_lag1
- severe_carryover_lag1
- D2_lag2
- PREC_lag1
- drought_carryover_lag1
- PREC_roll12_mean


---


**7. Scenario 4 (Weather only)**
Script: BiLSTM_Scenario4_WeatherOnly.py  
Output: results_summary.txt


Perubahan:
- Set fitur diganti menjadi hanya 6 raw weather variables.
- Menambahkan SCENARIO_NAME + blok logging fitur.
- OUTPUT folder dipisah per-scenario.
- Summary menulis daftar fitur terpilih.
- use_class_weight diset True di setiap trial.


Fitur (6):
- ALLSKY_SFC_SW_DWN
- PRECTOTCORR
- PS
- RH2M
- T2M
- WS2M


---


**8. Scenario 5 (Weather + lag only)**
Script: BiLSTM_Scenario5_WeatherLagOnly.py  
Output: results_summary.txt


Perubahan:
- Set fitur diganti menjadi raw weather + weather lags + week_sin/week_cos.
- Tidak ada rolling weather stats.
- Tidak ada drought history features.
- Menambahkan logging, folder output terpisah, dan summary fitur.
- use_class_weight = True di setiap trial.


Fitur (20):
- Raw weather 6
- PREC_lag1, PREC_lag2, PREC_lag4, PREC_lag8
- T2M_lag1, T2M_lag2, T2M_lag4, T2M_lag8
- RH2M_lag1, RH2M_lag2, RH2M_lag4, RH2M_lag8
- week_sin, week_cos


---


**9. Scenario 6 (No drought history)**
Script: BiLSTM_Scenario6_NoDroughtHistory.py  
Output: results_summary.txt


Perubahan:
- Menggunakan weather raw + lag + rolling + seasonal + heat_dry_stress.
- Menghapus semua drought history lag/carryover features.
- Menambahkan logging dan folder output terpisah.
- Summary menulis fitur.
- use_class_weight = True di setiap trial.


Fitur (27):
- Raw weather 6
- Weather lags 12
- Rolling stats 6
- week_sin, week_cos
- heat_dry_stress


---


**10. Scenario 7 (Drought history only)**
Script: BiLSTM_Scenario7_DroughtHistoryOnly.py  
Output: results_summary.txt


Perubahan:
- Menghapus semua weather features.
- Menyimpan hanya sinyal persistensi drought.
- Menambahkan logging dan folder output terpisah.
- Summary menulis fitur.
- use_class_weight = True di setiap trial.


Fitur (14):
- None_lag1, D0_lag1, D1_lag1, D2_lag1, D3_lag1, D4_lag1
- None_lag2, D0_lag2, D1_lag2, D2_lag2, D3_lag2, D4_lag2
- drought_carryover_lag1
- severe_carryover_lag1


---


**11. Clean baseline track (konteks dataset/split berbeda)**
Perubahan:
- Ini adalah clean-track runs terpisah (profil sampel/dukungan berbeda dari 20-county runs).
- Berguna sebagai family baseline lain, tetapi tidak langsung sebanding dengan outputs 4180-support.


---


**A. Change Matrix (relatif terhadap Baseline 20-county)**


| Scenario | Feature selection method | Raw weather | Weather lags | Rolling weather stats | Seasonal (`week_sin/cos`) | Drought lags (`*_lag1/2`) | Carryover drought (`drought_carryover_lag1`, `severe_carryover_lag1`) | Heat-dry stress | Class-weight policy |
|---|---|---|---|---|---|---|---|---|---|
| Baseline 20counties | None (full feature set) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Mixed by trial (not always true) |
| Scenario 2 | MI + corr pruning (target 15) | Partial | Partial | Partial | No | Very limited | Yes | No | Mixed by trial (best used class weight) |
| Scenario 2A | MI + corr pruning (target 20) | Yes | Partial | Yes | No | Very limited | Yes | No | Mixed by trial (best used class weight) |
| Scenario 2B | MI + corr pruning (target 25) | Yes | Yes | Yes | Partial (`week_sin` only) | Some | Yes | No | Mixed by trial (best did not use class weight) |
| Scenario 2C | MI + corr pruning (target 30 requested) | Same as 2B | Same as 2B | Same as 2B | Same as 2B | Same as 2B | Same as 2B | Same as 2B | Same as 2B |
| Scenario 3 | Permutation macro-F1 subset search (Top-10 selected) | Very limited | Minimal | Minimal | No | Yes | Yes | No | Mixed by trial (best did not use class weight) |
| Scenario 4 | Manual feature ablation | Yes | No | No | No | No | No | No | Always true in all trials |
| Scenario 5 | Manual feature ablation | Yes | Yes | No | Yes | No | No | No | Always true in all trials |
| Scenario 6 | Manual feature ablation | Yes | Yes | Yes | Yes | No | No | Yes | Always true in all trials |
| Scenario 7 | Manual feature ablation | No | No | No | No | Yes | Yes | No | Always true in all trials |


---


**B. Exact feature sets untuk manual ablation scenarios**


**Scenario 4 (Weather only, 6 features)**  
- ALLSKY_SFC_SW_DWN  
- PRECTOTCORR  
- PS  
- RH2M  
- T2M  
- WS2M


**Scenario 5 (Weather + lag only, 20 features)**  
- Raw weather 6  
- PREC_lag1, PREC_lag2, PREC_lag4, PREC_lag8  
- T2M_lag1, T2M_lag2, T2M_lag4, T2M_lag8  
- RH2M_lag1, RH2M_lag2, RH2M_lag4, RH2M_lag8  
- week_sin, week_cos


**Scenario 6 (No drought history, 27 features)**  
- Raw weather 6  
- Weather lags 12  
- Rolling: PREC_roll4_mean, PREC_roll4_std, PREC_roll12_mean, PREC_roll12_std, T2M_roll4_mean, T2M_roll12_mean  
- week_sin, week_cos  
- heat_dry_stress


**Scenario 7 (Drought history only, 14 features)**  
- None_lag1, D0_lag1, D1_lag1, D2_lag1, D3_lag1, D4_lag1  
- None_lag2, D0_lag2, D1_lag2, D2_lag2, D3_lag2, D4_lag2  
- drought_carryover_lag1  
- severe_carryover_lag1



---


**C. Perubahan output/reporting pada manual scenarios (4–7)**


Untuk tiap file:
- BiLSTM_Scenario4_WeatherOnly.py
- BiLSTM_Scenario5_WeatherLagOnly.py
- BiLSTM_Scenario6_NoDroughtHistory.py
- BiLSTM_Scenario7_DroughtHistoryOnly.py


Anda mengubah:
- `SCENARIO_NAME` ditambahkan
- `OUTPUT_FOLDER` per-scenario digunakan  
- Blok logging fitur ditambahkan (print feature count + daftar lengkap)
- results_summary.txt sekarang menulis daftar fitur terpilih
- `use_class_weight = True` dipaksa pada setiap trial config


---


**D. Ringkasan singkat perbedaan utama**
- Perbedaan struktural terbesar adalah ada/tidaknya drought history.
- Scenario 7 hanya menyimpan memori persistensi drought.
- Scenario 6 menghapus semua drought memory tetapi mempertahankan dinamika weather.
- Scenario 4/5 fokus pada weather dengan konteks temporal berbeda.
- Scenario 2/2A/2B/2C/3 adalah varian algorithmic untuk feature-reduction, bukan ablation domain murni.
