# Laporan Evaluasi Komprehensif: Generalisasi Model Prediksi Kekeringan dan Rekayasa Fitur (Kansas & Nebraska)

Dokumen ini menyediakan evaluasi terpadu dari model prediksi kekeringan BiLSTM yang dilatih dan diuji pada data dari Kansas dan Nebraska. Dokumen ini mensintesis hasil di seluruh berbagai skenario eksperimental untuk mengevaluasi proses integrasi data, strategi rekayasa fitur (feature engineering), dan kemampuan generalisasi model melintasi wilayah geografis yang berbeda.

---

## 1. Pengumpulan, Integrasi Data, dan Pra-pemrosesan

Fondasi dari pemodelan prediktif ini bergantung pada pengumpulan dan integrasi kumpulan data yang heterogen.

### A. Pengumpulan dan Penjelasan Dataset
Data yang digunakan dalam penelitian ini mencakup rentang waktu 15 tahun (2010 hingga 2025) pada skala *county* (kabupaten). Untuk memastikan perbandingan yang seimbang dan adil antar wilayah, data ini dikumpulkan secara spesifik dari **20 kabupaten terpilih di negara bagian Kansas** dan **20 kabupaten di negara bagian Nebraska** (total 40 kabupaten). Data ini bersumber dari dua penyedia utama:

1. **Data Meteorologi (NASA POWER):** 
   Berisi observasi iklim harian yang mencerminkan kondisi lingkungan yang berpotensi memicu kekeringan. Dataset ini mencakup 6 parameter utama:
   - **T2M (*Temperature at 2 Meters*):** Suhu rata-rata harian pada ketinggian 2 meter, mengindikasikan tingkat panas yang mendorong laju evaporasi.
   - **PREC (*Precipitation*):** Tingkat curah hujan/presipitasi, sumber utama asupan kelembapan.
   - **RH2M (*Relative Humidity at 2 Meters*):** Kelembapan relatif, memengaruhi seberapa cepat air menguap ke udara.
   - **WS2M (*Wind Speed at 2 Meters*):** Kecepatan angin, yang mempercepat laju penguapan permukaan.
   - **PS (*Surface Pressure*):** Tekanan udara permukaan, yang berkaitan dengan dinamika sistem cuaca lokal.
   - **ALLSKY_SFC_SW_DWN (*All Sky Surface Shortwave Downward Irradiance*):** Radiasi insolasi matahari yang mencapai permukaan, memengaruhi suhu dan kelembapan.

2. **Data Indeks Kekeringan (US Drought Monitor - USDM):** 
   Berfungsi sebagai *ground truth* (variabel target) yang mengkategorikan tingkat keparahan kekeringan mingguan. Kelas kekeringan ini meliputi:
   - **None:** Kondisi normal (tidak ada tanda kekeringan).
   - **D0 (*Abnormally Dry*):** Kering tidak normal (gejala awal akan masuk musim kemarau panjang, atau fase pemulihan dari kekeringan).
   - **D1 (*Moderate Drought*):** Kekeringan tingkat sedang (terjadi beberapa kerusakan ringan pada tanaman/padang rumput).
   - **D2 (*Severe Drought*):** Kekeringan tingkat parah (kehilangan hasil panen cukup besar, pembatasan air mulai berlaku).
   - **D3 (*Extreme Drought*):** Kekeringan tingkat ekstrem (kerugian panen besar, krisis air/defisit meluas).
   - **D4 (*Exceptional Drought*):** Kekeringan tingkat luar biasa (kerugian pertanian masif, keadaan darurat air secara umum).
   
   Interval pencatatan USDM secara sistematis selalu dimulai pada hari Selasa (*ValidStart*) setiap minggunya.

### B. Integrasi Data dan Pra-pemrosesan (Preprocessing)
Untuk menyelaraskan kedua sumber data tersebut, agregasi dan transformasi berikut dilakukan:
- **Penyelarasan Temporal:** Data meteorologi harian dari NASA diagregasi (dirata-rata) ke dalam format mingguan. Awal minggu untuk data meteorologi digeser agar sejajar dengan hari Selasa mengikuti standar pencatatan mingguan USDM.
- **Ekstraksi Fitur Tambahan (Feature Engineering):** Dari data yang sudah sejajar, fitur turunan dibuat untuk menangkap dependensi temporal, seperti:
  - **Lag & Jendela Musiman:** Rata-rata bergerak (*moving average* seperti `roll4_mean`, `roll12_mean`), standar deviasi (`roll4_std`, `roll12_std`), dan *lag* cuaca historis (`lag1`, `lag2`, `lag4`, `lag8`).
  - **Riwayat Kekeringan:** Tingkat kekeringan historis digunakan sebagai fitur *lag* (`D0_lag1`, `D1_lag2`, dll.) untuk memberikan inersia status kekeringan kepada model.
  - **Pengkodean Temporal:** Fitur siklikal `week_sin` dan `week_cos` diintegrasikan untuk menangkap ritme musim tahunan.

Seluruh data akhir yang terintegrasi diubah ke dalam bentuk sekuens mingguan (*weekly sequences*) untuk melatih model BiLSTM.

---

## 2. Evaluasi Rekayasa Fitur (Feature Engineering)

Eksperimen ini menguji berbagai subset dari fitur terintegrasi ini untuk mengisolasi kekuatan prediktifnya.

### A. Dominasi Riwayat Kekeringan
Membandingkan **Skenario 6 (Tanpa Riwayat Kekeringan)** dengan **Skenario 7 (Hanya Riwayat Kekeringan)** mengungkapkan perbedaan yang mencolok:
- **Hanya Cuaca (Skenario 4, 5, & 6):** Ketika model sangat bergantung pada data cuaca murni (bahkan dengan lag yang direkayasa dan rolling window), performanya anjlok. Skor Test F1 berkisar antara `0.11` dan `0.27`. Model gagal total untuk menangkap inersia dan manifestasi kekeringan yang kompleks dari variabel cuaca mentah saja.
- **Hanya Riwayat Kekeringan (Skenario 7):** Dengan menggunakan *hanya* 12 fitur (status lag dari kelas kekeringan dari 2 minggu sebelumnya), model mencapai **performa keseluruhan tertingginya**, mengungguli baseline yang menggunakan seluruh 41 fitur. Hal ini menunjukkan autokorelasi temporal yang masif dalam kondisi kekeringan; prediktor paling andal dari kekeringan minggu depan adalah status kekeringan itu sendiri yang baru-baru ini terjadi.

### B. Strategi Seleksi Fitur
- **Seleksi Korelasi (Skenario 2, 2A, 2B, 2C):** Memilih secara ketat Top-15 atau Top-20 fitur berdasarkan korelasi linear menghasilkan penurunan performa yang masif (Test F1 turun ke ~`0.46` di KS dan ~`0.39` di NE). Hanya ketika ambang batas diperluas ke Top-25 atau Top-30, performa kembali ke tingkat baseline. Hal ini menunjukkan bahwa korelasi linear sederhana dapat membuang interaksi non-linear krusial antara cuaca dan status kekeringan.
- **Importansi Permutasi F1 (Skenario 3):** Metode ini lebih berhasil dalam reduksi dimensionalitas dibandingkan korelasi ketat. Untuk Kansas, model ini mempertahankan Test F1 sebesar `0.7940` dengan hanya menggunakan 20 fitur. Untuk Nebraska, model ini mencapai `0.6761` dengan hanya menggunakan 10 fitur. Metode ini berhasil mengidentifikasi bahwa lag kekeringan terbaru adalah fitur paling krusial.

---

## 3. Performa Generalisasi: Kansas vs. Nebraska

Tabel di bawah ini merangkum Validation Macro F1 dan Tuned Test Macro F1 untuk kedua negara bagian di semua skenario.

| Skenario | KS Val F1 | KS Test F1 | NE Val F1 | NE Test F1 | Penilaian Generalisasi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Semua Fitur)** | 0.8553 | 0.7970 | 0.8670 | 0.8099 | **Kuat.** Penurunan kecil dari validation ke test. Konsisten di kedua negara bagian. |
| **Skenario 2 (Corr Top-15)** | 0.7108 | 0.4635 | 0.5343 | 0.3813 | **Buruk.** Overfitting parah / hilangnya sinyal. |
| **Skenario 2A (Corr Top-20)**| 0.8049 | 0.4560 | 0.5021 | 0.3975 | **Buruk.** Model gagal melakukan generalisasi dengan fitur terbatas. |
| **Skenario 2B (Corr Top-25)**| 0.8522 | 0.7896 | 0.6997 | 0.6895 | **Sedang.** KS pulih dengan baik; NE pulih sebagian. |
| **Skenario 2C (Corr Top-30)**| 0.8522 | 0.7896 | 0.8678 | 0.7799 | **Kuat.** Kedua negara bagian pulih mendekati generalisasi baseline. |
| **Skenario 3 (F1 Permutasi)**| 0.8516 | 0.7940 | 0.6719 | 0.6761 | **Sedang hingga Kuat.** Generalisasi sangat stabil, meskipun NE Val F1 lebih rendah secara keseluruhan. |
| **Skenario 4 (Hanya Cuaca)** | 0.2060 | 0.1185 | 0.2774 | 0.1516 | **Gagal.** Model tidak dapat mempelajari pola yang dapat digeneralisasi murni dari cuaca. |
| **Skenario 5 (Cuaca + Lag)** | 0.2498 | 0.1152 | 0.3183 | 0.2709 | **Gagal.** Penambahan lag tidak menyelesaikan kurangnya sinyal prediktif. |
| **Skenario 6 (Tanpa Riwayat)**| 0.2457 | 0.1715 | 0.3056 | 0.2565 | **Gagal.** Bahkan dengan rekayasa cuaca penuh, generalisasi minimal. |
| **Skenario 7 (Hanya Riwayat)**| 0.8481 | **0.8289** | 0.8711 | **0.8161** | **Sangat Baik.** Generalisasi terbaik. Model dengan sempurna memanfaatkan inersia kekeringan. |

### Kesimpulan Generalisasi
Model BiLSTM menunjukkan kemampuan generalisasi yang kuat di Kansas dan Nebraska *asalkan* memiliki akses ke riwayat kekeringan terbaru. Respons perilaku terhadap manipulasi fitur hampir identik di kedua negara bagian, menegaskan bahwa mekanika yang mendasari bagaimana model belajar untuk memprediksi kekeringan (sangat bergantung pada inersia status temporal daripada prakiraan meteorologi) secara universal konsisten pada dataset ini.

---

## 4. Evaluasi Jangka Panjang (Horizon Prediksi 4 Bulan)

Selain memprediksi kekeringan untuk 1 minggu ke depan (horizon pendek), pengujian tambahan dilakukan dengan mencoba memprediksi status kekeringan untuk 4 bulan ke depan (*4-Month Forecast Horizon*) menggunakan agregasi data bulanan. Pengujian ini difokuskan pada skenario **Baseline** (menggunakan semua fitur) dan **Skenario 7** (hanya menggunakan riwayat kekeringan) guna menguji kemampuan generalisasi jarak jauh.

Tabel berikut merangkum performa prediksi untuk horizon 4 bulan:

| Wilayah | Skenario | Val Macro F1 (Tuned) | Test Macro F1 (Raw) | Test Macro F1 (Tuned) |
| :--- | :--- | :---: | :---: | :---: |
| **Kansas** | Baseline (Semua Fitur) | 0.3301 | 0.1640 | 0.1487 |
| **Kansas** | Skenario 7 (Hanya Riwayat) | 0.1928 | 0.1423 | 0.1285 |
| **Nebraska** | Skenario 7 (Hanya Riwayat) | 0.2203 | 0.2072 | 0.2214 |

### Analisis Horizon Prediksi Jangka Panjang
Terjadi **penurunan performa yang drastis** ketika rentang waktu prediksi diperpanjang dari 1 minggu (di mana Test F1 mencapai > 0.80) menjadi 4 bulan ke depan (Test F1 anjlok drastis ke rentang 0.12 - 0.22). 
- Fakta ini membuktikan bahwa meskipun fitur riwayat kekeringan (*drought history*) memiliki inersia dan kekuatan prediktif yang sangat kuat untuk masa depan yang dekat, status kekeringan tersebut akan memudar dan kehilangan signifikansinya secara drastis saat memproyeksikan kekeringan sejauh 4 bulan.
- Prediksi jarak jauh yang hanya mengandalkan inersia (riwayat sebelumnya) terbukti sangat sulit dan tidak relevan, karena pola kekeringan 4 bulan ke depan sangat didikte oleh perubahan cuaca (misalnya hujan mendadak atau gelombang panas) yang belum terjadi.

---

## 5. Lampiran: Detail Hasil Skenario

<details>
<summary><b>Klik untuk melihat Hasil Kansas</b></summary>

### Kansas Baseline (20 counties)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.8553 | **Test Macro F1:** 0.7970
- **Fitur:** Semua 41 fitur.

### Kansas Skenario 2 (Correlation Top-15)
- **Trial Terbaik:** `ros_focal_inv_no_cw_128x64`
- **Val Macro F1:** 0.7108 | **Test Macro F1:** 0.4635

### Kansas Skenario 2A (Correlation Top-20)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.8049 | **Test Macro F1:** 0.4560

### Kansas Skenario 2B (Correlation Top-25)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.8522 | **Test Macro F1:** 0.7896

### Kansas Skenario 2C (Correlation Top-30)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.8522 | **Test Macro F1:** 0.7896

### Kansas Skenario 3 (F1 Permutation Selection)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.8516 | **Test Macro F1:** 0.7940

### Kansas Skenario 4 (Weather Only)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.2060 | **Test Macro F1:** 0.1185

### Kansas Skenario 5 (Weather + Lag Only)
- **Trial Terbaik:** `none_focal_inv_cw_64x32`
- **Val Macro F1:** 0.2498 | **Test Macro F1:** 0.1152

### Kansas Skenario 6 (No Drought History)
- **Trial Terbaik:** `ros_focal_inv_no_cw_128x64`
- **Val Macro F1:** 0.2457 | **Test Macro F1:** 0.1715

### Kansas Skenario 7 (Drought History Only)
- **Trial Terbaik:** `ros_focal_inv_no_cw_128x64`
- **Val Macro F1:** 0.8481 | **Test Macro F1:** 0.8289

</details>

<details>
<summary><b>Klik untuk melihat Hasil Nebraska</b></summary>

### Nebraska Baseline (20 counties)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.8670 | **Test Macro F1:** 0.8099
- **Fitur:** Semua 41 fitur.

### Nebraska Skenario 2 (Correlation Top-15)
- **Trial Terbaik:** `ros_focal_inv_no_cw_128x64`
- **Val Macro F1:** 0.5343 | **Test Macro F1:** 0.3813

### Nebraska Skenario 2A (Correlation Top-20)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.5021 | **Test Macro F1:** 0.3975

### Nebraska Skenario 2B (Correlation Top-25)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.6997 | **Test Macro F1:** 0.6895

### Nebraska Skenario 2C (Correlation Top-30)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.8678 | **Test Macro F1:** 0.7799

### Nebraska Skenario 3 (F1 Permutation Selection)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.6719 | **Test Macro F1:** 0.6761

### Nebraska Skenario 4 (Weather Only)
- **Trial Terbaik:** `none_focal_inv_cw_64x32`
- **Val Macro F1:** 0.2774 | **Test Macro F1:** 0.1516

### Nebraska Skenario 5 (Weather + Lag Only)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.3183 | **Test Macro F1:** 0.2709

### Nebraska Skenario 6 (No Drought History)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.3056 | **Test Macro F1:** 0.2565

### Nebraska Skenario 7 (Drought History Only)
- **Trial Terbaik:** `ros_focal_no_cw_96x48`
- **Val Macro F1:** 0.8711 | **Test Macro F1:** 0.8161

</details>
