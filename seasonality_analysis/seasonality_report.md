# Seasonality Analysis Report

## Metode

- Dataset: `Integrated_weekly_NEB_20counties.csv`.
- Jumlah baris: 16,720; jumlah county unik: 20.
- Rentang waktu: 2009-12-29 sampai 2025-12-30.
- Unit analisis: rata-rata seluruh county per `week_start` sebelum ACF, profil musiman, dan dekomposisi.
- Alasan agregasi: setiap county berbagi kalender mingguan yang sama, sehingga menghitung ACF pada seluruh baris county dapat memperlakukan observasi spasial sebagai replikasi temporal. Rata-rata regional mengurangi noise lokal dan menguji seasonality temporal dataset secara lebih langsung.
- ACF dihitung sampai lag 104 minggu. Confidence interval menggunakan pendekatan white-noise `+/- 1.96/sqrt(n)`.
- Seasonal profile dihitung dari rata-rata setiap ISO week-of-year 1-52.
- Baris dengan ISO week 53 yang tidak masuk profil 1-52: 3.
- Seasonal decomposition memakai STL dengan `period=52` jika `statsmodels` tersedia; jika tidak, script memakai fallback dekomposisi aditif klasik berbasis moving average 52 minggu.

## Output Grafik

- `acf_prec.png`: ACF PRECTOTCORR.
- `acf_t2m.png`: ACF T2M.
- `acf_rh2m.png`: ACF RH2M.
- `seasonal_profile.png`: profil rata-rata week-of-year.
- `decomposition_t2m.png`: trend, seasonal, residual T2M.
- `decomposition_prec.png`: trend, seasonal, residual PRECTOTCORR.

## Interpretasi ACF dan Profil Musiman

- **PRECTOTCORR**: ACF at lag 52 = 0.315, outside the 95% CI (+/- 0.068), and is a local maximum in lags 50-54. Seasonal profile range = 4.699; seasonal strength = 0.347. Overall evidence: **moderate**.
- **T2M**: ACF at lag 52 = 0.859, outside the 95% CI (+/- 0.068), and is a local maximum in lags 50-54. Seasonal profile range = 30.748; seasonal strength = 0.937. Overall evidence: **strong**.
- **RH2M**: ACF at lag 52 = 0.346, outside the 95% CI (+/- 0.068), and is a local maximum in lags 50-54. Seasonal profile range = 20.943; seasonal strength = not computed. Overall evidence: **moderate**.

## Metode Dekomposisi yang Dipakai

- **T2M**: STL decomposition dengan period 52 minggu.
- **PRECTOTCORR**: STL decomposition dengan period 52 minggu.

## Kesimpulan

Penggunaan sequence length 52 minggu memiliki dasar empiris yang layak, karena sebagian besar variabel utama menunjukkan indikasi siklus tahunan.

Kesimpulan ini bersifat objektif terhadap hasil deskriptif: lag 52 yang signifikan dan profil week-of-year yang jelas mendukung sequence 52 minggu, sedangkan variabel dengan ACF lag 52 lemah menunjukkan bahwa seasonality tahunan tidak sama kuat pada semua sinyal cuaca.