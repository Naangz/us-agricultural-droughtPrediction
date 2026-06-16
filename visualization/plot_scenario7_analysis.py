"""
plot_scenario7_analysis.py
===========================
Visualisasi hasil Skenario 7 (Drought History Only) di Wilayah Kansas dan Nebraska.
Menghasilkan 3 file gambar:
  1. scenario7_perclass_f1.png        — Perbandingan F1-Score Per Kelas Antar-Wilayah
  2. scenario7_dist_shift_kansas.png   — Pergeseran Distribusi Prediksi vs Aktual (Kansas)
  3. scenario7_dist_shift_nebraska.png — Pergeseran Distribusi Prediksi vs Aktual (Nebraska)

Jalankan dari folder root project:
  cd f:\\Projectan\\TA_new\\enhprota
  python visualization/plot_scenario7_analysis.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA — Skenario 7 (Drought History Only)
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = ['None', 'D0', 'D1', 'D2', 'D3', 'D4']
N_CLASSES = 6

# ── Kansas Skenario 7 ────────────────────────────────────────────────────────
KS_SC7_F1        = [0.9082, 0.7795, 0.7943, 0.7255, 0.7254, 0.9180]
KS_SC7_MACRO     = 0.8085
KS_SC7_PRECISION = [0.9001, 0.7573, 0.8324, 0.8265, 0.6361, 0.8690]
KS_SC7_RECALL    = [0.9165, 0.8031, 0.7595, 0.6464, 0.8438, 0.9730]
KS_SC7_SUPPORT   = [1150, 975, 948, 560, 288, 259]   # aktual
KS_SC7_TOTAL     = sum(KS_SC7_SUPPORT)               # 4180

# ── Nebraska Skenario 7 ──────────────────────────────────────────────────────
NE_SC7_F1        = [0.9285, 0.7526, 0.7941, 0.8202, 0.8008, 0.8418]
NE_SC7_MACRO     = 0.8230
NE_SC7_PRECISION = [0.9437, 0.7143, 0.8295, 0.8393, 0.7505, 0.7790]
NE_SC7_RECALL    = [0.9137, 0.7952, 0.7616, 0.8020, 0.8584, 0.9156]
NE_SC7_SUPPORT   = [973, 547, 952, 1081, 473, 154]    # aktual
NE_SC7_TOTAL     = sum(NE_SC7_SUPPORT)                # 4180

# ─────────────────────────────────────────────────────────────────────────────
# 2. UTILS
# ─────────────────────────────────────────────────────────────────────────────

def compute_predicted_counts(precision_arr, recall_arr, support_arr):
    """
    Menghitung jumlah sampel yang diprediksi per kelas berdasarkan
    precision, recall, dan support (actual) dari classification report.
    predicted_count_i = round(recall_i * support_i / precision_i)
    """
    predicted = []
    for p, r, s in zip(precision_arr, recall_arr, support_arr):
        if p == 0.0:
            predicted.append(0)
        else:
            tp = r * s
            pred_total = tp / p
            predicted.append(int(round(pred_total)))
    return predicted

ks7_actual    = KS_SC7_SUPPORT
ks7_predicted = compute_predicted_counts(KS_SC7_PRECISION, KS_SC7_RECALL, KS_SC7_SUPPORT)

ne7_actual    = NE_SC7_SUPPORT
ne7_predicted = compute_predicted_counts(NE_SC7_PRECISION, NE_SC7_RECALL, NE_SC7_SUPPORT)

# ─────────────────────────────────────────────────────────────────────────────
# 3. MATPLOTLIB STYLE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Segoe UI', 'Helvetica'],
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 0.8,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#222222',
    'axes.labelcolor': '#222222',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
})

# Palet warna
COLOR_KS = '#1E88E5'   # Biru Kansas
COLOR_NE = '#FF8F00'   # Amber/Orange Nebraska

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Perbandingan F1-Score Per Kelas Antar-Wilayah (Skenario 7)
# ─────────────────────────────────────────────────────────────────────────────

def plot_perclass_f1_comparison():
    fig, ax = plt.subplots(figsize=(11, 7), facecolor='#F7F9FC')
    ax.set_facecolor('#FFFFFF')
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CCCCCC', zorder=0)

    bar_width = 0.35
    x = np.arange(N_CLASSES)

    # Batang F1-Score
    bars_ks = ax.bar(x - bar_width/2, KS_SC7_F1, bar_width, label=f'Kansas (Macro F1 = {KS_SC7_MACRO:.4f})',
                     color=COLOR_KS, alpha=0.88, edgecolor='white', linewidth=0.5, zorder=3)
    bars_ne = ax.bar(x + bar_width/2, NE_SC7_F1, bar_width, label=f'Nebraska (Macro F1 = {NE_SC7_MACRO:.4f})',
                     color=COLOR_NE, alpha=0.88, edgecolor='white', linewidth=0.5, zorder=3)

    # Nilai di atas tiap bar
    for bars in [bars_ks, bars_ne]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.4f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color='#333333'
                )

    # Garis referensi Macro F1 (dashed)
    ax.axhline(KS_SC7_MACRO, color=COLOR_KS, linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'Kansas Macro F1 Ref ({KS_SC7_MACRO:.4f})')
    ax.axhline(NE_SC7_MACRO, color=COLOR_NE, linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'Nebraska Macro F1 Ref ({NE_SC7_MACRO:.4f})')

    ax.set_title("Perbandingan F1-Score Per Kelas Antar-Wilayah — Skenario 7 (Drought History Only)",
                 fontsize=14, fontweight='bold', color='#1A237E', pad=15)
    ax.set_xlabel('Kelas Kekeringan (Drought Class)', labelpad=10, fontweight='bold')
    ax.set_ylabel('F1-Score', labelpad=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='lower left', frameon=True, facecolor='#FAFAFA',
              edgecolor='#CCCCCC', fontsize=9.5)

    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), 'scenario7_perclass_f1.png')
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[1/3] Tersimpan: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 & 3 — Pergeseran Distribusi Prediksi vs Aktual Skenario 7
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution_shift(region_name, actual_counts, predicted_counts, total,
                             output_filename, bar_color_actual='#1565C0',
                             bar_color_pred='#FF7043'):
    """
    Membuat grafik batang berdampingan Aktual vs Prediksi + grafik pergeseran (delta).
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 9),
        gridspec_kw={'height_ratios': [2.2, 1]},
        facecolor='#F7F9FC'
    )
    fig.patch.set_facecolor('#F7F9FC')

    x = np.arange(N_CLASSES)
    bar_w = 0.35

    # ── Panel atas: Aktual vs Prediksi ────────────────────────────────────────
    ax_top.set_facecolor('#FFFFFF')
    ax_top.grid(axis='y', linestyle='--', alpha=0.4, color='#CCCCCC', zorder=0)

    bars_act  = ax_top.bar(x - bar_w/2, actual_counts,    bar_w,
                           label='Distribusi Aktual (Ground Truth)',
                           color=bar_color_actual, alpha=0.88,
                           edgecolor='white', linewidth=0.5, zorder=3)
    bars_pred = ax_top.bar(x + bar_w/2, predicted_counts, bar_w,
                           label='Distribusi Prediksi Model',
                           color=bar_color_pred, alpha=0.88,
                           edgecolor='white', linewidth=0.5, zorder=3)

    # Persen di atas bar
    for bars, counts in [(bars_act, actual_counts), (bars_pred, predicted_counts)]:
        for bar, cnt in zip(bars, counts):
            pct = cnt / total * 100
            if cnt > 0:
                ax_top.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 15,
                    f'{cnt}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold', color='#333333'
                )

    ax_top.set_title(
        f"Pergeseran Distribusi Prediksi vs Aktual — Skenario 7 (Drought History Only)\nWilayah {region_name}  |  Total Sampel Uji: {total:,}",
        fontweight='bold', color='#1A237E', pad=12
    )
    ax_top.set_ylabel('Jumlah Sampel (Count)', labelpad=8, fontweight='bold')
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(CLASSES, fontsize=10, fontweight='bold')
    ax_top.legend(loc='upper right', frameon=True, facecolor='#FAFAFA',
                  edgecolor='#CCCCCC', fontsize=9.5)
    ax_top.set_xlim(-0.6, N_CLASSES - 0.4)

    # ── Panel bawah: Delta (Prediksi − Aktual) ────────────────────────────────
    ax_bot.set_facecolor('#FFFFFF')
    ax_bot.grid(axis='y', linestyle='--', alpha=0.4, color='#CCCCCC', zorder=0)

    deltas = [pred - act for pred, act in zip(predicted_counts, actual_counts)]
    colors_delta = ['#2E7D32' if d >= 0 else '#C62828' for d in deltas]  # hijau = over-predict, merah = under-predict

    bars_delta = ax_bot.bar(x, deltas, 0.55,
                             color=colors_delta, alpha=0.85,
                             edgecolor='white', linewidth=0.5, zorder=3)

    for bar, d in zip(bars_delta, deltas):
        sign = '+' if d >= 0 else ''
        ax_bot.text(
            bar.get_x() + bar.get_width() / 2,
            d + (20 if d >= 0 else -40),
            f'{sign}{d}',
            ha='center', va='bottom', fontsize=9,
            fontweight='bold',
            color='#2E7D32' if d >= 0 else '#C62828'
        )

    ax_bot.axhline(0, color='#333333', linewidth=1.2, linestyle='-')
    ax_bot.set_title(
        'Δ Prediksi − Aktual  (Hijau = Over-predicted, Merah = Under-predicted)',
        fontsize=10, color='#444444', pad=8
    )
    ax_bot.set_ylabel('Δ Count', labelpad=8, fontweight='bold')
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(CLASSES, fontsize=10, fontweight='bold')
    ax_bot.set_xlim(-0.6, N_CLASSES - 0.4)

    # Penyesuaian batas y bawah agar teks delta negatif tidak terpotong
    min_d = min(deltas)
    max_d = max(deltas)
    ax_bot.set_ylim(min_d - (100 if min_d < 0 else 50), max_d + (100 if max_d > 0 else 50))

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), output_filename)
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[{2 if 'kansas' in output_filename else 3}/3] Tersimpan: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Visualisasi Skenario 7 (Drought History Only) — Kansas & Nebraska")
    print("=" * 65)

    # 1. Perbandingan F1 Per Kelas
    plot_perclass_f1_comparison()

    # 2. Distribusi Shift — Kansas Skenario 7
    print(f"\nKansas Sc7 — Distribusi Aktual   : {dict(zip(CLASSES, ks7_actual))}")
    print(f"Kansas Sc7 — Distribusi Prediksi  : {dict(zip(CLASSES, ks7_predicted))}")
    plot_distribution_shift(
        region_name='Kansas',
        actual_counts=ks7_actual,
        predicted_counts=ks7_predicted,
        total=KS_SC7_TOTAL,
        output_filename='scenario7_dist_shift_kansas.png',
        bar_color_actual='#1565C0',
        bar_color_pred='#FF7043',
    )

    # 3. Distribusi Shift — Nebraska Skenario 7
    print(f"\nNebraska Sc7 — Distribusi Aktual  : {dict(zip(CLASSES, ne7_actual))}")
    print(f"Nebraska Sc7 — Distribusi Prediksi: {dict(zip(CLASSES, ne7_predicted))}")
    plot_distribution_shift(
        region_name='Nebraska',
        actual_counts=ne7_actual,
        predicted_counts=ne7_predicted,
        total=NE_SC7_TOTAL,
        output_filename='scenario7_dist_shift_nebraska.png',
        bar_color_actual='#E65100',
        bar_color_pred='#6A1B9A',
    )

    print("\nSelesai! Semua file berhasil disimpan di folder visualization/.")


if __name__ == '__main__':
    main()
