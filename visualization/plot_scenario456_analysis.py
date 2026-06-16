"""
plot_scenario456_analysis.py
============================
Visualisasi hasil Skenario 4, 5, dan 6 di Wilayah Kansas dan Nebraska.
Menghasilkan 3 file gambar:
  1. scenario456_perclass_f1.png  — Perbandingan F1-Score Per Kelas Skenario 4, 5, 6
  2. scenario4_dist_shift_kansas.png   — Pergeseran Distribusi Prediksi vs Aktual (Kansas Sc4)
  3. scenario4_dist_shift_nebraska.png — Pergeseran Distribusi Prediksi vs Aktual (Nebraska Sc4)

Jalankan dari folder root project:
  cd f:\\Projectan\\TA_new\\enhprota
  python visualization/plot_scenario456_analysis.py
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
# 1. DATA — Per-class F1 (langsung dari results_summary.txt)
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = ['None', 'D0', 'D1', 'D2', 'D3', 'D4']
N_CLASSES = 6

# ── Kansas ────────────────────────────────────────────────────────────────────
KS_SC4_F1   = [0.1256, 0.0758, 0.1301, 0.2062, 0.0000, 0.1734]
KS_SC5_F1   = [0.4184, 0.0000, 0.1209, 0.0034, 0.1485, 0.0000]
KS_SC6_F1   = [0.2366, 0.2610, 0.2584, 0.1511, 0.0836, 0.2041]

# Macro F1 ringkasan
KS_SC4_MACRO = 0.1185
KS_SC5_MACRO = 0.1152
KS_SC6_MACRO = 0.1991

# ── Nebraska ──────────────────────────────────────────────────────────────────
NE_SC4_F1   = [0.0000, 0.0000, 0.1749, 0.2589, 0.2780, 0.1979]
NE_SC5_F1   = [0.5704, 0.2612, 0.3129, 0.0594, 0.4215, 0.0000]
NE_SC6_F1   = [0.6176, 0.2092, 0.3813, 0.0254, 0.2261, 0.1751]

NE_SC4_MACRO = 0.1516
NE_SC5_MACRO = 0.2709
NE_SC6_MACRO = 0.2725

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA — Distribusi Aktual vs Prediksi Skenario 4
#    Sumber: Classification Report (precision, recall, support) di results_summary.txt
#    recall = TP / actual_support  → TP = recall × support
#    precision = TP / predicted_total → predicted_total = TP / precision
# ─────────────────────────────────────────────────────────────────────────────

def compute_predicted_counts(precision_arr, recall_arr, support_arr):
    """
    Menghitung jumlah sampel yang diprediksi per kelas berdasarkan
    precision, recall, dan support (actual) dari classification report.

    predicted_count_i = round(recall_i * support_i / precision_i)
    Kasus precision = 0: predicted_count = 0 (model tidak pernah prediksi kelas ini)
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

# ── Kansas Sc4 ────────────────────────────────────────────────────────────────
KS_SC4_PRECISION = [0.5786, 0.2687, 0.1645, 0.1811, 0.0000, 0.0956]
KS_SC4_RECALL    = [0.0704, 0.0441, 0.1076, 0.2393, 0.0000, 0.9305]
KS_SC4_SUPPORT   = [1150, 975, 948, 560, 288, 259]   # aktual
KS_SC4_TOTAL     = sum(KS_SC4_SUPPORT)               # 4180

ks4_actual    = KS_SC4_SUPPORT
ks4_predicted = compute_predicted_counts(KS_SC4_PRECISION, KS_SC4_RECALL, KS_SC4_SUPPORT)

# ── Nebraska Sc4 ─────────────────────────────────────────────────────────────
NE_SC4_PRECISION = [0.0000, 0.0000, 0.2857, 0.1882, 0.1957, 0.1682]
NE_SC4_RECALL    = [0.0000, 0.0000, 0.1261, 0.4144, 0.4799, 0.2403]
NE_SC4_SUPPORT   = [973, 547, 952, 1081, 473, 154]    # aktual
NE_SC4_TOTAL     = sum(NE_SC4_SUPPORT)                # 4180

ne4_actual    = NE_SC4_SUPPORT
ne4_predicted = compute_predicted_counts(NE_SC4_PRECISION, NE_SC4_RECALL, NE_SC4_SUPPORT)

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

# Palet warna per skenario
COLOR_SC4 = '#D32F2F'   # Merah
COLOR_SC5 = '#F57C00'   # Oranye
COLOR_SC6 = '#1976D2'   # Biru

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Perbandingan F1-Score Per Kelas Skenario 4, 5, 6 (2 baris × 1 kolom)
# ─────────────────────────────────────────────────────────────────────────────

def plot_perclass_f1_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=False)
    fig.patch.set_facecolor('#F7F9FC')

    bar_width = 0.24
    x = np.arange(N_CLASSES)

    datasets = [
        {
            'ax': axes[0],
            'title': 'Wilayah Kansas',
            'sc4': KS_SC4_F1, 'sc5': KS_SC5_F1, 'sc6': KS_SC6_F1,
            'macro4': KS_SC4_MACRO, 'macro5': KS_SC5_MACRO, 'macro6': KS_SC6_MACRO,
        },
        {
            'ax': axes[1],
            'title': 'Wilayah Nebraska',
            'sc4': NE_SC4_F1, 'sc5': NE_SC5_F1, 'sc6': NE_SC6_F1,
            'macro4': NE_SC4_MACRO, 'macro5': NE_SC5_MACRO, 'macro6': NE_SC6_MACRO,
        },
    ]

    for d in datasets:
        ax = d['ax']
        ax.set_facecolor('#FFFFFF')

        bars4 = ax.bar(x - bar_width, d['sc4'], bar_width, label=f'Sc4 – Weather Only (Macro={d["macro4"]:.4f})',
                       color=COLOR_SC4, alpha=0.88, edgecolor='white', linewidth=0.5)
        bars5 = ax.bar(x,             d['sc5'], bar_width, label=f'Sc5 – Weather + Lag (Macro={d["macro5"]:.4f})',
                       color=COLOR_SC5, alpha=0.88, edgecolor='white', linewidth=0.5)
        bars6 = ax.bar(x + bar_width, d['sc6'], bar_width, label=f'Sc6 – No Drought History (Macro={d["macro6"]:.4f})',
                       color=COLOR_SC6, alpha=0.88, edgecolor='white', linewidth=0.5)

        # Nilai di atas tiap bar
        for bars in [bars4, bars5, bars6]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.005:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f'{h:.3f}', ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold', color='#333333',
                        rotation=90
                    )

        # Garis referensi Macro F1 tiap skenario (dashed)
        ax.axhline(d['macro4'], color=COLOR_SC4, linestyle='--', linewidth=1.0, alpha=0.5)
        ax.axhline(d['macro5'], color=COLOR_SC5, linestyle='--', linewidth=1.0, alpha=0.5)
        ax.axhline(d['macro6'], color=COLOR_SC6, linestyle='--', linewidth=1.0, alpha=0.5)

        ax.set_title(d['title'], fontweight='bold', pad=10)
        ax.set_xlabel('Kelas Kekeringan (Drought Class)', labelpad=8)
        ax.set_ylabel('F1-Score', labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES)
        ax.set_ylim(0, 0.72)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CCCCCC')
        ax.legend(loc='upper right', frameon=True, facecolor='#FAFAFA',
                  edgecolor='#CCCCCC', fontsize=8.5)

    plt.suptitle(
        "Perbandingan F1-Score Per Kelas\nSkenario 4 (Weather Only) · Skenario 5 (Weather + Lag) · Skenario 6 (No Drought History)",
        fontsize=14, fontweight='bold', color='#1A237E', y=0.98
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.93])

    out = os.path.join(os.path.dirname(__file__), 'scenario456_perclass_f1.png')
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[1/3] Tersimpan: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 & 3 — Pergeseran Distribusi Prediksi vs Aktual Skenario 4
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
        f"Pergeseran Distribusi Prediksi vs Aktual — Skenario 4 (Weather Only)\nWilayah {region_name}  |  Total Sampel Uji: {total:,}",
        fontweight='bold', color='#1A237E', pad=12
    )
    ax_top.set_ylabel('Jumlah Sampel (Count)', labelpad=8, fontweight='bold')
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(CLASSES, fontsize=10)
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
    ax_bot.set_xticklabels(CLASSES, fontsize=10)
    ax_bot.set_xlim(-0.6, N_CLASSES - 0.4)

    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    out = os.path.join(os.path.dirname(__file__), output_filename)
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[{2 if 'kansas' in output_filename else 3}/3] Tersimpan: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Visualisasi Skenario 4, 5, 6 — Kansas & Nebraska")
    print("=" * 65)

    # 1. Perbandingan F1 Per Kelas
    plot_perclass_f1_comparison()

    # 2. Distribusi Shift — Kansas Sc4
    print(f"\nKansas Sc4 — Distribusi Aktual   : {dict(zip(CLASSES, ks4_actual))}")
    print(f"Kansas Sc4 — Distribusi Prediksi  : {dict(zip(CLASSES, ks4_predicted))}")
    plot_distribution_shift(
        region_name='Kansas',
        actual_counts=ks4_actual,
        predicted_counts=ks4_predicted,
        total=KS_SC4_TOTAL,
        output_filename='scenario4_dist_shift_kansas.png',
        bar_color_actual='#1565C0',
        bar_color_pred='#FF7043',
    )

    # 3. Distribusi Shift — Nebraska Sc4
    print(f"\nNebraska Sc4 — Distribusi Aktual  : {dict(zip(CLASSES, ne4_actual))}")
    print(f"Nebraska Sc4 — Distribusi Prediksi: {dict(zip(CLASSES, ne4_predicted))}")
    plot_distribution_shift(
        region_name='Nebraska',
        actual_counts=ne4_actual,
        predicted_counts=ne4_predicted,
        total=NE_SC4_TOTAL,
        output_filename='scenario4_dist_shift_nebraska.png',
        bar_color_actual='#E65100',
        bar_color_pred='#6A1B9A',
    )

    print("\nSelesai! Semua file berhasil disimpan di folder visualization/.")


if __name__ == '__main__':
    main()
