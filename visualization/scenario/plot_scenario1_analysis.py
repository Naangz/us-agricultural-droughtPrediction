"""
plot_scenario1_analysis.py
===========================
Visualisasi hasil Skenario 1 (Baseline) di Wilayah Kansas dan Nebraska.
Menghasilkan 3 file gambar:
  1. scenario1_perclass_f1.png        — Perbandingan F1-Score Per Kelas Antar-Wilayah
  2. scenario1_dist_shift_kansas.png   — Pergeseran Distribusi Prediksi vs Aktual (Kansas)
  3. scenario1_dist_shift_nebraska.png — Pergeseran Distribusi Prediksi vs Aktual (Nebraska)

Jalankan dari folder root project:
  cd f:\\Projectan\\TA_new\\enhprota
  python visualization/plot_scenario1_analysis.py
"""

import ast
import os
import re
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
# 1. DATA — Skenario 1 (Baseline), dibaca dari hasil training aktual
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = ['None', 'D0', 'D1', 'D2', 'D3', 'D4']
N_CLASSES = 6
TRAIN_END_DATE = '2019-12-31'
TEST_START_DATE = '2022-01-01'
FEATURE_COLS = [
    'ALLSKY_SFC_SW_DWN', 'PRECTOTCORR', 'PS', 'RH2M', 'T2M', 'WS2M',
    'PREC_lag1', 'PREC_lag2', 'PREC_lag4', 'PREC_lag8',
    'T2M_lag1', 'T2M_lag2', 'T2M_lag4', 'T2M_lag8',
    'RH2M_lag1', 'RH2M_lag2', 'RH2M_lag4', 'RH2M_lag8',
    'PREC_roll4_mean', 'PREC_roll4_std', 'PREC_roll12_mean', 'PREC_roll12_std',
    'T2M_roll4_mean', 'T2M_roll12_mean',
    'week_sin', 'week_cos',
    'None_lag1', 'D0_lag1', 'D1_lag1', 'D2_lag1', 'D3_lag1', 'D4_lag1',
    'None_lag2', 'D0_lag2', 'D1_lag2', 'D2_lag2', 'D3_lag2', 'D4_lag2',
]
PMF_COLS = ['PMF_None', 'PMF_D0', 'PMF_D1', 'PMF_D2', 'PMF_D3', 'PMF_D4']
REGION_CONFIGS = {
    'kansas': {
        'display_name': 'Kansas',
        'dataset_path': 'Integrated_weekly_KAN_20counties.csv',
        'summary_candidates': [
            os.path.join('kansas', 'output_weekly_kansas_20counties', 'results_summary.txt'),
            os.path.join('kansas', 'output_weekly_kansas_20counties', 'results_summary - KAN.txt'),
        ],
        'model_candidates': [
            os.path.join('kansas', 'output_weekly_kansas_20counties', 'best_model.keras'),
        ],
    },
    'nebraska': {
        'display_name': 'Nebraska',
        'dataset_path': 'Integrated_weekly_NEB_20counties.csv',
        'summary_candidates': [
            os.path.join('nebraska', 'output_weekly_nebraska_20counties', 'results_summary.txt'),
            os.path.join('nebraska', 'output_weekly_nebraska_20counties', 'NE_results_summary.txt'),
            os.path.join('nebraska', 'output_weekly_nebraska_20counties', 'results_summary - NEB.txt'),
        ],
        'model_candidates': [
            os.path.join('nebraska', 'output_weekly_nebraska_20counties', 'best_model.keras'),
        ],
    },
}

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

def get_latest_existing_path(candidates, description):
    """
    Mengambil file kandidat terbaru berdasarkan waktu modifikasi.
    """
    existing = [candidate for candidate in candidates if os.path.exists(candidate)]
    if not existing:
        raise FileNotFoundError(
            f"Tidak menemukan {description}. Kandidat yang dicek: {candidates}"
        )
    return max(existing, key=os.path.getmtime)

def get_summary_path(region):
    return get_latest_existing_path(
        REGION_CONFIGS[region]['summary_candidates'],
        f"results_summary baseline untuk wilayah '{region}'"
    )

def get_model_path(region):
    return get_latest_existing_path(
        REGION_CONFIGS[region]['model_candidates'],
        f"best_model baseline untuk wilayah '{region}'"
    )

def decumulate_drought(row):
    pmf_d4 = row['D4']
    pmf_d3 = max(0.0, row['D3'] - row['D4'])
    pmf_d2 = max(0.0, row['D2'] - row['D3'])
    pmf_d1 = max(0.0, row['D1'] - row['D2'])
    pmf_d0 = max(0.0, row['D0'] - row['D1'])
    pmf_none = max(0.0, row['None'])
    return [pmf_none, pmf_d0, pmf_d1, pmf_d2, pmf_d3, pmf_d4]

def build_feature_frame(region):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    data_path = REGION_CONFIGS[region]['dataset_path']
    df = pd.read_csv(data_path)
    df['week_start'] = pd.to_datetime(df['week_start'])
    df = df.sort_values(['FIPS', 'week_start']).reset_index(drop=True)
    df[PMF_COLS] = df.apply(lambda row: decumulate_drought(row), axis=1, result_type='expand')
    df['Label'] = df[PMF_COLS].idxmax(axis=1).apply(lambda x: PMF_COLS.index(x))

    df_fe = df.copy().sort_values(['FIPS', 'week_start']).reset_index(drop=True)
    for lag in [1, 2, 4, 8]:
        df_fe[f'PREC_lag{lag}'] = df_fe.groupby('FIPS')['PRECTOTCORR'].shift(lag)
        df_fe[f'T2M_lag{lag}'] = df_fe.groupby('FIPS')['T2M'].shift(lag)
        df_fe[f'RH2M_lag{lag}'] = df_fe.groupby('FIPS')['RH2M'].shift(lag)

    for window in [4, 12]:
        df_fe[f'PREC_roll{window}_mean'] = df_fe.groupby('FIPS')['PRECTOTCORR'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df_fe[f'PREC_roll{window}_std'] = df_fe.groupby('FIPS')['PRECTOTCORR'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0.0)
        )
        df_fe[f'T2M_roll{window}_mean'] = df_fe.groupby('FIPS')['T2M'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    iso_week = df_fe['week_start'].dt.isocalendar().week.astype(int)
    df_fe['week_sin'] = np.sin(2 * np.pi * iso_week / 52.0)
    df_fe['week_cos'] = np.cos(2 * np.pi * iso_week / 52.0)

    for col in CLASSES:
        df_fe[f'{col}_lag1'] = df_fe.groupby('FIPS')[col].shift(1)
        df_fe[f'{col}_lag2'] = df_fe.groupby('FIPS')[col].shift(2)

    df_fe = df_fe.dropna(subset=FEATURE_COLS + ['Label']).reset_index(drop=True)
    train_df = df_fe[df_fe['week_start'] <= TRAIN_END_DATE].copy()
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURE_COLS])
    df_fe.loc[:, FEATURE_COLS] = scaler.transform(df_fe[FEATURE_COLS])
    return df_fe

def create_sequences_from_df(df_input, seq_length):
    import pandas as pd

    features = []
    labels = []
    for _, group in df_input.groupby('FIPS'):
        group = group.sort_values('week_start')
        feat_values = group[FEATURE_COLS].values
        label_values = group['Label'].values
        dates = group['week_start'].values
        if len(group) < seq_length:
            continue
        for i in range(seq_length - 1, len(group)):
            target_date = pd.Timestamp(dates[i])
            if target_date < pd.Timestamp(TEST_START_DATE):
                continue
            features.append(feat_values[i - seq_length + 1:i + 1])
            labels.append(int(label_values[i]))
    return np.array(features), np.array(labels)

def compute_exact_predicted_counts(region, parsed_summary):
    import tensorflow as tf

    if parsed_summary['seq_length'] is None or parsed_summary['class_multipliers'] is None:
        raise ValueError('Metadata seq_length atau class_multipliers tidak tersedia di summary.')

    feature_frame = build_feature_frame(region)
    x_test, y_test = create_sequences_from_df(feature_frame, parsed_summary['seq_length'])
    if len(y_test) != parsed_summary['total']:
        raise ValueError(
            f"Jumlah sampel inference ({len(y_test)}) tidak sama dengan total summary ({parsed_summary['total']})."
        )

    model_path = get_model_path(region)
    model = tf.keras.models.load_model(model_path, compile=False)
    class_multipliers = np.array(parsed_summary['class_multipliers'], dtype=float)
    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_prob * class_multipliers, axis=1)

    actual_counts = np.bincount(y_test, minlength=N_CLASSES).tolist()
    predicted_counts = np.bincount(y_pred, minlength=N_CLASSES).tolist()
    if actual_counts != parsed_summary['support']:
        raise ValueError(
            f"Distribusi aktual hasil inference {actual_counts} tidak sama dengan summary {parsed_summary['support']}."
        )
    return predicted_counts

def parse_scenario1_summary(summary_path):
    """
    Parse Macro F1, Per-class F1, serta precision/recall/support dari
    results_summary.txt hasil training baseline.
    """
    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()

    macro_matches = re.findall(r'^\s*Macro\s+F1:\s*([\d.]+)\s*$', content, flags=re.MULTILINE)
    if not macro_matches:
        raise ValueError(f"Gagal menemukan Macro F1 pada file: {summary_path}")
    macro_f1 = float(macro_matches[-1])
    summary_title = content.splitlines()[0].strip() if content.splitlines() else os.path.basename(summary_path)
    seq_length_match = re.search(r'^\s*Seq Length:\s*(\d+)\s*$', content, flags=re.MULTILINE)
    class_multipliers_match = re.search(r'^\s*Class multipliers:\s*(\[[^\]]+\])\s*$', content, flags=re.MULTILINE)
    best_trial_match = re.search(r'^\s*Best trial:\s*(.+?)\s*$', content, flags=re.MULTILINE)

    perclass_f1 = {}
    perclass_block_match = re.search(
        r'Per-class F1:\s*(.*?)(?:\n\s*=|\n\s*CLASSIFICATION REPORT)',
        content,
        flags=re.DOTALL
    )
    if perclass_block_match:
        for cls_name, value in re.findall(r'^\s*(None|D0|D1|D2|D3|D4):\s*([\d.]+)\s*$', perclass_block_match.group(1), flags=re.MULTILINE):
            perclass_f1[cls_name] = float(value)

    final_report_match = re.search(
        r'^\s*CLASSIFICATION REPORT\s*$'
        r'(?:\r?\n^\s*=+\s*$)?'
        r'(.*?)'
        r'^\s*=+\s*$',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    if not final_report_match:
        raise ValueError(f"Gagal menemukan blok CLASSIFICATION REPORT final pada file: {summary_path}")
    final_report = final_report_match.group(1)

    class_report_rows = re.findall(
        r'^\s*(None|D0|D1|D2|D3|D4)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$',
        final_report,
        flags=re.MULTILINE
    )
    if len(class_report_rows) != N_CLASSES:
        raise ValueError(f"Gagal parse classification report lengkap pada file: {summary_path}")

    precision = []
    recall = []
    support = []
    f1_scores = []
    for cls_name in CLASSES:
        row = next((row for row in class_report_rows if row[0] == cls_name), None)
        if row is None:
            raise ValueError(f"Kelas {cls_name} tidak ditemukan di classification report: {summary_path}")
        precision.append(float(row[1]))
        recall.append(float(row[2]))
        f1_scores.append(perclass_f1.get(cls_name, float(row[3])))
        support.append(int(row[4]))

    predicted = compute_predicted_counts(precision, recall, support)
    return {
        'summary_path': summary_path,
        'summary_title': summary_title,
        'best_trial': best_trial_match.group(1).strip() if best_trial_match else None,
        'seq_length': int(seq_length_match.group(1)) if seq_length_match else None,
        'class_multipliers': ast.literal_eval(class_multipliers_match.group(1)) if class_multipliers_match else None,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'support': support,
        'f1_scores': f1_scores,
        'predicted': predicted,
        'predicted_source': 'classification_report_estimate',
        'total': sum(support),
    }

def load_region_data():
    """
    Memuat data baseline aktual untuk Kansas dan Nebraska dari results_summary.
    """
    data = {}
    for region in ['kansas', 'nebraska']:
        summary_path = get_summary_path(region)
        parsed = parse_scenario1_summary(summary_path)
        try:
            parsed['predicted'] = compute_exact_predicted_counts(region, parsed)
            parsed['predicted_source'] = 'exact_model_inference'
        except Exception as exc:
            print(f"  Peringatan: gagal hitung prediksi exact untuk {REGION_CONFIGS[region]['display_name']}: {exc}")
            print("  Fallback ke estimasi dari classification report.")
        data[region] = parsed
        print(f"Memuat baseline {REGION_CONFIGS[region]['display_name']} dari: {summary_path}")
        print(f"  Macro F1 = {parsed['macro_f1']:.4f}")
        print(f"  Sumber distribusi prediksi = {parsed['predicted_source']}")
    return data

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
COLOR_KS = '#0288D1'   # Deep Blue Kansas
COLOR_NE = '#F57C00'   # Amber/Orange Nebraska

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Perbandingan F1-Score Per Kelas Antar-Wilayah (Skenario 1 Baseline)
# ─────────────────────────────────────────────────────────────────────────────

def plot_perclass_f1_comparison(kansas_data, nebraska_data):
    fig, ax = plt.subplots(figsize=(11, 7), facecolor='#F7F9FC')
    ax.set_facecolor('#FFFFFF')
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CCCCCC', zorder=0)

    bar_width = 0.35
    x = np.arange(N_CLASSES)

    # Batang F1-Score
    bars_ks = ax.bar(x - bar_width/2, kansas_data['f1_scores'], bar_width,
                     label=f'Kansas (Macro F1 = {kansas_data["macro_f1"]:.4f})',
                     color=COLOR_KS, alpha=0.88, edgecolor='white', linewidth=0.5, zorder=3)
    bars_ne = ax.bar(x + bar_width/2, nebraska_data['f1_scores'], bar_width,
                     label=f'Nebraska (Macro F1 = {nebraska_data["macro_f1"]:.4f})',
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
    ax.axhline(kansas_data['macro_f1'], color=COLOR_KS, linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'Kansas Macro F1 Ref ({kansas_data["macro_f1"]:.4f})')
    ax.axhline(nebraska_data['macro_f1'], color=COLOR_NE, linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'Nebraska Macro F1 Ref ({nebraska_data["macro_f1"]:.4f})')

    ax.set_title("Perbandingan F1-Score Per Kelas Antar-Wilayah — Skenario 1 (Baseline)",
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

    out = os.path.join(os.path.dirname(__file__), 'scenario1_perclass_f1.png')
    plt.savefig(out, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[1/3] Tersimpan: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 & 3 — Pergeseran Distribusi Prediksi vs Aktual Skenario 1
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
        f"Pergeseran Distribusi Prediksi vs Aktual — Skenario 1 (Baseline)\nWilayah {region_name}  |  Total Sampel Uji: {total:,}",
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
    print("  Visualisasi Skenario 1 (Baseline) — Kansas & Nebraska")
    print("=" * 65)

    region_data = load_region_data()
    kansas_data = region_data['kansas']
    nebraska_data = region_data['nebraska']

    # 1. Perbandingan F1 Per Kelas
    plot_perclass_f1_comparison(kansas_data, nebraska_data)

    # 2. Distribusi Shift — Kansas Skenario 1 Baseline
    print(f"\nKansas Sc1 — Distribusi Aktual   : {dict(zip(CLASSES, kansas_data['support']))}")
    print(f"Kansas Sc1 — Distribusi Prediksi  : {dict(zip(CLASSES, kansas_data['predicted']))}")
    plot_distribution_shift(
        region_name='Kansas',
        actual_counts=kansas_data['support'],
        predicted_counts=kansas_data['predicted'],
        total=kansas_data['total'],
        output_filename='scenario1_dist_shift_kansas.png',
        bar_color_actual='#1565C0',
        bar_color_pred='#FF7043',
    )

    # 3. Distribusi Shift — Nebraska Skenario 1 Baseline
    print(f"\nNebraska Sc1 — Distribusi Aktual  : {dict(zip(CLASSES, nebraska_data['support']))}")
    print(f"Nebraska Sc1 — Distribusi Prediksi: {dict(zip(CLASSES, nebraska_data['predicted']))}")
    plot_distribution_shift(
        region_name='Nebraska',
        actual_counts=nebraska_data['support'],
        predicted_counts=nebraska_data['predicted'],
        total=nebraska_data['total'],
        output_filename='scenario1_dist_shift_nebraska.png',
        bar_color_actual='#E65100',
        bar_color_pred='#6A1B9A',
    )

    print("\nSelesai! Semua file berhasil disimpan di folder visualization/.")

if __name__ == '__main__':
    main()
