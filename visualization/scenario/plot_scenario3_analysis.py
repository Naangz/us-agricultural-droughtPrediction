import ast
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import seaborn for high-quality plotting
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass

# Try to import sklearn and tensorflow for dynamic evaluation
HAS_ML_LIBRARIES = True
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import confusion_matrix, f1_score
    import tensorflow as tf
except ImportError:
    HAS_ML_LIBRARIES = False

CLASSES = ['None', 'D0', 'D1', 'D2', 'D3', 'D4']
PMF_COLS = ['PMF_None', 'PMF_D0', 'PMF_D1', 'PMF_D2', 'PMF_D3', 'PMF_D4']
TRAIN_END_DATE = '2019-12-31'
TEST_START_DATE = '2022-01-01'
SUMMARY_CANDIDATES = {
    'kansas': [os.path.join('kansas', 'output_weekly_kansas_scenario3', 'results_summary.txt')],
    'nebraska': [os.path.join('nebraska', 'output_weekly_nebraska_scenario3', 'results_summary.txt')],
}
MODEL_CANDIDATES = {
    'kansas': [os.path.join('kansas', 'output_weekly_kansas_scenario3', 'best_model.keras')],
    'nebraska': [os.path.join('nebraska', 'output_weekly_nebraska_scenario3', 'best_model.keras')],
}
DATASET_PATHS = {
    'kansas': 'Integrated_weekly_KAN_20counties.csv',
    'nebraska': 'Integrated_weekly_NEB_20counties.csv',
}
BASELINE_FEATURE_COLS = [
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
REGION_DISPLAY = {
    'kansas': 'Kansas',
    'nebraska': 'Nebraska',
}
REGION_COLORS = {
    'kansas': '#0288D1',
    'nebraska': '#F57C00',
}


def get_latest_existing_path(candidates, description):
    existing = [candidate for candidate in candidates if os.path.exists(candidate)]
    if not existing:
        raise FileNotFoundError(
            f"Tidak menemukan {description}. Kandidat yang dicek: {candidates}"
        )
    return max(existing, key=os.path.getmtime)


def get_summary_path(region):
    return get_latest_existing_path(
        SUMMARY_CANDIDATES[region],
        f"results_summary Scenario 3 untuk wilayah '{region}'",
    )


def get_model_path(region):
    return get_latest_existing_path(
        MODEL_CANDIDATES[region],
        f"best_model Scenario 3 untuk wilayah '{region}'",
    )


def parse_scenario3_summary(summary_path):
    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()

    trend_matches = re.findall(
        r'Top-(\d+):\s*Validation\s+Macro\s+F1\s*=\s*([\d.]+)',
        content,
    )
    validation_trend = {int(size): float(score) for size, score in trend_matches}
    if not validation_trend:
        raise ValueError(f"Gagal parse validation trend pada file: {summary_path}")

    best_subset_match = re.search(
        r'Best subset selected:\s*Top-(\d+)\s*\((\d+)\s+features\)\s+with val macro-F1\s*=\s*([\d.]+)',
        content,
    )
    if not best_subset_match:
        raise ValueError(f"Gagal parse best subset pada file: {summary_path}")

    selected_features_match = re.search(r'Selected features:\s*(\[[^\n]+\])', content)
    if not selected_features_match:
        raise ValueError(f"Gagal parse selected features pada file: {summary_path}")

    multipliers_match = re.search(r'Class multipliers:\s*(\[[^\n]+\])', content)
    if not multipliers_match:
        raise ValueError(f"Gagal parse class multipliers pada file: {summary_path}")

    seq_length_match = re.search(r'^\s*Seq Length:\s*(\d+)\s*$', content, flags=re.MULTILINE)
    if not seq_length_match:
        raise ValueError(f"Gagal parse Seq Length pada file: {summary_path}")

    macro_f1_match = re.findall(r'^\s*Macro\s+F1:\s*([\d.]+)\s*$', content, flags=re.MULTILINE)
    if not macro_f1_match:
        raise ValueError(f"Gagal parse Macro F1 pada file: {summary_path}")

    report_match = re.search(
        r'Classification Report:\s*(.*)$',
        content,
        flags=re.DOTALL,
    )
    if not report_match:
        raise ValueError(f"Gagal parse Classification Report pada file: {summary_path}")

    class_rows = re.findall(
        r'^\s*(None|D0|D1|D2|D3|D4)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$',
        report_match.group(1),
        flags=re.MULTILINE,
    )
    if len(class_rows) != len(CLASSES):
        raise ValueError(f"Gagal parse support classification report pada file: {summary_path}")

    support_map = {label: int(support) for label, _, _, _, support in class_rows}

    return {
        'summary_path': summary_path,
        'summary_title': content.splitlines()[0].strip(),
        'seq_length': int(seq_length_match.group(1)),
        'validation_trend': validation_trend,
        'best_subset_size': int(best_subset_match.group(1)),
        'best_subset_feature_count': int(best_subset_match.group(2)),
        'best_subset_val_f1': float(best_subset_match.group(3)),
        'selected_features': ast.literal_eval(selected_features_match.group(1)),
        'class_multipliers': ast.literal_eval(multipliers_match.group(1)),
        'macro_f1': float(macro_f1_match[-1]),
        'support': [support_map[label] for label in CLASSES],
    }

def decumulate_drought(row):
    """Decumulates USDM cumulative percentages into PMF categories."""
    pmf_d4 = row['D4']
    pmf_d3 = max(0.0, row['D3'] - row['D4'])
    pmf_d2 = max(0.0, row['D2'] - row['D3'])
    pmf_d1 = max(0.0, row['D1'] - row['D2'])
    pmf_d0 = max(0.0, row['D0'] - row['D1'])
    pmf_none = max(0.0, row['None'])
    return pd.Series([pmf_none, pmf_d0, pmf_d1, pmf_d2, pmf_d3, pmf_d4])

def engineer_features(df):
    """Reconstructs the full feature space matching Scenario 3 training scripts."""
    df_fe = df.copy()
    
    # 1. Weather Lags
    for lag in [1, 2, 4, 8]:
        df_fe[f'PREC_lag{lag}'] = df_fe.groupby('FIPS')['PRECTOTCORR'].shift(lag)
        df_fe[f'T2M_lag{lag}'] = df_fe.groupby('FIPS')['T2M'].shift(lag)
        df_fe[f'RH2M_lag{lag}'] = df_fe.groupby('FIPS')['RH2M'].shift(lag)

    # 2. Weather Rolling Statistics
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

    # 3. Seasonal features
    iso_week = df_fe['week_start'].dt.isocalendar().week.astype(int)
    df_fe['week_sin'] = np.sin(2 * np.pi * iso_week / 52.0)
    df_fe['week_cos'] = np.cos(2 * np.pi * iso_week / 52.0)

    # 4. Drought Lags
    for col in ['None', 'D0', 'D1', 'D2', 'D3', 'D4']:
        df_fe[f'{col}_lag1'] = df_fe.groupby('FIPS')[col].shift(1)
        df_fe[f'{col}_lag2'] = df_fe.groupby('FIPS')[col].shift(2)

    return df_fe


def prepare_feature_frames(region, selected_features):
    if not HAS_ML_LIBRARIES:
        raise ImportError("Sklearn tidak terinstall.")

    data_path = DATASET_PATHS[region]
    df = pd.read_csv(data_path)
    df['week_start'] = pd.to_datetime(df['week_start'])
    df['ValidEnd'] = pd.to_datetime(df['ValidEnd'])
    df = df.sort_values(['FIPS', 'week_start']).reset_index(drop=True)
    df[PMF_COLS] = df.apply(decumulate_drought, axis=1)
    df['Label'] = df[PMF_COLS].idxmax(axis=1).apply(lambda x: PMF_COLS.index(x))

    df_fe = engineer_features(df)
    df_fe = df_fe.dropna(subset=BASELINE_FEATURE_COLS + ['Label']).reset_index(drop=True)
    train_df = df_fe[df_fe['week_start'] <= TRAIN_END_DATE].copy()

    baseline_scaler = MinMaxScaler()
    baseline_scaler.fit(train_df[BASELINE_FEATURE_COLS])
    df_fe.loc[:, BASELINE_FEATURE_COLS] = baseline_scaler.transform(df_fe[BASELINE_FEATURE_COLS])

    final_scaler = MinMaxScaler()
    final_scaler.fit(train_df[selected_features])
    df_fe_final = df_fe.copy()
    # Meniru persis alur script training Scenario 3: scaler subset di-fit pada train_df
    # mentah, lalu diterapkan ke kolom selected_features dari df_fe yang sudah baseline-scaled.
    df_fe_final.loc[:, selected_features] = final_scaler.transform(df_fe[selected_features])
    return df_fe_final

def create_sequences_from_df(df_input, feature_columns, label_col, seq_length=52, id_col='FIPS', start_date=None):
    """Creates sequences for the LSTM model."""
    X, y = [], []
    for _, group in df_input.groupby(id_col):
        group = group.sort_values('week_start')
        feats = group[feature_columns].values
        labels = group[label_col].values
        dates = group['week_start'].values
        if len(group) < seq_length:
            continue
        for i in range(seq_length - 1, len(group)):
            target_date = pd.Timestamp(dates[i])
            if start_date is not None and target_date < pd.Timestamp(start_date):
                continue
            X.append(feats[i - seq_length + 1:i + 1])
            y.append(labels[i])
    return np.array(X), np.array(y)

def evaluate_model_dynamically(region, summary_data):
    """
    Loads data and model, runs predictions, and computes the confusion matrix dynamically.
    """
    if not HAS_ML_LIBRARIES:
        raise ImportError("Sklearn or TensorFlow tidak terinstall.")

    data_path = DATASET_PATHS[region]
    model_path = get_model_path(region)
    selected_features = summary_data['selected_features']
    class_multipliers = np.array(summary_data['class_multipliers'], dtype=float)
    seq_length = summary_data['seq_length']

    print(f"\n--- Memulai Evaluasi Dinamis untuk Wilayah {region.upper()} ---")
    print(f"Membaca data: {data_path}")
    df_fe = prepare_feature_frames(region, selected_features)
    
    # Create test sequences (2022-01-01 onwards)
    print("Membuat sekuens LSTM untuk data uji...")
    X_test_seq, y_test = create_sequences_from_df(
        df_fe, selected_features, 'Label', seq_length=seq_length, start_date=TEST_START_DATE
    )
    
    print(f"Memuat Model: {model_path}")
    # Load with compile=False to avoid missing custom loss issues
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print("Melakukan prediksi...")
    y_pred_prob = model.predict(X_test_seq, verbose=0)
    # Apply calibrated class multipliers
    y_pred = np.argmax(y_pred_prob * class_multipliers, axis=1)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(6)))
    support = np.bincount(y_test, minlength=len(CLASSES)).tolist()
    if support != summary_data['support']:
        raise ValueError(
            f"Support hasil evaluasi {support} tidak sama dengan summary {summary_data['support']}"
        )
    
    # Print metrics validation
    test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"Evaluasi Sukses! Test Macro-F1 = {test_f1:.4f}")
    
    return cm

def plot_validation_trend(kansas_data, nebraska_data):
    """
    Plots the bar chart of Validation Macro-F1 vs Feature Subset Size.
    """
    sizes = sorted(set(kansas_data['validation_trend']) | set(nebraska_data['validation_trend']))
    x = np.arange(len(sizes))
    bar_width = 0.34
    
    # Extract values in correct order
    ks_y = [kansas_data['validation_trend'][s] for s in sizes]
    ne_y = [nebraska_data['validation_trend'][s] for s in sizes]
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Segoe UI'],
        'text.color': '#222222',
        'axes.labelcolor': '#222222',
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })
    
    plt.figure(figsize=(10, 6), facecolor='#F8F9FA')
    ax = plt.subplot(1, 1, 1)
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='#CCCCCC')
    
    # Plot grouped bars
    ks_bars = ax.bar(
        x - bar_width / 2, ks_y, width=bar_width, color=REGION_COLORS['kansas'],
        edgecolor='white', linewidth=1.2, label='Kansas (Macro F1-Driven)', zorder=3
    )
    ne_bars = ax.bar(
        x + bar_width / 2, ne_y, width=bar_width, color=REGION_COLORS['nebraska'],
        edgecolor='white', linewidth=1.2, label='Nebraska (Macro F1-Driven)', zorder=3
    )
    
    # Annotate values
    for ks_bar, ne_bar, y_ks, y_ne in zip(ks_bars, ne_bars, ks_y, ne_y):
        ks_x = ks_bar.get_x() + ks_bar.get_width() / 2
        ne_x = ne_bar.get_x() + ne_bar.get_width() / 2
        ax.annotate(f"{y_ks:.4f}", (ks_x, y_ks), textcoords="offset points", xytext=(0, 10), 
                    ha='center', fontweight='bold', color='#01579B', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#E1F5FE", ec="none", alpha=0.8))
        ax.annotate(f"{y_ne:.4f}", (ne_x, y_ne), textcoords="offset points", xytext=(0, 10), 
                    ha='center', fontweight='bold', color='#E65100', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#FFE0B2", ec="none", alpha=0.8))
        
    # Highlight Best Subsets dynamically from summary output
    ks_best_idx = sizes.index(kansas_data['best_subset_size'])
    ks_bars[ks_best_idx].set_edgecolor('red')
    ks_bars[ks_best_idx].set_linewidth(2.5)
    ne_best_idx = sizes.index(nebraska_data['best_subset_size'])
    ne_bars[ne_best_idx].set_edgecolor('purple')
    ne_bars[ne_best_idx].set_linewidth(2.5)

    ax.bar([], [], color=REGION_COLORS['kansas'], edgecolor='red', linewidth=2.5,
           label=f"Kansas Best (Top-{kansas_data['best_subset_size']})")
    ax.bar([], [], color=REGION_COLORS['nebraska'], edgecolor='purple', linewidth=2.5,
           label=f"Nebraska Best (Top-{nebraska_data['best_subset_size']})")

    plt.title("Tren Performa Validation Macro-F1 vs Ukuran Subset Fitur\nSkenario 3 (Seleksi Fitur Berbasis Permutation Importance)", 
              fontsize=14, fontweight='bold', color='#1A237E', pad=15)
    plt.xlabel("Ukuran Subset Fitur (Top-K Features)", labelpad=10, fontweight='bold')
    plt.ylabel("Validation Macro F1-Score", labelpad=10, fontweight='bold')
    plt.xticks(x, sizes)
    plt.xlim(-0.7, len(sizes) - 0.3)
    plt.ylim(0.32, 0.88)
    plt.legend(loc='lower left', frameon=True, facecolor='#FFFFFF', edgecolor='#DDDDDD')
    
    # Save validation trend plot
    filename = "scenario3_validation_trend.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor=plt.gcf().get_facecolor())
    plt.close()
    print(f"Plot tren validasi berhasil disimpan: '{filename}'")

def plot_confusion_matrices(ks_cm, ne_cm, kansas_data, nebraska_data):
    """
    Plots the Confusion Matrix Heatmaps side-by-side.
    """
    labels = CLASSES
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.patch.set_facecolor('#F8F9FA')
    
    # Helper to generate beautiful annotation labels (counts + normalized percents)
    def get_cell_labels(cm):
        cm_norm = cm.astype(float) / np.maximum(1.0, cm.sum(axis=1, keepdims=True))
        cm_norm = np.nan_to_num(cm_norm)
        cell_labels = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cell_labels[i, j] = f"{cm[i,j]}\n({cm_norm[i,j]:.1%})"
        return cell_labels
    
    # Plot Kansas
    ks_labels = get_cell_labels(ks_cm)
    sns.heatmap(ks_cm, annot=ks_labels, fmt='', cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax1,
                annot_kws={"size": 10.5, "weight": "bold"}, linewidths=0.5, linecolor='#EEEEEE')
    ax1.set_title(
        f"Kansas Scenario 3 - Confusion Matrix (Test Data)\nBest Subset: Top-{kansas_data['best_subset_size']} Features",
        fontweight='bold', pad=12
    )
    ax1.set_xlabel('Predicted Class', fontweight='bold', labelpad=8)
    ax1.set_ylabel('Actual Class', fontweight='bold', labelpad=8)
    
    # Plot Nebraska
    ne_labels = get_cell_labels(ne_cm)
    sns.heatmap(ne_cm, annot=ne_labels, fmt='', cmap='Oranges', cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax2,
                annot_kws={"size": 10.5, "weight": "bold"}, linewidths=0.5, linecolor='#EEEEEE')
    ax2.set_title(
        f"Nebraska Scenario 3 - Confusion Matrix (Test Data)\nBest Subset: Top-{nebraska_data['best_subset_size']} Features",
        fontweight='bold', pad=12
    )
    ax2.set_xlabel('Predicted Class', fontweight='bold', labelpad=8)
    ax2.set_ylabel('Actual Class', fontweight='bold', labelpad=8)
    
    plt.suptitle("Heatmap Confusion Matrix Data Uji (Test Data) Skenario 3\nPerbandingan Berdampingan Wilayah Kansas dan Nebraska", 
                 fontsize=15, fontweight='bold', y=0.96, color='#1A237E')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
    filename = "scenario3_confusion_matrices.png"
    plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot confusion matrix berhasil disimpan: '{filename}'")

def main():
    # ------------------ 1. PARSE VALIDATION TREND ------------------
    print("Parsing Scenario 3 summaries...")
    kansas_data = parse_scenario3_summary(get_summary_path('kansas'))
    nebraska_data = parse_scenario3_summary(get_summary_path('nebraska'))

    print(f"Kansas Validation Macro-F1 Trend: {kansas_data['validation_trend']}")
    print(f"Nebraska Validation Macro-F1 Trend: {nebraska_data['validation_trend']}")

    plot_validation_trend(kansas_data, nebraska_data)

    if not HAS_ML_LIBRARIES:
        raise RuntimeError(
            "TensorFlow/Sklearn tidak tersedia di environment ini, sehingga confusion matrix tidak bisa dihitung tanpa hardcode."
        )

    ks_cm = evaluate_model_dynamically('kansas', kansas_data)
    ne_cm = evaluate_model_dynamically('nebraska', nebraska_data)
    plot_confusion_matrices(ks_cm, ne_cm, kansas_data, nebraska_data)

if __name__ == "__main__":
    main()
